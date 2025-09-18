import argparse
import torch
import os
import signal
from utils.evaluate import load_model, train_model
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

checkpoint_path = None

def save_checkpoint(model, tokenizer, save_path):
    """Save a checkpoint for the model and tokenizer."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    checkpoint_file = os.path.join(save_path, "checkpoint.pth")
    torch.save({'model_state_dict': model.state_dict(), 'tokenizer': tokenizer}, checkpoint_file)
    print(f"Checkpoint saved at {checkpoint_file}")

def signal_handler(sig, frame):
    """Handle signals for graceful termination."""
    if checkpoint_path and model and tokenizer:
        print("Signal received, saving checkpoint...")
        save_checkpoint(model, tokenizer, checkpoint_path)
    exit(0)

# Register signal handlers for SLURM time limit or manual interruption
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def main(model_name, save_path, loss_strategy, masking_strategy, dataset_path,  mask_prob, precomputed_csv_path=None):   
    global checkpoint_path, model, tokenizer
    checkpoint_path = save_path  # Set global checkpoint path
    
    try:
        # Check if a checkpoint exists
        checkpoint_file = os.path.join(save_path, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            print(f"Loading checkpoint from {checkpoint_file}...")
            checkpoint = torch.load(checkpoint_file)
            model, tokenizer = load_model(model_name)
            model.load_state_dict(checkpoint['model_state_dict'])
            tokenizer = checkpoint['tokenizer']
            print("Checkpoint loaded successfully.")
        else:
            print("No checkpoint found, loading model from scratch.")
            model, tokenizer = load_model(model_name)
        
        # Ensure save_path exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Train the model
        train_model(model, tokenizer, loss_strategy=loss_strategy, 
                    masking_strategy=masking_strategy, save_path=save_path, 
                    dataset_path=dataset_path, mask_prob=mask_prob, precomputed_csv_path=precomputed_csv_path)
        
        # Save the final model and tokenizer
        save_checkpoint(model, tokenizer, save_path)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        if model and tokenizer:
            print("Saving checkpoint due to error...")
            save_checkpoint(model, tokenizer, save_path)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and fine-tune a model with a specific dataset.")
    parser.add_argument("--model_name", type=str, required=True, 
                        choices=['roberta-large','roberta-base', 'bert-large', 'bert-base','electra-generator', 'electra-discriminator'],  
                        help="Model Name.")
    parser.add_argument("--dataset_path", type=str, required=True,  
                        help="Path to your dataset.")
    parser.add_argument("--masking_strategy", type=str, required=True, 
                        choices=['PMI', 'LDA', 'BERTopic'],  
                        help="Masking Strategy.")
    parser.add_argument("--loss_strategy", type=str, default='weighted', 
                        choices=['weighted', 'none'],  
                        help="Loss Strategy.")
    parser.add_argument("--save_path", type=str, required=True, 
                        help="Path to save the pretrained model and tokenizer.")
    parser.add_argument("--precomputed_csv", type=str, default=None, 
                    help="Path to the precomputed CSV file containing PMI scores.")
    parser.add_argument("--mask_prob", type=float, default=0.15, 
                        help="Probability of masking a token.")
    args = parser.parse_args()
    main(model_name=args.model_name, save_path=args.save_path,
         loss_strategy=args.loss_strategy, masking_strategy=args.masking_strategy, 
         dataset_path=args.dataset_path, mask_prob=args.mask_prob, precomputed_csv_path=args.precomputed_csv)






