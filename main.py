import argparse
import torch
import os
from utils.evaluate import load_model, train_model
import torch.nn.functional as F

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()

def main(model_name,save_path,loss_strategy,masking_strategy, dataset_path):   
    model, tokenizer = load_model(model_name)
    train_model(model, tokenizer, loss_strategy=loss_strategy, masking_strategy=masking_strategy, save_path=save_path, dataset_path=dataset_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and fine-tune a model with a specific dataset.")
    parser.add_argument("--model_name", type=str, required=True, choices=['roberta', 'bert', 'electra'],  help="Model Name.")
    parser.add_argument("--dataset_path", type=str,required=True,  help="Path to your dataset.")
    parser.add_argument("--masking_strategy", type=str, required=True, choices=['PMI', 'LDA', 'BERTopic'],  help="Masking Strategy.")
    parser.add_argument("--loss_strategy", type=str, default='weighted', choices=['weighted',],  help="Masking Strategy.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the pretrained model and tokenizer.")
    args = parser.parse_args()
    main(model_name=args.model_name, save_path=args.save_path,loss_strategy=args.loss_strategy, masking_strategy=args.masking_strategy, dataset_path=args.dataset_path)