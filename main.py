import argparse
import torch
import os
from utils.evaluate import train_model, finetune_model, finetune_supervised_classifier
import torch.nn.functional as F

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(description="Train and fine-tune a model with a specific dataset.")
    parser.add_argument("--experiment_name", type=str, required=True, choices=['finetune_nli','supervised_classification', 'train'])
    parser.add_argument("--model_name", type=str, choices=['roberta', 'bert', 'electra'],  help="Model Name.")
    parser.add_argument("--pretrained_model_path", type=str, help="Path to the pretrained model.")
    parser.add_argument("--dataset_path", type=str, help="Path to your dataset.")
    parser.add_argument("--masking_strategy", default='PMI' , type=str, choices=['PMI', 'LDA', 'BERTopic'],  help="Masking Strategy.")
    parser.add_argument("--loss_strategy", type=str, default='weighted', choices=['weighted', 'none'],  help="Masking Strategy.")
    parser.add_argument("--nli_dataset_name", type=str,  choices=['mnli', 'qnli', 'snli'], help="Dataset to be used for fine-tuning (e.g., 'mnli', 'qnli', 'snli').")
    parser.add_argument("--save_path", default='./Trained_models/' ,  type=str, help="Path to save the pretrained model and tokenizer.")
    
    args = parser.parse_args()

    if args.experiment_name == 'finetune_nli':
        finetune_supervised_classifier(args.model_path, save_path=args.save_path,  dataset_name=args.dataset_name, output_dir='./output', logging_dir='./logs')
    
    elif args.experiment_name == 'supervised_classification':
        finetune_model(args.pretrained_model_path, save_path=args.save_path,  dataset_path=args.dataset_path)

    elif args.experiment_name == 'train':
        train_model(args.model_name, loss_strategy=args.loss_strategy, masking_strategy=args.masking_strategy, save_path=args.save_path, dataset_path=args.dataset_path)