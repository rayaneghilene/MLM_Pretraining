import os
import torch
import argparse
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()

# model_path = "/data/ARENAS_Automatic_Extremist_Analysis/ARENAS_Automatic_Extremist_Analysis/ZSL/roBERTa/roberta_large_cetautomatix"

def load_and_tokenize_dataset(tokenizer, dataset_name='mnli'):
    try:
        if dataset_name == 'mnli':
            dataset = load_dataset('glue', 'mnli')
            train_dataset = dataset['train'] 
            eval_dataset = dataset['validation_matched']
        elif dataset_name == 'qnli':
            dataset = load_dataset('glue', 'qnli')
            train_dataset = dataset['train'] 
            eval_dataset = dataset['validation']
        elif dataset_name == 'snli':
            dataset = load_dataset('snli')
            train_dataset = dataset['train'] 
            eval_dataset = dataset['validation']
        else:
            raise ValueError(f"Dataset '{dataset_name}' not supported")

        # Tokenization and formatting
        train_dataset = train_dataset.map(lambda examples: tokenizer(examples['premise'], examples['hypothesis'], 
                                                         truncation=True, padding='max_length', max_length=128), 
                              batched=True)
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        eval_dataset = eval_dataset.map(lambda examples: tokenizer(examples['premise'], examples['hypothesis'], 
                                                         truncation=True, padding='max_length', max_length=128), 
                              batched=True)
        eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        return train_dataset, eval_dataset
    
    except Exception as e:
        print(f"Dataset '{dataset_name}' unavailable or could not be loaded: {e}")
        return None

def save_model(model, tokenizer, output_path):
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

from transformers import Trainer, TrainingArguments

def train_model(model, tokenizer, dataset_name, output_path, output_dir='./output', logging_dir='./logs'):
    train_dataset, eval_dataset = load_and_tokenize_dataset(tokenizer, dataset_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
        eval_strategy="steps",  ## Can change to "epoch" if needed
        eval_steps=1000,
        logging_dir=logging_dir,
        logging_steps=100,
        fp16=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        metric_for_best_model="accuracy",
    )

    # Set up the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset 
    )

    # Train the model
    trainer.train()
    
    save_model(model, tokenizer, output_path)

def main(model_path, output_path, dataset_name):
    ## load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3) 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ## fine tune the model
    train_model(model=model, tokenizer=tokenizer, output_path=output_path,  dataset_name=dataset_name, output_dir='./output', logging_dir='./logs')

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train and fine-tune a model with a specific dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the fine-tuned model and tokenizer.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=['mnli', 'qnli', 'snli'], help="Dataset to be used for fine-tuning (e.g., 'mnli', 'qnli', 'snli').")
    args = parser.parse_args()

    main(model_path=args.model_path, output_path=args.output_path, dataset_name=args.dataset_name)