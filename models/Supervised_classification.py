import torch
import argparse
import pandas as pd
from sklearn.metrics import  f1_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import  AutoTokenizer
from transformers import Trainer, TrainingArguments,AutoModelForSequenceClassification
from datasets import Dataset
import numpy as np
import os

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.empty_cache()

def run_experiment(model_path, data_path):
    torch.cuda.empty_cache()
    SEED = 42
    train_percentage = 0.8
    validation_percentage = 0.1
    test_percentage = 0.1

    
    df = pd.read_csv(data_path)
    distinct_count = df['class'].nunique()

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=distinct_count) 
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    X = df.text.values
    y = pd.get_dummies(df["class"])
    labels = list(y.keys())
    y = y.values 

    X_train, X_rest, y_train , y_rest = train_test_split(X, y, test_size = 1 - train_percentage, train_size = train_percentage, random_state = SEED, shuffle = True, stratify = y)
    X_valid, X_test, y_valid , y_test = train_test_split(X_rest, y_rest, test_size = test_percentage / (validation_percentage + test_percentage), train_size = validation_percentage / (validation_percentage + test_percentage), random_state = SEED, shuffle = True, stratify = y_rest)
        
    y_train = np.argmax(y_train, axis=1)
    y_valid = np.argmax(y_valid, axis=1)

    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=512)
    valid_encodings = tokenizer(X_valid.tolist(), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=512)

    # Create Hugging Face datasets
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': y_train
    })

    eval_dataset = Dataset.from_dict({
        'input_ids': valid_encodings['input_ids'],
        'attention_mask': valid_encodings['attention_mask'],
        'labels': y_valid
    })
    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': np.argmax(y_test, axis=1) 
    })

    training_args = TrainingArguments(
        output_dir='./outputs',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
        eval_strategy="epoch",
        # eval_steps=100,
        logging_dir='./logs',
        logging_steps=100,
        fp16=True,
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    # Run inference (get predictions)
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    print("Classification Report:")
    print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=labels))
    print("f1_score:", f1_score(np.argmax(y_test, axis=1), y_pred, average='macro'))

def main(model_path, data_path):
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if os.path.isfile(file_path) and file_path.endswith('.csv'):
            print(f"Processing CSV file: {filename}")
            run_experiment(model_path, file_path)
        else:
            print(f"Skipping non-CSV file: {file_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train and fine-tune a model with a specific dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to save the training datasets.")
    args = parser.parse_args()

    main(model_path=args.model_path, data_path=args.data_path)