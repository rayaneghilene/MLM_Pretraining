import argparse
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import torch
import os
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments
from Datasets_with_masks import PMI_BERT_masks, LDA_masks, PMI_roBERTa_masks, LDA_masks, BERTopic_masks
import random

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()

def load_masks(masking_strategy='PMI'):
    if masking_strategy == 'PMI':
        Masks = PMI_roBERTa_masks
    elif masking_strategy == 'LDA':
        Masks = LDA_masks
    elif masking_strategy == 'BERTopic':
        Masks = BERTopic_masks
    else:
        raise ValueError(f"Masking strategy '{model_name}' not supported")

    return Masks
def load_model(model_name='roberta'):
    if model_name == 'roberta':
        model_name = 'FacebookAI/roberta-large'
    elif model_name == 'bert':
        model_name = 'bert-base-uncased'
    elif model_name == 'electra':
        model_name = 'google/electra-base-discriminator'
    else:
        raise ValueError(f"Model '{model_name}' not supported")
    
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def mask_words_by_class(text, tokenizer, class_label, class_masks, mask_prob=0.15, mask = '<mask>'):
    tokens = tokenizer.tokenize(text)
    num_tokens = len(tokens)
    required_num_masks = max(1, int(mask_prob * num_tokens))  # Ensure at least 15% of tokens are masked
    
    if class_label in class_masks:
        word_set = set(class_masks[class_label])
        
        # Mask tokens based on the class label
        masked_tokens = [mask if token in word_set else token for token in tokens]
        Tokens_to_predict = [token for token in tokens if token in word_set]
        
        # If not enough tokens were masked, randomly mask additional tokens
        current_num_masks = len(Tokens_to_predict)
        if current_num_masks < required_num_masks:
            remaining_mask_count = required_num_masks - current_num_masks
            # Get the indices of tokens that are not yet masked
            unmasked_indices = [i for i, token in enumerate(masked_tokens) if token != mask]
            random_indices = random.sample(unmasked_indices, remaining_mask_count)
            
            for idx in random_indices:
                Tokens_to_predict.append(tokens[idx])
                masked_tokens[idx] = mask
        
        masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
        return masked_text, Tokens_to_predict
    else:
        # If class_label is not in class_masks, apply random masking to 15% of the tokens
        num_to_mask = required_num_masks
        mask_indices = random.sample(range(num_tokens), num_to_mask)
        
        masked_tokens = tokens.copy()
        Tokens_to_predict = []
        
        for idx in mask_indices:
            # Mask the token and add to Tokens_to_predict
            Tokens_to_predict.append(tokens[idx])
            masked_tokens[idx] = mask
        
        masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
        
        return masked_text, Tokens_to_predict

def preprocess_function(examples, tokenizer):
    # Tokenize the masked text (input to the model)
    inputs = tokenizer(examples['masked_text'], padding='max_length', truncation=True, max_length=128, return_tensors='pt')

    # Initialize labels tensor with -100 (to ignore)
    labels_ids = []
    
    for masked_text, tokens_to_predict in zip(examples['masked_text'], examples['tokens_to_predict']):
        # Tokenize the masked text
        input_ids = tokenizer(masked_text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')['input_ids'][0]

        # Convert tokens_to_predict to token IDs
        token_ids = tokenizer.convert_tokens_to_ids(tokens_to_predict)
        
        # Create labels where only masked positions have values
        label_ids = [-100] * len(input_ids)
        for token_id in token_ids:
            if token_id in input_ids:
                mask_index = (input_ids == token_id).nonzero(as_tuple=True)[0].tolist()
                if mask_index:
                    label_ids[mask_index[0]] = token_id
        
        labels_ids.append(label_ids)

    inputs['labels'] = torch.tensor(labels_ids)
    return inputs

def get_dataset(test_datasets, tokenizer):
    SEED = 42
    train_percentage = 0.8
    validation_percentage = 0.1
    test_percentage = 0.1
    df = pd.DataFrame()

    for test_dataset in test_datasets:
        # print(test_dataset)
        test_data = pd.read_csv(test_dataset['data_path'])

        class_masks = test_dataset['masks']
        # print("The class masks are: ", class_masks)

        test_data[['masked_text', 'tokens_to_predict']] = test_data.apply(
            lambda row: pd.Series(mask_words_by_class(text=row['text'], tokenizer=tokenizer, class_label=row['class'], class_masks=class_masks)), axis=1
        )

        df = pd.concat([df, test_data], ignore_index=True)

    print(len(df))
    X = df.masked_text.values
    y_tokens_to_predict = df.tokens_to_predict.values

    # Split the data
    X_train, X_rest, y_tokens_train, y_tokens_rest = train_test_split(X, y_tokens_to_predict, test_size=1 - train_percentage, train_size=train_percentage, random_state=SEED, shuffle=True)
    X_valid, X_test, y_tokens_valid, y_tokens_test = train_test_split(X_rest, y_tokens_rest, test_size=test_percentage / (validation_percentage + test_percentage), train_size=validation_percentage / (validation_percentage + test_percentage), random_state=SEED, shuffle=True)

    # Create datasets
    train_dataset = Dataset.from_dict({'masked_text': X_train, 'tokens_to_predict': y_tokens_train})
    valid_dataset = Dataset.from_dict({'masked_text': X_valid, 'tokens_to_predict': y_tokens_valid})
    test_dataset = Dataset.from_dict({'masked_text': X_test, 'tokens_to_predict': y_tokens_test})

    # Apply preprocess_function to datasets
    # train_dataset = train_dataset.map(preprocess_function, batched=True)
    # valid_dataset = valid_dataset.map(preprocess_function, batched=True)
    # test_dataset = test_dataset.map(preprocess_function, batched=True)

    train_dataset = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    valid_dataset = valid_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

    return train_dataset, valid_dataset, test_dataset

def save_model(model, tokenizer, output_path):
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

def train_model(model, tokenizer, Masks, output_path, output_dir='./output', logging_dir='./logs'):
    train_dataset, valid_dataset, _ = get_dataset(Masks, tokenizer)

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
        eval_dataset=valid_dataset 
    )

    # Train the model
    trainer.train()
    
    save_model(model, tokenizer, output_path)

def main(model_name,output_path, Masks):   
    model, tokenizer = load_model(model_name)
    train_model(model, tokenizer, Masks=Masks, output_path=output_path)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train and fine-tune a model with a specific dataset.")
    parser.add_argument("--model_name", type=str, required=True, choices=['roberta', 'bert', 'electra'],  help="Model Name.")
    parser.add_argument("--masking_strategy", type=str, required=True, choices=['PMI', 'LDA', 'BERTopic'],  help="Masking Strategy.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the pretrained model and tokenizer.")
    args = parser.parse_args()
    Masks = load_masks(args.masking_strategy)
    main(model_name=args.model_name, output_path=args.output_path, Masks=Masks)