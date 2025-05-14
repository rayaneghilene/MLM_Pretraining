import os
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from Data.data_preparation import MLM_Dataset
from Data.Extract_NPMI import NPMI
from Data.Extract_BERTopic import BERTopicProcessor


def get_csv(path):
    try:
        # Check if the path is a file
        if os.path.isfile(path):
            # Check if it's a CSV file
            if path.endswith('.csv'):
                print(f"Processing single CSV file: {path}")
                df = pd.read_csv(path)
                return df
            else:
                raise ValueError("The provided file is not a CSV.")
        
        # Check if the path is a directory
        elif os.path.isdir(path):
            print(f"Processing directory: {path}")
            all_dfs = []  
            # Iterate over files in the folder
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path) and file_path.endswith('.csv'):
                    print(f"Processing CSV file: {file_path}")
                    df = pd.read_csv(file_path)
                    all_dfs.append(df)  # Append each dataframe to the list
                else:
                    print(f"Skipping non-CSV file: {file_path}")

            # Merge all dataframes into one
            if all_dfs:
                merged_df = pd.concat(all_dfs, ignore_index=True)
                print(f"All CSV files merged into a single DataFrame with {len(merged_df)} rows.")
                return merged_df
            else:
                raise ValueError("No valid CSV files found in the directory.")

        else:
            raise FileNotFoundError(f"Invalid path: {path}")
    
    except Exception as e:
        print(f"Error: {e}")

## load a trained model and tokenizer:
def load_model(model_name='roberta'):
    if model_name == 'roberta':
        model_name = 'FacebookAI/roberta-large'
    elif model_name == 'bert':
        model_name = 'bert-base-uncased'
    # elif model_name == 'electra-discriminator':
    #     model_name = 'google/electra-base-discriminator'
    # elif model_name == 'electra-generator':
    #     model_name = 'google/electra-base-generator'
    else:
        raise ValueError(f"Model '{model_name}' not supported")
    
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, pad_to_multiple_of=8)
    return model, tokenizer

## Save a trained model and tokenizer:
def save_model(model, tokenizer, output_path):
    model.save_pretrained(output_path, safe_serialization=False)
    tokenizer.save_pretrained(output_path)


## Load a dataset and PMI importance scores:
def load_data(masking_strategy, tokenizer, dataset_path):
    print(f"Loading dataset from {dataset_path}...")
    df = get_csv(dataset_path)
    # df = pd.read_csv(dataset_path)
    if masking_strategy == 'PMI':
        npmi_calculator = NPMI()
        Importance_scores_df = npmi_calculator.extract_npmi_for_dataset(df, text_column='text', class_column='class')
        Importance_scores_df.rename(columns={'npmi': 'score'}, inplace=True)
    elif masking_strategy == 'BERTopic':
        BERTOPIC_Calculator = BERTopicProcessor()
        Importance_scores_df = BERTOPIC_Calculator.process(df)
        Importance_scores_df.rename(columns={'avg_score': 'score'}, inplace=True)

    else:
        raise ValueError(f"Masking strategy '{masking_strategy}' not supported")
    
    mlm_dataset = MLM_Dataset(dataframe=df, tokenizer=tokenizer, mask_prob=0.15)#, importance_range=(0.5, 1))
    dataset_dict = mlm_dataset.get_dataset()
    return dataset_dict, Importance_scores_df

## Load and tokenize a dataset for NLI:
def load_and_tokenize_nli_dataset(tokenizer, dataset_name='mnli'):
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