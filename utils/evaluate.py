from transformers import Trainer
from sklearn.metrics import accuracy_score
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments
from Data.data_preparation import MLM_Dataset
from Data.Extract_NPMI import NPMI
import torch.nn.functional as F
from itertools import product
from datasets import load_dataset
import pandas as pd
import torch
import os

class WeightedLossTrainer(Trainer):
    def __init__(self, importance_scores, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.importance_scores = importance_scores

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Ensure the logits are of the correct shape (batch_size, sequence_length, vocab_size)
        vocab_size = logits.size(-1)
        # Flatten the logits and labels for compatibility with F.nll_loss
        log_probs = F.log_softmax(logits.view(-1, vocab_size), dim=-1)
        # Calculate NLL loss (Negative Log Likelihood) for MLM task
        masked_lm_loss = F.nll_loss(
            log_probs,
            labels.view(-1),
            reduction='none',  # Compute loss for each token separately
            ignore_index=-100 
        )
        # Apply importance weights to masked tokens 
        importance_weights = self.get_importance_weights(inputs['input_ids'])
        # Mask out the positions where label is -100
        active_loss = labels.view(-1) != -100
        active_loss = active_loss.float()
        # Apply the importance weights and mask
        weighted_loss = (masked_lm_loss * importance_weights.view(-1) * active_loss).sum() / active_loss.sum()
        return (weighted_loss, outputs) if return_outputs else weighted_loss

    def get_importance_weights(self, input_ids):
        # Align tokens with their importance scores
        importance_weights = torch.ones_like(input_ids, dtype=torch.float)
        for i, input_id_sequence in enumerate(input_ids):
            tokens = self.tokenizer.convert_ids_to_tokens(input_id_sequence)
            for j, token in enumerate(tokens):
                if token in self.importance_scores:
                    importance_weights[i, j] = self.importance_scores[token]
        return importance_weights
    from sklearn.metrics import accuracy_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Convert logits to predictions (choose the class with highest probability)
        predictions = logits.argmax(dim=-1)
        # Compute accuracy using sklearn's accuracy_score
        accuracy = accuracy_score(labels.flatten(), predictions.flatten())
        return {"accuracy": accuracy}

### Methods:
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
    elif model_name == 'electra':
        model_name = 'google/electra-base-discriminator'
    else:
        raise ValueError(f"Model '{model_name}' not supported")
    
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, pad_to_multiple_of=8)
    return model, tokenizer

## Save a trained model and tokenizer:
def save_model(model, tokenizer, output_path):
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

## Load a dataset and PMI importance scores:
def load_data(masking_strategy, tokenizer, dataset_path):
    df = get_csv(dataset_path)
    # df = pd.read_csv(dataset_path)
    if masking_strategy == 'PMI':
        npmi_calculator = NPMI()
        Importance_scores_df = npmi_calculator.extract_npmi_for_dataset(df, text_column='text', class_column='class')
    # elif masking_strategy == 'LDA':
    #     Masks = LDA_masks
    # elif masking_strategy == 'BERTopic':
    #     Masks = BERTopic_masks
    else:
        raise ValueError(f"Masking strategy '{masking_strategy}' not supported")
    
    mlm_dataset = MLM_Dataset(dataframe=df, tokenizer=tokenizer, mask_prob=0.15)
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

## Train a model:
def train_model(model, tokenizer, loss_strategy, masking_strategy, save_path, dataset_path):
    # Load dataset and PMI importance scores
    # dataset, PMI_df = load_data(masking_strategy, tokenizer, dataset_path='/data/ARENAS_Automatic_Extremist_Analysis/ARENAS_Automatic_Extremist_Analysis/Data/SUD_data/fox_all.csv')
    dataset, PMI_df = load_data(masking_strategy, tokenizer, dataset_path)
    
    # Convert PMI_df into a dictionary for fast lookup during training
    importance_scores = {row['token']: row['npmi'] for _, row in PMI_df.iterrows()}

    # Define parameter grid
    param_grid = {
        'learning_rate': [5e-5, 2e-5, 1e-5],
        'weight_decay': [0.1, 0.01, 0.001],
        'batch_size': [8, 16],
        # 'num_epochs': [3, 5, 10]
        'num_epochs': [1]
    }

    # Track the best model and score
    best_model = None
    best_score = float('-inf')
    best_params = {}

    # Iterate over all parameter combinations
    for lr, wd, batch_size, epochs in product(param_grid['learning_rate'], param_grid['weight_decay'], param_grid['batch_size'], param_grid['num_epochs']):
        # Update the training arguments
        training_args = TrainingArguments(
            output_dir='./output',
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=10_000,
            save_total_limit=2,
            eval_strategy="epoch",
            logging_dir='./logs',
            logging_steps=100,
            fp16=True,
            learning_rate=lr,
            weight_decay=wd,
            metric_for_best_model="accuracy",
        )
        
        # Choose trainer based on loss strategy
        if loss_strategy == 'weighted':
            trainer = WeightedLossTrainer(
                importance_scores=importance_scores,
                model=model,
                args=training_args,
                train_dataset=dataset['train'],
                eval_dataset=dataset['test'],
                tokenizer=tokenizer
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset['train'],
                eval_dataset=dataset['test'] 
            )

        # Train the model
        trainer.train()

        # Evaluate the model on the validation set
        eval_result = trainer.evaluate()
        print("eval_result : ", eval_result)
        
        best_loss = float('inf')  # Initialize the best loss to infinity
        if eval_result['eval_loss'] < best_loss:
            best_loss = eval_result['eval_loss']
            best_model = model
            best_params = {
                'learning_rate': lr,
                'weight_decay': wd,
                'batch_size': batch_size,
                'num_epochs': epochs
            }
            # Save the best model
    print(f"Best Params: {best_params}")
    print(f"Best Loss: {best_loss}")
    save_model(best_model, tokenizer, save_path)