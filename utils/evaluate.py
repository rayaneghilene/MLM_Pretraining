from transformers import Trainer
from sklearn.metrics import accuracy_score
from sklearn.metrics import  f1_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from Data.data_preparation import MLM_Dataset
from Data.Extract_NPMI import NPMI
import torch.nn.functional as F
from itertools import product
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import torch
import os
from transformers import ElectraForMaskedLM, ElectraForPreTraining, AdamW, AutoTokenizer
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
def train_model(model_name, loss_strategy, masking_strategy, save_path, dataset_path):

    # Define parameter grid
    param_grid = {'learning_rate': [5e-5, 2e-5, 1e-5],
                  'weight_decay': [0.1, 0.01, 0.001],
                  'batch_size': [64, 128],
                  'num_epochs': [3, 5],}
    ## For quick testing use the followign param_grid:
    # param_grid = {'learning_rate': [2e-5],
    #               'weight_decay': [0.1],
    #               'batch_size': [64],
    #               'num_epochs': [3],}
    # Track the best model and score
    best_model = None
    best_params = {}

    
    if model_name == 'electra':
        generator = ElectraForMaskedLM.from_pretrained("google/electra-small-generator")
        generator.to(device)
        discriminator = ElectraForPreTraining.from_pretrained("google/electra-small-discriminator")
        discriminator.to(device)
        tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
        dataset_path = '/data/ARENAS_Automatic_Extremist_Analysis/ARENAS_Automatic_Extremist_Analysis/PMI_extraction_test/test_datasets'
        dataset, PMI_df = load_data('PMI', tokenizer, dataset_path)
        # importance_scores = {row['token']: row['npmi'] for _, row in PMI_df.iterrows()}
        importance_scores = dict(zip(PMI_df['token'], PMI_df['npmi']))
        train_dataset = dataset['train']
        val_dataset = dataset['test']
        for lr, batch_size, epochs in product(param_grid['learning_rate'], param_grid['batch_size'], param_grid['num_epochs']):
            optimizer_G = AdamW(generator.parameters(), lr=5e-5)
            optimizer_D = AdamW(discriminator.parameters(), lr=5e-5)

            mlm_loss_fn = torch.nn.CrossEntropyLoss()  
            replaced_loss_fn = torch.nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss

            ## Training Loop
            train_indices = list(range(len(train_dataset)))

            # Training Loop
            for epoch in range(epochs):
                generator.train()
                discriminator.train()
                total_generator_loss = 0.0
                total_discriminator_loss = 0.0
                num_batches = 0

                for i in range(0, len(train_dataset), batch_size):
                    # Prepare batch
                    batch_indices = train_indices[i:i + batch_size]
                    batch = [train_dataset[idx] for idx in batch_indices]
                    input_ids = [item["input_ids"] for item in batch]
                    attention_mask = [item["attention_mask"] for item in batch]
                    labels = [item["labels"] for item in batch]

                    max_length = max(len(ids) for ids in input_ids)
                    input_ids = [ids + [0] * (max_length - len(ids)) for ids in input_ids]
                    attention_mask = [mask + [0] * (max_length - len(mask)) for mask in attention_mask]
                    labels = [label + [0] * (max_length - len(label)) for label in labels]

                    input_ids = torch.tensor(input_ids).to(device)
                    attention_mask = torch.tensor(attention_mask).to(device)
                    labels = torch.tensor(labels).to(device)

                    # Train the generator
                    optimizer_G.zero_grad()
                    generator_outputs = generator(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    ## If you're reading this, thank you for reviewing the code!
                    if loss_strategy == 'weighted':
                        generator_loss = compute_weighted_loss(
                            logits=generator_outputs.logits,
                            labels=labels,
                            input_ids=input_ids,
                            tokenizer=tokenizer,
                            importance_scores=importance_scores  # Pass the dictionary
                        )
                    elif loss_strategy == 'none':
                        generator_loss = generator_outputs.loss
                    generator_loss.backward()
                    optimizer_G.step()

                    # Train the discriminator
                    with torch.no_grad():
                        predictions = generator_outputs.logits.argmax(dim=-1)
                        replaced_input_ids = input_ids.clone()
                        masked_positions = labels != -100
                        replaced_input_ids[masked_positions] = predictions[masked_positions]

                    optimizer_D.zero_grad()
                    discriminator_labels = (input_ids != replaced_input_ids).float()
                    discriminator_outputs = discriminator(input_ids=replaced_input_ids, attention_mask=attention_mask)       
                    discriminator_loss = F.binary_cross_entropy_with_logits(
                        discriminator_outputs.logits.squeeze(-1), discriminator_labels
                    )
                    discriminator_loss.backward()
                    optimizer_D.step()

                    # Accumulate losses
                    total_generator_loss += generator_loss.item()
                    total_discriminator_loss += discriminator_loss.item()
                    num_batches += 1
                    print(f"Epoch {epoch + 1} | Eval Generator Loss: {generator_loss.item():.4f} | Eval Discriminator Loss: {discriminator_loss.item():.4f}")
                avg_generator_loss = total_generator_loss / num_batches
                avg_discriminator_loss = total_discriminator_loss / num_batches

                print(f"Epoch {epoch + 1} | Final Generator Loss: {avg_generator_loss:.4f} | Final Discriminator Loss: {avg_discriminator_loss:.4f}")
            ### Eval loop
            val_indices = list(range(len(val_dataset)))

            generator.eval()
            discriminator.eval()

            total_generator_loss = 0.0
            total_discriminator_loss = 0.0
            num_batches = 0

            with torch.no_grad():
                for i in range(0, len(val_dataset), batch_size):
                    batch_indices = val_indices[i:i + batch_size]
                    batch = [val_dataset[idx] for idx in batch_indices]
                    input_ids = [item["input_ids"] for item in batch]
                    attention_mask = [item["attention_mask"] for item in batch]
                    labels = [item["labels"] for item in batch]

                    # Padding and tensor conversion
                    max_length = max([len(ids) for ids in input_ids])
                    input_ids = [ids + [0] * (max_length - len(ids)) for ids in input_ids]
                    attention_mask = [mask + [0] * (max_length - len(mask)) for mask in attention_mask]
                    labels = [label + [0] * (max_length - len(label)) for label in labels]

                    input_ids = torch.tensor(input_ids).to(device)
                    attention_mask = torch.tensor(attention_mask).to(device)
                    labels = torch.tensor(labels).to(device)

                    # Generator evaluation
                    generator_outputs = generator(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    generator_loss = generator_outputs.loss

                    # Replace masked tokens
                    predictions = generator_outputs.logits.argmax(dim=-1)
                    replaced_input_ids = input_ids.clone()
                    masked_positions = labels != -100
                    replaced_input_ids[masked_positions] = predictions[masked_positions]

                    # Discriminator evaluation
                    discriminator_labels = (input_ids != replaced_input_ids).float()
                    discriminator_outputs = discriminator(input_ids=replaced_input_ids, attention_mask=attention_mask)
                    discriminator_loss = replaced_loss_fn(discriminator_outputs.logits.squeeze(-1), discriminator_labels)
                    print(f"Epoch {epoch + 1} | Eval Generator Loss: {generator_loss.item():.4f} | Eval Discriminator Loss: {discriminator_loss.item():.4f}")
                    # Accumulate losses
                    total_generator_loss += generator_loss.item()
                    total_discriminator_loss += discriminator_loss.item()
                    num_batches += 1
                    
            avg_generator_loss = total_generator_loss / num_batches
            avg_discriminator_loss = total_discriminator_loss / num_batches

            print(f"Validation | Final Generator Loss: {avg_generator_loss:.4f} | Final Discriminator Loss: {avg_discriminator_loss:.4f}")
            avg_combined_eval_loss = (avg_generator_loss + avg_discriminator_loss) / 2
            print(f"Validation | Avg Combined eval Loss: {avg_combined_eval_loss:.4f}")

            best_loss = float('inf')  # Initialize the best loss to infinity
            if avg_combined_eval_loss < best_loss:
                best_loss = avg_combined_eval_loss
                best_model = discriminator
                best_params = {
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'num_epochs': epochs
                }
            print(f"Model trained with LR={lr}, BS={batch_size}, Epochs={epochs}.")
        # Save the best model
        print(f"Best Params: {best_params}")
        print(f"Best Loss: {best_loss}")
        model_save_path = save_path + f"_lr_{best_params['learning_rate']}_bs_{best_params['batch_size']}_epochs_{best_params['num_epochs']}"
        save_model(best_model, tokenizer, model_save_path)
    else:
        # Iterate over all parameter combinations
        model, tokenizer = load_model(model_name)
        # Load dataset and PMI importance scores
        dataset, PMI_df = load_data(masking_strategy, tokenizer, dataset_path)
        # Convert PMI_df into a dictionary for fast lookup during training
        importance_scores = {row['token']: row['npmi'] for _, row in PMI_df.iterrows()}
        
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
                # save_only_model=True,
                save_safetensors=False,
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
            elif loss_strategy == 'none':
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
            print(f"Model trained with LR={lr}, WD={wd}, BS={batch_size}, Epochs={epochs}.")
            print(f"Validation Loss: {eval_result['eval_loss']}")
        # Save the best model
        print(f"Best Params: {best_params}")
        print(f"Best Loss: {best_loss}")
        model_save_path = save_path + model_name + f"_lr_{best_params['learning_rate']}_wd_{best_params['weight_decay']}_bs_{best_params['batch_size']}_epochs_{best_params['num_epochs']}"
        save_model(best_model, tokenizer, model_save_path)

    
def finetune_model(model_path, dataset_name, save_path, output_dir='./output', logging_dir='./logs'):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3) 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_dataset, eval_dataset = load_and_tokenize_nli_dataset(tokenizer, dataset_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
    overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
        eval_strategy="epoch",
        save_only_model=True,
    
        save_safetensors=False,
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
        
    save_model(model, tokenizer, save_path)


### electra
def get_importance_weights(input_ids, tokenizer, importance_scores):
    """
    Compute importance weights for input tokens using importance scores from a dictionary.
    """
    importance_weights = torch.ones_like(input_ids, dtype=torch.float)
    for i, input_id_sequence in enumerate(input_ids):
        tokens = tokenizer.convert_ids_to_tokens(input_id_sequence)
        for j, token in enumerate(tokens):
            if token in importance_scores:
                importance_weights[i, j] = importance_scores[token]
    return importance_weights


def compute_weighted_loss(logits, labels, input_ids, tokenizer, importance_scores):
    """
    Compute weighted loss for ELECTRA training.
    """
    vocab_size = logits.size(-1)
    log_probs = F.log_softmax(logits.view(-1, vocab_size), dim=-1)
    
    # Negative Log Likelihood loss
    masked_lm_loss = F.nll_loss(
        log_probs,
        labels.view(-1),
        reduction="none",  # Compute per-token loss
        ignore_index=-100  # Ignore non-masked tokens
    )
    
    # Compute importance weights
    importance_weights = get_importance_weights(input_ids, tokenizer, importance_scores)
    
    # Mask out positions where label is -100
    active_loss = labels.view(-1) != -100
    active_loss = active_loss.float()
    
    # Apply weights and compute final loss
    weighted_loss = (masked_lm_loss * importance_weights.view(-1) * active_loss).sum() / active_loss.sum()
    return weighted_loss



def finetune_supervised_classifier(model_path, data_path, save_path):
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

    save_model(model, tokenizer, save_path)