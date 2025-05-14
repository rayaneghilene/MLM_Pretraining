import random
import torch
import argparse
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import pandas as pd

class MLM_Dataset:
    def __init__(self, dataframe, tokenizer, mask_prob=0.15):
        """
        Args:
            dataframe (pd.DataFrame): A DataFrame containing text data.
            tokenizer: Pretrained tokenizer (BERT, RoBERTa, etc.).
            mlm_probability (float): The probability of masking a token (default 0.15).
            max_length (int): The maximum length of sequences after padding/truncation (default 128).
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

    def mask_tokens(self, text):
        tokens = self.tokenizer.tokenize(text,truncation=True ,max_length=512)
        num_tokens = len(tokens)
        required_num_masks = max(1, int(self.mask_prob * num_tokens))  # Ensure at least 15% of tokens are masked
        # print(f"Num tokens: {num_tokens}, Required masks: {required_num_masks}")
        # Randomly select positions to mask
        mask_indices = random.sample(range(num_tokens), required_num_masks)
        
        masked_tokens = tokens[:]
        tokens_to_predict = []
        
        for idx in mask_indices:
            # Mask the token and store the original token for prediction
            tokens_to_predict.append(tokens[idx])
            masked_tokens[idx] = self.mask_token  # Insert [MASK] token
            # print("mask token is: ", self.mask_token)
        # Reconstruct the masked text
        masked_text = self.tokenizer.convert_tokens_to_string(masked_tokens)
        # print("masked text is: ", masked_text)

        return masked_text, tokens_to_predict

    def preprocess_function(self, examples):
        # Tokenize the masked text (input to the model)
        inputs = self.tokenizer(examples['masked_text'], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        
        # Prepare labels tensor with -100 (to ignore)
        labels_ids = []
        
        input_ids = inputs['input_ids']  # Tokenized input ids for all examples
        
        for i, tokens_to_predict in enumerate(examples['tokens_to_predict']):
            # print("The tokens to predict are:", tokens_to_predict)
            # Convert tokens_to_predict to token IDs
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens_to_predict)

            # Initialize label ids with -100 to ignore non-masked tokens
            label_ids = [-100] * input_ids.size(1)

            # Find all mask positions (i.e., where [MASK] token occurs)
            mask_indices = (input_ids[i] == self.mask_token_id).nonzero(as_tuple=True)[0].tolist()

            # Assign token ids to the corresponding mask positions
            for idx, mask_index in enumerate(mask_indices):
                if idx < len(token_ids):
                    label_ids[mask_index] = token_ids[idx]  # Assign token_id to the masked position

            labels_ids.append(label_ids)

        # Convert list of labels to tensor
        inputs['labels'] = torch.tensor(labels_ids)

        return inputs
    def get_dataset(self):
        # Create train/validation datasets
        masked_texts, tokens_to_predict = [], []
        for text in self.dataframe['text']:
            masked_text, tokens = self.mask_tokens(text)
            masked_texts.append(masked_text)
            tokens_to_predict.append(tokens)
        masked_texts_padded = []
        for i in masked_texts:
            padded = self.tokenizer(i, truncation=True, padding='max_length', max_length=128)
            masked_texts_padded.append(self.tokenizer.decode(padded['input_ids'], skip_special_tokens=True))

        # dataset = Dataset.from_dict({'masked_text': masked_texts_padded, 'tokens_to_predict': tokens_to_predict})
        dataset = Dataset.from_dict({'masked_text': masked_texts, 'tokens_to_predict': tokens_to_predict})
        train_test_split = dataset.train_test_split(test_size=0.2)

        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']

        # Preprocess the datasets (tokenization + labels creation)
        train_dataset_processed = train_dataset.map(self.preprocess_function, batched=True)
        # print(f"the keys are: {train_dataset_processed.column_names}")
        # print("Input IDs:", train_dataset_processed['input_ids'])
        # print("Max Input ID:", max(train_dataset_processed['input_ids']))
        test_dataset_processed = test_dataset.map(self.preprocess_function, batched=True)

        # Return DatasetDict for HuggingFace models
        dataset_dict = DatasetDict({
            "train": train_dataset_processed,
            "test": test_dataset_processed
        })

        return dataset_dict

    def save_datasets(self, dataset_dict, output_dir="data/"):
        """
        Save the train and test datasets to CSV files.
        
        Args:
            dataset_dict (DatasetDict): A HuggingFace DatasetDict containing train and test datasets.
            output_dir (str): Directory to save the datasets (default is "data/").
        """
        train_file = f"{output_dir}/train.csv"
        test_file = f"{output_dir}/test.csv"

        dataset_dict["train"].to_csv(train_file)
        dataset_dict["test"].to_csv(test_file)
        print(f"Datasets saved to {output_dir} as CSV.")

def load_tokenizer(model_name='roberta'):
    if model_name == 'roberta':
        model_name = 'FacebookAI/roberta-large'
    elif model_name == 'bert':
        model_name = 'bert-base-uncased'
    elif model_name == 'electra':
        model_name = 'google/electra-base-discriminator'
    else:
        raise ValueError(f"Model '{model_name}' not supported")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def main(dataset_path, model_name, output_dir):
    """
    Create and preprocess a dataset from a DataFrame.
    This method can be run standalone to test the class functionality.
    """
    df = pd.read_csv(dataset_path)
    tokenizer = load_tokenizer(model_name)
    mlm_dataset = MLM_Dataset(dataframe=df, tokenizer=tokenizer, mask_prob=0.15)
    # Get the preprocessed dataset
    dataset_dict = mlm_dataset.get_dataset()
    mlm_dataset.save_datasets(dataset_dict, output_dir)

    # Print the train and test dataset
    # print(dataset_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a masked language model dataset from a text dataset.")
    parser.add_argument("--model_name", type=str, default='bert', choices=['roberta', 'bert', 'electra'],  help="Model Name.")
    parser.add_argument("--dataset_path", type=str, required=True,  help="Path to your dataset.")
    parser.add_argument("--output_dir", type=str, default='./Dataset',  help="Path to store your train and test dataset.")
    args = parser.parse_args()
    main(args.dataset_path, args.model_name, args.output_dir)