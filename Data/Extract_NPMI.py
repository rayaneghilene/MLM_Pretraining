import argparse
import os
import pandas as pd
from collections import Counter, defaultdict
import math
from transformers import AutoTokenizer

class NPMI:
    def __init__(self, tokenizer_model="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    
    def compute_npmi(self, class_token_counts, total_class_tokens, overall_token_counts, total_tokens):
        """
        Compute NPMI scores for a given class.
        
        :param class_token_counts: Counter of tokens for the specific class.
        :param total_class_tokens: Total number of tokens in the class.
        :param overall_token_counts: Counter of overall tokens across all classes.
        :param total_tokens: Total number of tokens across all classes.
        :return: Dictionary of NPMI scores for each token in the class.
        """
        npmi_scores = {}
        for token, count in class_token_counts.items():
            # Calculate probabilities
            p_token_given_class = count / total_class_tokens
            p_token = overall_token_counts[token] / total_tokens
            p_class = total_class_tokens / total_tokens
            p_token_and_class = count / total_tokens
            
            # Calculate PMI
            if p_token_given_class > 0 and p_token > 0 and p_token_and_class > 0:
                pmi = math.log(p_token_given_class / (p_token * p_class), 2)
                npmi = pmi / (-math.log(p_token_and_class, 2))  # Normalize PMI
                npmi_scores[token] = npmi
            else:
                npmi_scores[token] = 0
        return npmi_scores

    def extract_npmi_for_dataset(self, df, text_column, class_column):
        """
        Extract NPMI scores for a dataset.
        
        :param df: The input dataframe containing the text and class columns.
        :param text_column: The name of the column containing the text.
        :param class_column: The name of the column containing the class labels.
        :return: DataFrame containing tokens and their NPMI scores.
        """
        # Create counters for tokens
        overall_token_counts = Counter()
        class_token_counts = defaultdict(Counter)
        total_tokens = 0
        class_totals = defaultdict(int)

        # Tokenize and update counters
        for _, row in df.iterrows():
            text = row[text_column]
            class_label = row[class_column]
            
            tokens = self.tokenizer.tokenize(text, truncation=True, max_length=512)#, clean_up_tokenization_spaces=True)

            class_token_counts[class_label].update(tokens)
            overall_token_counts.update(tokens)
            class_totals[class_label] += len(tokens)
            total_tokens += len(tokens)
        
        # Calculate NPMI scores for each class
        token_npmi_scores = defaultdict(list)
        for class_label, token_counts in class_token_counts.items():
            class_npmi_scores = self.compute_npmi(token_counts, class_totals[class_label], overall_token_counts, total_tokens)
            for token, npmi_score in class_npmi_scores.items():
                token_npmi_scores[token].append(npmi_score)
        
        # Average NPMI scores for tokens across classes
        avg_npmi_scores = {token: sum(scores) / len(scores) for token, scores in token_npmi_scores.items()}
        
        # Create a dataframe from the averaged NPMI scores
        npmi_df = pd.DataFrame(list(avg_npmi_scores.items()), columns=['token', 'npmi'])
        
        return npmi_df

def main(dataset_path, output_path):
    npmi_calculator = NPMI()
    df = pd.read_csv(dataset_path)
    npmi_df = npmi_calculator.extract_npmi_for_dataset(df, text_column='text', class_column='class')
    npmi_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and fine-tune a model with a specific dataset.")
    parser.add_argument("--dataset_path", type=str, required=True,  help="Path to your dataset.")
    parser.add_argument("--output_path", default="npmi_scores.csv", type=str, help="Path to save the pretrained NPMI scores.")
    args = parser.parse_args()
    main(args.dataset_path, output_path=args.output_path)