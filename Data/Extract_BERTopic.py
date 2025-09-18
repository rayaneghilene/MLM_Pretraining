import argparse
import pandas as pd
from bertopic import BERTopic
import nltk
from collections import Counter
from transformers import BertTokenizer  # Added for tokenization

# Specify the custom directory where the stopwords are stored
custom_dir = '/path/to/nltk_data'

# Add this directory to NLTK's data path
nltk.data.path.append(custom_dir)

from nltk.corpus import stopwords

class BERTopicProcessor:
    def __init__(self):
        # Load English stopwords
        self.stop_words = set(stopwords.words('english'))

    def fit_bertopic(self, texts): 
        # Fit the BERTopic model on the given texts.
        print("Fitting BERTopic model...")
        
        model = BERTopic(
            verbose=True, 
            embedding_model="/path/to/paraphrase-MiniLM-L3-v2/", 
            min_topic_size=30
        )
        
        topics, _ = model.fit_transform(texts)
        return model

    def process(self, df):
        """
        Process the dataset for BERTopic topic modeling and token analysis.
        """
        print("Processing dataset...")
        
        texts = df['text'].tolist()
        model = self.fit_bertopic(texts)
        
        topic_details = []
        for topic_id, words_and_scores in model.get_topics().items():
            if topic_id == -1:  # Skip outliers
                continue
            for word, score in words_and_scores:
                topic_details.append({"topic_id": topic_id, "token": word, "bertopic_score": score})
        
        topic_df = pd.DataFrame(topic_details)

        # Remove blank cells from 'token' column
        topic_df = topic_df[topic_df['token'].notna()]
        topic_df = topic_df[topic_df['token'].str.strip() != ""]

        print("Removing stopwords...")
        topic_df = topic_df[~topic_df['token'].isin(self.stop_words)]

        print("Normalizing scores within topics...")
        topic_df['normalized_score'] = topic_df.groupby('topic_id')['bertopic_score'].transform(
            lambda x: x / x.sum()
        )

        print("Calculating average scores across topics...")
        avg_scores = topic_df.groupby('token')['normalized_score'].mean().reset_index()
        avg_scores.rename(columns={'normalized_score': 'avg_score'}, inplace=True)

        return avg_scores

def split_token(row, tokenizer):
    """
    Tokenize the input token into subwords and replicate the avg_score for each subword.
    """
    subwords = tokenizer.tokenize(row['token'])
    return pd.DataFrame({
        'token': subwords,
        'avg_score': [row['avg_score']] * len(subwords)
    })

def main(dataset_path, output_path, expanded_output_path):
    """
    Main function to process the dataset with BERTopic and then expand tokens with BERT tokenizer.
    """
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    processor = BERTopicProcessor()
    avg_scores_df = processor.process(df)

    print(f"Saving BERTopic scores to {output_path}...")
    avg_scores_df.to_csv(output_path, index=False)

    # Load tokenizer
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("Expanding tokens into subwords...")
    expanded_df = pd.concat([split_token(row, tokenizer) for _, row in avg_scores_df.iterrows()], ignore_index=True)

    print(f"Saving expanded tokenized scores to {expanded_output_path}...")
    expanded_df.to_csv(expanded_output_path, index=False)

    print("Preview of expanded tokens:")
    print(expanded_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataset with BERTopic and calculate token scores, then expand tokens using BERT tokenizer.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to your input dataset CSV.")
    parser.add_argument("--output_path", default="bertopic_scores.csv", type=str, help="Path to save BERTopic token scores.")
    parser.add_argument("--expanded_output_path", default="bertopic_scores_with_bertokeniser.csv", type=str, help="Path to save expanded tokenized scores.")
    args = parser.parse_args()

    main(args.dataset_path, args.output_path, args.expanded_output_path)

