import argparse
import pandas as pd
from bertopic import BERTopic
import nltk

# Specify the custom directory where the stopwords are stored
custom_dir = '/data/ARENAS_Automatic_Extremist_Analysis/ARENAS_Automatic_Extremist_Analysis/downloaded_models/nltk_data'

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
            embedding_model="/data/ARENAS_Automatic_Extremist_Analysis/ARENAS_Automatic_Extremist_Analysis/downloaded_models/paraphrase-MiniLM-L3-v2/", 
            min_topic_size=30
        )

        model.fit(texts)
        return model
	
    def process(self, df):
        """
        Process the dataset for BERTopic topic modeling and token analysis.
        """

        print("Processing dataset...")
        
        # Step 1: Extract text and fit BERTopic
        texts = df['text'].tolist()
        model = self.fit_bertopic(texts)
        
        # Step 2: Extract topic details
        topic_details = []
        for topic_id, words_and_scores in model.get_topics().items():
            if topic_id == -1:  # Skip outliers
                continue
            for word, score in words_and_scores:
                topic_details.append({"topic_id": topic_id, "token": word, "bertopic_score": score})
        topic_df = pd.DataFrame(topic_details)

        # Remove blank cells from the 'token' column
        topic_df = topic_df[topic_df['token'].notna()]  # Step to remove NaN values
        topic_df = topic_df[topic_df['token'].str.strip() != ""]  # Step to remove empty strings

        # Step 3: Remove stopwords
        print("Removing stopwords...")
        topic_df = topic_df[~topic_df['token'].isin(self.stop_words)]

        # Step 4: Normalize BERTopic scores
        print("Normalizing scores within topics...")
        topic_df['normalized_score'] = topic_df.groupby('topic_id')['bertopic_score'].transform(
            lambda x: x / x.sum()
        )

        # Step 5: Average bertopic (importance) scores of tokens across topics to get a final and unique score per token
        print("Calculating average scores across topics...")
        avg_scores = topic_df.groupby('token')['normalized_score'].mean().reset_index()
        avg_scores.rename(columns={'normalized_score': 'avg_score'}, inplace=True)

        return avg_scores

def main(dataset_path, output_path):
    """
    Main function to process the dataset and save the results.
    """
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    processor = BERTopicProcessor()
    avg_scores_df = processor.process(df)

    print(f"Saving results to {output_path}...")
    avg_scores_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataset with BERTopic and calculate token scores.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to your dataset.")
    parser.add_argument("--output_path", default="bertopic_scores.csv", type=str, help="Path to save the results.")
    args = parser.parse_args()
    main(args.dataset_path, output_path=args.output_path)