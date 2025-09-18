import pandas as pd
import nltk
from collections import Counter
import glob
import os

from nltk.corpus import stopwords

# --- Downloads ---
nltk.download('punkt')
nltk.download('stopwords')

# --- Configuration ---
DATA_DIR = "/path/to/identified_errors_per_dataset"  # Update this path with the directory of identified_errors_per_dataset created by detect_label_issues.py
FILES = glob.glob(os.path.join(DATA_DIR, "*.csv"))
stop_words = set(stopwords.words('english'))

# --- Load and Combine CSVs ---
dataframes = []
for file in FILES:
    df = pd.read_csv(file)
    df['is_error'] = df['given_label'] != df['predicted_label']
    dataframes.append(df)

df_all = pd.concat(dataframes, ignore_index=True)

# --- Extract Tokens from Misclassified Texts ---
error_texts = df_all[df_all['is_error']]['text']
all_tokens = [
    token.lower()
    for text in error_texts
    for token in nltk.word_tokenize(str(text))
    if token.isalpha() and token.lower() not in stop_words
]

token_freq = Counter(all_tokens)

# --- Select Top 25% Most Frequent Tokens ---
top_n = int(len(token_freq) * 0.25)
top_confusing_tokens = dict(token_freq.most_common(top_n))
max_freq = max(top_confusing_tokens.values())

# --- Normalize and Save ---
confusion_token_scores = {
    token: freq / max_freq
    for token, freq in top_confusing_tokens.items()
}

confusion_df = pd.DataFrame(list(confusion_token_scores.items()), columns=["token", "score"])
confusion_df.to_csv("noise_based_scores.csv", index=False)

print(f" Saved top {top_n} confusing tokens to 'noise_based_scores.csv'")

