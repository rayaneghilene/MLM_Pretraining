import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, models
from cleanlab.classification import CleanLearning

# ========== CONFIG ==========
INPUT_FILE = "/path/to/dataset.csv" # path to original dataset
OUTPUT_DIR = "/path/to/dataset_label_issues_output" # path to the identified errors dataset
TRAIN_CSV = os.path.join(OUTPUT_DIR, "train_identified_errors_dataset.csv") # train identified errors csv file
TEST_CSV = os.path.join(OUTPUT_DIR, "test_identified_errors_dataset.csv")# test identified errors csv file
COMBINED_CSV = os.path.join(OUTPUT_DIR, "dataset_identified_errors.csv") # train and test identified errors csv file
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ============================

# Load data
data = pd.read_csv(INPUT_FILE)
raw_texts, raw_labels = data["text"].values, data["class"].values

# Train/test split
raw_train_texts, raw_test_texts, raw_train_labels, raw_test_labels = train_test_split(
    raw_texts, raw_labels, test_size=0.1, random_state=42
)

# Encode labels
encoder = LabelEncoder()
encoder.fit(raw_labels)
train_labels = encoder.transform(raw_train_labels)
test_labels = encoder.transform(raw_test_labels)

# Load BERT model for embeddings
word_embedding_model = models.Transformer('google/bert_uncased_L-4_H-256_A-4')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
transformer = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Encode texts
train_embeddings = transformer.encode(raw_train_texts, show_progress_bar=True)
test_embeddings = transformer.encode(raw_test_texts, show_progress_bar=True)

# Function to find and export label issues
def detect_label_issues(X, y, raw_texts, raw_labels, filename):
    model = LogisticRegression(max_iter=1000)
    cl = CleanLearning(model, cv_n_folds=5)
    label_issues = cl.find_label_issues(X=X, labels=y)

    # Filter only identified issues
    identified_issues = label_issues[label_issues["is_label_issue"] == True]
    num_issues = len(identified_issues)

    # Extract details into a DataFrame
    df = pd.DataFrame({
        "text": raw_texts,
        "given_label": raw_labels,
        "predicted_label": encoder.inverse_transform(label_issues["predicted_label"]),
        "label_quality": label_issues["label_quality"],
        "is_label_issue": label_issues["is_label_issue"]
    }).iloc[identified_issues.index]

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Saved {num_issues} identified label issues to {filename}")

# Run label issue detection for both train and test
detect_label_issues(train_embeddings, train_labels, raw_train_texts, raw_train_labels, TRAIN_CSV)
detect_label_issues(test_embeddings, test_labels, raw_test_texts, raw_test_labels, TEST_CSV)

# Combine train and test issue files
def combine_csv(file1, file2, output_file):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f" Combined label issues saved to {output_file}")

combine_csv(TRAIN_CSV, TEST_CSV, COMBINED_CSV)


