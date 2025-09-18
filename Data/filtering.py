import os
import pandas as pd
from collections import defaultdict, Counter
from transformers import BertTokenizer
import json

# ---------------------------------------
# Load Pretrained BERT Tokenizer
# ---------------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ---------------------------------------
# File Paths
# ---------------------------------------
filtered_tokens_path = "/path/to/tokens_above_threshold.csv"  # Update with your actual path with tokens extracted from the token vs score plot threshold, extracted from Plot_loss_scores.ipynb

# Set the path to the input directory containing datasets to relabel
input_dir = "/path/to/dataset_directory_to_be_filtered/"     # Update with your actual path of the dataset that you want to filter

# Set the output directory (same structure but different folder name)
output_dir = input_dir.replace(
    "/path/to/dataset_directory_to_be_filtered/",
    "/path/to/Filtered_dataset_above_threshold/" # output of the new dataset after the filtering mechanism
)

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------
# Load Filtered Tokens (Ground Truth)
# ---------------------------------------
filtered_tokens = pd.read_csv(filtered_tokens_path)['Ground_Truth'].tolist() # ground truth or else tokens 

# Initialize a dictionary to store logs
log = {}

# ---------------------------------------
# Process Each CSV File in Input Directory
# ---------------------------------------
for file_name in os.listdir(input_dir):
    # Process only CSV files
    if not file_name.endswith(".csv"):
        continue

    df_path = os.path.join(input_dir, file_name)
    df = pd.read_csv(df_path)

    # Skip files that lack required columns
    if "text" not in df.columns or "class" not in df.columns:
        continue

    print(f" Processing: {file_name}")

    # Step 1: Build Token-Class Frequency Map
    token_class_counts = defaultdict(Counter)
    for _, row in df.iterrows():
        tokens = tokenizer.tokenize(str(row["text"]))
        for token in tokens:
            if token in filtered_tokens:
                token_class_counts[token][row["class"]] += 1

    # Get Majority Class for Each Token
    token_majority_class = {
        token: counts.most_common(1)[0][0]
        for token, counts in token_class_counts.items()
    }

    # Step 2: Remove Rows Where Class Does NOT Match Token Majority
    original_len = len(df)
    rows_to_keep = []
    removed_rows = []

    for idx, row in df.iterrows():
        tokens = tokenizer.tokenize(str(row["text"]))
        keep = True

        for token in tokens:
            if token in token_majority_class:
                if row["class"] != token_majority_class[token]:
                    keep = False
                    removed_rows.append({
                        "index": idx,
                        "original_class": row["class"],
                        "expected_class": token_majority_class[token],
                        "text": row["text"]
                    })
                    break  # Stop checking once a mismatch is found

        if keep:
            rows_to_keep.append(idx)

    # Create a filtered DataFrame
    df_filtered = df.loc[rows_to_keep]

    # Step 3: Save Filtered Dataset
    output_path = os.path.join(output_dir, file_name)
    df_filtered.to_csv(output_path, index=False)

    # Log Removal Statistics
    removed_count = len(removed_rows)
    log[file_name] = {
        "original_rows": original_len,
        "rows_removed": removed_count,
        "rows_remaining": len(df_filtered),
        "removed_samples": removed_rows
    }

    print(f" Saved cleaned dataset: {file_name} | Removed: {removed_count} rows")

# ---------------------------------------
# Save Full Log to JSON File
# ---------------------------------------
log_file = "token_based_filtering_log.json"
with open(log_file, "w") as f:
    json.dump(log, f, indent=4)

print(f"\nLog saved to {log_file}")






















