from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import argparse
import os
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def run_experiment_with_labels(model_path, data_path):
    SEED = 42
    train_percentage = 0.8
    validation_percentage = 0.1
    test_percentage = 0.1

    # Load data
    df = pd.read_csv(data_path)
    df['class'] = df['class'].apply(lambda x: x.replace('neither', 'neutral').replace('hate', 'hateful'))

    X = df.text.values
    y = pd.get_dummies(df["class"])
    candidate_labels = list(y.columns)
    print("the candidate labels are", candidate_labels)
    y = y.values
    # Split data
    _, X_rest, _, y_rest = train_test_split(X, y, test_size=1 - train_percentage, random_state=SEED, shuffle=True, stratify=y)
    _, X_test, _, y_test = train_test_split(X_rest, y_rest, test_size=test_percentage / (validation_percentage + test_percentage), random_state=SEED, shuffle=True, stratify=y_rest)

    sequences_to_classify = X_test.tolist()
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()
    entailed_probs = []
    for seq in sequences_to_classify:
        entailment_scores = []
        for label in candidate_labels:
            inputs = tokenizer(seq,
                            f"This text contains {label} speech.",
                            return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            # 0 for entailment, 1 for contradiction, 2 for neutral
            score = F.softmax(logits, dim=1)[:, 0].item()
            entailment_scores.append(score)
        
        entailed_probs.append(entailment_scores)

    entailed_probs = np.array(entailed_probs)
    print("the entailed probs are", entailed_probs)

    # Select the best label for each sequence
    predicted_labels = [candidate_labels[np.argmax(scores)] for scores in entailed_probs]

    #ground truth labels for comparison
    true_labels = [candidate_labels[np.argmax(label)] for label in y_test]

    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    precision = precision_score(true_labels, predicted_labels, average='macro')
    confusion = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, target_names=candidate_labels)

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(report)

    return f1


def main(model_path, data_path):
    Average = []
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if os.path.isfile(file_path) and file_path.endswith('.csv'):
            print(f"Processing CSV file: {filename}")
            f1 = run_experiment_with_labels(model_path, file_path)
            Average.append(f1)
        else:
            print(f"Skipping non-CSV file: {file_path}")
    print("The average F1 score is: ", np.mean(Average))


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train and fine-tune a model with a specific dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to save the training datasets.")
    args = parser.parse_args()

    main(model_path=args.model_path, data_path=args.data_path)