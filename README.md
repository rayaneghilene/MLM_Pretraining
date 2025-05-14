# [MASK]ed Language Modeling using Socially Unacceptable Discourse Artifacts

> [!NOTE]  
> The argument parser has been modified in the latest update. Please review the documentation before running any scripts to ensure correct usage of the new arguments.

This repository contains the code for Token Importance Assessment, Masked Language Modeling (MLM) Pretraining, and Natural Language Inference (NLI) Finetuning. An inference script is provided to test the tuned models. 

![Pipeline](https://github.com/user-attachments/assets/40a57d06-b2f0-43a9-8868-e0d545fcafe5#gh-light-mode-only)
![Pipeline](https://github.com/user-attachments/assets/ed5e10f2-85a1-493a-9dcb-53ae0909e94b#gh-dark-mode-only)

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)  
   - [Masked Language Modelling](#masked-language-modelling)  
   - [NLI Finetuning](#nli-finetuning)  
   - [Fine-tune a Model for Supervised Classification](#fine-tune-a-model-for-supervised-classification)  
   - [Inference Testing](#for-inference-testing-of-a-model)  
3. [Token Importance Assessment](#i-token-importance-assessment)
   - [Example Tokens with high PMI scores in a tweet from the corpus:](#example-tokens-with-high-pmi-scores-in-a-tweet-from-the-corpus)
5. [Masked Language Modelling](#ii-masked-language-modelling)  
   - [A. Token Masking](#a-token-masking)  
   - [B. Training Dataset Preparation](#b-training-dataset-preparation)  
   - [C. Training Loss Optimisation](#c-training-loss-optimisation)  
6. [Downstream Performance Evaluation](#iii-downstream-performance-evaluation)  
   - [Supervised Socially Unacceptable Discourse Classification](#1-supervised-socially-unacceptable-discourse-classification)  
   - [NLI Zero-Shot Classification Performance](#2-nli-zero-shot-classification-performance)  
7. [Acknowledgements](#acknowledgements)  
8. [Contributing](#contributing)  
9. [Contact](#contact)

## Installation

Clone the repo using the following command:
```ruby
git clone https://github.com/rayaneghilene/MLM_Pretraining.git
cd MLM_Pretraining
```

We recommend creating a virtual environment (optional):
```bash
python3 -m venv myenv
source myenv/bin/activate 
```

To install the requirements run the following command: 
```
pip install -r requirements.txt
```


## Usage
All experiments should be ran using the main.py file. The arguments are as follows:

* `--experiment_name`: can be either 'train' for MLM training, or 'finetune_nli' $$\textcolor{red}{required}$$
* `--model_name`: can be either 'roberta', 'bert', or 'electra'
* `--GPU`: Specifies the GPU device number to use. If not set, the training will default to using the CPU. Leave this option unset if you don’t have a GPU or prefer not to use one.
* `--pretrained_model_path`: is the Path to the pretrained model.
* `--dataset_path`: is the Path to your dataset.
* `--masking_strategy`: can be either 'PMI', or 'BERTopic' (PMI is the default option)
* `--loss_strategy`: is used for optimisation of the loss (with PMI or LDA..), and can be either 'weighted', or for no optimisation 'none' (weighted is the default option)
* `--nli_dataset_name`: can be either 'mnli', 'qnli', or 'snli' ('mnli' is the default option)
* `--save_path`: is the Path to save the pretrained model and tokenizer (the default path is ''./Trained_models/')




### Masked Language Modelling
Here's an example command to train a model for masked Language Modelling:

```ruby
nohup python main.py 
--experiment_name 'train' 
--GPU '1'
--model_name 'roberta'
--dataset_path Path_to_the_dataset 
--save_path Path_to_save_the_trained_model 
--masking_strategy 'BERTopic'
> Pretraining_logs.log 2>&1 &
```

You can visualize the training progress via terminal using the following command
```ruby
tail -f Pretraining_logs.log
```

### NLI Finetuning
Here's an example command to Fine tune a pretrained model for NLI:

```ruby
nohup python main.py 
--experiment_name 'finetune_nli'
--GPU '1'
--model_path path_to_your_pretrained_model 
--save_path Path_to_save_the_finetuned_model 
--dataset_name 'mnli' 
> Finetuning_logs.log 2>&1 &
```

You can visualize the finetuning progress via terminal using the following command
```ruby
tail -f Finetuning_logs.log
```


### Fine tune a model for Supervised Classification 
Here's an example command to Fine tune a pretrained model in a supiervised fashion:
```ruby
nohup python main.py
--pretrained_model_path path_to_your_pretrained_model 
--GPU '1'
--data_path Path_to_you_data
--save_path Path_to_save_the_finetuned_model
> Supervised_logs.log 2>&1 &
```

You can visualize the finetuning progress via terminal using the following command
```ruby
tail -f Supervised_logs.log
```

### For Inference testing of a model
Run the following command
```ruby
nohup python utils/Inference_test.py
--model_path path_to_your_pretrained_model 
--data_path Path_to_you_data
> Inference_logs.log 2>&1 &
```

You can visualize the inference progress via terminal using the following command
```ruby
tail -f Inference_logs.log
```



## I. Token Importance Assessment

We compute the Pointwise Mutual Information (PMI) score for each token based on its co-occurrence with a specific class label in a professionally annotated corpus of approximately 470K Tweets. The dataset contains annotations for categories related to Socially Unacceptable Discourse, such as hateful, offensive, and toxic content.

$\text{PMI}(x, y) = \log \frac{P(x, y)}{P(x)P(y)}$

Where: 
    - P(x, y) is the probability of both events x and y occurring together
    - P(x) and  P(y) are the probabilities of the individual events x and y occurring independently.

A higher PMI score indicates a stronger association between the token and the specific class. To obtain a final importance score for each token, we compute its PMI score for all class labels and take the average across them. This approach ensures that tokens frequently associated with socially unacceptable discourse receive higher importance scores, guiding our token selection process during masked language model pre-training. 

For a detailed mathematical breakdown of PMI and its role in importance assessment, refer to [this link](https://github.com/rayaneghilene/MLM_Pretraining/blob/main/Data/README.md).

#### Example Tokens with high PMI scores in a tweet from the corpus:

![tweet](https://github.com/user-attachments/assets/d9fa20ab-184c-4a78-9138-53ecaeed5f69#gh-light-mode-only)
![tweet](https://github.com/user-attachments/assets/ad709088-a6eb-433d-acd3-7dc9726547a4#gh-dark-mode-only)





### II. Masked Language Modelling

#### A. Token Masking
The process involves randomly masking a certain percentage of words in a given Tokenized sentence (usually 15%) and training a model to predict the original words based on the surrounding context. The masked tokens are replaced with a [MASK] token for BERT (\<mask\> for roBERTa), and the original tokens are stored as targets for prediction.

<!-- ##### 1. Static Token Masking: -->
We employ a Static Token masking strategy; The masked tokens are selected once during data preprocessing and remain the same across all training epochs to ensure consistency.
<!-- ##### 2. Dynamic Token Masking:
The masked tokens are randomly selected at each pre-training epoch, exposing the model to diverse masking patterns and improving generalization. -->


#### B. Training Dataset preparation
In the dataset,  the ground-truth token IDs, masked in the inputs, are present in the label tensor and all other tokens are ignored (set to -100) by the default behaviour of nn.CrossEntropyLoss\(\) as illustrated:

![MLM_dataset](https://github.com/user-attachments/assets/48344369-e10e-4634-b556-f46ed563dfcc#gh-light-mode-only)

![MLM_dataset](https://github.com/user-attachments/assets/c3e6ebbe-75db-4275-a370-dba32bfc3c69#gh-dark-mode-only)





During preprocessing, labels are initialized to -100 for all tokens, indicating they should be ignored during loss calculation. For positions where tokens were masked, their corresponding token IDs are assigned as labels. The dataset is split into training and test sets, and the masked text, along with the labels, is prepared for model training.

#### C. Training Loss Optimisation

During the training, the model is optimized by minimizing the loss between its predictions and the original tokens. The importance of the masked tokens, guided by their Importance scores, is incorporated into the loss function to emphasize learning from socially unacceptable discourse tokens. Specifically, tokens with higher scores are weighted more heavily in the loss calculation, encouraging the model to focus on learning the contextual relationships involving these tokens.

For a detailed mathematical breakdown of weighted loss optimisation, refer to [this link](https://github.com/rayaneghilene/MLM_Pretraining/blob/main/utils/README.md).

### III. Downstream Performance Evaluation

To assess the impact of our pretraining approach, we fine-tune the trained models on downstream tasks. We evaluate their performance on supervised classification for SUD detection and test their NLI capabilities on benchmark datasets.

#### 1. Supervised Socially Unacceptable Discourse Classification

We fine-tune the models on a collection of datasets focused on detecting hateful, offensive, toxic, and other forms of socially unacceptable discourse. Each dataset contains professionally annotated samples, ensuring robust and reliable evaluation.
	•	**Task:** Given a text input, classify it into predefined categories such as hateful, offensive, or neutral.
	•	**Objective:** Measure whether pretraining with importance weighted masking improves classification accuracy compared to baseline models trained with standard MLM.
	•	**Metrics:** We report macro-F1, accuracy, and precision-recall curves to capture overall performance and class-specific behavior.

#### 2. NLI Zero-Shot Classification Performance

To evaluate general language understanding capabilities, we fine-tune our models using three widely used NLI datasets and test them as zero-shot classifiers:

* **MNLI (Multi-Genre NLI):** A benchmark dataset requiring models to determine if a given premise entails, contradicts, or is neutral to a hypothesis across multiple domains.
* **QNLI (Question NLI):** A dataset derived from the Stanford Question Answering Dataset (SQuAD), where the task is to determine whether a given sentence contains an answer to a question.
* **SNLI (Stanford NLI):** A collection of human-written sentence pairs annotated for textual entailment.

By evaluating the models across both SUD classification and NLI tasks, we aim to understand how our Importance based masking strategy affects both domain-specific performance and general linguistic capabilities.



## Acknowledgements
This work was conducted as part of the European [Arenas](https://arenasproject.eu/) project, funded by Horizon Europe.
Its objective is to characterize, measure, and understand the role of extremist narratives in discourses that have an impact not only on political and social spheres but importantly on the stakeholders themselves.  Leading an innovative and ambitious research program, ARENAS will significantly contribute to filling the gap in contemporary research, make recommendations to policymakers, media, lawyers, social inclusion professionals, and educational institutions, and propose solutions for countering extreme narratives for developing more inclusive and respectful European societies.

<p align="center">
  <img src="https://github.com/rayaneghilene/ARENAS/blob/main/DSML_Research_Project/Images/Arenas-final-GIF.gif" width=100% >
</p>

## Contributing
We welcome contributions from the community to enhance work. If you have ideas for features, improvements, or bug fixes, please submit a pull request or open an issue on GitHub.

## Contact
Feel free to reach out about any questions/suggestions at rayane.ghilene@ensea.fr
