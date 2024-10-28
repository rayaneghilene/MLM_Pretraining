# Artifact Pretraining and Finetuning of Masked Language Models for Zero-Shot Socially Unacceptable Discourse Classification

This repository contains the code for Artifact extraction, MLM Pretraining, and Natural language inference Finetuning. An inference script is provided to test the tuned models. 

![Pretraining_pipeline](https://github.com/rayaneghilene/MLM_Pretraining/blob/main/Images/Pretraining_pipeline.png)


## Installation

Clone the repo using the following command:
```ruby
git clone https://github.com/rayaneghilene/MLM_Pretraining.git
cd MLM_Pretraining
```

We recommend creating a virtual environment (optional)
```bash
python3 -m venv myenv
source myenv/bin/activate 
```

To install the requirements run the following command: 
```
pip install -r requirements.txt
```


## Usage
### Masked Language Modelling
To train a model for masked Language Modelling run the following command:
```ruby
nohup python main.py 
--model_name 'roberta' #or , 'bert', 'electra-generator', or 'electra-discriminator'
--dataset_path Path_to_the_dataset 
--save_path Path_to_save_the_trained_model 
--masking_strategy 'PMI' # or 'LDA', or 'BERTopic' 
> Pretraining_logs.log 2>&1 &
```

You can visualize the training progress via terminal using the following command
```ruby
tail -f Pretraining_logs.log
```

### NLI Finetuning
To Fine tune the pretrained model run the following command
```ruby
nohup python models/fine_tune_nli.py 
--model_path path_to_your_pretrained_model 
--output_path Path_to_save_the_finetuned_model 
--dataset_name 'mnli' # or 'snli', or 'qnli' 
> Finetuning_logs.log 2>&1 &
```

You can visualize the finetuning progress via terminal using the following command
```ruby
tail -f Finetuning_logs.log
```


### Fine tune a model for Supervised Classification 
To Fine tune the pretrained model run the following command
```ruby
nohup python models/Supervised_classification.py
--model_path path_to_your_pretrained_model 
--data_path Path_to_you_data
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


## How This works?
### I. Token Extraction:
#### 1. Pointwise mutual information (PMI) Token extraction
P(x, y) is the probability of both events x and y occurring together, and P(x) and  P(y) are the probabilities of the individual events x and y occurring independently.

$\text{PMI}(x, y) = \log \frac{P(x, y)}{P(x)P(y)}$

#### Example Token Extraction from the Fox dataset using PMI

![PMI Example](https://github.com/rayaneghilene/MLM_Pretraining/blob/main/Images/PMI_example.png)




#### 2. BERTopic Similarity Token extraction
another widely used topic modeling technique that leverages transformer-based embeddings to identify and represent topics in text data.

#### 3. Latent Dirichlet Allocation (LDA) Token extraction
LDA estimates the topic distribution for each document and the word distribution for each topic, allowing us to identify common themes across the document corpus.

### II. Token Masking
Using the extracted tokens, we mask 15% of the data as follows:

* **No meaningful tokens:** If no significant tokens are identified, the method defaults to random masking for 15% of tokens.

* **Partial meaningful** tokens: If fewer than 15% of meaningful tokens are found, random masking is applied to reach the 15% target.

* **Excessive meaningful tokens:** If more than 15% of meaningful tokens are identified, only 15\% of the tokens will be selected for masking.

### III. Masked Language Modelling
The process involves randomly masking a certain percentage of words in a given text (usually 15%) and training a model to predict the original words based on the surrounding context.
Here the model is trained to predict the masked artifacts.


In the dataset these masked tokens are replaced with a [MASK] token (<mask> for roBERTa), and the original tokens are stored as targets for prediction. 
During preprocessing, labels are initialized to -100 for all tokens, indicating they should be ignored during loss calculation. For positions where tokens were masked, their corresponding token IDs are assigned as labels. The dataset is split into training and test sets, and the masked text, along with the labels, is prepared for model training.

#### Example:

**Tokenized text:** [CLS] token1 token2 token3 [MASK] token5 [SEP].

**Label tensor:** [-100,   -100,     -100,   -100, Token_Id,-100, -100].

### IV. Finetuning For NLI
To  use these models in a Zero-shot Setting, we finetune them on Natural Language Inference Datasets ```'mnli', 'qnli', and 'snli'```. This allows us to approach a multi class classification task with a Binary one, meaning that for each text we ask if the label is a match or not, and if it is how confident are we?

## Acknowledgements
This work was conducted as part of the European [Arenas](https://arenasproject.eu/) project, funded by Horizon Europe.

## Contributing
We welcome contributions from the community to enhance work. If you have ideas for features, improvements, or bug fixes, please submit a pull request or open an issue on GitHub.

## Contact
Feel free to reach out about any questions/suggestions at rayane.ghilene@ensea.fr
