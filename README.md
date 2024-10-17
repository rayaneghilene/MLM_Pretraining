# Artifact Pretraining and Finetuning of Masked Language Models for Zero-Shot Socially Unacceptable Discourse Classification

This repository contains the code for Artifact extraction, MLM Pretraining, and Natural language inference Finetuning. An inference script is provided to test the tuned models. 

![Pretraining_pipeline](https://github.com/user-attachments/assets/08be893c-38e4-4674-8000-4982dccc70da)



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
nohup python Pretraining.py 
--model_path path_to_your_pretrained_model 
--output_path Path_to_save_the_finetuned_model 
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
nohup python fine_tune_nli.py 
--model_path path_to_your_pretrained_model 
--output_path Path_to_save_the_finetuned_model 
--dataset_name 'mnli' # or 'snli', or 'qnli' 
> Finetuning_logs.log 2>&1 &
```

You can visualize the finetuning progress via terminal using the following command
```ruby
tail -f Finetuning_logs.log
```


## Contributing
We welcome contributions from the community to enhance work. If you have ideas for features, improvements, or bug fixes, please submit a pull request or open an issue on GitHub.

## Contact
Feel free to reach out about any questions/suggestions at rayane.ghilene@ensea.fr
