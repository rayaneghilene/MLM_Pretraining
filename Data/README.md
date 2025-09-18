# Token Importance Normalisation 

## Pointwise Mutual Information (PMI)

PMI is a technique that scores tokens within different classes across multiple datasets. 

* **Token Counting**:  we count the occurrences of each token in each class, as well as globally across the entire dataset.
* **PMI Calculation**: For each class, we computes the PMI score for each token using the following formula:

   $PMI(token, class) = \log_2 \left( \frac{P(token|class)}{P(token) \cdot P(class)} \right)$
   
   Where:
   - $P(token|class)$ is the probability of a token occurring in a class.
   - $P(token)$ is the probability of the token across the entire dataset.
   - $P(class)$ is the probability of a class in the dataset.


Sometimes a token is present in mutiple classes, and thus has multiple scores. To adress this issus, we propose the following normalisation methods



1. Normalized PMI (NPMI)

Normalized PMI rescales the PMI values to a range between -1 and 1. This normalization can help in reducing the effect of corpus size while still retaining the relative association between tokens.

The formula for NPMI is:


$NPMI(x, y) = \frac{PMI(x, y)}{-\log P(x, y)}$

Where:
* $PMI(x, y)$ is the PMI score between tokens x and y.
* $P(x, y)$ is the joint probability of x and y.

This ensures that the scores are bounded, making them more comparable across different datasets or token occurrences.



## 2. BERTopic 

The `Extract_BERTopic.py` file processes a text dataset using **BERTopic** topic modeling to extract important topic tokens and their relevance scores. It then expands these tokens into subword units using the **BERT tokenizer**, assigning the original token’s score to each subword. This helps prepare fine-grained token-level scores for downstream tasks like masked language modeling or token filtering.

It follows the followng steps:
- Fits BERTopic model on input text data.
- Extracts topic tokens and calculates normalized relevance scores.
- Removes stopwords to focus on meaningful tokens.
- Averages token scores across topics.
- Uses BERT tokenizer to split tokens into subwords.
- Assigns average scores to each subword.
- Saves both original token scores and expanded subword scores as CSV files.

You can run the script seperatly and extract BERTopic scors using the following arguments:

```bash
python your_script.py --dataset_path path/to/input_dataset.csv --output_path path/to/bertopic_scores.csv --expanded_output_path path/to/expanded_token_scores.csv
```




## Contribution

Please If there are any Normalisation methods that can be applied for this use case, submit a pull request, or directly reach out to rayane.ghilene@ensea.fr

