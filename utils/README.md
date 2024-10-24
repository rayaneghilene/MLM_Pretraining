
# WeightedLossTrainer

`WeightedLossTrainer` is a custom trainer class designed to extend the Hugging Face `Trainer` class by incorporating token-specific importance scores into the loss calculation for Masked Language Modeling.

## Key Features

The `WeightedLossTrainer` applies importance weights to the masked tokens during loss calculation:
- Uses Negative Log-Likelihood (NLL) loss for MLM tasks.
- Applies importance weights (PMI, LDA, BERTopic, ..) to tokens, allowing tokens to contribute more or less to the overall loss based on their importance scores.


### 1.	Negative Log-Likelihood (NLL) loss:

$\text{masked\_lm\_loss}(i) = - \log(p(\text{label}_i \mid \text{input}))$

for each token i, where  $p(\text{label}_i \mid \text{input})$  is the probability of the correct token (from the logits) after applying softmax.

### 2.	Importance weighting:

$\text{weighted\_loss}(i) = \text{masked\_lm\_loss}(i) \times \text{importance\_weights}(i) \times \text{active\_loss}(i)$

Here:
* $\text{importance\_weights}(i)$  is the importance score for token i (from the importance_scores dictionary).

*  $\text{active\_loss}(i)$  is a binary mask that is 1 if the token label is not -100 (indicating it is not masked out) and 0 otherwise.

### 3.	Final loss calculation:

$\text{final\_loss} = \frac{\sum_{i=1}^{n} \text{weighted\_loss}(i)}{\sum_{i=1}^{n} \text{active\_loss}(i)}$

This sums up the weighted losses over all tokens and normalizes by the total number of non-masked tokens (where  $\text{active\_loss}(i)=1$).

## Contribution

Please If there are any errors, bugs, or inconsistencies, submit a pull request, or directly reach out to rayane.ghilene@ensea.fr