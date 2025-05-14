import torch
import torch.nn.functional as F
from transformers import Trainer
from sklearn.metrics import accuracy_score

class WeightedLossTrainer(Trainer):
    def __init__(self, importance_scores, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.importance_scores = importance_scores
        self.loss_history = []

    def get_importance_weights(self, input_ids):
        ## Align tokens with their importance scores
        importance_weights = torch.ones_like(input_ids, dtype=torch.float)
        for i, input_id_sequence in enumerate(input_ids):
            tokens = self.tokenizer.convert_ids_to_tokens(input_id_sequence)
            for j, token in enumerate(tokens):
                if token in self.importance_scores:
                    importance_weights[i, j] = self.importance_scores[token]
        return importance_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Ensure the logits are of the correct shape (batch_size, sequence_length, vocab_size)
        vocab_size = logits.size(-1)
        # Flatten the logits and labels for compatibility with F.nll_loss
        log_probs = F.log_softmax(logits.view(-1, vocab_size), dim=-1)
        # Calculate NLL loss (Negative Log Likelihood) for MLM task
        masked_lm_loss = F.nll_loss(
            log_probs,
            labels.view(-1),
            reduction='none',  # Compute loss for each token separately
            ignore_index=-100 
        )
        # Apply importance weights to masked tokens 
        importance_weights = self.get_importance_weights(inputs['input_ids'])
        # Mask out the positions where label is -100
        active_loss = labels.view(-1) != -100
        active_loss = active_loss.float()
        # Apply the importance weights and mask
        weighted_loss = (masked_lm_loss * importance_weights.view(-1) * active_loss).sum() / active_loss.sum()
        self.loss_history.append(weighted_loss.item())  # Store loss

        return (weighted_loss, outputs) if return_outputs else weighted_loss


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Convert logits to predictions (choose the class with highest probability)
        predictions = logits.argmax(dim=-1)
        # Compute accuracy using sklearn's accuracy_score
        accuracy = accuracy_score(labels.flatten(), predictions.flatten())
        return {"accuracy": accuracy}
    
    def get_loss_history(self):
        return {"train_loss": self.loss_history}