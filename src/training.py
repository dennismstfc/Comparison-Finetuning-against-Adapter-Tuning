from transformers import TrainingArguments, Trainer
from transformers.adapters import AdapterTrainer
import torch.nn as nn

def get_training_args(output_dir, logging_dir):
    return TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        logging_strategy="epoch",
        logging_steps=100,
        num_train_epochs=100000,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=3e-6,
        save_strategy="epoch",
        save_steps=100,
        evaluation_strategy="epoch",
        eval_steps=100,
        load_best_model_at_end=True
    )


class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


class MultilabelAdapterTrainer(AdapterTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss