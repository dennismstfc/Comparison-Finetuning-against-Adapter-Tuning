from transformers import TrainingArguments, Trainer, PreTrainedModel
from transformers.adapters import AdapterTrainer

import torch
import torch.nn as nn
from typing import Dict, Tuple, Union

def get_training_args(
        output_dir: str, 
        logging_dir: str
        ) -> TrainingArguments:
    return TrainingArguments(
        do_predict=True,
        output_dir=output_dir,
        logging_dir=logging_dir,
        logging_strategy="epoch",
        logging_steps=100,
        num_train_epochs=1000,
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
    def compute_loss(self, 
                     model: PreTrainedModel, 
                     inputs: Dict[str, torch.Tensor], 
                     return_outputs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


class MultilabelAdapterTrainer(AdapterTrainer):
    def compute_loss(self, 
                     model: PreTrainedModel, 
                     inputs: Dict[str, torch.Tensor], 
                     return_outputs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss