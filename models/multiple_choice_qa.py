import torch
import torch.nn as nn
from typing import Optional

from transformers import AutoModelForMultipleChoice
from transformers.adapters import BertAdapterModel


def get_base_model(checkpoint, config):
    return AutoModelForMultipleChoice.from_pretrained(
        checkpoint,
        config=config
    )



'''
Modified model from:
\transformers\src\transformers\models\bert\modeling_bert.py
'''
class BertAdapterMultipleChoiceModel(BertAdapterModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertAdapterModel(config)
        classifier_dropout = (
            config.classifer_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)
        
        # Let's see if we need that shit
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)


        output = (reshaped_logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


def get_adapter_model(
        checkpoint,
        config,
        adapter_config,
        adapter_name="bottleneck_adapter"
        ):
    
    model = BertAdapterMultipleChoiceModel.from_pretrained(
        checkpoint,
        config
    )

    # Adding the actual adpater model
    model.add_adapter(adapter_name, config=adapter_config)

    # Freezing all BERT layers and enable the adpater tuning
    model.train_adapter(adapter_name)

    # No classifier is needed here, because it was already implemented in BertAdapterMultipleChoiceModel class
    return model

