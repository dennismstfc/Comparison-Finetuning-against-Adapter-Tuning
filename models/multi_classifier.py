from transformers import AutoModelForSequenceClassification
from transformers.adapters import BertAdapterModel


def get_base_model(checkpoint, config):
    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        config=config
    )



def get_adapter_model(
        checkpoint, 
        config, 
        adapter_config, 
        task_name,
        num_labels,
        adapter_name="bottleneck_adapter"
        ):
    model = BertAdapterModel.from_pretrained(
        checkpoint,
        config
    )
    
    # Adding the actual adapter model
    model.add_adapter(adapter_name, config=adapter_config)
    
    # Freezing all BERT Layers and enable the adapter tuning
    model.train_adpater(adapter_name)
    
    # Adding the classifier
    return model.add_classification_head(
        task_name,
        num_labels=num_labels,
        activation_function=None,
        overwrite_ok=True
    )