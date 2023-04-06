from transformers.adapters import AdapterConfig

adapter_config = AdapterConfig(
    mh_adapter=True, 
    output_adapter=True, 
    reduction_factor=16, 
    non_linearity="relu"
    )