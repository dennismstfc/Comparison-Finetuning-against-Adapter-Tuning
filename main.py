from src.task import TASK_DATA
from models.multi_classifier import get_adapter_model, get_base_model
from src.data_preprocessing import get_preprocessed_data
from src.metrics import *
from src.training import get_training_args
from src.utils import *

import os
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    default_data_collator, 
    Trainer,
    EarlyStoppingCallback
    )


if __name__ == "__main__":
    print(TASK_DATA)
    actual_task = "scotus"
    checkpoint = "bert-base-cased"
    label_list = list(range(TASK_DATA[actual_task][0]))


    if TASK_DATA[actual_task][1] == "multi_class":
        # First Train the base model, second train the corresponding adapter model
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint
        )

        config = AutoConfig.from_pretrained(
            checkpoint,
            num_labels=TASK_DATA[actual_task][0],
            finetuning_task=actual_task
        )

        model = get_base_model(
            checkpoint,
            config=config
        )

        print("=======================================================================")
        print(model)
        print("=======================================================================")

        
        if actual_task == "scotus":
            def _preprocess_scotus(examples):
                batch = tokenizer(
                    examples["text"],
                    padding="max_length",
                    max_length=512,
                    truncation=True
                )

                batch["label"] = [label_list.index(labels) for labels in examples["label"]]
                return batch

            # Select preprocess function and task for correct data processing
            train_dataset, test_dataset, eval_dataset = get_preprocessed_data(actual_task, _preprocess_scotus)
            data_collator = default_data_collator
            
        # Defining output paths for training args
        output_dir, logging_dir = get_output_logging_paths(actual_task)
        print(output_dir, logging_dir)
        training_args = get_training_args(output_dir, logging_dir)
        
        create_output_folder(actual_task)
        
        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            compute_metrics = compute_multi_class_metrics,
            tokenizer = tokenizer,
            data_collator = data_collator,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        )

    
    train_result = trainer.train()




