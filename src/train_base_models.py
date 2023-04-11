from task import TASK_DATA
from metrics import *
from training import get_training_args, MultilabelTrainer
from utils import *
from data_preprocessing import DataClass, MultipleChoiceDataset, Split

from transformers import (
    AutoConfig,
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    default_data_collator, 
    Trainer,
    EarlyStoppingCallback,
    AutoModelForMultipleChoice
)

'''
TODO:
- Add listings
- Select for all tasks the hyperparamters in the src/task.py file (use dict)
- Implement adapter training for case_hold in train_adapter_models.py
- Write script, which trains all 14 models sequentially
- Add comments for all functions
'''

def start_vanilla_finetuning(
        checkpoint, 
        actual_task
        ):

    label_list = list(range(TASK_DATA[actual_task][0]))

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        use_fast=False if not actual_task == "case_hold" else True
    )

    config = AutoConfig.from_pretrained(
        checkpoint,
        num_labels=TASK_DATA[actual_task][0],
        finetuning_task=actual_task,
        use_fast=False if not actual_task == "case_hold" else True
    )

    # Defining output paths for training args
    output_dir, logging_dir = get_output_logging_paths(actual_task)
    training_args = get_training_args(output_dir, logging_dir)

    if TASK_DATA[actual_task][1] == "multi_class":
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            config=config
        )

        data_helper = DataClass(
            actual_task,
            TASK_DATA[actual_task][1],
            tokenizer=tokenizer,
            padding="max_length",
            max_seq_length=512,
            trunc=True,
            label_list=label_list       
        )
        train_dataset, test_dataset, eval_dataset = data_helper.get_preprocessed_data()
        data_collator = default_data_collator
            
        
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


    if TASK_DATA[actual_task][1] == "multi_label":
        config = AutoConfig.from_pretrained(
            checkpoint,
            num_labels=TASK_DATA[actual_task][0],
            finetuning_task=TASK_DATA[actual_task][1]
        )

        model = AutoModelForSequenceClassification.from_pretrained(
                    checkpoint,
                    config=config
                )

        data_helper = DataClass(
            actual_task,
            tokenizer=tokenizer,
            variant=TASK_DATA[actual_task][1],
            padding="max_length",
            max_seq_length=128,
            trunc=True,
            label_list=label_list        
            )

        train_dataset, test_dataset, eval_dataset = data_helper.get_preprocessed_data()
        data_collator = default_data_collator


        trainer = MultilabelTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_multi_label_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    
    
    if actual_task == "case_hold":
        model = AutoModelForMultipleChoice.from_pretrained(
            checkpoint,
            config=config
        )

        train_dataset = MultipleChoiceDataset(
            tokenizer=tokenizer,
            task=actual_task,
            max_seq_length=256,
            mode=Split.train
        )

        eval_dataset = MultipleChoiceDataset(
            tokenizer = tokenizer,
            task="case_hold",
            max_seq_length=256,
            mode=Split.dev
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_multiple_choice_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

    
    trainer.train()
    trainer.save_model()


    