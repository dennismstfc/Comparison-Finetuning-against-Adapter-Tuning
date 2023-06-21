from task import TASK_DATA
from metrics import *
from training import get_training_args, MultilabelTrainer
from utils import *
from data_preprocessing import DataClass, MultipleChoiceDataset, Split
from custom_callback import TimeCallBack

from transformers import (
    AutoConfig,
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    default_data_collator, 
    Trainer,
    EarlyStoppingCallback,
    AutoModelForMultipleChoice,
    set_seed,
)

'''
TODO:
- Add listings
- Add comments for all functions
'''


def start_vanilla_finetuning(
        checkpoint: str, 
        actual_task: str,
        seed: int,
        train_duration: float,
        early_stopping_patience: int
        ) -> None:

    set_seed(seed)

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
    create_output_folder(actual_task)
    output_dir, logging_dir = get_output_logging_paths(actual_task)
    training_args = get_training_args(output_dir, logging_dir)

    if actual_task != "case_hold":
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            config=config
        )
        
        data_collator = default_data_collator

        data_helper = DataClass(
            actual_task,
            tokenizer=tokenizer,
            variant=TASK_DATA[actual_task][1],
            padding="max_length",
            max_seq_length=512,
            trunc=True,
            label_list=label_list        
        )

        train_dataset, eval_dataset, test_dataset = data_helper.get_preprocessed_data()
    else:
        model = AutoModelForMultipleChoice.from_pretrained(
            checkpoint,
            config=config
        )

        train_dataset = MultipleChoiceDataset(
            tokenizer=tokenizer,
            task=actual_task,
            max_seq_length=512,
            mode=Split.train
        )

        eval_dataset = MultipleChoiceDataset(
            tokenizer = tokenizer,
            task="case_hold",
            max_seq_length=512,
            mode=Split.dev
        )


    if TASK_DATA[actual_task][1] == "multi_class":
        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            compute_metrics = compute_multi_class_metrics,
            tokenizer = tokenizer,
            data_collator = data_collator,
            callbacks=[TimeCallBack(actual_task, train_duration, early_stopping_patience)]
        )


    if TASK_DATA[actual_task][1] == "multi_label":
        trainer = MultilabelTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_multi_label_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=[TimeCallBack(actual_task, train_duration, early_stopping_patience)]
        )
    
    
    if actual_task == "case_hold":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_multiple_choice_metrics,
            callbacks=[TimeCallBack(actual_task, train_duration, early_stopping_patience)]
        )


    trainer.train()
    trainer.save_model()