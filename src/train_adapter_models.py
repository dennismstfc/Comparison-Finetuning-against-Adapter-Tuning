from task import TASK_DATA
from metrics import *
from model import BertAdapterMultipleChoiceModel
from training import get_training_args, MultilabelAdapterTrainer 
from utils import *
from data_preprocessing import DataClass, MultipleChoiceDataset, Split
from custom_callback import TimeCallBack

from transformers import (
    AutoConfig,
    AutoTokenizer, 
    default_data_collator, 
    EarlyStoppingCallback,
    Trainer,
    set_seed
)

from transformers.adapters import (
    AdapterConfig,
    AdapterTrainer,
    BertAdapterModel
)

def start_adapter_tuning(
        checkpoint,
        actual_task,
        adapter_name,
        seed,
        train_duration,
        early_stopping_patience
        ):
    
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

    adapter_config = AdapterConfig(
        mh_adapter=True,
        output_adapter=True,
        reduction_factor=16,
        non_linearity="relu"
    )

    if actual_task != "case_hold":
        model = BertAdapterModel.from_pretrained(
                checkpoint,
                config=config
        )
    else:
        model = BertAdapterMultipleChoiceModel.from_pretrained(
            checkpoint,
            config=config
        )

    # Adding the actual adapter model
    if adapter_name not in model.config.adapters:
        model.add_adapter(adapter_name, config=adapter_config)
    
    # Freezing all BERT Layers and enable the adapter tuning
    model.train_adapter(adapter_name)
    
    '''
    Adding the classifier except for the case_hold model.
    BertAdapterMultipleChoice already implements a classification
    head in the class. Also a data_collator is only needed if multi-class
    or multi-label classifier are trained.
    '''
    if actual_task != "case_hold":
        model.add_classification_head(
            actual_task,
            num_labels=TASK_DATA[actual_task][0],
            activation_function=None,
            overwrite_ok=True
        )

        data_collator = default_data_collator

    # Defining output paths for training args
    create_output_folder(actual_task, "adapter_output")
    output_dir, logging_dir = get_output_logging_paths(actual_task, "adapter_output")
    training_args = get_training_args(output_dir, logging_dir)


    if TASK_DATA[actual_task][1] == "multi_class":
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

        trainer = AdapterTrainer(
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

        trainer = MultilabelAdapterTrainer(
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
            callbacks=[TimeCallBack(actual_task, train_duration, early_stopping_patience)]
        )

    

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    start_adapter_tuning(
        "bert-base-cased",
        actual_task="ecthr_a",
        adapter_name="bottle_neck_adapter"
    )
    