from train_adapter_models import start_adapter_tuning
from train_base_models import start_vanilla_finetuning
from task import TASK_DATA
from utils import write_error_log


'''
This function trains all downstream models with vanilla finetuning for 
a given amount of time. 
'''
def run_vanilla_finetuning_all(
        checkpoint: str,
        train_duration: float,
        early_stopping_patience: int
        ) -> None:

    print("Starting Vanilla Finetuning for all tasks.")
    print(f"Each downstream model gets trained for {train_duration / (60 * 60)} h.")
    for actual_task in TASK_DATA.keys():
        try:
            start_vanilla_finetuning(
                checkpoint=checkpoint,
                actual_task=actual_task,
                seed=110110,
                train_duration=train_duration,
                early_stopping_patience=early_stopping_patience
            )
        except Exception as e:
            print(f"Training for {actual_task} failed. Contuning with the next task.")
            write_error_log(actual_task, str(e))


'''
This function trains all downstream models with adapter tuning for a given
amount of time. 
'''
def run_adapter_tuning_all(
        checkpoint: str,
        train_duration: float,
        early_stopping_patience: int
        ) -> None:

    print("Starting Adapter Tuning for all tasks.")
    print(f"Each downstream model gets trained for {train_duration / (60 * 60)} h.")
    for actual_task in TASK_DATA.keys():
        try:
            start_adapter_tuning(
                checkpoint=checkpoint,
                actual_task=actual_task,
                adapter_name="bottleneck_adapter",
                seed=110110,
                train_duration=train_duration,
                early_stopping_patience=early_stopping_patience
            )
        except Exception as e:
            print(f"Training for {actual_task} failed. Contuning with the next task.")
            write_error_log(actual_task, str(e))


if __name__ == "__main__":
    checkpoint = "bert-base-cased"    
    actual_task = "unfair_tos"
    
    # 24 hours in seconds
    train_duration = 24 * 60 * 60

    start_vanilla_finetuning(
        checkpoint=checkpoint,
        actual_task=actual_task,
        seed=110110,
        train_duration=train_duration,
        early_stopping_patience=3
    )

    start_adapter_tuning(
        checkpoint=checkpoint,
        actual_task=actual_task,
        adapter_name="bottlneck_adapter",
        seed=110110,
        train_duration=train_duration,
        early_stopping_patience=3
    )
