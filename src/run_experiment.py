from train_adapter_models import start_adapter_tuning
from train_base_models import start_vanilla_finetuning
from task import TASK_DATA

def run_experiment_all(checkpoint):
    for actual_task in TASK_DATA.keys():
        start_vanilla_finetuning(
            checkpoint=checkpoint,
            actual_task=actual_task
        )

        start_adapter_tuning(
            checkpoint=checkpoint,
            actual_task=actual_task,
            adapter_name="bottlneck_adapter"
        )



if __name__ == "__main__":
    checkpoint = "bert-base-cased"    
    actual_task = "unfair_tos"
    
    start_vanilla_finetuning(
        checkpoint=checkpoint,
        actual_task=actual_task,
        seed=110110
    )

    start_adapter_tuning(
        checkpoint=checkpoint,
        actual_task=actual_task,
        adapter_name="bottlneck_adapter",
        seed=110110
    )
