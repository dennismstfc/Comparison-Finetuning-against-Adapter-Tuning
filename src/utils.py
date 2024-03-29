import os
from typing import Tuple

'''
This function creates based on task_name a folder, that is used to store the
outputs of the model during training.
'''
def create_output_folder(
        task_name: str, 
        base_dir: str = "output"
        ) -> None:
    base_dir = os.path.join("..", base_dir)
    # First check if the general output folder exists
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    
    # Create the task specific 
    task_dir = os.path.join(base_dir, task_name)
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    # Define the subdir for logs
    log_dir = os.path.join(task_dir, "log")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)


# Returns the paths to a task-specific output folder and log subdirectory
def get_output_logging_paths(
        task_name: str, 
        base_dir: str ="output"
        ) -> Tuple[str, str]:
    base_dir = os.path.join("..", base_dir)
    task_dir = os.path.join(base_dir, task_name)
    return task_dir, os.path.join(task_dir, "log") 

'''
Logs the current training state into a .txt-file. This function is used in order
to track where the EarlyStopping Callback would have applied.
'''
def log_training_state(
        task_name: str, 
        epoch: float,
        total_flos: float,
        best_model_checkpoint: str,
        base_dir: str ="output"
        ) -> None:
    base_dir = os.path.join("..", base_dir)
    task_dir = os.path.join(base_dir, task_name)
    file_dir = os.path.join(task_dir, "early_stopping_log.txt")

    with open(file_dir, "w") as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Total floating operations: {total_flos}\n")
        f.write(f"Best model checkpoint: {best_model_checkpoint}\n")
    
'''
Creates a .txt-file that stores informations about failed training attempt.
'''
def write_error_log(task_name: str, error_msg: str) -> None:
    file_dir = os.path.join("..", task_name)
    file_dir = os.path.join(file_dir, "_training_error_log.txt")
    
    with open(file_dir, "w") as f:
        f.write(f"Training failed for: {task_name}\n")
        f.write(f"Occured error: {error_msg}\n")