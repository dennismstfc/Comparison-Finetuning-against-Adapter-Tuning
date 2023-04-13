import os

'''
This function creates based on task_name a folder, that is used to store the
outputs of the model during training.
'''
def create_output_folder(task_name, base_dir = "output"):
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
def get_output_logging_paths(task_name, base_dir="output"):
    base_dir = os.path.join("..", base_dir)
    task_dir = os.path.join(base_dir, task_name)
    return task_dir, os.path.join(task_dir, "log") 