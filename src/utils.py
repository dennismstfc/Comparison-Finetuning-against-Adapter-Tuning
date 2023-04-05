import os

'''
This function creates based on task_name a folder, that is used to store the
outputs of the model during training.
'''
def create_output_folder(task_name, base_dir = "output"):
    # First check if the general output folder exists
    if not os.path.exists("output"):
        os.mkdir("output")
    
    # Create the task specific 
    task_dir = os.path.join(base_dir, task_name)
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    # Define the subdir for logs
    log_dir = os.path.join(task_dir, "log")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)