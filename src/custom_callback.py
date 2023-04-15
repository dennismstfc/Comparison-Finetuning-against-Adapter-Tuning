from utils import log_training_state

import numpy as np
import time
import os
import shutil

from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
    )

'''
Custom Callback that stops the training after a given training duration,
which can be e.g. 24 hours. Also this class implements EarlyStopping
functionality which determines the best model with given metrics. Instead of 
stopping the training, the training continues and the checkpoint, where the
EarlyStopping applies is logged into a textfile. This is needed for further 
evaluation.
'''
class TimeCallBack(TrainerCallback):
    def __init__(
            self,
            task_name: str,
            training_duration: int, 
            early_stopping_patience: int, 
            early_stopping_threshold: float = 0.0,
            base_dir: str = "output" 
            ):
        self.task_name = task_name
        self.base_dir = base_dir
        self.training_duration = training_duration
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_patience_counter = 0
        self.determined_early_stop_checkpoint = False
        self.best_model = None


    def check_metric_value(
            self, 
            args: TrainingArguments, 
            state: TrainerState, 
            control: TrainerControl, 
            metric_value
            ) -> None:
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1


    def on_evaluate(
            self, 
            args: TrainingArguments, 
            state: TrainerState, 
            control: TrainerControl, 
            metrics, 
            **kwargs
            ) -> None:
        if not self.determined_early_stop_checkpoint:
            metric_to_check = args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics.get(metric_to_check)

            self.check_metric_value(args, state, control, metric_value)
            if self.early_stopping_patience_counter >= self.early_stopping_patience:
                print("\n\n The current epoch is according to the early stop the best! Logging the parameters... \n\n")
                # Log the current state
                log_training_state(
                    task_name=self.task_name,
                    epoch= state.epoch,
                    total_flos=state.total_flos,
                    best_model_checkpoint=state.best_model_checkpoint,
                    base_dir=self.base_dir
                )
                self.determined_early_stop_checkpoint = True
                self.best_model = state.best_model_checkpoint
                


    def on_train_begin(
            self, 
            args: TrainingArguments, 
            state: TrainerState, 
            control: TrainerControl, 
            **kwargs
            ):
        self.training_start = time.time()
        start_time_formatted = time.gmtime(self.training_start)
        print(f"Training start: {time.strftime('%Y-%m-%d %H:%M:%S', start_time_formatted)}")


    def on_epoch_begin(
            self, 
            args: TrainingArguments, 
            state: TrainerState, 
            control: TrainerControl, 
            **kwargs
            ):
        actual_time = time.time()
        proceeded_training_time = actual_time - self.training_start

        if proceeded_training_time >= self.training_duration:
            print("\n\nThe training time has ended. Finishing the training!\n\n")
            control.should_training_stop = True

        print(f"Proceeded training time: {(proceeded_training_time / (60 * 60))} h of {self.training_duration / (60 * 60)} h")
        print(f"Done: {proceeded_training_time / self.training_duration} %")


        '''
        Delete all checkpoints except the checkpoint where the early stop would have applied. 
        This is needed in order to not flood the GPU-Cluster with a lot of data, that is produced
        during the training process.
        '''
        if self.determined_early_stop_checkpoint:
            output_path = os.path.join("..", self.base_dir)
            output_path = os.path.join(output_path, self.task_name)

            for root, dirs, _ in os.walk(output_path, topdown=True):
                for sub_dir in dirs:
                    if "checkpoint" in sub_dir and sub_dir != self.best_model:
                        print(self.best_model)
                        tmp_path = os.path.join(root, sub_dir)
                        try:
                            shutil.rmtree(tmp_path)
                        except:
                            print(f"Error: couldn't delete {tmp_path}")
