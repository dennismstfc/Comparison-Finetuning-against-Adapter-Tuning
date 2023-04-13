from utils import log_training_state

from transformers import TrainerCallback
import numpy as np
import time


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
            task_name,
            training_duration, 
            early_stopping_patience, 
            early_stopping_threshold = 0.0,
            base_dir = "output" 
            ):
        self.task_name = task_name
        self.base_dir = base_dir
        self.training_duration = training_duration
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_treshold = early_stopping_threshold
        self.early_stopping_patience_counter = 0
        self.determined_early_stop_checkpoint = False


    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1


    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if not self.determined_early_stop_checkpoint:
            metric_to_check = args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics.get(metric_to_check)

            self.check_metric_value(args, state, control, metric_value)
            if self.early_stopping_patience_counter >= self.early_stopping_patience:
                # Log the current state
                log_training_state(
                    task_name=self.task_name,
                    epoch= state.epoch,
                    total_flos=state.total_flos,
                    best_model_checkpoint=state.best_model_checkpoint,
                    base_dir=self.base_dir
                )
                self.determined_early_stop_checkpoint = True

    def on_train_begin(self, args, state, control, **kwargs):
        self.training_start = time.time()
        start_time_formatted = time.gmtime(self.training_start)
        print(f"Training start: {time.strftime('%Y-%m-%d %H:%M:%S', start_time_formatted)}")


    def on_epoch_begin(self, args, state, control, **kwargs):
        actual_time = time.time()
        proceeded_training_time = actual_time - self.training_start

        if proceeded_training_time >= self.training_duration:
            print("\n\nThe training time has ended. Finishing the training!\n\n")
            control.should_training_stop = True

        print(f"Proceeded training time: {(proceeded_training_time / (60)) / 60} h")
    


