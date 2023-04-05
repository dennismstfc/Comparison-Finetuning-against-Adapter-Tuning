from sklearn.metrics import f1_score
import numpy as np
from scipy.special import expit

def compute_multi_class_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(logits, axis=1)
    macro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
    return {'macro-f1': macro_f1, 'micro-f1': micro_f1}


def compute_multi_label_metrics(p):
    # Fix gold labels
    y_true = np.zeros((p.label_ids.shape[0], p.label_ids.shape[1] + 1), dtype=np.int32)
    y_true[:, :-1] = p.label_ids
    y_true[:, -1] = (np.sum(p.label_ids, axis=1) == 0).astype('int32')
    # Fix predictions
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = (expit(logits) > 0.5).astype('int32')
    y_pred = np.zeros((p.label_ids.shape[0], p.label_ids.shape[1] + 1), dtype=np.int32)
    y_pred[:, :-1] = preds
    y_pred[:, -1] = (np.sum(preds, axis=1) == 0).astype('int32')
    # Compute scores
    macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
    return {'macro-f1': macro_f1, 'micro-f1': micro_f1}


def compute_multiple_choice_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    # Compute macro and micro F1 for 5-class CaseHOLD task
    macro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
    return {'macro-f1': macro_f1, 'micro-f1': micro_f1}