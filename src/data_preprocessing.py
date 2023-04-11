from datasets import load_dataset
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from torch.utils.data.dataset import Dataset
from typing import List, Optional
from transformers import PreTrainedTokenizer
from filelock import FileLock

import torch
import tqdm
import datasets
import re
import os


class DataClass:
    def __init__(self,
                task, 
                variant,
                tokenizer,
                padding, 
                max_seq_length, 
                trunc,
                label_list
                ):
        self.task = task
        self.variant = variant
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_seq_length = max_seq_length
        self.trunc = trunc
        self.label_list = label_list


    def _get_splitted_data(self):
        train = load_dataset("lex_glue", self.task, split="train")
        test = load_dataset("lex_glue", self.task, split="test")
        eval = load_dataset("lex_glue", self.task, split="validation")
        return train, test, eval


    def _preprocess_function(self, examples):
        if self.task == "ecthr_a" or self.task == "ecthr_b":
            cases = []
            for case in examples["text"]:
                cases.append(f"\n".join(case))
            
            batch = self.tokenizer(
                cases,
                padding=self.padding,
                max_length=self.max_seq_length,
                truncation=self.trunc
            )
        
        else:
            batch = self.tokenizer(
                examples["text"],
                padding=self.padding,
                max_length=self.max_seq_length,
                truncation=self.trunc
            )

        if self.variant == "multi_class":
            batch["label"] = [self.label_list.index(labels) for labels in examples["label"]]
        elif self.variant == "multi_label":
            batch["labels"] = [[1 if label in labels else 0 for label in self.label_list] for labels in examples["labels"]]


        return batch

    def get_preprocessed_data(self):
        train_dataset, test_dataset, eval_dataset = self._get_splitted_data()

        train_dataset = train_dataset.map(
            self._preprocess_function,
            batched=True,
            desc="Running tokenizer on train_dataset"
        )

        test_dataset = test_dataset.map(
            self._preprocess_function,
            batched=True,
            desc="Running tokenizer on test_dataset"
        )

        eval_dataset = eval_dataset.map(
            self._preprocess_function,
            batched=True,
            desc="Running tokenizer on eval_dataset"
        )

        return train_dataset, test_dataset, eval_dataset 



@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]

def convert_examples_to_features(
    examples: datasets.Dataset,
    max_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        choices_inputs = []
        for ending_idx, ending in enumerate(example['endings']):
            context = example['context']
            inputs = tokenizer(
                context,
                ending,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )

            choices_inputs.append(inputs)
        
        label = example['label']

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )

    return features


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class MultipleChoiceDataset(Dataset):
    """
    PyTorch multiple choice dataset class
    """

    features: List[InputFeatures]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        task: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Split = Split.train,
    ):
        dataset = datasets.load_dataset('lex_glue', task)
        tokenizer_name = re.sub('[^a-z]+', ' ', tokenizer.name_or_path).title().replace(' ', '')
        cached_features_file = os.path.join(
            '.cache',
            task,
            "cached_{}_{}_{}_{}".format(
                mode.value,
                tokenizer_name,
                str(max_seq_length),
                task,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        if not os.path.exists(os.path.join('.cache', task)):
            if not os.path.exists('.cache'):
                os.mkdir('.cache')
            os.mkdir(os.path.join('.cache', task))
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                self.features = torch.load(cached_features_file)
            else:
                if mode == Split.dev:
                    examples = dataset['validation']
                elif mode == Split.test:
                    examples = dataset['test']
                elif mode == Split.train:
                    examples = dataset['train']
                self.features = convert_examples_to_features(
                    examples,
                    max_seq_length,
                    tokenizer,
                )
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]