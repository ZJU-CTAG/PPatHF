import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import json

import torch
import transformers
from torch.utils.data import Dataset


IGNORE_INDEX = -100
PROMPT_TEMPLATE = {
    "instruction": (
        "Below is a patch (including function before and function after) from vim, paired with a corresponding function before from neovim. "
        "Adapt the patch from vim to neovim by generating the function after based on the given function before.\n\n"
    ),
    "context": (
        "### Function Before (vim):\n{func_before_source}\n\n"
        "### Function After (vim):\n{func_after_source}\n\n"
        "### Function Before (neovim):\n{func_before_target}\n\n"
        "### Function After (neovim):\n"
    ),
    "output": "{func_after_target}"
}
PROMPT_TEMPLATE_TRANS_MSG = {
    "instruction": (
        "Below is a commit message, paired with a corresponding function before from {repo}. "
        "Apply the patch according to the commit message by generating the function after based on the given function before.\n\n"
    ),
    "context": (
        "### Commit Message:\n{commit_msg}\n\n"
        "### Function Before ({repo}):\n{func_before}\n\n"
        "### Function After ({repo}):\n"
    ),
    "output": "{func_after}"
}
PROMPT_TEMPLATE_DICT = {
    "trans_msg": PROMPT_TEMPLATE_TRANS_MSG,
    "trans_patch": PROMPT_TEMPLATE,
}


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = [
        len(_) for _ in input_ids
    ]

    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    def add_eos_token(x):
        if (x[-1] != tokenizer.eos_token_id) and (len(x) < tokenizer.model_max_length):
            x = torch.cat([x, x.new(data=[tokenizer.eos_token_id])])
        return x
    input_ids = [add_eos_token(_) for _ in input_ids]
    
    labels = copy.deepcopy(input_ids)
    for y, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        y[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, 'r'))

        logging.warning("Formatting inputs...")
        sources, targets = [], []
        for example in list_data_dict:
            task = example.get("task", "trans_patch")
            sources.append(
                PROMPT_TEMPLATE_DICT[task]["instruction"].format_map(example) + PROMPT_TEMPLATE_DICT[task]["context"].format_map(example)
            )
            targets.append(
                PROMPT_TEMPLATE_DICT[task]["output"].format_map(example)
            )
        
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class Testset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, key_output: str="func_after_target"):
        super(Testset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, 'r'))

        logging.warning("Formatting inputs...")

        sources, targets = [], []
        for example in list_data_dict:
            task = example.get("task", "trans_patch")
            sources.append(
                PROMPT_TEMPLATE_DICT[task]["instruction"].format_map(example) + PROMPT_TEMPLATE_DICT[task]["context"].format_map(example)
            )
            targets.append(
                PROMPT_TEMPLATE_DICT[task]["output"].format_map(example)
            )

        logging.warning("Tokenizing inputs... This may take some time...")
        input_ids = [
            tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                max_length=tokenizer.model_max_length,
                truncation=True,
            )["input_ids"]
            for text in sources
        ]

        self.input_ids = input_ids
        self.targets = targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i])
    
    def get_targets(self):
        return self.targets


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
        )