import pandas as pd
import json
from transformers import AutoTokenizer
import numpy as np
import git
import os
import re
from datetime import datetime
import copy
import pydriller
from typing import List, Optional
import random
import warnings
import difflib
from pprint import pprint

from porting.data import PROMPT_TEMPLATE_DICT
from reduction.fcu import FunctionCompareUtilities
from reduction.reducer import Reducer

from config import LLM_PATH, LLM_NAME_PATH_MAPPING, DATA_PATH


def filter_by_max_length(dataset: List[dict], max_token_num=2048, include_output=True):
    print("#dataset before length filtering:", len(dataset))

    model_path = LLM_PATH + f"{LLM_NAME_PATH_MAPPING['starcoder']}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset_selected = []
    for sample in dataset:
        task = sample.get("task", "trans_patch")
        input_prompt = PROMPT_TEMPLATE_DICT[task]["instruction"].format_map(sample) + PROMPT_TEMPLATE_DICT[task]["context"].format_map(sample)
        if include_output:
            input_prompt += PROMPT_TEMPLATE_DICT[task]["output"].format_map(sample)

        token_num = len(tokenizer(input_prompt)["input_ids"])

        if token_num < max_token_num:
            dataset_selected.append(sample)
    
    print("#dataset after length filtering:", len(dataset_selected))

    return dataset_selected


def filter_by_date(dataset: List[dict], split_date, date_key="neovim_committer_date"):
    print("#dataset before date filtering:", len(dataset))

    dataset.sort(key=lambda x: datetime.strptime(x[date_key], DATE_FORMAT))

    split_date = datetime.strptime(split_date, DATE_FORMAT)
    print("split_date:", split_date)

    split_idx = -1
    for idx, sample in enumerate(dataset):
        sample_date = datetime.strptime(sample[date_key], DATE_FORMAT)
        if sample_date >= split_date:
            split_idx = idx
            break

    dataset_before = dataset[:split_idx]
    dataset_after = dataset[split_idx:]
    print("first_date of dataset_after:", datetime.strptime(dataset_after[0][date_key], DATE_FORMAT))
    print("#dataset before split_date:", len(dataset_before))
    print("#dataset after split_date:", len(dataset_after))
    
    return dataset_before, dataset_after


def prepare_testset(
        data_path,
        split_date: Optional[str] = None):
    
    dataset = json.load(open(data_path))
    print(dataset[0].keys())
    print("#dataset:", len(dataset))

    date_key = "neovim_committer_date"
    _, dataset = filter_by_date(dataset=dataset, date_key=date_key, split_date=split_date)

    # 2k, 4k, 8k
    max_token_num = 1024 * 8
    dataset = filter_by_max_length(dataset=dataset, max_token_num=max_token_num, include_output=False)

    with open(DATA_PATH + f"vim_neovim_test_{max_token_num}.json", 'w') as f:
        json.dump(dataset, f, indent=4)


def prepare_testset_sliced(
        data_path,
        source_type = "slice+placeholder",
        target_type = "slice+placeholder",
        split_date: Optional[str] = None):
    
    dataset = json.load(open(data_path))
    df = pd.DataFrame(dataset)
    print(df.columns)
    print("#dataset:", len(dataset))

    for key in ["func_before", "func_after"]:
        for suffix in ["source", "target"]:
            df[f"{key}_{suffix}_origin"] = df[f"{key}_{suffix}"]

    if source_type == "slice+placeholder":
        df["func_before_source"] = df["func_before_sliced_source"]
        df["func_after_source"] = df["func_after_sliced_source"]
    else:
        raise ValueError(f"invalid source_type={source_type}")
    
    if target_type == "slice+placeholder":
        df["func_before_target"] = df["func_before_sliced_target"]
    else:
        raise ValueError(f"invalid target_type={target_type}")

    dataset = df.to_dict(orient="records")

    date_key = "neovim_committer_date"
    _, dataset = filter_by_date(dataset=dataset, date_key=date_key, split_date=split_date)

    # 2k, 4k, 8k
    max_token_num = 1024 * 8
    dataset = filter_by_max_length(dataset=dataset, max_token_num=max_token_num, include_output=False)

    with open(DATA_PATH + SLICE_PATH + f"vim_neovim_test_sliced_{max_token_num}.json", 'w') as f:
        json.dump(dataset, f, indent=4)


DATE_FORMAT = '%Y-%m-%d %H:%M:%S%z'

# switch between diffrent slice settings
# SLICE_PATH = ""  # for non-sliced dataset
SLICE_PATH = "sliced/"  # for sliced dataset

if __name__ == "__main__":
    data_path = DATA_PATH + "vim_neovim_test_all.json"
    split_date = "2022-07-01 00:00:00+00:00"  # pretrain data of StarCoder is collected before 2022-07-01
    prepare_testset(data_path=data_path, split_date=split_date)

    # data_path = DATA_PATH + SLICE_PATH + "vim_neovim_test_sliced_all.json"
    # split_date = "2022-07-01 00:00:00+00:00"  # pretrain data of StarCoder is collected before 2022-07-01
    # prepare_testset_sliced(
    #     data_path = data_path,
    #     split_date=split_date,
    # )