import torch
torch.backends.cuda.matmul.allow_tf32 = True  # tf32 to replace fp32 to allow higher throughout

import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from typing import List
import re
import difflib
from pprint import pprint

from predict import generate
from metrics import MetricsSampleWise
from reduction.fcu import FunctionCompareUtilities
from config import LLM_PATH, LLM_NAME_PATH_MAPPING, MODEL_PATH, DATA_PATH, OUTPUT_PATH


def generate_dummy(data_path, output_path):
    test_set = json.load(open(data_path))

    key_str_before_target = "### Function After (neovim):"

    dummy_output_key = "func_before_target"
    gens = [_[dummy_output_key] for _ in test_set]

    with open(output_path, 'w') as f:
        json.dump(gens, f, indent=4)


def cal_metrics(
        data_path,
        output_path,
        do_postprocess: bool = True,
        do_recover: bool = False,
        reference_key: str = "func_after_target",
        reference_before_key: str = "func_before_target",
        save_sample_wise_results: bool = False):
    test_set = json.load(open(data_path))
    generations = json.load(open(output_path))
    
    if len(test_set) != len(generations):
        raise ValueError("the lengths of test set and output are not equal")

    key_str_before_target = "### Function After (neovim):"
    key_str_after_target = ["\n###"]
    metrics = MetricsSampleWise(
        generations=generations,
        test_set=test_set,
        do_postprocess=do_postprocess,
        do_recover=do_recover,
        key_str_before_target=key_str_before_target,
        key_str_after_target=key_str_after_target,
        reference_key=reference_key,
        reference_before_key=reference_before_key,
    )

    m = metrics.cal_metrics()
    
    m["test_set"] = data_path.split("/")[-1]
    print(m)

    metrics_path = output_path.replace("result.json", "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(m, f, indent=4)

    if save_sample_wise_results:
        results_path = output_path.replace("result.json", "result_processed.json")
        metrics.save_sample_wise_test_results(save_path=results_path, save_metadata=True)


def analyze_test_results(result_path: str, save_analyzed_results=False):
    print(f"########## {result_path.split('/')[-1]} ##########")

    col_id = "commit_id_target"

    result = json.load(open(result_path))

    result_df = pd.DataFrame(result)
    print("#test samples:", len(result_df))
    print(result_df.columns)

    result_df["with_code_change"] = result_df["src_distance"].map(lambda x: True if x != 0 else False)
    fcu = FunctionCompareUtilities()
    result_df["with_code_change_source"] = result_df.apply(lambda row: False if np.array_equal(fcu.get_cleaned_tokens(row["func_before_source"]), fcu.get_cleaned_tokens(row["func_after_source"])) else True, axis=1)
    result_df["generate_new_func_source"] = result_df["func_before_source"].map(lambda x: True if x.strip() == "" else False)
    result_df["generate_new_func_target"] = result_df["func_before_target"].map(lambda x: True if x.strip() == "" else False)
    
    result_df["complete_generation"] = result_df["gen_processed"].map(lambda x: True if x.strip().splitlines()[-1][0] == '}' else False)
    
    if len(result_df[result_df["with_code_change_source"] == False]) > 0:
        print("find no_code_change_source in aligned test set:", result_df[result_df["with_code_change_source"] == False][col_id].to_list())
    if len(result_df[result_df["generate_new_func_source"] == True]) > 0:
        raise ValueError("find generate_new_func_source in aligned test set")
    if len(result_df[result_df["generate_new_func_target"] == True]) > 0:
        raise ValueError("find generate_new_func_target in aligned test set")

    print("========== no code change ==========")
    reuslt_df_no_code_change = result_df[result_df["with_code_change"] == False]
    print("#test_sample_no_code_change:", len(reuslt_df_no_code_change))

    print("========== complete generation ==========")
    inconsistent = result_df[(result_df["complete_generation"] == False) & (result_df["exact_match"] == True)]
    print("[inconsistency] incomplete yet correct:", len(inconsistent))
    if len(inconsistent) > 0:
        raise ValueError("generation incomplete yet exact_match=True")
    result_df_complete = result_df[result_df["complete_generation"] == True]
    print("#test_sample_generation_completed:", len(result_df_complete))
    
    if save_analyzed_results:
        analyzed_result_path = result_path.replace("result_processed.json", "result_processed_analyzed.json")
        with open(analyzed_result_path, 'w') as f:
            json.dump(result_df.to_dict(orient="records"), f, indent=4)


def align_test_metrics(result: pd.DataFrame, dummy_result: pd.DataFrame):
    if any(result["complete_generation"].map(lambda x: type(x) != bool).to_list()):
        raise ValueError("complete_generation contains value not in {True, False}")

    col_id = "commit_id_target"
    col_metrics = ["exact_match", "rel_distance", "gen_distance", "src_distance"]
    
    result_aligned = result.merge(dummy_result, how="outer", on=col_id, suffixes=('_x', '_y'), validate="1:1")
    if len(result_aligned) != len(dummy_result):
        raise ValueError(f"test set not correctly aligned. {len(result_aligned)} vs. {len(dummy_result)}")
    result_aligned["use_dummy"] = result_aligned["complete_generation"].map(lambda x: x != True)
    print("#use_dummy:", len(result_aligned[result_aligned["use_dummy"] == True]))

    metrics = {}
    for m in col_metrics:
        result_aligned[m] = result_aligned.apply(lambda row: row[f"{m}_x"] if not row["use_dummy"] else row[f"{m}_y"], axis=1)
        
        metrics[f"{m}_aligned"] = result_aligned[m].mean()
        metrics[f"{m}_model"] = result_aligned[result_aligned["use_dummy"] == False][m].mean()
        metrics[f"{m}_model(dummy)"] = result_aligned[result_aligned["use_dummy"] == False][f"{m}_y"].mean()
        metrics[f"{m}_dummy"] = result_aligned[result_aligned["use_dummy"] == True][m].mean()
        if m in ["exact_match"]:
            metrics[f"{m}_sum_aligned"] = sum(result_aligned[m].to_list())
            metrics[f"{m}_sum_model"] = sum(result_aligned[result_aligned["use_dummy"] == False][m].to_list())
            metrics[f"{m}_sum_model(dummy)"] = sum(result_aligned[result_aligned["use_dummy"] == False][f"{m}_y"].to_list())
            metrics[f"{m}_sum_dummy"] = sum(result_aligned[result_aligned["use_dummy"] == True][m].to_list())

    metrics["#able_to_handle"] = len(result)
    metrics["#unable_to_handle(dummy)"] = len(dummy_result) - metrics["#able_to_handle"]
    metrics["#complete_generate"] = len(result[result["complete_generation"] == True])
    metrics["#incomplete_generate"] = len(dummy_result) - metrics["#complete_generate"]
    metrics["#total"] = len(dummy_result)

    print(json.dumps(metrics, indent=4))

    return metrics, result_aligned


def compare_test_metrics(result_list: List[pd.DataFrame], dummy_result):
    for result in result_list:
        if any(result["complete_generation"].map(lambda x: type(x) != bool).to_list()):
            raise ValueError("complete_generation contains value not in {True, False}")

    col_id = "commit_id_target"
    col_metrics = ["exact_match", "rel_distance", "gen_distance", "src_distance"]
    total_sample_num = len(dummy_result)
    
    result_list = [r[r["complete_generation"] == True].copy() for r in result_list]
    
    result_id_shared = set(dummy_result[col_id].to_list())
    for result in result_list:
        result_id_shared &= set(result[col_id].to_list())

    metric_list = []
    for result in result_list:
        print(f"==========")

        metric = {"#total": total_sample_num, "#incomplete_generate": total_sample_num - len(result), "#complete_generate": len(result)}
        
        result["is_shared"] = result[col_id].map(lambda x: x in result_id_shared)
        result_shared = result[result["is_shared"] == True]
        result_unique = result[result["is_shared"] == False]

        metric["#shared"] = len(result_shared)
        metric["#unique"] = len(result_unique)

        for m in col_metrics:
            metric[f"{m}_shared"] = result_shared[m].mean()
            metric[f"{m}_unique"] = result_unique[m].mean()
            if m in ["exact_match"]:
                metric[f"{m}_sum_shared"] = sum(result_shared[m].to_list())
                metric[f"{m}_sum_unique"] = sum(result_unique[m].to_list())

        print(json.dumps(metric, indent=4))
        metric_list.append(metric)
    
    return metric_list


DIFF_HEAD = re.compile(r"^@@[^@]+@@$")
PLACEHOLDER = re.compile(r"/\* placeholder_(\d+) \*/")

# SLICE_PATH = ""  # for non-sliced dataset
# SLICE_PATH = "sliced/"  # for sliced dataset

if __name__ == "__main__":
    model_max_length = "all"
    data_path = DATA_PATH + f"vim_neovim_test_{model_max_length}.json"
    output_path = OUTPUT_PATH + f"out_dummy_{model_max_length}_result.json"
    generate_dummy(
        data_path=data_path,
        output_path=output_path
    )
    
    # cal_metrics(
    #     data_path=data_path,
    #     output_path=output_path,
    #     do_postprocess=False,  # skip the post process
    #     # do_recover=False,
    #     # reference_key=reference_key,
    #     # reference_before_key=reference_before_key,
    #     save_sample_wise_results=True,
    # )
