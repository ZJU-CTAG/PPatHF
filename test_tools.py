import json
import torch
import pandas as pd
import os

from test import cal_metrics, analyze_test_results, align_test_metrics, compare_test_metrics
from config import LLM_PATH, LLM_NAME_PATH_MAPPING, MODEL_PATH, DATA_PATH, OUTPUT_PATH, EXTRACTED_METRIC_PATH

def tool_run_tests():
    model_name = "starcoder"
    base_model_name_or_path = LLM_PATH + f"{LLM_NAME_PATH_MAPPING[model_name]}/"
    # 2k, 4k, 8k
    model_max_length = 1024 * 8

    peft_output_list = [
        None,  # the off-the-shelf LLM
        "xxx",  # finetuning output
    ]

    data_path = f"vim_neovim_test_{model_max_length}.json"
    data_path_sliced = f"vim_neovim_test_sliced_{model_max_length}.json"

    data_path_list = data_path
    
    output_path_list = [
        None,
        # "sliced"
        "xxx",
        # "xxx_sliced"
    ]

    df = pd.DataFrame({
        "peft_output": peft_output_list,
        "output_path": output_path_list,
        "data_path": data_path_list,
    })
    df["base_model_name_or_path"] = base_model_name_or_path
    df["model_max_length"] = model_max_length
    df["max_length"] = model_max_length

    df["peft_output"] = df["peft_output"].map(lambda x: (MODEL_PATH + x) if x is not None else None)
    df["output_path"] = df["output_path"].map(lambda x: OUTPUT_PATH + SLICE_PATH + (f"out_{model_name}_{model_max_length}_{x}_result.json" if x is not None else f"out_{model_name}_{model_max_length}_result.json"))
    df["data_path"] = df["data_path"].map(lambda x: DATA_PATH + SLICE_PATH + x)

    print("#test task num:", len(df))

    avail_cuda = [0]
    num_avail_cuda = len(avail_cuda)

    cmd_param_base = "--base_model_name_or_path '{base_model_name_or_path}' --data_path '{data_path}' --output_path '{output_path}' --model_max_length {model_max_length} --max_length {model_max_length} --device '{device}'"
    for idx, task in enumerate(df.to_dict(orient="records")):
        print(f"========== task {idx} ==========")

        task["device"] = f"cuda:{avail_cuda[idx % num_avail_cuda]}"
        task["stop_words"] = "[\"\\n###\",\"\\n\\n\\n\\n\\n\"]" if task["peft_output"] is None else None

        cmd_param = cmd_param_base
        if task["peft_output"] is not None:
            cmd_param += " --peft_output '{peft_output}'"
        if task["stop_words"] is not None:
            cmd_param += " --stop_words '{stop_words}'"
            if task["peft_output"] is not None:
                raise ValueError("stop_words should only be used for original LLM")

        cmd_param = cmd_param.format_map(task)
        print(cmd_param)

        cmd = f"nohup python predict.py {cmd_param} >test.log 2>&1 &"
        os.system(cmd)


def tool_cal_metrics():
    # 2k, 4k, 8k
    model_max_length = 1024 * 8
    output_dir = OUTPUT_PATH + SLICE_PATH

    is_sliced = True
    save_sample_wise_results = True

    reference_key = "func_after_target"
    reference_before_key = "func_before_target"
    if is_sliced:
        reference_key += "_origin"
        reference_before_key += "_origin"

    mapping_model_name_suffix = {
        "starcoder": [
            # None,
            # "sliced",
            "xxx",
            # "xxx_sliced"
        ]
    }

    # data_path = f"vim_neovim_test_{model_max_length}.json"
    data_path_sliced = f"vim_neovim_test_sliced_{model_max_length}.json"

    mapping_model_name_data_path = {
        "starcoder": [data_path_sliced] * len(mapping_model_name_suffix["starcoder"])
    }

    for model_name, model_suffix_list in mapping_model_name_suffix.items():
        if model_name not in mapping_model_name_data_path:
            raise ValueError(f"model_name={model_name} not found in mapping_model_name_data_path")
        if len(model_suffix_list) != len(mapping_model_name_data_path[model_name]):
            raise ValueError(f"for model_name={model_name}, the length of model_suffix_list and data_path_list are not equal")
        
        for model_suffix, data_path in zip(model_suffix_list, mapping_model_name_data_path[model_name]):
            if model_suffix:
                output_path = output_dir + f"out_{model_name}_{model_max_length}_{model_suffix}_result.json"
            else:
                output_path = output_dir + f"out_{model_name}_{model_max_length}_result.json"

            absolute_data_path = DATA_PATH + SLICE_PATH + data_path

            do_recover = False
            if is_sliced:
                do_recover = True

            cal_metrics(
                data_path=absolute_data_path,
                output_path=output_path,
                do_recover=do_recover,
                reference_key=reference_key,
                reference_before_key=reference_before_key,
                save_sample_wise_results=save_sample_wise_results,
            )

            if save_sample_wise_results:
                output_path = output_path.replace("result.json", "result_processed.json")
                analyze_test_results(
                    result_path=output_path,
                    save_analyzed_results=True,
                )


def tool_align_test_metrics():
    col_id = "commit_id_target"
    col_metrics = ["exact_match", "rel_distance", "gen_distance", "src_distance", "exact_match_sum"]

    dummy_result_path = OUTPUT_PATH + "out_dummy_all_result_processed.json"
    dummy_result = json.load(open(dummy_result_path))
    dummy_result = pd.DataFrame(dummy_result)
    print("===== dummy_result =====")
    test_sample_num = len(dummy_result)
    print("#dummy_result:", test_sample_num)
    print(dummy_result.columns)
    
    def _check_dummy_dataset(row):
        if row["gen_distance"] != row["src_distance"]:
            raise ValueError(f"gen_distance != src_distance for target commit {row[col_id]}")
        
        if row["exact_match"] and row["src_distance"] != 0:
            raise ValueError(f"exact_match=True, but src_distance!=0 for target commit {row[col_id]}")
    
    dummy_result.apply(_check_dummy_dataset, axis=1)
    print("#no code change:", len(dummy_result[dummy_result["exact_match"] == True]))

    # 2k, 4k, 8k
    model_max_length = 1024 * 8

    result_suffix = "result_processed_analyzed"

    slice_setting_default = "sliced/"

    slice_setting_head_list = [None, slice_setting_default]  # None refers to no slicing

    mapping_model_name_suffix = {
        "starcoder": [
            None,
            "xxx",    
        ]
    }

    result_path_list = []
    for model_name, model_suffix_list in mapping_model_name_suffix.items():
        for model_suffix in model_suffix_list:
            for slice_head in slice_setting_head_list:
                if slice_head is None:
                    result_path_list.append(
                        f"out_{model_name}_{model_max_length}_{model_suffix}" if model_suffix is not None
                        else f"out_{model_name}_{model_max_length}"
                    )
                else:
                    result_path_list.append(
                            f"{slice_head}/out_{model_name}_{model_max_length}_{model_suffix}_sliced" if model_suffix is not None
                            else f"{slice_head}/out_{model_name}_{model_max_length}_sliced"                            
                        )

    metric_list = []
    for path in result_path_list:
        print(f"===== {path} =====")
        result_save_path = OUTPUT_PATH + f"{path}_{result_suffix}.json"
        result = json.load(open(result_save_path))
        result = pd.DataFrame(result)
        metric, _ = align_test_metrics(result=result, dummy_result=dummy_result)
        
        metric_list.append(metric)
        
        metric_save_path = OUTPUT_PATH + f"{path}_metrics_aligned.json"
        with open(metric_save_path, 'w') as f:
            json.dump(metric, f, indent=4)
    
    df = pd.DataFrame(metric_list, index=result_path_list)
    selected_cols = [f"{_}_aligned" for _ in col_metrics] + ["#total"] + \
                    [f"{_}_model" for _ in col_metrics] + ["#complete_generate"] + \
                    [f"{_}_dummy" for _ in col_metrics] + ["#incomplete_generate"] + \
                    ["#able_to_handle", "#unable_to_handle(dummy)"]
    df = df[selected_cols]

    df.to_excel(EXTRACTED_METRIC_PATH + f"{model_max_length}_aligned.xlsx")


def tool_compare_test_metrics():
    col_id = "commit_id_target"
    col_metrics = ["exact_match", "rel_distance", "gen_distance", "src_distance", "exact_match_sum"]

    dummy_result_path = OUTPUT_PATH + "out_dummy_all_result_processed.json"
    dummy_result = json.load(open(dummy_result_path))
    dummy_result = pd.DataFrame(dummy_result)
    print("===== dummy_result =====")
    test_sample_num = len(dummy_result)
    print("#dummy_result:", test_sample_num)
    print(dummy_result.columns)
    
    def _check_dummy_dataset(row):
        if row["gen_distance"] != row["src_distance"]:
            raise ValueError(f"gen_distance != src_distance for target commit {row[col_id]}")
        
        if row["exact_match"] and row["src_distance"] != 0:
            raise ValueError(f"exact_match=True, but src_distance!=0 for target commit {row[col_id]}")
    
    dummy_result.apply(_check_dummy_dataset, axis=1)
    print("#no code change:", len(dummy_result[dummy_result["exact_match"] == True]))

    # 2k, 4k, 8k
    model_max_length = 1024 * 2

    result_suffix = "result_processed_analyzed"

    slice_setting_default = "sliced/"

    slice_setting_head_list = [None, slice_setting_default]

    mapping_model_name_suffix = {
        "starcoder": [
            # None,
            "xxx",
        ]
    }

    different_slicing_setting = len(slice_setting_head_list) > 1
    different_model = sum(len(v) for v in mapping_model_name_suffix.values()) > 1
    if different_slicing_setting and different_model:
        raise ValueError("it is meaningless to compare with different models and different slicing settings")
    elif different_slicing_setting:
        print("##### compare different slicing settings with the same model #####")
    elif different_model:
        print("##### compare different model with the same slicing setting #####")
    else:
        raise ValueError("same slicing setting and same model")

    result_path_list = []
    for model_name, model_suffix_list in mapping_model_name_suffix.items():
        for model_suffix in model_suffix_list:
            for slice_head in slice_setting_head_list:
                if slice_head is None:
                    result_path_list.append(
                        f"out_{model_name}_{model_max_length}_{model_suffix}" if model_suffix is not None
                        else f"out_{model_name}_{model_max_length}"
                    )
                else:
                    result_path_list.append(
                            f"{slice_head}/out_{model_name}_{model_max_length}_{model_suffix}_sliced" if model_suffix is not None
                            else f"{slice_head}/out_{model_name}_{model_max_length}_sliced"                            
                        )

    def _load_test_results(p):
        result_save_path = OUTPUT_PATH + f"{p}_{result_suffix}.json"
        result = json.load(open(result_save_path))
        result = pd.DataFrame(result)
        return result
    
    result_list = [_load_test_results(p) for p in result_path_list]

    metric_list = compare_test_metrics(result_list=result_list, dummy_result=dummy_result)

    df = pd.DataFrame(metric_list, index=result_path_list)
    selected_cols = [f"{_}_shared" for _ in col_metrics] + ["#shared"] + [f"{_}_unique" for _ in col_metrics] + ["#unique", "#complete_generate", "#incomplete_generate", "#total"]
    df = df[selected_cols]

    df.to_excel(EXTRACTED_METRIC_PATH + f"{model_max_length}_compared.xlsx")


# switch between diffrent slice settings
SLICE_PATH = ""  # for non-sliced dataset
# SLICE_PATH = "sliced/"  # for sliced dataset

if __name__ == "__main__":
    # 1. run LLM to get the generation
    tool_run_tests()

    # 2. analyze the sample-wise generation results
    # tool_cal_metrics()

    # 3. we use the func_target_before as the default output for samples that LLM can not complete the generation
    # tool_align_test_metrics()

    # 4. compare the performance of same model with different slice setting or different model with the same slice setting
    # tool_compare_test_metrics()