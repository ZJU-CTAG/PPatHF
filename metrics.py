
import pandas as pd
import json
import numpy as np
import warnings
from typing import List, Union, Optional
import editdistance
import copy

from reduction.fcu import FunctionCompareUtilities
from reduction.reducer import Reducer


def postprocess(s: str, key_str_before_target: str, key_str_after_target: Union[str, List[str]]):
    if key_str_before_target not in s:
        raise ValueError("can not find the key str to locate the target")
    if type(key_str_after_target) == str:
        key_str_after_target = [key_str_after_target]
    
    if len(s.split(key_str_before_target)) > 2:
        warnings.warn(f"find multiple key str to locate the target\n\n{s}")
    s = s.split(key_str_before_target)[1]
    s = s.strip()

    for key_str in key_str_after_target:
        s = s.split(key_str)[0].strip()

    return s


class MetricsSampleWise:
    def __init__(self, 
                 generations: List[str], 
                 test_set: List[dict],
                 do_postprocess: bool = True,
                 do_recover: bool = False,
                 reference_key: str = "func_after_target",
                 reference_before_key: str = "func_before_target", 
                 key_str_before_target: str = "### Function After (neovim):",
                 key_str_after_target: Union[str, List[str]] = ["\n###"]):
        self.generations = generations

        self.test_set = test_set
        self._reference_before_key = reference_before_key
        self._reference_key = reference_key

        if len(generations) != len(test_set):
            raise ValueError("the lengths of generations and test_set are not equal")

        self._key_str_before_target = key_str_before_target
        self._key_str_after_target = key_str_after_target

        self.sample_wise_test_result = None
        self._do_recover = do_recover
        self._do_postprocess = do_postprocess
    
    def get_sample_wise_test_results(self):
        if self.sample_wise_test_result is not None:
            return self.sample_wise_test_result

        fcu = FunctionCompareUtilities()
        slicer = Reducer(tolerant=True)
        
        self.sample_wise_test_result = []
        for gen, sample in zip(self.generations, self.test_set):
            gen_processed = postprocess(s=gen, key_str_before_target=self._key_str_before_target, key_str_after_target=self._key_str_after_target) if self._do_postprocess else gen
            if self._do_recover and len(sample["removed_pieces_sliced_target"]) > 0:
                gen_processed = slicer.do_recovering(reduced_func=gen_processed, removed_pieces=sample["removed_pieces_sliced_target"])
            
            ref = sample[self._reference_key]
            ref_before = sample[self._reference_before_key]
            
            gen_processed_tokens = fcu.get_cleaned_tokens(gen_processed)
            ref_tokens = fcu.get_cleaned_tokens(ref)
            ref_before_tokens = fcu.get_cleaned_tokens(ref_before)

            exact_match = np.array_equal(gen_processed_tokens, ref_tokens)
            gen_ref_dis = editdistance.eval(gen_processed_tokens, ref_tokens)
            src_ref_dis = editdistance.eval(ref_before_tokens, ref_tokens)
            if src_ref_dis == 0:
                print(f"src_ref is the same as ref. {sample['commit_id_target']}")
            rel_dis = gen_ref_dis / (src_ref_dis if src_ref_dis != 0 else 1)
            
            self.sample_wise_test_result.append({
                "gen": gen,
                "gen_processed": gen_processed,
                "exact_match": exact_match,
                "rel_distance": rel_dis,
                "gen_distance": gen_ref_dis,
                "src_distance": src_ref_dis,
            })
        
        return self.sample_wise_test_result
    
    def cal_metrics(self):
        sample_wise_test_results = self.get_sample_wise_test_results()
        df = pd.DataFrame(sample_wise_test_results)
        
        metrics = {}
        selected_metrics = ["exact_match", "rel_distance", "gen_distance", "src_distance"]
        for m in selected_metrics:
            metrics[m] = df[m].mean()
            if m in ["exact_match"]:
                metrics[f"{m}_sum"] = sum(df[m].to_list())
        metrics["#sample"] = len(df)
        
        return metrics

    def save_sample_wise_test_results(self, save_path: str, save_metadata: bool = False, metadata_keys: Optional[List[str]] = None):
        sample_wise_test_results = self.get_sample_wise_test_results()
        df_result = pd.DataFrame(sample_wise_test_results)

        if save_metadata:
            df_metadata = pd.DataFrame(self.test_set)
            if metadata_keys is not None:
                df_metadata = df_metadata[metadata_keys]
        
            df_result = pd.concat([df_result, df_metadata], axis=1)

        with open(save_path, 'w') as f:
            json.dump(df_result.to_dict(orient="records"), f, indent=4)