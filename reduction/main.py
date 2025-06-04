import logging
import multiprocessing
import os
import json
import time
from reducer import Reducer
import argparse

from reducerConfig import reducing_log_path, reducing_output_path, reducing_input_file

parser = argparse.ArgumentParser(description='Reduce c program')
parser.add_argument('-i', '--id', type=int, help='id of concurrency')
parser.add_argument('-b', '--base', type=int, help='base of concurrency')
parser.add_argument('-m', '--merge', type=int, help='results to merge')
args = parser.parse_args()


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        print(self.time_string())

    def elapsed_time(self):
        return self.end_time - self.start_time

    def time_string(self):
        return time.strftime("%H:%M:%S", time.gmtime(self.elapsed_time()))


def do_reducing_merge(output_path, counts):
    result = []
    for id in range(0, counts):
        with open(output_path + f"_{str(id)}", "r") as input:
            result_tmp = json.load(input)
            result += result_tmp
        os.remove(output_path + f"_{str(id)}")
    with open(output_path, "w") as output:
        result = sorted(result, key=lambda pa: len(pa["func_after_source"]), reverse=True)
        json.dump(result, output)


def do_reducing_serial(content_path, output_path, id, base):
    reducer = Reducer()
    result_list = []
    with open(content_path, 'r') as f:
        content = json.load(f)
        content = sorted(content, key=lambda pa: len(pa["func_after_source"]), reverse=True)

        for patch_no in range(len(content)):
            if patch_no % base == id:
                patch = content[patch_no]
                start_time = time.time()
                try:
                    vim_before_commit_process = patch["func_before_source"]
                    vim_after_commit_process = patch["func_after_source"]
                    neovim_before_commit_process = patch["func_before_target"]

                    reduced_dict_process = reducer.do_reducing(vim_before_commit_process,
                                                               vim_after_commit_process, neovim_before_commit_process)
                    func_reduced_before_fin_process = reduced_dict_process["func_before_reduced"]
                    func_reduced_after_fin_process = reduced_dict_process["func_after_reduced"]
                    func_reduced_mapping_fin_process = reduced_dict_process["func_mapped_reduced"]
                    removed_pieces_before_process = reduced_dict_process["removed_pieces_before"]
                    removed_pieces_mapping_process = reduced_dict_process["removed_pieces_mapped"]

                    end_time = time.time()
                    result = patch.copy()
                    result["func_before_sliced_source"] = func_reduced_before_fin_process
                    result["func_after_sliced_source"] = func_reduced_after_fin_process
                    result["func_before_sliced_target"] = func_reduced_mapping_fin_process
                    result["removed_pieces_sliced_source"] = removed_pieces_before_process
                    result["removed_pieces_sliced_target"] = removed_pieces_mapping_process
                    result["reducing_time"] = end_time - start_time
                    result_list.append(result)
                except Exception as e:
                    print("\nerr", patch["neovim_commit_id"])
                    print(e)
                    print("-" * 100)
    with open(output_path + f"_{str(id)}", "w") as output:
        json.dump(result_list, output)


if __name__ == "__main__":
    with Timer() as timer:
        output_file = f"{reducing_output_path}/vim-neovim-reduced.json"
        if args.id is not None and args.base is not None:
            do_reducing_serial(reducing_input_file, output_file, args.id, args.base)
        elif args.merge is not None:
            do_reducing_merge(output_file, args.merge)
