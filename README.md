# PPatHF

## Project Description

This is the replication package for our paper `Automating Zero-Shot Patch Porting for Hard Forks`.

We introduce the task of automating zero-shot patch porting across hard forks.

We propose a LLM-based approach (namely PPatHF) to automate the patch porting on a function-wise basis, which is composed of a reduction module and a porting module. 

We collect Neovim patches ported from Vim to investigate the necessities of patch porting and use patches ported after 2022-07-01 to evaluate model performance.

## Environments

1. Language: Python (Python 3.9)

2. Key Python Packages:
  * [PyTorch 2.0.1](https://pytorch.org/)
  * [Transformers 4.29.1](https://huggingface.co/docs/transformers/index)
  * [PEFT 0.4.0 ](https://huggingface.co/docs/peft/index)
  * [Pydriller 2.5.1](https://github.com/ishepard/pydriller)
  * [tree-sitter 0.20.1](https://github.com/tree-sitter/py-tree-sitter)
  * [unidiff 0.7.5](https://github.com/matiasb/python-unidiff)

3. Setup:

  * Clone [tree-sitter-c](https://github.com/tree-sitter/tree-sitter-c) into your local directory and provide the path to the [reducerConfig.py](reduction/reducerConfig.py) (used to setup the treesitter python binding). See the documentation of [py-tree-sitter](https://github.com/tree-sitter/py-tree-sitter) for more details.
  * Download the [StarCoder weights from Huggingface](https://huggingface.co/bigcode/starcoder) into your local directory and provide the path to [config.py](config.py).


## Dataset

**NOTE:** Please dowload the data from [here](https://drive.google.com/drive/folders/15cHr1s7B3ldL7hPC5Hw1fkSYYYdbfVLP?usp=sharing)

* `vim_neovim_test_all.json`: the collected 310 patches ported from Vim to Neovim, used as the test set in our evaluations. 
Note that we only use patches ported after 2022-07-01 (the date when the pretrain corpus of StarCoder is collected) for evaluation to avoid possible leakage of the test data

Example of a data record (This example omits the details of the code fields):

```
    {
        "commit_id_source": "ce79353ace9e21238f13655089363cd23cbb6b32",
        "file_path_source": "src/menu.c",
        "func_before_source": "void\nexecute_menu(exarg_T *eap, vimmenu_T *menu, int mode_idx)<...>",
        "func_after_source": "void\nexecute_menu(exarg_T *eap, vimmenu_T *menu, int mode_idx)<...>",
        "diff_source": "@@ -71,7 +71,8 @@<...>",
        "commit_id_target": "236947ab20b82417eb4f3f69dd46740f299b7fdf",
        "file_path_target": "src/nvim/menu.c",
        "func_before_target": "void execute_menu(const exarg_T *eap, vimmenu_T *menu, int mode_idx)<...>",
        "func_after_target": "void execute_menu(const exarg_T *eap, vimmenu_T *menu, int mode_idx)<...>",
        "diff_target": "@@ -67,7 +67,7 @@<...>",
        "neovim_committer_date": "2022-07-01 10:17:39+08:00"
    }
```
*commit_id_source:* commit hash of the Vim commit.

*file_path_source:* path of the modified C file in the Vim commit.

*func_before_source:* code of the modified function before applying the Vim commit.

*func_after_source:* code of the modified function after applying the Vim commit.

*commit_id_target:* commit hash of the corresponding Neovim commit. 

*file_path_target:* path of the Modified C file in the corresponding Neovim commit. 

*func_before_target:* code of the modified function before applying the corresponding Neovim commit.

*func_after_target:* code of the modified function after applying the corresponding Neovim commit.

*neovim_committer_date:* commit date of the corresponding Neovim commit.

* `vim_neovim_test_sliced_all.json`: the reduced test set (i.e., slimmed by the reduction module). It contains following extra fields compared with `vim_neovim_test_all.json`:

*func_before_sliced_source/func_after_sliced_source/func_before_sliced_target:* the reduced version of the corresponding functions.

*removed_pieces_sliced_source/removed_pieces_sliced_target:* the removed code snippets for the functions from the source project and the hard fork, respectively.

* `vim_neovim_test_{2048\4096\8192}.json`: the test set after length filtering (i.e., input length < length limit). For PPatHF and StarCoder (i.e., nueral models), they can not deal with samples with inputs exceeding the lenght limit.

* `vim_neovim_test_sliced_{2048\4096\8192}.json`: the reduced test set after length filtering.

* `finetune.json`: the collected datasets (including 2,747 Vim commits and 2,082 Neovim commits before 2022-07-01) used to conduct finetuning.


## File Organization

There are basically two directories corresponding to the reduction module (`reduction/`) and the porting module (`porting/`), and several files that support evaluations.


   * `reduction/`: 
      * `fcu.py`: utilities for parsing and analyzing functions using treesitter
      * `reducer.py`: implementation of the reduction module
      * `reducerConfig.py`: config file of the reduction module

  * `porting/`: 
    * `data.py`: data utilities for the LLM, e.g., dataloader, tokenization
    * `train.py`: tune LLM with the proposed finetuning task using LoRA
    * `run.sh`: a script to run the finetuning

  * `predict.py`: run the LLM to get the generations
  * `test.py`: evaluate the patch porting performance
  * `test_tools.py`: tools (wrappers of the methods in `test.py`) to conveniently run tests in batch and get the formatted results (write into excel files)
  * `metrics.py`: calculate metrics including accuracy (exact_match), AED, and RED
  * `utils.py`: util functions such as buiding test sets


## Training & Evaluation

### Run Finetuning

To run the finetuning, follow the instructions in `porting/run.sh` to provide:
  * MODEL_PATH: the path of the pretrained LLM weights downloaded from Huggingface
  * DATA_PATH: the path of dataset used for finetuning
  * OUTDIR: the path to save the trained LoRA adapter

    Run `sh run.sh` in command line. For other hyperparameters of mode training, we provide the value used in our experiments as default.

To get the generations of LLM (either with or without tuning), use `predict.py`.
Specifically, we provide a wrapper (`tool_run_tests()` in `test_tools.py`) for you to easility run generations in batch.


### Evaluation

We provide tools in `test_tools.py` (wrappers of the methods in `test.py`) for you to conveniently run evaluations (get the generations of LLM and further calculate the metrics) in batch and get the formatted results (write into excel files).

Follow the instructions in `test_tools.py` to get the results presented in the paper. 
