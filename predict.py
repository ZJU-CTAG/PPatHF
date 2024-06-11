import torch
torch.backends.cuda.matmul.allow_tf32 = True

import json
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
from evaluate import load
from tqdm import tqdm
import pandas as pd
from typing import List

import fire

from porting.data import Testset


class EndOfFunctionCriteria(StoppingCriteria):
    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)


def generate(
        base_model_name_or_path: str,
        data_path: str,
        output_path: str,
        peft_output: str = None,
        model_max_length: int = 2048,
        max_length: int = 2048,
        stop_words: List[str] = None,
        device: str = "cuda",
        model_torch_dtype=torch.float16,
        **kwargs
    ):
    device_map = "auto" if device=="cuda" else {"": device}
    model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=model_torch_dtype, device_map=device_map)
    if peft_output:
        model = PeftModel.from_pretrained(model, peft_output, torch_dtype=model_torch_dtype, device_map=device_map)
    print(type(model))

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path,
        model_max_length=model_max_length,
        truncation_side="left",
        padding_side="right",
    )

    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    generation_config = GenerationConfig(
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id,
        max_length=max_length,
        **kwargs,
    )

    stopping_criteria = None
    if stop_words:
        if tokenizer.eos_token:
            stop_words.append(tokenizer.eos_token)
        stopping_criteria = StoppingCriteriaList(
            [EndOfFunctionCriteria(0, stop_words, tokenizer)]
        )

    model.eval()

    test_set = Testset(data_path=data_path, tokenizer=tokenizer)

    gens = []

    for _, sample in enumerate(tqdm(test_set)):
        if stopping_criteria:
            stopping_criteria[0].start_length = (
                sample["input_ids"].shape[-1]
            )
        with torch.no_grad():
            outputs = model.generate(
                inputs=sample["input_ids"].to(device),
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
            )
        
        outputs = outputs.tolist()
        outputs_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        gens.append(outputs_decoded)

    with open(output_path, 'w') as f:
        json.dump(gens, f, indent=4)


if __name__ == "__main__":
    fire.Fire(generate)