import re
import logging
import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset, concatenate_datasets
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from multiprocessing import Pool
from .abstract_selector import AbstractCandidateSelector
from .loss_selector import map_len_token, LossSelector


logger = logging.getLogger(__name__)


def compute_rank_on_device(args):
    inter_dataset_slice, device, model_name_or_path, infer_batch_size, max_seq_length, ignore_pre_pii, log_rank = args
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype="bfloat16")
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    inter_dataset_slice = inter_dataset_slice.map(map_len_token, fn_kwargs={"tokenizer": tokenizer}, num_proc=1)

    all_ranks = []

    for i in tqdm(range(0, len(inter_dataset_slice), infer_batch_size), desc="Computing rank"):
        inputs = tokenizer(
            inter_dataset_slice["seq"][i:i + infer_batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )

        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, return_dict=True)
            logits = outputs.logits # shape: (batch_size, seq_length, vocab_size)
            labels = inputs["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            if ignore_pre_pii:
                len_pre_pii_tokens = inter_dataset_slice["len_pre_pii_tokens"][i:i + infer_batch_size]
                for j in range(len(len_pre_pii_tokens)):
                    labels[j, :len_pre_pii_tokens[j]] = -100
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            bs, seq_len = shift_labels.shape

            probs = torch.softmax(shift_logits, dim=-1)
            shift_labels_for_gather = shift_labels.clone().unsqueeze(-1)
            shift_labels_for_gather[shift_labels_for_gather == -100] = 0
            probs_actual = probs.gather(-1, shift_labels_for_gather).squeeze(-1)  # [bs, seq_len]
            ranks = (probs > probs_actual.unsqueeze(-1)).sum(dim=-1) # [bs, seq_len]

            for bi in range(bs):
                current_ranks = ranks[bi, :]
                current_shift_labels = shift_labels[bi, :]
                current_ranks = current_ranks[current_shift_labels != -100]
                # sum(log(ranks + 1))
                rank_score = np.sum(np.log(current_ranks.cpu().float().numpy() + 1)) if log_rank else np.sum(current_ranks.cpu().float().numpy())
                all_ranks.append(rank_score)

    inter_dataset_slice = inter_dataset_slice.add_column("rank", all_ranks)
    return inter_dataset_slice


def map_rank_to_mean(example, ignore_pre_pii: bool = True):
    if ignore_pre_pii:
        mean_rank = example['rank'] / max(example['len_tokens'] - example['len_pre_pii_tokens'], 1)
    else:
        mean_rank = example['rank'] / example['len_tokens']
    return {'mean_rank': mean_rank}


class RankSelector(LossSelector):
    
    def select_candidates(
        self,
        test_dataset: Dataset,
        candidate_outputs: list[dict],
        legal_pii_types: Optional[list[str]] = None,
        device_parallel_size: int = 4,
        ignore_pre_pii: bool = True,
        processed_inter_dataset: Optional[Dataset] = None,
        mean_score: bool = True,
        log_rank: bool = True,
    ) -> Dataset:
        inter_dataset = self.construct_inference_data(test_dataset, candidate_outputs, legal_pii_types)
        num_examples = len(inter_dataset)
        indices_list = [list(range(i, num_examples, device_parallel_size)) for i in range(device_parallel_size)]
        dataset_slices = [inter_dataset.select(indices) for indices in indices_list]

        if not processed_inter_dataset:
            # prepare args
            args_for_devices = []
            for idx in range(device_parallel_size):
                inter_dataset_slice = dataset_slices[idx]
                device = f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
                model_name_or_path = self.model_name_or_path
                infer_batch_size = self.infer_batch_size
                max_seq_length = self.max_seq_length
                args_for_devices.append((inter_dataset_slice, device, model_name_or_path, infer_batch_size, max_seq_length, ignore_pre_pii, log_rank))

            # parallel inference
            with Pool(device_parallel_size, maxtasksperchild=1) as p:
                results = p.map(compute_rank_on_device, args_for_devices)
            # results = [compute_rank_on_device(args) for args in args_for_devices] # for debugging

            # merge results
            processed_inter_dataset = concatenate_datasets(results)

            if mean_score:
                processed_inter_dataset = processed_inter_dataset.map(map_rank_to_mean, fn_kwargs={"ignore_pre_pii": ignore_pre_pii}, num_proc=8)
            else:
                processed_inter_dataset = processed_inter_dataset.add_column("mean_rank", processed_inter_dataset["rank"])

        inter_df = processed_inter_dataset.to_pandas()
        inter_df = inter_df.groupby(["row_idx", "pii_key", "candidate"])["mean_rank"].sum().reset_index() # sum maked_seq
        inter_df = inter_df.groupby(["row_idx", "pii_key"])

        # add scores to the candidate_outputs
        def add_scores(example):
            row_idx = example["row_idx"]
            pii_key = example["pii_key"]
            candidates = example["candidates"]
            rank_scores = []
            for candidate in candidates:
                candidate_df = inter_df.get_group((row_idx, pii_key))
                candidate_df = candidate_df[candidate_df["candidate"] == candidate]
                rank_score = candidate_df["mean_rank"].values[0]
                rank_scores.append(rank_score)
            scores = rank_scores
            return {'scores': scores, 'rank_scores': rank_scores}
        
        candidate_outputs = candidate_outputs.map(add_scores, num_proc=8)

        return candidate_outputs
