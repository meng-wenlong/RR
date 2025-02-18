import re
import os
import logging
import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset, concatenate_datasets, load_from_disk
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from multiprocessing import Pool
from .abstract_selector import AbstractCandidateSelector


logger = logging.getLogger(__name__)


def normlize(scores: list):
    arr = np.array(scores)
    mean = np.mean(arr)
    std = np.std(arr)
    if std <= 1e-6 and std >= -1e-6:
        normalized = arr - mean
        logger.warning(f"std is too small: {std}")
    else:
        normalized = (arr - mean) / std
    return normalized


def map_len_token(example, tokenizer):
    tokens = tokenizer.tokenize(example['seq'], add_special_tokens=False)
    len_tokens = len(tokens)

    pre_pii_seq = example['masked_seq'].split('[' + example['pii_key'] + ']')[0]
    if pre_pii_seq != '' and pre_pii_seq[-1] == ' ':
        pre_pii_seq = pre_pii_seq[:-1]
    pre_pii_tokens = tokenizer.tokenize(pre_pii_seq, add_special_tokens=False)
    len_pre_pii_tokens = len(pre_pii_tokens)

    return {'len_tokens': len_tokens, 'len_pre_pii_tokens': len_pre_pii_tokens}


def compute_loss_on_device(args):
    inter_dataset_slice, device, model_name_or_path, infer_batch_size, max_seq_length, ignore_pre_pii = args
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype="bfloat16")
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    inter_dataset_slice = inter_dataset_slice.map(map_len_token, fn_kwargs={"tokenizer": tokenizer}, num_proc=1)

    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    all_logits = []
    all_losses = []

    for i in tqdm(range(0, len(inter_dataset_slice), infer_batch_size), desc="Computing loss"):
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
            labels = inputs["input_ids"].clone()
            mask = inputs["attention_mask"].clone()
            labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding
            if ignore_pre_pii: # ignore pre_pii_tokens
                len_pre_pii_tokens = inter_dataset_slice["len_pre_pii_tokens"][i:i + infer_batch_size]
                for j in range(len(len_pre_pii_tokens)):
                    labels[j, :len_pre_pii_tokens[j]] = -100
                mask = mask * (labels != -100).long()

            outputs = model(**inputs)
            logits = outputs["logits"]

            # Shift logits and labels to calculate loss for each token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            bs, seq_len = shift_labels.shape
            loss = loss_fct(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))
            loss = torch.sum(loss.view(bs, seq_len), dim=1)

            # Gather logits for correct labels
            shift_mask = mask[..., 1:].contiguous()
            shift_labels_for_gather = shift_labels.clone().unsqueeze(-1)
            shift_labels_for_gather[shift_labels_for_gather == -100] = 0
            
            selected_values = torch.gather(shift_logits, 2, shift_labels_for_gather)
            selected_values = selected_values.squeeze(-1)
            selected_values[shift_labels == -100] = 0

            logit = torch.sum(selected_values * shift_mask, dim=1)

            all_logits += logit.cpu().tolist()
            all_losses += loss.cpu().tolist()

    inter_dataset_slice = inter_dataset_slice.add_column("logits", all_logits)
    inter_dataset_slice = inter_dataset_slice.add_column("loss", all_losses)
    return inter_dataset_slice


def map_ll_to_mean(example, ignore_pre_pii=True):
    if ignore_pre_pii:
        mean_logit = example['logits'] / max(example['len_tokens'] - example['len_pre_pii_tokens'], 1)
        mean_loss = example['loss'] / max(example['len_tokens'] - example['len_pre_pii_tokens'], 1)
    else:
        mean_logit = example['logits'] / example['len_tokens']
        mean_loss = example['loss'] / example['len_tokens']
    return {'mean_logits': mean_logit, 'mean_loss': mean_loss}


class LossSelector(AbstractCandidateSelector):
    def __init__(self, model_name_or_path: str,
                 infer_batch_size: int = 24,
                 max_seq_length: int = 512,
                 refer_model_name_or_path: Optional[str] = None):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.infer_batch_size = infer_batch_size
        self.max_seq_length = max_seq_length
        self.refer_model_name_or_path = refer_model_name_or_path

    @staticmethod
    def extract_subsequences(text, target_label, content_length=300):
        # Find all positions of the target label in the text
        label_positions = [m.start() for m in re.finditer(re.escape(target_label), text)]
        extracted = []

        for pos in label_positions:
            # Find the nearest ']' before the target label
            left_boundary = text.rfind(']', 0, pos)
            # Find the nearest '[' after the target label
            right_boundary = text.find('[', pos + len(target_label))

            start = max(0, pos - content_length, left_boundary+1)
            end = min(len(text), pos + len(target_label) + content_length, right_boundary)
            segment = text[start:end].strip()
            
            extracted.append(segment)

        return extracted

    def construct_inference_data(self, test_dataset: Dataset, candidate_outputs: Dataset, legal_pii_types: Optional[list[str]] = None) -> Dataset:
        assert "masked_seq" in test_dataset.column_names, "Test dataset must contain 'masked_seq' field"
        assert "pii_mask" in test_dataset.column_names, "Test dataset must contain 'pii_mask' field"
        assert "row_idx" in candidate_outputs.column_names, "Candidates must contain 'row_idx' field"
        assert "pii_key" in candidate_outputs.column_names, "Candidates must contain 'pii_key' field"
        assert "candidates" in candidate_outputs.column_names, "Candidates must contain 'candidates' field"

        # Accelerating queries with numpy vectorization operations
        row_idxs = np.array(candidate_outputs["row_idx"])
        pii_keys = np.array(candidate_outputs["pii_key"])

        inter_data = []
        for row_idx, row in enumerate(test_dataset):
            for mask in row["pii_mask"]["labels"]:
                if legal_pii_types and mask.split('-')[0] not in legal_pii_types:
                    logger.warning(f"Skipping {mask} as it is not in the legal_pii_types list")
                    continue

                pii_label = '[' + mask + ']'
                sequences = self.extract_subsequences(row["masked_seq"], pii_label)
                assert len(sequences) > 0, f"Could not find any sequences for {pii_label} in {row['masked_seq']}"

                # candidate_output = candidate_outputs.filter(
                #     lambda batch: [row_idx == r and mask == p for r, p in zip(batch["row_idx"], batch["pii_key"])],
                #     batched=True,
                # )[0]
                condition = (row_idxs == row_idx) & (pii_keys == mask)
                indices = np.nonzero(condition)[0]
                # print(f"row_idx: {row_idx}")
                # print(f"masks: {mask}")
                # print(f"row_idxs: {row_idxs}")
                # print(f"pii_keys: {pii_keys}")
                # print(f"indices[0]: {indices[0]}")
                candidate_output = candidate_outputs[indices[0].item()]

                candidate_idx = 0
                for candidate in candidate_output["candidates"]:
                    for sequence in sequences:
                        replaced_sequence = sequence.replace(pii_label, candidate)
                        inter_data.append({
                            "row_idx": row_idx,
                            "pii_key": mask,
                            "seq": replaced_sequence,
                            "masked_seq": sequence,
                            "candidate_idx": candidate_idx,
                            "candidate": candidate,
                        })
                    candidate_idx += 1

        inter_dataset = Dataset.from_list(inter_data)
        return inter_dataset

    def select_candidates(
        self, 
        test_dataset: Dataset, 
        candidate_outputs: list[dict], 
        legal_pii_types: Optional[list[str]] = None, 
        device_parallel_size: int = 4,
        ignore_pre_pii: bool = True,
        processed_inter_dataset_path: Optional[str] = None, # path
        refer_inter_dataset_path: Optional[str] = None, # path
        refer_calibrate_method: str = "divide", # "divide" or "subtract"
        refer_bias: float = 0.,
    ) -> Dataset:
        inter_dataset = self.construct_inference_data(test_dataset, candidate_outputs, legal_pii_types)
        num_examples = len(inter_dataset)
        indices_list = [list(range(i, num_examples, device_parallel_size)) for i in range(device_parallel_size)]
        dataset_slices = [inter_dataset.select(indices) for indices in indices_list]

        if processed_inter_dataset_path is None or not os.path.exists(processed_inter_dataset_path):
            # prepare args
            args_for_devices = []
            for idx in range(device_parallel_size):
                inter_dataset_slice = dataset_slices[idx]
                device = f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
                model_name_or_path = self.model_name_or_path
                infer_batch_size = self.infer_batch_size
                max_seq_length = self.max_seq_length
                args_for_devices.append((inter_dataset_slice, device, model_name_or_path, infer_batch_size, max_seq_length, ignore_pre_pii))
            
            # parallel inference
            with Pool(device_parallel_size, maxtasksperchild=1) as p:
                results = p.map(compute_loss_on_device, args_for_devices)
            # results = [compute_loss_on_device(args) for args in args_for_devices] # for debugging

            # merge results
            processed_inter_dataset = concatenate_datasets(results)

            # compute mean loss and logits
            processed_inter_dataset = processed_inter_dataset.map(map_ll_to_mean, fn_kwargs={"ignore_pre_pii": ignore_pre_pii}, num_proc=8)
            if processed_inter_dataset_path is not None:
                processed_inter_dataset.save_to_disk(processed_inter_dataset_path)#############
        else:
            processed_inter_dataset = load_from_disk(processed_inter_dataset_path)

        if not self.refer_model_name_or_path:
            refer_inter_dataset = None
        elif refer_inter_dataset_path is None or not os.path.exists(refer_inter_dataset_path):
            # prepare args
            args_for_devices = []
            for idx in range(device_parallel_size):
                inter_dataset_slice = dataset_slices[idx]
                device = f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
                model_name_or_path = self.refer_model_name_or_path
                infer_batch_size = self.infer_batch_size
                max_seq_length = self.max_seq_length
                args_for_devices.append((inter_dataset_slice, device, model_name_or_path, infer_batch_size, max_seq_length, ignore_pre_pii))
            
            # parallel inference
            with Pool(device_parallel_size, maxtasksperchild=1) as p:
                results = p.map(compute_loss_on_device, args_for_devices)

            # merge results
            refer_inter_dataset = concatenate_datasets(results)

            # compute mean loss and logits
            refer_inter_dataset = refer_inter_dataset.map(map_ll_to_mean, fn_kwargs={"ignore_pre_pii": ignore_pre_pii}, num_proc=8)
            if refer_inter_dataset_path is not None:
                refer_inter_dataset.save_to_disk(refer_inter_dataset_path)
        else:
            refer_inter_dataset = load_from_disk(refer_inter_dataset_path)

        if refer_inter_dataset:
            assert len(refer_inter_dataset) == len(processed_inter_dataset), "The number of examples in the refer_inter_dataset must be the same as the processed_inter_dataset"
            processed_inter_dataset = processed_inter_dataset.add_column("refer_mean_logits", refer_inter_dataset["mean_logits"])
            processed_inter_dataset = processed_inter_dataset.add_column("refer_mean_loss", refer_inter_dataset["mean_loss"])
            if refer_calibrate_method == "divide":
                processed_inter_dataset = processed_inter_dataset.map(lambda x: {'mean_logits': x['mean_logits'] - x['refer_mean_logits'], 'mean_loss': (x['mean_loss'] - x['refer_mean_loss'] - refer_bias) / (x['refer_mean_loss'] + refer_bias)}, num_proc=8)
            elif refer_calibrate_method == "subtract":
                processed_inter_dataset = processed_inter_dataset.map(lambda x: {'mean_logits': x['mean_logits'] - x['refer_mean_logits'], 'mean_loss': x['mean_loss'] - x['refer_mean_loss']}, num_proc=8)
            # processed_inter_dataset = processed_inter_dataset.map(lambda x: {'loss_drop': (x['mean_loss'] - x['refer_mean_loss'])/x['refer_mean_loss']}, num_proc=8)
            # normed_loss_drop = normlize(processed_inter_dataset['loss_drop'])
            # normed_loss = normlize(processed_inter_dataset['mean_loss'])
            # processed_inter_dataset = processed_inter_dataset.add_column("normed_loss_drop", normed_loss_drop)
            # processed_inter_dataset = processed_inter_dataset.add_column("normed_loss", normed_loss)
            # processed_inter_dataset = processed_inter_dataset.map(lambda x: {'mean_logits': x['mean_logits'] - x['refer_mean_logits'], 'mean_loss': refer_bias*x['normed_loss_drop'] + (1-refer_bias)*x['normed_loss']}, num_proc=8)

        inter_df = processed_inter_dataset.to_pandas()
        inter_df = inter_df.groupby(["row_idx", "pii_key", "candidate"])[['mean_logits', 'mean_loss']].sum().reset_index() # sum masked_seq
        inter_df = inter_df.groupby(["row_idx", "pii_key"])

        # add loss scores and logits scores to the candidate_outputs
        def add_scores(example):
            row_idx = example['row_idx']
            pii_key = example['pii_key']
            candidates = example['candidates']
            loss_scores = []
            logits_scores = []
            for candidate in candidates:
                candidate_df = inter_df.get_group((row_idx, pii_key))
                candidate_df = candidate_df[candidate_df['candidate'] == candidate]
                loss_score = candidate_df['mean_loss'].values[0]
                logits_score = candidate_df['mean_logits'].values[0]
                loss_scores.append(loss_score)
                logits_scores.append(logits_score)
            scores = loss_scores
            return {'scores': scores, 'logits_scores': logits_scores, 'loss_scores': loss_scores}

        candidate_outputs_with_scores = candidate_outputs.map(add_scores, num_proc=8)

        return candidate_outputs_with_scores