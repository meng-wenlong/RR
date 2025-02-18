import pandas as pd
from dataclasses import dataclass, field
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import HfArgumentParser
from typing import Optional

from pii.datas import get_loader_class


@dataclass
class EvalArguments:
    dataset_path: str = field(
        default='selected_candidates/loss/llama3.1-8b_echr',
        metadata={"help": "The path to the selected data"}
    )
    save_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to save the generated candidates"}
    )
    split_strategy: Optional[str] = field(
        default='pii_type', # 'pii_type' or 'pii_times' or 'text_length'
        metadata={"help": "The dataset split to attack"}
    )

    dataset_name: str = field(
        default="echr",
        metadata={"help": "The name of the dataset to attack"}
    )
    dataset_split: Optional[str] = field(
        default='train',
        metadata={"help": "The dataset split to attack"}
    )
    new_chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "The new chat template"}
    )


def split_pii_type(ds):
    def add_pii_type(example):
        example['pii_type'] = example['pii_key'].split('-')[0]
        return example
    
    ds = ds.map(add_pii_type)
    df = ds.to_pandas()
    grouped = df.groupby('pii_type')
    grouped_datasets = DatasetDict({name: Dataset.from_pandas(group) for name, group in grouped})
    return grouped_datasets


def split_pii_times(ds, raw_dataset):
    def add_pii_times(example):
        masked_text = raw_dataset[example['row_idx']]['masked_seq']
        example['pii_times'] = masked_text.count(example['pii_key'])
        return example
    
    ds = ds.map(add_pii_times)
    df = ds.to_pandas()
    grouped = df.groupby('pii_times')
    grouped_datasets = DatasetDict({str(name): Dataset.from_pandas(group) for name, group in grouped})
    return grouped_datasets


def split_text_length(ds, raw_dataset):
    def add_text_length(example):
        unmasked_text = raw_dataset[example['row_idx']]['unmasked_seq']
        text_length = len(unmasked_text.split())
        if text_length < 50:
            example['text_length'] = '<50'
        elif text_length < 100:
            example['text_length'] = '50-100'
        elif text_length < 150:
            example['text_length'] = '100-150'
        elif text_length < 200:
            example['text_length'] = '150-200'
        elif text_length < 250:
            example['text_length'] = '200-250'
        else:
            example['text_length'] = '>250'
        return example
    
    ds = ds.map(add_text_length)
    df = ds.to_pandas()
    grouped = df.groupby('text_length')
    grouped_datasets = DatasetDict({name: Dataset.from_pandas(group) for name, group in grouped})
    return grouped_datasets


def main():
    parser = HfArgumentParser((EvalArguments,))
    eval_args = parser.parse_args_into_dataclasses()[0]

    #################### get dataset ####################
    DatasetLoader = get_loader_class(eval_args.dataset_name)
    loader = DatasetLoader()
    dataset = loader.get_dataset(new_chat_template=eval_args.new_chat_template) if eval_args.dataset_name == "llm_pc" else loader.get_dataset()
    if eval_args.dataset_split:
        dataset = dataset[eval_args.dataset_split]

    #################### eval ####################
    ori_ds = load_from_disk(eval_args.dataset_path)

    if eval_args.split_strategy == 'pii_type':
        res = split_pii_type(ori_ds)
    elif eval_args.split_strategy == 'pii_times':
        res = split_pii_times(ori_ds, dataset)
    elif eval_args.split_strategy == 'text_length':
        res = split_text_length(ori_ds, dataset)
    else:
        raise ValueError(f"Unknown split strategy: {eval_args.split_strategy}")

    if eval_args.save_path:
        res.save_to_disk(eval_args.save_path)


if __name__ == "__main__":
    main()