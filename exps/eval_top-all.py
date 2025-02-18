import os
import json
from typing import Optional
from dataclasses import dataclass, field
from datasets import load_from_disk
from transformers import HfArgumentParser
from pii.datas import get_loader_class
from pii.eval import top_n_accuracy
from pii.utils import merge_and_deduplicate


@dataclass
class EvalArguments:
    dataset_name: str = field(
        default="echr",
        metadata={"help": "The name of the dataset to attack"}
    )
    dataset_split: Optional[str] = field(
        default="train",
        metadata={"help": "The dataset split to attack"}
    )
    selected_data: str = field(
        default="",
        metadata={"help": "The path to save the generated candidates"}
    )
    new_chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "The new chat template"}
    )
    iter_num: int = field(
        default=0,
        metadata={"help": "The number of iterations to attack"}
    )
    save_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to save the generated candidates"}
    )


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
    candidates = load_from_disk(eval_args.selected_data)
    if eval_args.iter_num > 0:
        def postporocess(example):
            example["candidates"] = merge_and_deduplicate(example["raw_candidates"][:eval_args.iter_num])
            return example
        candidates = candidates.map(postporocess)

    top_all_accuracy = top_n_accuracy(dataset, candidates, sort_using_scores=False)
    print(f"Top all accuracy: {top_all_accuracy}")

    if eval_args.save_path:
        save_dir = os.path.dirname(eval_args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(eval_args.save_path, "w") as f:
            json.dump({"top_all_accuracy": top_all_accuracy}, f)


if __name__ == "__main__":
    main()