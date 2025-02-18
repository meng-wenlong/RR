import json
from typing import Optional
from dataclasses import dataclass, field
from datasets import load_from_disk, DatasetDict
from transformers import HfArgumentParser
from pii.datas import get_loader_class
from pii.eval import top_one_accuracy, top_n_accuracy


@dataclass
class EvalArguments:
    dataset_name: str = field(
        default="echr",
        metadata={"help": "The name of the dataset to attack"}
    )
    dataset_split: Optional[str] = field(
        default='train',
        metadata={"help": "The dataset split to attack"}
    )
    selected_data: str = field(
        default='selected_candidates/loss/llama3.1-8b_echr',
        metadata={"help": "The path to the selected data"}
    )
    save_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to save the generated candidates"}
    )
    new_chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "The new chat template"}
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
    candidates_with_scores: DatasetDict = load_from_disk(eval_args.selected_data)
    
    save_data = {}
    for label, ds in candidates_with_scores.items():
        accuracy = top_one_accuracy(dataset, ds)
        print(f"Top 1 accuracy: {accuracy}")

        top_2_accuracy = top_n_accuracy(dataset, ds, n=2)
        print(f"Top 2 accuracy: {top_2_accuracy}")

        top_3_accuracy = top_n_accuracy(dataset, ds, n=3)
        print(f"Top 3 accuracy: {top_3_accuracy}")

        top_5_accuracy = top_n_accuracy(dataset, ds, n=5)
        print(f"Top 5 accuracy: {top_5_accuracy}")

        top_10_accuracy = top_n_accuracy(dataset, ds, n=10)
        print(f"Top 10 accuracy: {top_10_accuracy}")

        top_all_accuracy = top_n_accuracy(dataset, ds, n=-1)
        print(f"Top all accuracy: {top_all_accuracy}")

        save_data[label] = {
            "top_1_accuracy": accuracy,
            "top_2_accuracy": top_2_accuracy,
            "top_3_accuracy": top_3_accuracy,
            "top_5_accuracy": top_5_accuracy,
            "top_10_accuracy": top_10_accuracy,
            "top_all_accuracy": top_all_accuracy,
            "total": len(ds),
        }

    if eval_args.save_path:
        # save json
        with open(eval_args.save_path, "w") as f:
            json.dump(save_data, f, indent=4)


if __name__ == "__main__":
    main()