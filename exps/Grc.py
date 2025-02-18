from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from vllm import SamplingParams
from pii.candidate_generators import RecollectGenerator
from pii.datas import get_loader_class
from pii.utils import get_template_name
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    is_npu_available = True
except:
    is_npu_available = False


@dataclass
class AttackArguments:
    model_name_or_path: str = field(
        default="llm_ft/outputs/llama3.1-8b-llm_pc",
        metadata={"help": "The model checkpoint for the target model"}
    )
    dataset_name: str = field(
        default="echr",
        metadata={"help": "The name of the dataset to attack"}
    )
    dataset_split: Optional[str] = field(
        default="train",
        metadata={"help": "The dataset split to attack"}
    )
    iter_num: int = field(
        default=40,
        metadata={"help": "The number of iterations to run the self-query generator"}
    )
    generated_candidates_path: str = field(
        default = "generated_candidates/recollect/llama3.1-8b_echr",
        metadata={"help": "The path to save the generated candidates"}
    )
    max_tokens: int = field(
        default=381,
        metadata={"help": "The max tokens for the model"}
    )


def main():
    parser = HfArgumentParser((AttackArguments,))

    attack_args = parser.parse_args_into_dataclasses()[0]

    #################### get dataset ####################
    DatasetLoader = get_loader_class(attack_args.dataset_name)
    loader = DatasetLoader()
    dataset = loader.get_dataset(new_chat_template=get_template_name(attack_args.model_name_or_path)) if attack_args.dataset_name == "llm_pc" else loader.get_dataset()
    # dataset['train'] = dataset['train'].select(range(10))
    if attack_args.dataset_split:
        dataset = dataset[attack_args.dataset_split]
    legal_pii_types = list(loader.PII_ASC.keys())

    #################### generate candidates ####################
    
    generator = RecollectGenerator(
        model_name_or_path=attack_args.model_name_or_path, 
        iter_num=attack_args.iter_num, 
        tensor_parallel_size=4,
        sampling_params=SamplingParams(
            max_tokens=attack_args.max_tokens,
            temperature=1.2,
            top_k=30,
            top_p=0.8,
        )
    )
    generated_candidates = generator.generate_candidates(
        dataset,
        legal_pii_types=legal_pii_types,
        prompt_method="pii_specific" if attack_args.dataset_name == "llm_pc" else "whole",
        # chat=True if get_template_name(attack_args.model_name_or_path) == "gemma" and attack_args.dataset_name == 'enron' else False,
    )
    generated_candidates.save_to_disk(attack_args.generated_candidates_path)


if __name__ == "__main__":
    if not is_npu_available:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn")
    main()