from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from vllm import SamplingParams
from pii.candidate_generators import SelfQueryGenerator
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
        default="llm_ft/outputs/gemma-2-9b-it-echr/checkpoint-141",
        metadata={"help": "The model checkpoint for the target model"}
    )
    dataset_name: str = field(
        default="llm_pc",
        metadata={"help": "The name of the dataset to attack"}
    )
    dataset_split: Optional[str] = field(
        default="development",
        metadata={"help": "The dataset split to attack"}
    )
    iter_num: int = field(
        default=8,
        metadata={"help": "The number of iterations to run the self-query generator"}
    )
    generated_candidates_path: str = field(
        default = "generated_candidates/self_query/gemma-2-9b-it-echr",
        metadata={"help": "The path to save the generated candidates"}
    )


def main():
    parser = HfArgumentParser((AttackArguments,))

    attack_args = parser.parse_args_into_dataclasses()[0]

    #################### get dataset ####################
    DatasetLoader = get_loader_class(attack_args.dataset_name)
    loader = DatasetLoader()
    dataset = loader.get_dataset(new_chat_template=get_template_name(attack_args.model_name_or_path)) if attack_args.dataset_name == "llm_pc" else loader.get_dataset()
    if attack_args.dataset_split:
        dataset = dataset[attack_args.dataset_split]
    legal_pii_types = list(loader.PII_ASC.keys())

    #################### generate candidates ####################
    generator = SelfQueryGenerator(
        model_name_or_path=attack_args.model_name_or_path,
        iter_num=attack_args.iter_num,
        tensor_parallel_size=4,
        sampling_params=SamplingParams(
            max_tokens=128,
            temperature=1.2,
            top_k=30,
            top_p=0.8,
        )
    )
    generated_candidates = generator.generate_candidates(
        dataset, 
        legal_pii_types=legal_pii_types, 
        context_leverage_method="no_same_pii" if attack_args.dataset_name == "echr" or attack_args.dataset_name == "enron" else "no_pii",
        chat=False,
    )
    generated_candidates.save_to_disk(attack_args.generated_candidates_path)
    

if __name__ == "__main__":
    if not is_npu_available:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn")
    main()