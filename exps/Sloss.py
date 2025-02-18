from typing import Optional
from dataclasses import dataclass, field
from datasets import load_from_disk
from transformers import HfArgumentParser
from pii.candidate_selectors import LossSelector
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
        default="llm_ft/outputs/deepseek-llm-7b-chat-echr/checkpoint-375",
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
    refer_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for the reference model"}
    )
    generated_candidates_path: Optional[str] = field(
        default="generated_candidates/echr_presidio-deepseek-7b-recollect-iter40",
        metadata={"help": "The path to save the generated candidates"}
    )
    refer_bias: float = field(
        default=1e-6,
        metadata={"help": "The bias of refer"}
    )
    save_path: str = field(
        default="selected_candidates/loss/llama3.1-8b_echr",
        metadata={"help": "The path to save the generated candidates"}
    )
    
    processed_inter_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to save the generated candidates"}
    )
    refer_inter_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to save the generated candidates"}
    )

    ignore_pre_pii: bool = field(
        default=True,
        metadata={"help": "Ignore pre pii text"}
    )
    refer_calibrate_method: str = field(
        default="divide",
        metadata={"help": "The method to calibrate refer"}
    )


def main():
    parser = HfArgumentParser((AttackArguments,))

    attack_args = parser.parse_args_into_dataclasses()[0]
    if attack_args.refer_bias == 0:
        attack_args.refer_bias = 1e-6

    #################### get dataset ####################
    DatasetLoader = get_loader_class(attack_args.dataset_name)
    loader = DatasetLoader()
    dataset = loader.get_dataset(new_chat_template=get_template_name(attack_args.model_name_or_path)) if attack_args.dataset_name == "llm_pc" else loader.get_dataset()
    # dataset['train']=dataset['train'].select(range(196))
    if attack_args.dataset_split:
        dataset = dataset[attack_args.dataset_split]
    legal_pii_types = list(loader.PII_ASC.keys())
    
    
    generated_candidates = load_from_disk(attack_args.generated_candidates_path)
    # generated_candidates = generated_candidates.select(range(196))
    ################### select candidates ####################
    selector = LossSelector(
        model_name_or_path=attack_args.model_name_or_path,
        refer_model_name_or_path=attack_args.refer_model_name_or_path,
    )
    candidates_with_scores = selector.select_candidates(
        dataset, 
        generated_candidates, 
        legal_pii_types=legal_pii_types, 
        device_parallel_size=4,
        refer_bias=attack_args.refer_bias,
        processed_inter_dataset_path=attack_args.processed_inter_dataset_path,
        refer_inter_dataset_path=attack_args.refer_inter_dataset_path,
        ignore_pre_pii=attack_args.ignore_pre_pii,
        refer_calibrate_method=attack_args.refer_calibrate_method,
    )
    candidates_with_scores.save_to_disk(attack_args.save_path)
    

if __name__ == "__main__":
    if not is_npu_available:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn")
    main()