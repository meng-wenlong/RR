from .llm_pc import RawLoader as llm_pc_RawLoader
from .echr import RawLoader as echr_RawLoader
from .enron import RawLoader as enron_RawLoader

def get_loader_class(dataset_name: str):
    if dataset_name == "llm_pc":
        return llm_pc_RawLoader
    elif dataset_name == "echr":
        return echr_RawLoader
    elif dataset_name == "enron":
        return enron_RawLoader
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported")