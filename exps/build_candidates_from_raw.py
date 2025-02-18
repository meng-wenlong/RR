import os
import argparse
from datasets import load_from_disk
from pii.utils import merge_and_deduplicate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter_num", type=int, default=40)
    parser.add_argument("--source_dir", type=str, default="generated_candidates/recollect")
    parser.add_argument("--target_dir", type=str, default="generated_candidates/recollect-40")

    args = parser.parse_args()

    datas = [
        # "gemma2-9b_echr",
        # "gemma2-9b_enron",
        # "gemma2-9b_llm_pc",
        # "llama3.1-8b_echr",
        # "llama3.1-8b_enron",
        # "llama3.1-8b_llm_pc",
        # "phi3.5-mini_echr",
        # "phi3.5-mini_enron",
        # "phi3.5-mini_llm_pc",
        # "qwen2.5-7b_echr",
        # "qwen2.5-7b_enron",
        # "qwen2.5-7b_llm_pc",
        # "deepseek-llama3.1-8b_echr",
        # "deepseek-llama3.1-8b_enron",
        # "deepseek-llama3.1-8b_llm_pc",
        "llama3.2-3b_echr",
        "llama3.2-3b_enron",
        "llama3.2-3b_llm_pc",
    ]

    for data in datas:
        print(data)
        generated_candidates = load_from_disk(os.path.join(args.source_dir, data))
        def postporocess(example):
            example["candidates"] = merge_and_deduplicate(example["raw_candidates"][:args.iter_num])
            return example
        generated_candidates = generated_candidates.map(postporocess)
        generated_candidates.save_to_disk(os.path.join(args.target_dir, data))


if __name__ == "__main__":
    main()