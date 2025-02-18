from datasets import Dataset, DatasetDict, concatenate_datasets
from pii.datas import get_loader_class


def main():
    DatasetLoader = get_loader_class("llm_pc")
    loader = DatasetLoader()
    for template in [None, "qwen", "gemma", "phi"]:
        data = loader.get_dataset(new_chat_template=template)

        if isinstance(data, Dataset):
            if 'text' in data.column_names:
                data = data.remove_columns('text')
            data = data.add_column('text', data['unmasked_seq'])
        elif isinstance(data, DatasetDict):
            for split in data.keys():
                if 'text' in data[split].column_names:
                    data[split] = data[split].remove_columns('text')
                data[split] = data[split].add_column('text', data[split]['unmasked_seq'])
        else:
            raise ValueError("data should be either a Dataset or a DatasetDict")

        data['train'] = data['development']
        data['test'] = data['development']
        data.save_to_disk(f'datas/llm_pc_{template}') if template is not None else data.save_to_disk('datas/llm_pc_llama')


if __name__ == "__main__":
    main()