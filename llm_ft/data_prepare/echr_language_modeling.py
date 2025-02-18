from datasets import Dataset, DatasetDict, concatenate_datasets
from pii.datas import get_loader_class


def main():
    DatasetLoader = get_loader_class("echr")
    loader = DatasetLoader()
    data = loader.get_dataset()

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

    data['train'] = concatenate_datasets([data['train'], data['test']])
    data.save_to_disk('datas/echr')


if __name__ == "__main__":
    main()