import os
from datasets import load_dataset


def filter_bad_cases(example):
    masked_seq = example["masked_seq"]
    pii_mask_idx = example["pii_mask_idx"]
    pii_mask = example["pii_mask"]
    if len(pii_mask_idx) != len(pii_mask['labels']):
        return False
    
    for pii_mask_label in pii_mask['labels']:
        if pii_mask_label not in masked_seq:
            return False
    
    if len(set(pii_mask['labels'])) < len(pii_mask['labels']):
        return False
    
    return True


class RawLoader:
    def __init__(self):
        base_path = os.path.dirname(__file__)
        self.data_path = os.path.join(base_path, "raw/enron.jsonl")
        self.PII_ASC = {
            "CREDIT_CARD": "A credit card number is between 12 to 19 digits. https://en.wikipedia.org/wiki/Payment_card_number",
            "CRYPTO": "A Crypto wallet number. Currently only Bitcoin address is supported",
            "DATE_TIME": "Absolute or relative dates or periods or times smaller than a day.",
            "EMAIL_ADDRESS": "An email address identifies an email box to which email messages are delivered",
            "IBAN_CODE": "The International Bank Account Number (IBAN) is an internationally agreed system of identifying bank accounts across national borders to facilitate the communication and processing of cross border transactions with a reduced risk of transcription errors.",
            "IP_ADDRESS": "An Internet Protocol (IP) address (either IPv4 or IPv6).",
            "NRP": "A person’s Nationality, religious or political group.",
            "LOCATION": "Name of politically or geographically defined location (cities, provinces, countries, international regions, bodies of water, mountains",
            "PERSON": "A full person name, which can include first names, middle names or initials, and last names.",
            "PHONE_NUMBER": "A telephone number",
            "MEDICAL_LICENSE": "Common medical license numbers.",
            "URL": "A URL (Uniform Resource Locator), unique identifier used to locate a resource on the Internet",
            "US_BANK_NUMBER": "A US bank account number is between 8 to 17 digits.",
            "US_DRIVER_LICENSE": "A US driver license according to https://ntsi.com/drivers-license-format/",
            "US_ITIN": "US Individual Taxpayer Identification Number (ITIN). Nine digits that start with a \"9\" and contain a \"7\" or \"8\" as the 4 digit.",
            "US_PASSPORT": "A US passport number with 9 digits.",
            "US_SSN": "A US Social Security Number (SSN) with 9 digits.",
            "ID": "Identity number.",
            "ORGANIZATION": "Name of an organization.",
        }

    def get_dataset(self, max_size=4000, test_size=0.5):
        all_dataset = load_dataset("json", data_files=self.data_path)["train"]

        def format_pii_mask(example):
            pii_masks: list[dict] = example["pii_mask"]
            new_pii_mask: dict[list] = {'labels': [], 'values': []}
            for pii_mask in pii_masks:
                new_pii_mask['labels'].append(pii_mask['label'])
                new_pii_mask['values'].append(pii_mask['value'])

            example["pii_mask"] = new_pii_mask
            return example
        
        all_dataset = all_dataset.map(format_pii_mask)
        all_dataset = all_dataset.filter(filter_bad_cases)

        if max_size > 0:
            all_dataset = all_dataset.select(range(max_size))

        all_dataset = all_dataset.train_test_split(test_size=test_size, shuffle=False)

        return all_dataset
