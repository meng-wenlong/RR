import os
import jsonlines
import re
import spacy
from collections import defaultdict
from datasets import Dataset, DatasetDict, load_from_disk
from presidio_analyzer.nlp_engine import NlpEngineProvider, TransformersNlpEngine, NerModelConfiguration
from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine
from typing import Optional
from ...utils import get_unmask_seq, find_pii_indices, mask_pii_in_seq, illegal_pii_list

spacy.require_gpu()


def is_lowercase_and_spcaes(s: str) -> bool:
    return bool(re.fullmatch(r'[a-z ]*', s))


class RawLoader:
    def __init__(self):
        base_path = os.path.dirname(__file__)
        self.dev_pii_dict_path = os.path.join(base_path, "raw/development/LLM-PC-development-pii.jsonl")
        self.dev_scrubbed_data_path = os.path.join(base_path, "raw/development/LLM-PC-development-scrubbed-data.jsonl")
        self.test_pii_dict_path = os.path.join(base_path, "raw/test/test_pii_revised.jsonl")
        self.test_scrubbed_data_path = os.path.join(base_path, "raw/test/test_scrubbed.jsonl")

        self.PII_ASC = {
            "AC": "Academic fields.",  
            "ACC": "Account numbers.",  
            "AGE": "Age-related information, including specific ages and age ranges.",  
            "ARTIST-NAME": "Names of artists.",  
            "ARTIST": "Names of artists.",  
            "AUTHOR": "Names of authors.",  
            "CHARACTER": "Names of characters in literature, film, or other media.",  
            "DATE": "All elements of dates (except year) for dates directly related to an individual, including birth date, admission date, discharge date, date of death; and all ages over 89 and all elements of dates (including year) indicative of such age, except that such ages and elements may be aggregated into a single category of age 90 or older.",  
            "DETAIL": "Details.",  
            "DEVICE_ID": "Device identifiers and serial numbers.",
            "DIRECTOR": "Director names",
            "EMAIL": "Electronic mail addresses",  
            "ENTITY_TYPE": "Types of entities, such as organizations or companies.",  
            "EVENT": "Events, such as conferences, concerts, or meetings.", 
            "FAX": "Fax numbers.",   
            "HISTORICAL-FIGURE" : "Historical figures",
            "HOBBY": "Hobbies or personal interests.", 
            "HPB_NUM": "Health plan beneficiary numbers.",   
            "IP": "Internet Protocol (IP) address numbers.",  
            "LANG": "Programming languages.",  
            "LICENSE": "Certificate/license numbers.",  
            "LOCAL-PARK": "Names of local parks.",
            "LOCAL": "Names of local parks.",
            "LOC": "All geographical subdivisions smaller than a State, including street address, city, county, precinct, zip code, and their equivalent geocodes, except for the initial three digits of a zip code, if according to the current publicly available data from the Bureau of the Census: (1) The geographic unit formed by combining all zip codes with the same three initial digits contains more than 20,000 people; and (2) The initial three digits of a zip code for all such geographic units containing 20,000 or fewer people is changed to 000.",
            "LOCATION": "Locations.",
            "MED_NUM": "Medical record numbers.",  
            "MUS": "Names of musicians.",  
            "NAME": "Names.",  
            "NUMBER": "Numbers.",
            "NOT": "Notable landmarks.", 
            "OBJECTIVE": "Objectives.",
            "OPTION": "Options.",
            "PARK-NAME": "Names of parks.",
            "PHI": "Personal health information.",  
            "PHI_TYPE": "Types of personal health information, such as medical conditions, treatments, or medications.",  
            "PHONE": "Phone numbers.",  
            "PLACE": "Places", 
            "PLATFORM": "Platforms, such as social media or software platforms.",  
            "PROFESSION": "Professions",
            "RESOURCE": "Resources, such as books, articles, or websites.",  
            "SONG": "Names of songs",
            "STATUE": "Statues",
            "SSN": "Social Security numbers.",  
            "TEAM": "Teams, such as sports teams or project teams.",  
            "TOOL": "Tools, such as software or hardware tools.",  
            "TOPIC": "Topics of discussion or interest.",  
            "URL": "Web Universal Resource Locators (URLs).",  
            "VEHICLE_ID": "Vehicle identifiers and serial numbers, including license plate numbers.",  
        }

        self.pii_type_to_presidio_entity_mapping = {
            "AC": "PERSON",  # inaccurate
            "ACC": "US_BANK_NUMBER",
            "AGE": "DATE_TIME",  # inaccurate
            "ARTIST-NAME": "PERSON",
            "ARTIST": "PERSON",
            "AUTHOR": "PERSON",
            "CHARACTER": "PERSON",
            "DATE": "DATE_TIME",
            "DETAIL": "PERSON",  # inaccurate
            "DEVICE_ID": "IP_ADDRESS",  # inaccurate
            "DIRECTOR": "PERSON",
            "EMAIL": "EMAIL_ADDRESS",
            "ENTITY_TYPE": "PERSON",  # inaccurate
            "EVENT": "PERSON",  # inaccurate
            "FAX": "PHONE_NUMBER",
            "HISTORICAL-FIGURE": "PERSON",
            "HOBBY": "PERSON",  # inaccurate
            "HPB_NUM": "US_SSN",  # inaccurate
            "IP": "IP_ADDRESS",
            "LANG": "PERSON",  # inaccurate
            "LICENSE": "US_DRIVER_LICENSE",
            "LOCAL-PARK": "LOCATION",
            "LOCAL": "LOCATION",
            "LOC": "LOCATION",
            "LOCATION": "LOCATION",
            "MED_NUM": "US_SSN",  # inaccurate
            "MUS": "PERSON",
            "NAME": "PERSON",
            "NUMBER": "US_SSN",  # inaccurate
            "NOT": "LOCATION",
            "OBJECTIVE": "PERSON",  # inaccurate
            "OPTION": "PERSON",  # inaccurate
            "PARK-NAME": "LOCATION",
            "PHI": "PERSON",  # inaccurate
            "PHI_TYPE": "PERSON",  # inaccurate
            "PHONE": "PHONE_NUMBER",
            "PLACE": "LOCATION",
            "PLATFORM": "URL",  # inaccurate
            "PROFESSION": "PERSON",  # inaccurate
            "RESOURCE": "PERSON",  # inaccurate
            "SONG": "PERSON",  # inaccurate
            "STATUE": "LOCATION",  # inaccurate
            "SSN": "US_SSN",
            "TEAM": "PERSON",  # inaccurate
            "TOOL": "PERSON",  # inaccurate
            "TOPIC": "PERSON",  # inaccurate
            "URL": "URL",
            "VEHICLE_ID": "US_DRIVER_LICENSE",  # inaccurate
        }

        self.inaccurate_pii_types = [
            "AC",
            "AGE",
            "DETAIL",
            "DEVICE_ID",
            "ENTITY_TYPE",
            "EVENT",
            "HOBBY",
            "HPB_NUM",
            "LANG",
            "MED_NUM",
            "NUMBER",
            "OBJECTIVE",
            "OPTION",
            "PHI",
            "PHI_TYPE",
            "PROFESSION",
            "RESOURCE",
            "SONG",
            "TEAM",
            "TOOL",
            "TOPIC",
            "VEHICLE_ID",
        ]

        self.accurate_pii_types = list(set(self.PII_ASC.keys()) - set(self.inaccurate_pii_types))

    def _remove_system_prompt(self, dataset: Dataset):
        system_prompt = "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>"
        def remove_system_prompt_from_dataset(example):
            example["unmasked_seq"] = example["unmasked_seq"].replace(system_prompt, "")
            example["masked_seq"] = example["masked_seq"].replace(system_prompt, "")
            return example
        return dataset.map(remove_system_prompt_from_dataset)
    
    def _replace_chat_template(self, dataset: Dataset, new_chat_template: str):
        def replace_chat_template_qwen(example):
            def replace_strings(text: str) -> str:
                return text.replace(
                    "<|begin_of_text|>", ""
                ).replace(
                    "<|start_header_id|>system<|end_header_id|>\n\n", "<|im_start|>system\n"
                ).replace(
                    "<|eot_id|>", "<|im_end|>\n"
                ).replace(
                    "<|start_header_id|>user<|end_header_id|>\n\n", "<|im_start|>user\n"
                ).replace(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n", "<|im_start|>assistant\n"
                )
            example["unmasked_seq"] = replace_strings(example["unmasked_seq"])
            example["masked_seq"] = replace_strings(example["masked_seq"])
            return example
        
        def replace_chat_template_phi(example):
            def replace_strings(text: str) -> str:
                return text.replace(
                    "<|begin_of_text|>", ""
                ).replace(
                    "<|start_header_id|>system<|end_header_id|>\n\n", "<|system|>\n"
                ).replace(
                    "<|eot_id|>", "<|end|>\n"
                ).replace(
                    "<|start_header_id|>user<|end_header_id|>\n\n", "<|user|>\n"
                ).replace(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n", "<|assistant|>\n"
                )
            example["unmasked_seq"] = replace_strings(example["unmasked_seq"])
            example["masked_seq"] = replace_strings(example["masked_seq"])
            example["unamsked_seq"] = example["unmasked_seq"] + "<|endoftext|>"
            example["masked_seq"] = example["masked_seq"] + "<|endoftext|>"
            return example

        def replace_chat_template_gemma(example):
            def replace_strings(text: str) -> str:
                return text.replace(
                    "<|begin_of_text|>", "<bos>"
                ).replace(
                    "<|start_header_id|>system<|end_header_id|>\n\n", "<start_of_turn>user\n"
                ).replace(
                    "<|eot_id|>", "<end_of_turn>\n"
                ).replace(
                    "<|start_header_id|>user<|end_header_id|>\n\n", "<start_of_turn>user\n"
                ).replace(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n", "<start_of_turn>model\n"
                )
            example["unmasked_seq"] = replace_strings(example["unmasked_seq"])
            example["masked_seq"] = replace_strings(example["masked_seq"])
            return example

        def replace_chat_template_deepseek(example):
            def replace_strings(text: str) -> str:
                return text.replace(
                    "<|begin_of_text|>", "<｜begin▁of▁sentence｜>"
                ).replace(
                    "<|start_header_id|>system<|end_header_id|>\n\n", ""
                ).replace(
                    "<|eot_id|>", ""
                ).replace(
                    "<|start_header_id|>user<|end_header_id|>\n\n", "<｜User｜>"
                ).replace(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n", "<｜Assistant｜>"
                )
            example["unmasked_seq"] = replace_strings(example["unmasked_seq"])
            example["masked_seq"] = replace_strings(example["masked_seq"])
            example["unamsked_seq"] = example["unmasked_seq"] + "<｜end▁of▁sentence｜>"
            example["masked_seq"] = example["masked_seq"] + "<｜end▁of▁sentence｜>"

        if new_chat_template == "qwen":
            return dataset.map(replace_chat_template_qwen)
        elif new_chat_template == "phi":
            return dataset.map(replace_chat_template_phi)
        elif new_chat_template == "gemma":
            return dataset.map(replace_chat_template_gemma)
        elif new_chat_template == "deepseek":
            return dataset.map(replace_chat_template_deepseek)
        else:
            raise ValueError(f"Unknown chat template: {new_chat_template}")

    def reidentify_piis(
        self,
        dataset: Dataset,
        analyzer: Optional[AnalyzerEngine] = None,
    ):
        if analyzer is None:
            '''
            # Create configuration containing engine name and models
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_trf"}],
            }

            # Create NLP engine based on configuration
            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine = provider.create_engine()

            # Pass the created NLP engine and supported_languages to the AnalyzerEngine
            analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine, supported_languages=["en"], default_score_threshold=0.8
            )
            '''
            # Transformer model config
            model_config = [
                {"lang_code": "en",
                "model_name": {
                    "spacy": "en_core_web_trf", # for tokenization, lemmatization
                    "transformers": "StanfordAIMI/stanford-deidentifier-base" # for NER
                }
            }]

            # Entity mappings between the model's and Presidio's
            mapping = dict(
                # PER="PERSON",
                # LOC="LOCATION",
                # ORG="ORGANIZATION",
                # AGE="AGE",
                ID="ID",
                # EMAIL="EMAIL",
                DATE="DATE_TIME",
                PHONE="PHONE_NUMBER",
                # PERSON="PERSON",
                # LOCATION="LOCATION",
                # GPE="LOCATION",
                # ORGANIZATION="ORGANIZATION",
                # NORP="NRP",
                PATIENT="PERSON",
                # STAFF="PERSON",
                # HOSP="LOCATION",
                # PATORG="ORGANIZATION",
                # TIME="DATE_TIME",
                HCW="PERSON",
                HOSPITAL="LOCATION",
                # FACILITY="LOCATION",
                VENDOR="ORGANIZATION",
            )

            labels_to_ignore = ["O"]

            ner_model_configuration = NerModelConfiguration(
                model_to_presidio_entity_mapping=mapping,
                alignment_mode="expand", # "strict", "contract", "expand"
                aggregation_strategy="max", # "simple", "first", "average", "max"
                labels_to_ignore = labels_to_ignore,
            )

            transformers_nlp_engine = TransformersNlpEngine(
                models=model_config,
                ner_model_configuration=ner_model_configuration,
            )

            # Transformer-based analyzer
            analyzer = AnalyzerEngine(
                nlp_engine=transformers_nlp_engine, 
                supported_languages=["en"],
                default_score_threshold=0.8
            )

        batch_analyzer = BatchAnalyzerEngine(analyzer_engine=analyzer)
        entities = [self.pii_type_to_presidio_entity_mapping[pii_type] for pii_type in self.accurate_pii_types]
        entities = list(set(entities))
        analyze_results = batch_analyzer.analyze_iterator(
            texts=dataset["unmasked_seq"],
            entities=entities,
            language="en",
            batch_size=256,
        )

        def assign_pii_for_dataset(example, idx):
            analyze_result = analyze_results[idx]
            # sort analyze_result by start
            analyze_result = sorted(analyze_result, key=lambda x: x.start)
            pii_values = [example['unmasked_seq'][entity.start:entity.end] for entity in analyze_result]

            def is_legal_value(value: str) -> bool:
                if value in illegal_pii_list:
                    return False
                if pii_values.count(value) == 1 and is_lowercase_and_spcaes(value):
                    return False
                if '<|' in value or '|>' in value:
                    return False
                
                return True

            label_counters = defaultdict(int) # key: entity_type, val: current counter
            et_value2label = {}  # key: (entity_type, value), value: label string

            pii_mask_idx = []
            for ei, entity in enumerate(analyze_result):
                value = pii_values[ei]
                if not is_legal_value(value):
                    continue
                # remove overlaped piis (we only maintain the one with the highest score)
                if self._is_repeated_pii(analyze_result, entity):
                    continue

                # if (entity_type, value) has not been seen, assign a new label
                key = (entity.entity_type, value)
                if key not in et_value2label:
                    label_counters[entity.entity_type] += 1
                    label_with_id = f"{entity.entity_type}-{label_counters[entity.entity_type]}"
                    et_value2label[key] = label_with_id
                else:
                    label_with_id = et_value2label[key]

                pii_mask_idx.append({
                    "start": entity.start,
                    "end": entity.end,
                    "label": label_with_id,
                    "value": value,
                })
            example["pii_mask_idx"] = pii_mask_idx

            unique_pii_dict = {}
            for item in pii_mask_idx:
                lbl = item["label"]
                val = item["value"]
                if lbl not in unique_pii_dict:
                    unique_pii_dict[lbl] = val

            example["pii_mask"]["labels"] = list(unique_pii_dict.keys())
            example["pii_mask"]["values"] = list(unique_pii_dict.values())

            example["masked_seq"] = mask_pii_in_seq(example["unmasked_seq"], pii_mask_idx)
            return example
        
        return dataset.map(assign_pii_for_dataset, with_indices=True)


    def _is_repeated_pii(self, analyze_result, entity):
        assert entity in analyze_result
        overlapping = []
        for item in analyze_result:
            if item == entity:
                continue
            if not (item.end < entity.start or item.start > entity.end):
                overlapping.append(item)

        for item in overlapping:
            if item.score > entity.score:
                return True

            if item.score == entity.score:
                if analyze_result.index(item) < analyze_result.index(entity):
                    return True 
        return False

    def get_dataset(
        self,
        remove_system_prompt: bool = True,
        new_chat_template: Optional[str] = None,
        reidentify_piis: bool = True,
        analyzer: Optional[AnalyzerEngine] = None,
        use_cache: bool = True
    ):
        if new_chat_template == "llama":
            new_chat_template = None
        if reidentify_piis:
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
        
        dataset_path = os.path.join(os.path.dirname(__file__), f"cache_data_system-{remove_system_prompt}_template-{new_chat_template}_reidentify-{reidentify_piis}")
        if use_cache and os.path.exists(dataset_path):
            return load_from_disk(dataset_path)

        # Load development
        dev_pii_dict = []
        with jsonlines.open(self.dev_pii_dict_path) as reader:
            for obj in reader:
                dev_pii_dict.append(obj)
        
        dev_scrubbed_data = []
        with jsonlines.open(self.dev_scrubbed_data_path) as reader:
            for obj in reader:
                dev_scrubbed_data.append(obj)

        assert len(dev_pii_dict) == len(dev_scrubbed_data), "Development dataset length mismatch"

        dev_rows = []
        for i in range(len(dev_pii_dict)):
            piis = dev_pii_dict[i]
            scrubbed = dev_scrubbed_data[i]
            
            # unmask_seq | mask_seq | pii_mask_idx | pii_mask
            mask_seq = scrubbed["messages"]
            pii_mask = piis
            unmask_seq = get_unmask_seq(mask_seq, pii_mask)
            pii_mask_idx = find_pii_indices(pii_mask, unmask_seq)

            row = {
                "unmasked_seq": unmask_seq,
                "masked_seq": mask_seq,
                "pii_mask_idx": pii_mask_idx,
                "pii_mask": {"labels": list(pii_mask.keys()), "values": list(pii_mask.values())},
            }
            dev_rows.append(row)

        dev_dataset = Dataset.from_list(dev_rows)

        # Load test
        test_pii_dict = []
        with jsonlines.open(self.test_pii_dict_path) as reader:
            for obj in reader:
                test_pii_dict.append(obj)

        test_scrubbed_data = []
        with jsonlines.open(self.test_scrubbed_data_path) as reader:
            for obj in reader:
                test_scrubbed_data.append(obj)

        assert len(test_pii_dict) == len(test_scrubbed_data), "Test dataset length mismatch"

        test_rows = []
        for i in range(len(test_pii_dict)):
            piis = test_pii_dict[i]
            scrubbed = test_scrubbed_data[i]
            
            # unmask_seq (none) | mask_seq | pii_mask_idx | pii_mask (key: "")
            mask_seq = scrubbed["messages"]
            pii_mask = piis

            row = {
                "unmasked_seq": "",
                "masked_seq": mask_seq,
                "pii_mask_idx": [],
                "pii_mask": {"labels": list(pii_mask.keys()), "values": list(pii_mask.values())},
            }
            test_rows.append(row)

        test_dataset = Dataset.from_list(test_rows)

        if remove_system_prompt:
            dev_dataset = self._remove_system_prompt(dev_dataset)
            test_dataset = self._remove_system_prompt(test_dataset)

        if new_chat_template is not None:
            dev_dataset = self._replace_chat_template(dev_dataset, new_chat_template)
            test_dataset = self._replace_chat_template(test_dataset, new_chat_template)

        if reidentify_piis:
            dev_dataset = self.reidentify_piis(dev_dataset, analyzer=analyzer)

        final_datasets = DatasetDict({
            "development": dev_dataset,
            "test": test_dataset,
        })
        final_datasets.save_to_disk(dataset_path)
        return final_datasets


if __name__ == "__main__":
    loader = RawLoader()
    data = loader.get_dataset()