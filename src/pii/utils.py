import re
import sys
import ray
import gc
import contextlib
import itertools
import torch
import spacy
from concurrent.futures import ProcessPoolExecutor, as_completed
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
from typing import Optional
from presidio_analyzer.nlp_engine import NlpEngineProvider, NerModelConfiguration, TransformersNlpEngine
from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine
from transformers import AutoConfig


def get_unmask_seq(
    mask_seq: str, 
    pii_mask: dict[str, str], 
    mask_prefix: str = "[",
    mask_suffix: str = "]",
) -> str:
    unmask_seq = mask_seq
    for mask, entity in pii_mask.items():
        unmask_seq = unmask_seq.replace(mask_prefix + mask + mask_suffix, entity)
    
    return unmask_seq


def find_pii_indices(
    pii_mask: dict[str, str],
    unmask_seq: str,
) -> list[dict]:
    pii_mask_idx = []
    for mask, entity in pii_mask.items():
        start_positions = [m.start() for m in re.finditer(re.escape(entity), unmask_seq)]
        for pos in start_positions:
            end_pos = pos + len(entity)
            pii_mask_idx.append({
                "start": pos,
                "end": end_pos,
                "label": mask,
                "value": entity,
            })

    return pii_mask_idx


def mask_pii_in_seq(unmasked_seq: str, pii_mask_idx: list) -> str:
    masked_seq = unmasked_seq
    # backward loop to avoid index shift
    for item in reversed(pii_mask_idx):
        s = item["start"]
        e = item["end"]
        lbl = item["label"]
        # mask the entity
        masked_seq = masked_seq[:s] + f"[{lbl}]" + masked_seq[e:]
    return masked_seq


def batch_data(data_list, batch_size=200):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n - 1):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data.append(data_list[start:end])

    last_start = (n - 1) * batch_size
    last_end = sys.maxsize
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def vllm_cleanup(llm):
    del llm.llm_engine.model_executor
    del llm
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()


def find_substring_locations(main_string, substring):
    return [m.start() for m in re.finditer(re.escape(substring), main_string)]


def has_pii_mask(text: str, pii_key: Optional[str] = None) -> bool:
    if pii_key is not None:
        pattern = r'\[' + pii_key + r'\]'
        return pattern in text

    pattern = r'\[[A-Z]+-\d+\]'
    match = re.search(pattern, text)
    return match is not None


illegal_pii_list = ['One moment', 'today', 'Today', 'the day', 'the days', 'the year', 'the years', 'the month', 'the months', 'my 92 years', 'Just a moment', 'just a moment', 'One evening', 'one evening', 'night', 'Every year', 'every year', 'the holidays', 'Just last summer', 'that day', 'these days', 'that year', 'this year', 'this month', 'this week', 'this day', 'this evening', 'this morning', 'this afternoon', 'this night', 'this summer', 'this winter', 'this spring', 'this fall', 'this holiday', 'this weekend', 'this week', 'this month', 'this year', 'this decade', 'this century', 'this millennium', 'this morning', 'this afternoon', 'this evening', 'this night', 'this summer', 'this winter', 'this spring', 'this fall', 'this holiday', 'this weekend', 'this week', 'this month', 'this year', 'this decade', 'this century', 'this millennium', 'this morning', 'this afternoon', 'this evening', 'this night', 'this summer', 'this winter', 'this spring', 'this fall', 'this holiday', 'this weekend', 'this week', 'this month', 'this year', 'this decade', 'this century', 'this millennium', 'this morning', 'this afternoon', 'this evening', 'this night', 'this summer', 'this winter', 'this spring', 'this fall', 'this holiday', 'this weekend', 'this week', 'this month', 'this year', 'this decade', 'this century', 'this millennium', 'this morning', 'this afternoon', 'this evening', 'this night', 'this summer', 'this winter', 'this spring', 'this fall', 'this holiday', 'this weekend', 'this week', 'this month', 'this year', 'this decade', 'this century', 'this millennium', 'this morning', 'this afternoon', 'this evening', 'this night', 'this summer', 'this winter', 'this spring', 'this fall', 'this holiday', 'this weekend', 'this week', 'this month', 'this year', 'this decade', 'this century', 'this millennium', 'this morning', 'this afternoon', 'this evening', 'this night', 'this summer', 'this winter', 'this spring', 'this fall', 'this holiday', 'this weekend', 'this week', 'this month', 'this year', 'this decade', 'this century', 'this millennium', 'this morning', 'this afternoon', 'this evening', 'this night', 'this summer', 'this winter', 'this spring', 'this fall', 'this holiday', 'a few years ago', 'a few years later', 'a few years earlier', 'a few years before', 'a few years after', 'a few years from now', 'a few years hence', 'Summer', 'summer', 'Winter', 'winter', 'Spring', 'spring', 'Fall', 'fall', 'Holiday', 'holiday', 'Weekend', 'weekend', 'Week', 'week', 'Month', 'month', 'Year', 'year', 'Those days', 'those days', 'Those years', 'those years', 'Those months', 'those months', 'Those weeks', 'those weeks', 'Those days', 'those days', 'Those evenings', 'those evenings', 'Those mornings', 'those mornings', 'Those afternoons', 'those afternoons', 'Those nights', 'those nights', 'Those summers', 'those summers', 'Those winters', 'those winters', 'Those springs', 'those springs', 'Those falls', 'those falls', 'Those holidays', 'those holidays', 'Those weekends', 'those weekends', 'Those weeks', 'those weeks', 'Those months', 'those months', 'Those years', 'those years', 'Those decades', 'those decades', 'Those centuries', 'those centuries', 'Those millenniums', 'those millenniums', 'Those mornings', 'those mornings', 'Those afternoons', 'those afternoons', 'Those evenings', 'those evenings', 'Those nights', 'those nights', 'Those summers', 'those summers', 'Those winters', 'those winters', 'Those springs', 'those springs', 'Those falls', 'those falls', 'Those holidays', 'those holidays', 'Those weekends', 'those weekends', 'Those weeks', 'those weeks', 'Those months', 'those months', 'Those years', 'those years', 'Those decades', 'those decades', 'Those centuries', 'those centuries', 'Those millenniums', 'those millenniums', 'Those mornings', 'those mornings', 'Those afternoons', 'those afternoons', 'Those evenings', 'those evenings', 'Those nights', 'those nights', 'Those summers', 'those summers', 'Those winters', 'those winters', 'Those springs', 'those springs', 'Those falls', 'those falls', 'Those holidays', 'those holidays', 'Those weekends', 'those weekends', 'Those weeks', 'those weeks', 'Those months', 'those months', 'Those years', 'those years', 'Those decades', 'those decades', 'Those centuries', 'those centuries', 'Those millenniums', 'those millenniums', 'Those mornings', 'those mornings', 'Those afternoons', 'those afternoons', 'Those evenings', 'those evenings', 'Those nights', 'those nights', 'Those summers', 'those summers', 'Those winters', 'those winters', 'Those springs', 'those springs', 'Those falls', 'my day', 'my days', "Hello", "Hi, Dorothy", "user", "assistant", "system", "Wow, Linda Ernest", "Hmm", "Wow, Heidi Reid", "Wow, Bessie Cronin"]


def create_analyzer():
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

    return analyzer


def is_legal_value(value: str) -> bool:
    if value in illegal_pii_list:
        return False
    if '<|' in value or '|>' in value:
        return False

    return True


def remove_chat_template(text: str) -> str:
    text = text.replace(
        "<|begin_of_text|>", ""
    ).replace(
        "<|start_header_id|>system<|end_header_id|>\n\n", "",
    ).replace(
        "<|eot_id|>", " "
    ).replace(
        "<|start_header_id|>user<|end_header_id|>\n\n", ""
    ).replace(
        "<|start_header_id|>assistant<|end_header_id|>\n\n", ""
    )

    text = text.replace(
        "<bos>", ""
    ).replace(
        "<start_of_turn>user\n", ""
    ).replace(
        "<end_of_turn>\n", " "
    ).replace(
        "<start_of_turn>model\n", ""
    )

    text = text.replace(
        "<|im_start|>system\n", ""
    ).replace(
        "<|im_end|>\n", " "
    ).replace(
        "<|im_start|>user\n", ""
    ).replace(
        "<|im_start|>assistant\n", ""
    )

    text = text.replace(
        "<|system|>\n", ""
    ).replace(
        "<|end|>\n", " "
    ).replace(
        "<|user|>\n", ""
    ).replace(
        "<|assistant|>\n", ""
    ).replace(
        "<|endoftext|>", ""
    )

    text = text.replace(
        "<｜begin▁of▁sentence｜>", ""
    ).replace(
        "<｜User｜>", " "
    ).replace(
        "<｜Assistant｜>", " "
    ).replace(
        "<｜end▁of▁sentence｜>", ""
    )

    return text


def get_template_name(model_name_or_path: str) -> str:
    config = AutoConfig.from_pretrained(model_name_or_path)
    if 'deepseek' in getattr(config, "_name_or_path", "") or "DeepSeek" in getattr(config, "_name_or_path", ""):
        return "deepseek"

    type2template = {
        "llama": "llama",
        "llama3": "llama",
        "qwen2": "qwen",
        "qwen": "qwen",
        "gemma2": "gemma",
        "gemma": "gemma",
        "phi2": "phi",
        "phi3": "phi",
        "phi": "phi",
    }
    
    return type2template.get(config.model_type, "llama")


def analyze_chunk(texts, gpu_id):
    spacy.require_gpu(gpu_id)
    analyzer = create_analyzer()
    batch_analyzer = BatchAnalyzerEngine(analyzer_engine=analyzer)
    all_entities = analyzer.get_supported_entities()
    supported_entities = [entity for entity in all_entities if not entity.startswith('UK_') and not entity.startswith('IN_') and not entity.startswith('SG_') and not entity.startswith('AU_')]
    analyze_results = batch_analyzer.analyze_iterator(
        texts=texts,
        entities=supported_entities,
        language='en',
        batch_size=256,
    )
    return analyze_results


def identify_piis(
    texts: list[str],
    device_parallel_size: int,
):
    chunk_size = len(texts) // device_parallel_size
    remainder = len(texts) % device_parallel_size
    chunks = []
    start = 0
    for i in range(device_parallel_size):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(texts[start:end])
        start = end

    with ProcessPoolExecutor(max_workers=device_parallel_size) as executor:
        futures = []
        for i in range(device_parallel_size):
            future = executor.submit(analyze_chunk, chunks[i], i)
            futures.append(future)
        
        results = [None] * device_parallel_size
        for fut in as_completed(futures):
            index = futures.index(fut)
            results[index] = fut.result()

    analyze_results = []
    for result in results:
        analyze_results.extend(result)

    return analyze_results


def cut_off_text(text: str, word_len: int = 20) -> str:
    words = text.split()
    if len(words) > word_len:
        return ' '.join(words[-word_len:]) 
    return text


def merge_and_deduplicate(lists):
    return list(set(itertools.chain.from_iterable(lists)))