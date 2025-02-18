import re
import logging
import spacy
from collections import defaultdict
from datasets import Dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm
from typing import Optional
from presidio_analyzer import BatchAnalyzerEngine
from .sq_generator import SelfQueryGenerator
from ..utils import batch_data, is_legal_value, identify_piis, cut_off_text


spacy.require_gpu()
logger = logging.getLogger(__name__)


class TabAnalyzeGenerator(SelfQueryGenerator):
    def __init__(self, model_name_or_path: str,
                 tensor_parallel_size: int = 4,
                 gpu_memory_utilization: float = 0.9,
                 sampling_params: Optional[SamplingParams] = None,
                 iter_num = 10,):
        super().__init__(
            model_name_or_path=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            sampling_params=sampling_params,
            iter_num=iter_num
        )
        self.sampling_params.n = 1

    def compose_prompts_for_tab(
        self,
        masked_seq: str,
        pii_mask: dict[str, list[str]],
        legal_pii_types: Optional[list[str]] = None,
    ):
        prompts = []
        for pii_key in pii_mask['labels']:
            pii_type = pii_key.split('-')[0]
            if legal_pii_types and pii_type not in legal_pii_types:
                logger.warning(f"Skipping {pii_type} as it is not in legal_pii_types")
                continue

            def get_pii_key_prefixes(masked_seq, pii_key) -> list[str]:
                pattern = re.compile(r'\[[^\]]+-\d+\]')
                matches = list(pattern.finditer(masked_seq))
                results = []
                prev_end = 0

                for m in matches:
                    bracket_str = m.group(0)
                    start_idx = m.start()
                    end_idx = m.end()

                    # If we have found the bracket matching pii_key, collect the preceding text.
                    if bracket_str == f'[{pii_key}]':
                        prefix = masked_seq[prev_end:start_idx]
                        if prefix and prefix[-1] == ' ':
                            prefix = prefix[:-1]
                        if prefix:
                            results.append({
                                "pii_key": pii_key,
                                "prompt": prefix,
                            })
                        prev_end = end_idx
                    else:
                        # If it is a different key, skip past it so we do not collect this part.
                        prev_end = end_idx

                return results

            pii_key_prefixes = get_pii_key_prefixes(masked_seq, pii_key)
            if len(pii_key_prefixes) == 0:
                pii_key_prefixes.append({
                    "pii_key": pii_key,
                    "prompt": "Generate a piece of text:",
                })
                # logger.warning(f"Skipping {pii_key} as it does not have any prefixes")
                # continue
            sorted_pii_key_prefixes = sorted(
                pii_key_prefixes, key=lambda x: len(x["prompt"]), reverse=True
            )
            for item in sorted_pii_key_prefixes:
                item['prompt'] = cut_off_text(item['prompt'])
            
            def adjust_list_length(lst: list, n: int) -> list:
                if len(lst) >= n:
                    return lst[:n]
                else:
                    return (lst * (n // len(lst) + 1))[:n]

            prompts.extend(adjust_list_length(sorted_pii_key_prefixes, self.iter_num))
        return prompts

    def generate_candidates(
        self,
        test_dataset: Dataset,
        legal_pii_types: Optional[list[str]] = None,
        pii_type_to_presidio_entity_mapping: dict[str, str] = None,
    ) -> Dataset:
        all_prompts = []
        for row_idx, row in enumerate(test_dataset):
            masked_seq = row["masked_seq"]
            pii_mask = row["pii_mask"]
            prompts = self.compose_prompts_for_tab(masked_seq, pii_mask, legal_pii_types)
            for prompt in prompts:
                prompt['row_idx'] = row_idx
            all_prompts.extend(prompts)

        batched_prompts = batch_data(all_prompts)
        completions = []

        # Inference
        for br in tqdm(batched_prompts, desc="Generating candidates"):
            if isinstance(br, list):
                pass
            else:
                br = [br]
            input_prompts = [b['prompt'] for b in br]
            completion = self.llm.generate(input_prompts, self.sampling_params)
            completions.extend(completion)

        self.destroy_llm()

        # Extract candidates from generated text
        assert len(all_prompts) == len(completions)
        all_generated_texts = [completion.outputs[0].text for completion in completions]
        all_analyze_results = identify_piis(
            all_generated_texts,
            device_parallel_size=self.llm.llm_engine.parallel_config.tensor_parallel_size,
        )
        for i in range(len(all_prompts)):
            all_prompts[i]['completion'] = completions[i]
            all_prompts[i]['analyze_result'] = all_analyze_results[i]
        row_idx_to_prompt = defaultdict(list)
        for prompt in all_prompts:
            row_idx_to_prompt[prompt['row_idx']].append(prompt)

        generated_candidates = []
        # gather texts for analyze
        for i in tqdm(range(len(test_dataset)), desc="Extracting candidates"):
            # gather completions whose row_idx is i
            row_prompts = row_idx_to_prompt[i]
            pii_keys = set([prompt['pii_key'] for prompt in row_prompts])
            for pii_key in pii_keys:
                pii_type = pii_key.split('-')[0]
                presidio_entity = pii_type_to_presidio_entity_mapping[pii_type] if pii_type_to_presidio_entity_mapping else pii_type
                completions = [prompt['completion'] for prompt in row_prompts if prompt['pii_key'] == pii_key]
                generated_texts = [completion.outputs[0].text for completion in completions]
                analyze_results = [prompt['analyze_result'] for prompt in row_prompts if prompt['pii_key'] == pii_key]
                data = {
                    "row_idx": i,
                    "pii_key": pii_key,
                    "candidates": [],
                    "raw_candidates": [],
                }
                # analyze_results = self.batch_analyzer.analyze_iterator(
                #     texts=generated_texts,
                #     language='en',
                #     entities=[presidio_entity],
                #     batch_size=256,
                # )
                for idx, analyze_result in enumerate(analyze_results):
                    current_candidates = [generated_texts[idx][entity.start:entity.end] for entity in analyze_result if entity.entity_type == presidio_entity]
                    current_candidates = [candidate for candidate in current_candidates if is_legal_value(candidate)]
                    data['candidates'].extend(current_candidates)
                    data['raw_candidates'].append(current_candidates)
                
                data['candidates'] = list(set(data['candidates']))
                generated_candidates.append(data)

        return Dataset.from_list(generated_candidates)
