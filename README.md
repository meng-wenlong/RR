### <div align="center">R.R.: Unveiling LLM Training Privacy through Recollection and Ranking<div> 

## Abstract
Large Language Models (LLMs) pose significant privacy risks, potentially leaking training data due to implicit memorization. Existing privacy attacks primarily focus on membership inference attacks (MIAs) or data extraction attacks, but reconstructing specific personally identifiable information (PII) in LLM's training data remains challenging. In this paper, we propose R.R. (Recollect and Rank), a novel two-step privacy stealing attack that enables attackers to reconstruct PII entities from scrubbed training data where the PII entities have been masked. In the first stage, we introduce a prompt paradigm named recollection, which instructs the LLM to repeat a masked text but fill in masks. Then we can use PII identifiers to extract recollected PII candidates. In the second stage, we design a new criterion to score each PII candidate and rank them. Motivated by membership inference, we leverage the reference model as a calibration to our criterion. Experiments across three popular PII datasets demonstrate that the R.R. achieves better PII identical performance compared to baselines. These results highlight the vulnerability of LLMs to PII leakage even when training data has been scrubbed.


##  Overview
### Pipeline
<p align="center">
<img src="Images/overview.png">
</p>

The pipeline of R.R. is illustrated above. R.R. has two steps: candidate generation and selection. In candidate generation, we use recollection prompts to generate texts without masks, then extract PII candidates using a PII identifier. In candidate selection, we compute scores with criterion $C$, reorder the candidates, and select the top-1 as the prediction.

### Results

The top-1 PII prediction accuracy is as follows.

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
</head>
<body>

<table>
  <thead>
    <tr>
      <th rowspan="2">Stealer</th>
      <th colspan="3">Llama3.1-8B</th>
      <th colspan="3">Llama3.2-3B</th>
      <th colspan="3">Qwen2.5-7B</th>
      <th colspan="3">Phi3.5-Mini</th>
    </tr>
    <tr>
      <th>ECHR</th><th>Enron</th><th>LLM-PC</th>
      <th>ECHR</th><th>Enron</th><th>LLM-PC</th>
      <th>ECHR</th><th>Enron</th><th>LLM-PC</th>
      <th>ECHR</th><th>Enron</th><th>LLM-PC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DirectPrompt</td>
      <td>6.07</td><td>2.55</td><td>10.33</td>
      <td>3.56</td><td>2.11</td><td>10.73</td>
      <td>3.24</td><td>2.09</td><td>12.58</td>
      <td>2.11</td><td>0.62</td><td>10.95</td>
    </tr>
    <tr>
      <td>TAB</td>
      <td>13.51</td><td>19.00</td><td>8.30</td>
      <td>7.20</td><td>9.12</td><td>6.77</td>
      <td>8.61</td><td>13.31</td><td>6.95</td>
      <td>4.09</td><td>5.62</td><td>4.86</td>
    </tr>
    <tr>
      <td>P2P</td>
      <td>13.19</td><td>19.14</td><td>11.68</td>
      <td>6.91</td><td>8.38</td><td>8.65</td>
      <td>8.99</td><td>13.50</td><td>10.31</td>
      <td>4.28</td><td>5.74</td><td>7.41</td>
    </tr>
    <tr>
      <td>R.R.</td>
      <td><strong>25.68</strong></td><td><strong>33.31</strong></td><td><strong>28.93</strong></td>
      <td><strong>14.79</strong></td><td><strong>20.61</strong></td><td><strong>26.48</strong></td>
      <td><strong>16.35</strong></td><td><strong>25.38</strong></td><td><strong>26.41</strong></td>
      <td><strong>11.10</strong></td><td><strong>16.71</strong></td><td><strong>22.13</strong></td>
    </tr>
  </tbody>
</table>

</body>
</html>



## Get Start

### Set environment
```bash
conda create -n pii python=3.10
cd RR
chmod a+x install.sh
./install.sh
```

### Download raw data

Please download raw datas from this [link](https://drive.google.com/drive/folders/1ANd0aHo_f3gqURHD3ZTjAEEfNjsl6muG?usp=sharing).

For data xxx, download the corresponding `raw/` folder and copy it to `src/pii/datas/xxx/`.

### Fine-tune an LLM

```bash
cd llm_ft/data_prepare
python echr_language_modeling.py
cd ..
accelerate launch scripts/run_sft.py --config_file recipes/sft/echr_llama3.1-8b_config.yaml
```

### Candidate generation

```bash
cd RR

MODEL=llama3.1-8b
DATASET=echr
CHECKPOINT=checkpoint-375
DATASET_SPLIT=train
ITER=40
MAX_TOKENS=381

python exps/Grc.py \
--model_name_or_path llm_ft/outputs/${MODEL}-${DATASET}/${CHECKPOINT} \
--dataset_name ${DATASET} \
--dataset_split ${DATASET_SPLIT} \
--iter_num ${ITER} \
--max_tokens ${MAX_TOKENS} \
--generated_candidates_path generated_candidates/recollect/${MODEL}_${DATASET}
```

### Candidate selection

```bash
cd RR

MODEL=llama3.1-8b
DATASET=echr
CHECKPOINT=checkpoint-375
DATASET_SPLIT=train

REFER_MODEL=meta-llama/Llama-3.1-8B-Instruct

python exps/Sloss.py \
--model_name_or_path llm_ft/outputs/${MODEL}-${DATASET}/${CHECKPOINT} \
--dataset_name ${DATASET} \
--dataset_split ${DATASET_SPLIT} \
--refer_model_name_or_path ${REFER_MODEL} \
--generated_candidates_path generated_candidates/recollect/${MODEL}_${DATASET} \
--save_path selected_candidates/loss/${MODEL}_${DATASET} \
--processed_inter_dataset_path processed_inter_datasets/${MODEL}_${DATASET} \
--refer_inter_dataset_path refer_inter_datasets/${MODEL}_${DATASET}
```

### Evaluate accuracy

```bash
MODEL=llama3.1-8b
DATASET=echr
DATASET_SPLIT=train

TEMPLATE=llama

python exps/eval_topn.py \
--dataset_name ${DATASET} \
--dataset_split ${DATASET_SPLIT} \
--selected_data selected_candidates/loss/${MODEL}_${DATASET} \
--new_chat_template ${TEMPLATE}
```