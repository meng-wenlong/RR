MODEL=deepseek-llama3.1-8b
DATASET=echr
CHECKPOINT=checkpoint-375
DATASET_SPLIT=train
ITER=50
MAX_TOKENS=381

python exps/Grc.py \
--model_name_or_path llm_ft/outputs/${MODEL}-${DATASET}/${CHECKPOINT} \
--dataset_name ${DATASET} \
--dataset_split ${DATASET_SPLIT} \
--iter_num ${ITER} \
--max_tokens ${MAX_TOKENS} \
--generated_candidates_path generated_candidates/recollect/${MODEL}_${DATASET} \
