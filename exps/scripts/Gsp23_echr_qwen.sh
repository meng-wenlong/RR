MODEL=qwen2.5-7b
DATASET=echr
CHECKPOINT=checkpoint-375
DATASET_SPLIT=train
ITER=64

python exps/Gtab_analyze.py \
--model_name_or_path llm_ft/outputs/${MODEL}-${DATASET}/${CHECKPOINT} \
--dataset_name ${DATASET} \
--dataset_split ${DATASET_SPLIT} \
--iter_num ${ITER} \
--generated_candidates_path generated_candidates/sp23-o/${MODEL}_${DATASET} \
