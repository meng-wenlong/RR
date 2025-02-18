MODEL=llama3.1-8b
DATASET=echr
CHECKPOINT=checkpoint-375
DATASET_SPLIT=train
ITER=1

python exps/Gsq1.py \
--model_name_or_path llm_ft/outputs/${MODEL}-${DATASET}/${CHECKPOINT} \
--dataset_name ${DATASET} \
--dataset_split ${DATASET_SPLIT} \
--iter_num ${ITER} \
--generated_candidates_path generated_candidates/sq1/${MODEL}_${DATASET}