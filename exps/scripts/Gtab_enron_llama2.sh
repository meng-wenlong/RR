MODEL=llama3.2-3b
DATASET=enron
CHECKPOINT=checkpoint-375
DATASET_SPLIT=train
ITER=1

python exps/Gtab.py \
--model_name_or_path llm_ft/outputs/${MODEL}-${DATASET}/${CHECKPOINT} \
--dataset_name ${DATASET} \
--dataset_split ${DATASET_SPLIT} \
--iter_num ${ITER} \
--generated_candidates_path generated_candidates/tab/${MODEL}_${DATASET}