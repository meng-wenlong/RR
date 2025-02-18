MODEL=phi3.5-mini
DATASET=enron
CHECKPOINT=checkpoint-375
DATASET_SPLIT=train
ITER=50
MAX_TOKENS=512

REFER_MODEL=microsoft/Phi-3.5-mini-instruct

python exps/Sloss.py \
--model_name_or_path llm_ft/outputs/${MODEL}-${DATASET}/${CHECKPOINT} \
--dataset_name ${DATASET} \
--dataset_split ${DATASET_SPLIT} \
--refer_model_name_or_path ${REFER_MODEL} \
--generated_candidates_path generated_candidates/recollect/${MODEL}_${DATASET} \
--save_path selected_candidates/loss/${MODEL}_${DATASET} \
--processed_inter_dataset_path processed_inter_datasets/${MODEL}_${DATASET} \
--refer_inter_dataset_path refer_inter_datasets/${MODEL}_${DATASET}