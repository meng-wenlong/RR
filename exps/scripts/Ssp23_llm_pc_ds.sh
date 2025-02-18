MODEL=deepseek-llama3.1-8b
DATASET=llm_pc
CHECKPOINT=checkpoint-564
DATASET_SPLIT=development

python exps/Sloss.py \
--model_name_or_path llm_ft/outputs/${MODEL}-${DATASET}/${CHECKPOINT} \
--dataset_name ${DATASET} \
--dataset_split ${DATASET_SPLIT} \
--generated_candidates_path generated_candidates/sp23-o/${MODEL}_${DATASET} \
--save_path selected_candidates/sp23-o/${MODEL}_${DATASET} \
--processed_inter_dataset_path processed_inter_datasets/sp23-o/${MODEL}_${DATASET} \
--ignore_pre_pii False