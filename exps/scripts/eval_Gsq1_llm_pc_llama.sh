MODEL=llama3.1-8b
DATASET=llm_pc
DATASET_SPLIT=development

TEMPLATE=llama

python exps/eval_top-all.py \
--dataset_name ${DATASET} \
--dataset_split ${DATASET_SPLIT} \
--selected_data generated_candidates/sq1/${MODEL}_${DATASET} \
--new_chat_template ${TEMPLATE}