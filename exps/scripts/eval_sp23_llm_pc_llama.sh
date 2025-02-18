MODEL=llama3.1-8b
DATASET=llm_pc
DATASET_SPLIT=development

TEMPLATE=llama

python exps/eval_topn.py \
--dataset_name ${DATASET} \
--dataset_split ${DATASET_SPLIT} \
--selected_data selected_candidates/sp23-o/${MODEL}_${DATASET} \
--new_chat_template ${TEMPLATE}