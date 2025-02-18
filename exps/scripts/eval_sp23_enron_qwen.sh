MODEL=qwen2.5-7b
DATASET=enron
DATASET_SPLIT=train

TEMPLATE=qwen

python exps/eval_topn.py \
--dataset_name ${DATASET} \
--dataset_split ${DATASET_SPLIT} \
--selected_data selected_candidates/sp23-o/${MODEL}_${DATASET} \
--new_chat_template ${TEMPLATE}