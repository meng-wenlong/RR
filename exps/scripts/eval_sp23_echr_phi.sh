MODEL=phi3.5-mini
DATASET=echr
DATASET_SPLIT=train

TEMPLATE=phi

python exps/eval_topn.py \
--dataset_name ${DATASET} \
--dataset_split ${DATASET_SPLIT} \
--selected_data selected_candidates/sp23-o/${MODEL}_${DATASET} \
--new_chat_template ${TEMPLATE}