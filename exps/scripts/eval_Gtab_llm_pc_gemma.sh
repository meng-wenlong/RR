MODEL=gemma2-9b
DATASET=echr
DATASET_SPLIT=train

TEMPLATE=gemma

python exps/eval_top-all.py \
--dataset_name ${DATASET} \
--dataset_split ${DATASET_SPLIT} \
--selected_data generated_candidates/tab_analyze/${MODEL}_${DATASET} \
--new_chat_template ${TEMPLATE}