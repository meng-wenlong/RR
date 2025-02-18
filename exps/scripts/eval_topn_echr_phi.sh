MODEL=phi3.5-mini
DATASET=echr
CHECKPOINT=checkpoint-375
DATASET_SPLIT=train
ITER=50
MAX_TOKENS=381

TEMPLATE=phi

python exps/eval_topn.py \
--dataset_name ${DATASET} \
--dataset_split ${DATASET_SPLIT} \
--selected_data selected_candidates/loss/${MODEL}_${DATASET} \
--new_chat_template ${TEMPLATE}