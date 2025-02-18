MODEL=gemma2-9b
DATASET=enron
CHECKPOINT=checkpoint-750
DATASET_SPLIT=train
ITER=50
MAX_TOKENS=512

TEMPLATE=gemma
python exps/eval_topn.py \
--dataset_name ${DATASET} \
--dataset_split ${DATASET_SPLIT} \
--selected_data selected_candidates/loss/${MODEL}_${DATASET} \
--new_chat_template ${TEMPLATE}