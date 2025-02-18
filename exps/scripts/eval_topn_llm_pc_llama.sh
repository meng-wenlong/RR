MODEL=llama3.1-8b
DATASET=llm_pc
CHECKPOINT=checkpoint-564
DATASET_SPLIT=development
ITER=50
MAX_TOKENS=512

TEMPLATE=llama

python exps/eval_topn.py \
--dataset_name ${DATASET} \
--dataset_split ${DATASET_SPLIT} \
--selected_data selected_candidates/loss/${MODEL}_${DATASET} \
--new_chat_template ${TEMPLATE}