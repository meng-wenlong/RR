MODEL=gemma2-9b
DATASET=enron
CHECKPOINT=checkpoint-750
DATASET_SPLIT=train
ITER=50
MAX_TOKENS=512

REFER_MODEL=google/gemma-2-9b-it

for REFER_BIAS in $(seq 0.1 0.1 4.0)
do
  python exps/Sloss.py \
    --model_name_or_path llm_ft/outputs/${MODEL}-${DATASET}/${CHECKPOINT} \
    --dataset_name ${DATASET} \
    --dataset_split ${DATASET_SPLIT} \
    --refer_model_name_or_path ${REFER_MODEL} \
    --generated_candidates_path generated_candidates/recollect-40/${MODEL}_${DATASET} \
    --save_path selected_candidates/loss/${MODEL}_${DATASET}_b${REFER_BIAS} \
    --processed_inter_dataset_path processed_inter_datasets/${MODEL}_${DATASET} \
    --refer_inter_dataset_path refer_inter_datasets/${MODEL}_${DATASET} \
    --refer_bias ${REFER_BIAS}
done