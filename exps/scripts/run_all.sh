MODEL=phi
DATASET=echr

bash exps/scripts/Grc_${DATASET}_${MODEL}.sh > Logs_wln/gsq_${DATASET}_${MODEL}.log 2>&1 
bash exps/scripts/Sloss_${DATASET}_${MODEL}.sh > Logs_wln/sloss_${DATASET}_${MODEL}.log 2>&1 
bash exps/scripts/eval_topn_${DATASET}_${MODEL}.sh > Logs_wln/eval_topn_${DATASET}_${MODEL}.log 2>&1 

#google/gemma-2-9b-it
#Qwen/Qwen2.5-7B-Instruct
#microsoft/Phi-3.5-mini-instruct