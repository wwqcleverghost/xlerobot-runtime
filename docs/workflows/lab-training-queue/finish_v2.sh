#!/bin/bash
# Finish the d2t_*_v2 REACT runs on handover2-trimmed-v2 (50 episodes):
# resume from the last checkpoint, or start from scratch if there is none; retry up to
# 5 times (server4 crashes DataLoader workers now and then). server6 / server4 layout (~/wwq).
# Usage: setsid -f bash ~/wwq/finish_v2.sh d2t_react_h32_k4_s8_v2 d2t_react_h64_k8_s8_v2 >> ~/wwq/logs/finish_v2.log 2>&1 < /dev/null
W=$HOME/wwq DATA_ROOT=$HOME/wwq/data/handover2-trimmed-v2
source $W/secrets.env
: "${WANDB_API_KEY:?no wandb key}"
export WANDB_API_KEY HF_HOME=$W/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0
for name in "$@"; do
  case $name in d2t_*) ;; *) name=d2t_$name ;; esac
  log=$W/logs/$name.log
  grep -q "End of training" $log 2>/dev/null && continue
  IFS=_ read -r _ h k _ <<< "${name#d2t_}"
  for try in 0 1 2 3 4 5; do
    last=$W/outputs/$name/checkpoints/last/pretrained_model/train_config.json
    if [ -f $last ]; then
      echo "$(date '+%F %T') $name: resume from $(readlink $W/outputs/$name/checkpoints/last)"
      $W/envs/lerobot/bin/python -m lerobot.scripts.lerobot_train --resume=true --config_path=$last \
        --dataset.repo_id=wwqcleverghost/handover2-trimmed-v2 --dataset.root=$DATA_ROOT >> $log 2>&1 && break
    else
      [ -d $W/outputs/$name ] && mv $W/outputs/$name $W/outputs/$name.crashed_$(date +%H%M%S)
      echo "$(date '+%F %T') $name: start from scratch"
      GPU=0 bash $W/train_react.sh ${h#h} ${k#k} --dataset.repo_id=wwqcleverghost/handover2-trimmed-v2 --dataset.root=$DATA_ROOT \
        --output_dir=$W/outputs/$name --job_name=$name --wandb.project=handover >> $log 2>&1 && break
    fi
  done
  grep -q "End of training" $log && echo "$(date '+%F %T') done $name" || echo "$(date '+%F %T') FAILED $name, see $log"
done
