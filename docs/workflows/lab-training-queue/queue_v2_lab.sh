#!/bin/bash
# Lab computer (b2-unit-366, RTX 3080 Ti): handover2-trimmed-v2 REACT runs one after another with the same recipe as
# server6's train_react.sh, in the lerobot-wwq env. A run whose log says it finished is skipped.
# Jobs: react:<horizon>:<blocks>. No tmux here, so run it detached:
#   setsid nohup bash ~/wwq/queue_v2_lab.sh react:32:8 react:80:10 >> ~/wwq/logs/queue_v2.log 2>&1 < /dev/null &
W=$HOME/wwq
mkdir -p $W/logs $W/outputs
WANDB_API_KEY=$(head -n1 $W/.wandb_api_key | tr -d '[:space:]')
export WANDB_API_KEY HF_HUB_OFFLINE=1
cd $W/lerobot
for job in "$@"; do
  IFS=: read -r _ H K <<< "$job"
  S=$((H / K)) && name=d2t_react_h${H}_k${K}_s${S}_v2 && log=$W/logs/$name.log
  grep -q "End of training" $log 2>/dev/null && continue
  echo "$(date '+%F %T') start $name"
  ~/miniconda3/envs/lerobot-wwq/bin/python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=wwqcleverghost/handover2-trimmed-v2 --dataset.root=$W/data/handover2-trimmed-v2 \
    --dataset.image_transforms.enable=true --dataset.image_transforms.on_gpu=true \
    --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
    --policy.resize_shape=[240,320] --policy.crop_shape=[216,288] --policy.use_amp=true \
    --policy.horizon=$H --policy.n_action_steps=$S --policy.react_blocks=$K \
    --batch_size=24 --steps=100000 --save_freq=20000 --ema_decay=0.9999 --num_workers=8 \
    --output_dir=$W/outputs/$name --job_name=$name \
    --wandb.enable=true --wandb.entity=wwqcleverghost --wandb.project=handover --wandb.disable_artifact=true \
    > $log 2>&1 && echo "$(date '+%F %T') done $name" || echo "$(date '+%F %T') FAILED $name, see $log"
done
