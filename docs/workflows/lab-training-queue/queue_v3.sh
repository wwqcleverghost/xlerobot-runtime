#!/bin/bash
# Train handover-perfect runs one after another on this machine's GPU 0: server6, server4 (~/wwq/envs/lerobot,
# key in ~/wwq/secrets.env) or the lab computer b2-unit-366 (lerobot-wwq env, key in ~/wwq/.wandb_api_key).
# Same recipe as train_flow.sh / train_react.sh. A run whose log says it finished is skipped, so the queue can
# simply be restarted. DRY=1 only prints the commands.
# Jobs:
#   flow                 flow matching, horizon 16, 8 steps per chunk (the baseline)
#   flow:strict          baseline on handover-perfect-strict (64 cleanest episodes), ablation A1
#   flow:sepenc          baseline with one image encoder per camera
#   flow:noshift         baseline without the image-moving augmentations (center crop, no affine)
# Run detached (no tmux on the lab computer):
#   setsid nohup bash ~/wwq/queue_v3.sh flow flow:sepenc >> ~/wwq/logs/queue_v3.log 2>&1 < /dev/null &
W=${WQ:-$HOME/wwq}  # server1/server2 keep everything on /media/data/wwq: WQ=/media/data/wwq
if [ -x $W/envs/lerobot/bin/python ]; then
  PY=$W/envs/lerobot/bin/python CODE=$W/xlerobot-runtime
  export HF_HOME=$W/hf PYTHONNOUSERSITE=1
  [ -f $W/secrets.env ] && source $W/secrets.env
else
  PY=$HOME/miniconda3/envs/lerobot-wwq/bin/python CODE=$W/lerobot
  unset PYTHONNOUSERSITE  # This lab env uses its owner's torch installation in ~/.local.
  WANDB_API_KEY=$(head -n1 $W/.wandb_api_key 2>/dev/null | tr -d '[:space:]')
fi
# Shared accounts: never fall back to a ~/.netrc wandb login that may belong to someone else.
[ -n "$WANDB_API_KEY" ] || [ -n "$DRY" ] || { echo "no wandb key on this machine"; exit 1; }
# PYTHONNOUSERSITE: shared accounts keep other people's packages in ~/.local (server1, server4), which would
# otherwise come before the env's own site-packages.
export WANDB_API_KEY HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-0}
# The default color jitters without "affine" (shift +-5%, rotate +-5 deg), for flow:noshift.
COLOR='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}}}'
FLOW=(--policy.horizon=16 --policy.n_action_steps=8 --policy.noise_scheduler_type=FLOW --policy.num_inference_steps=10)
mkdir -p $W/logs $W/outputs
cd $CODE || exit 1
for job in "$@"; do
  data=handover-perfect
  case $job in
    flow) name=dp_flow_h16_s8 policy=("${FLOW[@]}") ;;
    flow:strict) name=dp_flow_h16_s8_strict data=handover-perfect-strict policy=("${FLOW[@]}") ;;
    flow:sepenc) name=dp_flow_h16_s8_sepenc policy=("${FLOW[@]}" --policy.use_separate_rgb_encoder_per_camera=true) ;;
    flow:noshift) name=dp_flow_h16_s8_noshift policy=("${FLOW[@]}" --policy.crop_is_random=false "--dataset.image_transforms.tfs=$COLOR") ;;
    *) echo "$(date '+%F %T') unknown job $job"; continue ;;
  esac
  log=$W/logs/$name.log
  grep -q "End of training" $log 2>/dev/null && continue
  cmd=($PY -m lerobot.scripts.lerobot_train
    --dataset.repo_id=wwqcleverghost/$data --dataset.root=$W/data/$data
    --dataset.image_transforms.enable=true --dataset.image_transforms.on_gpu=true
    --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false
    --policy.resize_shape=[240,320] --policy.crop_shape=[216,288] --policy.use_amp=true "${policy[@]}"
    --batch_size=24 --steps=100000 --save_freq=20000 --ema_decay=0.9999 --num_workers=${NUM_WORKERS:-12}
    --output_dir=$W/outputs/$name --job_name=$name
    --wandb.enable=true --wandb.entity=wwqcleverghost --wandb.project=handover --wandb.disable_artifact=true)
  last=$W/outputs/$name/checkpoints/last/pretrained_model/train_config.json
  if [ -f "$last" ]; then
    cmd=($PY -m lerobot.scripts.lerobot_train --resume=true --config_path=$last)
  fi
  if [ -n "$DRY" ]; then echo "$name: ${cmd[*]}"; continue; fi
  echo "$(date '+%F %T') start $name"
  echo "=== $(date "+%F %T") start $name" >> $log
  # server4 crashes DataLoader workers now and then (suspected bad RAM): resume from the last checkpoint, or
  # restart from scratch when there is none yet, at most 5 times.
  for try in 0 1 2 3 4 5; do
    "${cmd[@]}" >> $log 2>&1 && { echo "$(date '+%F %T') done $name"; break; }
    [ $try = 5 ] && { echo "$(date '+%F %T') FAILED $name, see $log"; break; }
    last=$W/outputs/$name/checkpoints/last/pretrained_model/train_config.json
    if [ -f $last ]; then
      cmd=($PY -m lerobot.scripts.lerobot_train --resume=true --config_path=$last)
      echo "$(date '+%F %T') $name crashed, resuming from $(readlink $W/outputs/$name/checkpoints/last)"
    else
      mv $W/outputs/$name $W/outputs/$name.crashed_$try 2>/dev/null
      echo "$(date '+%F %T') $name crashed before its first checkpoint, restarting"
    fi
  done
done
