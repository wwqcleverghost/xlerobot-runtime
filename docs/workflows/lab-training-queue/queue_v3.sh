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
#   flow:v6              baseline on v6-handover-curated (58 graded episodes, 12-dim), 140K steps
#   flow:v6ft            fine-tunes the finished v6_flow_h16_s8 on v6-corrections (20 short recovery segments):
#                        8K steps, lr 2e-5, padded actions masked; waits until v6_flow_h16_s8 has finished.
#                        Data dir v6-corrections-v6stats = v6-corrections with v6-handover-curated's meta/stats.json,
#                        so the normalization stays the one the base model was trained with.
#   fb:<base>            feedback denoising (arXiv 2609.21022, --policy.feedback) on top of the finished flow run
#                        <base>, on that run's own dataset: <base>_fb. The planner is frozen and only the small
#                        module trains: 40K steps, batch 12, 6 workers by default, since it loads 9 frames per
#                        camera instead of 2 (about 1.2 GB per batch). Waits until <base> has finished.
#   fbstale:<base>       control for fb: the same module, but it only ever sees the planner's own frame
#   fbstate:<base>       fb that also sees the joint state
#   fb:<base>@<step>     (also fbstale/fbstate) uses <base>'s checkpoint at <step> as the planner instead of its
#                        final one and starts as soon as that checkpoint is saved: <base>_fb_p<step/1000>k
#   flow:v7              baseline on v7-handover-clean (53 episodes, 17-dim), 140K steps, named v7-flow_h16_s8;
#                        when it has finished only the last three checkpoints are kept, all of them complete
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
FLOW=(--policy.type=diffusion --policy.resize_shape=[240,320] --policy.crop_shape=[216,288] --policy.use_amp=true
  --policy.horizon=16 --policy.n_action_steps=8 --policy.noise_scheduler_type=FLOW --policy.num_inference_steps=10)
mkdir -p $W/logs $W/outputs
cd $CODE || exit 1
for job in "$@"; do
  data=handover-perfect steps=100000 save=20000 base= ck=last keep= root= bs=24 workers=${NUM_WORKERS:-12}
  case $job in
    flow) name=dp_flow_h16_s8 policy=("${FLOW[@]}") ;;
    flow:strict) name=dp_flow_h16_s8_strict data=handover-perfect-strict policy=("${FLOW[@]}") ;;
    flow:sepenc) name=dp_flow_h16_s8_sepenc policy=("${FLOW[@]}" --policy.use_separate_rgb_encoder_per_camera=true) ;;
    flow:v6) name=v6_flow_h16_s8 data=v6-handover-curated steps=140000 policy=("${FLOW[@]}") ;;
    flow:v7) name=v7-flow_h16_s8 data=v7-handover-clean steps=140000 keep=3 policy=("${FLOW[@]}") ;;
    flow:v6ft) name=v6ft_corrections_flow_h16_s8 data=v6-corrections root=$W/data/v6-corrections-v6stats steps=8000 save=2000
      base=v6_flow_h16_s8 policy=(--policy.path=$W/outputs/v6_flow_h16_s8/checkpoints/last/pretrained_model
      --policy.optimizer_lr=2e-5 --policy.do_mask_loss_for_padding=true) ;;
    flow:noshift) name=dp_flow_h16_s8_noshift policy=("${FLOW[@]}" --policy.crop_is_random=false "--dataset.image_transforms.tfs=$COLOR") ;;
    fb:*|fbstale:*|fbstate:*) base=${job#*:} name=${base%@*}_${job%%:*}
      if [[ $base == *@* ]]; then ck=$(printf %06d ${base#*@}) name+=_p$((${base#*@} / 1000))k base=${base%@*}; fi
      steps=40000 save=10000 bs=12 workers=${NUM_WORKERS:-6}
      policy=(--policy.path=$W/outputs/$base/checkpoints/$ck/pretrained_model --policy.feedback=true)
      [ ${job%%:*} = fbstale ] && policy+=(--policy.feedback_every=8)  # n_action_steps of the FLOW recipe
      [ ${job%%:*} = fbstate ] && policy+=(--policy.feedback_state=true) ;;
    *) echo "$(date '+%F %T') unknown job $job"; continue ;;
  esac
  log=$W/logs/$name.log
  grep -q "End of training" $log 2>/dev/null && continue
  # A fine-tune starts from the finished base run: wait for it while it is still training.
  # ponytail: only checks for a running trainer, a base queued later in another queue is not waited for.
  # With a fixed planner step, wait until the last published checkpoint has reached it.
  ready() {
    if [ $ck = last ]; then grep -q "End of training" $W/logs/$base.log 2>/dev/null
    else l=$(readlink $W/outputs/$base/checkpoints/last); [[ -n $l && ! ${l##*/} < $ck ]]; fi
  }
  while [ -n "$base" ] && [ -z "$DRY" ] && ! ready; do
    pgrep -f "lerobot_train.*(job_name=$base |config_path=$W/outputs/$base/)" >/dev/null || { echo "$(date '+%F %T') $base has not finished, skip $name"; continue 2; }
    sleep 300
  done
  if [[ $job == fb* ]]; then  # same dataset as the base run, so its normalization stats still apply
    bcfg=$W/outputs/$base/checkpoints/$ck/pretrained_model/train_config.json
    [ -f $bcfg ] || { echo "$(date '+%F %T') no $bcfg, skip $name"; continue; }
    data=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["dataset"]["repo_id"].split("/")[1])' $bcfg)
  fi
  cmd=($PY -m lerobot.scripts.lerobot_train
    --dataset.repo_id=wwqcleverghost/$data --dataset.root=${root:-$W/data/$data}
    --dataset.image_transforms.enable=true --dataset.image_transforms.on_gpu=true
    --policy.device=cuda --policy.push_to_hub=false "${policy[@]}"
    --batch_size=$bs --steps=$steps --save_freq=$save --ema_decay=0.9999 --num_workers=$workers
    --output_dir=$W/outputs/$name --job_name=$name
    --wandb.enable=true --wandb.entity=wwqcleverghost --wandb.project=handover --wandb.disable_artifact=true)
  last=$W/outputs/$name/checkpoints/last/pretrained_model/train_config.json
  if [ -f "$last" ]; then
    cmd=($PY -m lerobot.scripts.lerobot_train --resume=true --config_path=$last --num_workers=$workers)
  fi
  if [ -n "$DRY" ]; then echo "$name: ${cmd[*]}"; continue; fi
  echo "$(date '+%F %T') start $name"
  echo "=== $(date "+%F %T") start $name" >> $log
  # server4 crashes DataLoader workers now and then (suspected bad RAM): resume from the last checkpoint, or
  # restart from scratch when there is none yet, at most 5 times.
  for try in 0 1 2 3 4 5; do
    if "${cmd[@]}" >> $log 2>&1; then
      echo "$(date '+%F %T') done $name"
      [ -n "$keep" ] && ls -d $W/outputs/$name/checkpoints/[0-9]* | head -n -$keep | xargs -r rm -rf
      break
    fi
    [ $try = 5 ] && { echo "$(date '+%F %T') FAILED $name, see $log"; break; }
    last=$W/outputs/$name/checkpoints/last/pretrained_model/train_config.json
    if [ -f $last ]; then
      cmd=($PY -m lerobot.scripts.lerobot_train --resume=true --config_path=$last --num_workers=$workers)
      echo "$(date '+%F %T') $name crashed, resuming from $(readlink $W/outputs/$name/checkpoints/last)"
    else
      mv $W/outputs/$name $W/outputs/$name.crashed_$try 2>/dev/null
      echo "$(date '+%F %T') $name crashed before its first checkpoint, restarting"
    fi
  done
done
