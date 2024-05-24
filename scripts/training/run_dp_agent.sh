

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=TranDPagent \
experiment_name=training_transformer_dp_agent \
scene_filter=all_scenes \
split=mini \
use_cache_without_dataset=true \
cache_path=/cephfs/zhanjh/exp/train_tmp_cache \
