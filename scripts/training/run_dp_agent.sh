

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=TranDPagent \
experiment_name=training_transformer_dp_agent \
scene_filter=all_scenes \
split=trainval \
use_cache_without_dataset=True \
cache_path=/cephfs/zhanjh/exp/training_cache_trainval \
