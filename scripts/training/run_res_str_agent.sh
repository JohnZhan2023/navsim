

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=RESSTRagent \
experiment_name=training_res_str_agent \
scene_filter=all_scenes \
split=trainval \
use_cache_without_dataset=true \
cache_path=/cephfs/zhanjh/exp/training_cache_trainval \
