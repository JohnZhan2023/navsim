CHECKPOINT="/cephfs/zhanjh/exp/training_oneframe_str_agent/2024.05.27.16.58.54/lightning_logs/temp45.ckpt"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=Tran_1StrMoeAgent \
experiment_name=training_oneframe_str_moe_agent \
scene_filter=all_scenes \
split=trainval \
use_cache_without_dataset=true \
cache_path=/cephfs/zhanjh/exp/training_cache_trainval \
