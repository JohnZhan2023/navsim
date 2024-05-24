SPLIT=mini

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
split=$SPLIT \
cache_path=/cephfs/zhanjh/exp/training_cache_mini \
agent=DPagent \
experiment_name=training_dp_agent \
force_cache_computation=False \
