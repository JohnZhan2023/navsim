SPLIT=mini

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
split=$SPLIT \
cache_path=/cephfs/zhanjh/exp/train_tmp_cache \
agent=DPagent \
experiment_name=training_dp_agent \