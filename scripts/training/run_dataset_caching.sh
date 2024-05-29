SPLIT=trainval

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
split=$SPLIT \
cache_path=/cephfs/zhanjh/exp/training_cache_trainval_single \
agent=Tran_1StrMoeAgent \
experiment_name=Tran_1StrMoeAgent_cache \
force_cache_computation=False \
