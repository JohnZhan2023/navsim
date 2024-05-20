SPLIT=private_test_e2e

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
split=$SPLIT \
cache.cache_path=/cephfs/zhanjh/exp/metric_cache \
scene_filter.frame_interval=1 \
