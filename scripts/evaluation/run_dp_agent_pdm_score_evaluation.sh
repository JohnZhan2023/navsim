SPLIT=mini
CHECKPOINT="/cephfs/zhanjh/exp/training_transformer_dp_agent/Newest_dp/lightning_logs/version_0/checkpoints/200transformer_dp.ckpt"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
agent=TranDPagent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=TranDPagent \
split=$SPLIT \
scene_filter=warmup_test_e2e \
