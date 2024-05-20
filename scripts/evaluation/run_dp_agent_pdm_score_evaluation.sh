SPLIT=private_test_e2e
CHECKPOINT="/cephfs/zhanjh/exp/training_transformer_dp_agent/2024.05.18.14.37.25/lightning_logs/version_0/checkpoints/50epoch.ckpt"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
agent=TranDPagent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=TranDPagent \
split=$SPLIT \
scene_filter=private_test_e2e \
