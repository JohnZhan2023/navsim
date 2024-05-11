SPLIT=mini
CHECKPOINT='/cephfs/zhanjh/exp/training_transfuser_agent/2024.05.11.11.28.09/lightning_logs/version_0/checkpoints/tranfuser_mini.ckpt'

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
agent=transfuser_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=transfuser_agent_eval \
split=$SPLIT \
