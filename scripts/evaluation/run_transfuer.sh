SPLIT=test
CHECKPOINT='/cephfs/zhanjh/exp/training_transfuser_agent/2024.05.20.11.11.33/lightning_logs/version_0/checkpoints/transfuser300.ckpt'

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
agent=transfuser_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=transfuser_agent_eval \
split=$SPLIT \

