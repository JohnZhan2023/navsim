SPLIT=test
CHECKPOINT="/cephfs/zhanjh/exp/training_res_str_agent/RESstrSota/lightning_logs/version_0/checkpoints/res_str.ckpt"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
agent=RESSTRagent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=RESSTRagent \
split=$SPLIT \
scene_filter=all_scenes \
