SPLIT=mini
CHECKPOINT="/cephfs/zhanjh/exp/training_oneframe_str_agent/2024.05.27.22.11.26/lightning_logs/version_0/checkpoints/oneframeSTR.ckpt"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
agent=Tran_1StrAgent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=Tran_1StrAgent \
split=$SPLIT \
scene_filter=warmup_test_e2e \
