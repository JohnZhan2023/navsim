SPLIT=mini
CHECKPOINT="/cephfs/zhanjh/exp/training_oneframe_str_moe_agent/2024.05.28.22.24.14/37largerMoe.ckpt"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
agent=Tran_1StrMoeAgent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=Tran_1StrMoeAgent \
split=$SPLIT \
scene_filter=warmup_test_e2e \
