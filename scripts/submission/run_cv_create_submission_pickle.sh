TEAM_NAME="MARS"
AUTHORS="JiahaoZhan"
EMAIL="22307140116@m.fudan.edu.cn"
INSTITUTION="MarsLab"
COUNTRY="China"
CHECKPOINT="/cephfs/zhanjh/exp/training_oneframe_str_moe_agent/2024.05.28.13.04.49/checkpoints/121moe.ckpt"
#/cephfs/zhanjh/exp/training_transfuser_agent/2024.05.20.11.11.33/lightning_logs/version_0/checkpoints/transfuser300.ckpt
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle.py \
agent=Tran_1StrMoeAgent \
agent.checkpoint_path=$CHECKPOINT \
split=private_test_e2e \
experiment_name=submission_Tran_1StrMoeAgent_agent \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \
