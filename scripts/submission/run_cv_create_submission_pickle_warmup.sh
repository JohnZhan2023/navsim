TEAM_NAME="MARS"
AUTHORS="JiahaoZhan"
EMAIL="22307140116@m.fudan.edu.cn"
INSTITUTION="MarsLab"
COUNTRY="China"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle.py \
agent=constant_velocity_agent \
split=mini \
scene_filter=warmup_test_e2e \
experiment_name=submission_cv_agent_warmup \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \
