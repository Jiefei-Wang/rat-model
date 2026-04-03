


REM test single sweep
set PYTHONPATH=%CD%
python scripts\sweep_params\sweep_params_peak.py
wandb agent szwjf08-utmb/rat-model-lg/7hd2ctwm


REM run multiple wandb agents in parallel
python scripts/sweep_params/parallel_sweep.py --num-proc 12 --agent-path szwjf08-utmb/rat-model-peaks/7hd2ctwm




REM Logistic Regression sweep
set PYTHONPATH=%CD%
python scripts\sweep_params\sweep_params_lg.py
wandb agent szwjf08-utmb/rat-model-lg/oykt6i8g

REM run multiple wandb agents in parallel
set PYTHONPATH=%CD%
python scripts/sweep_params/parallel_sweep.py --num-proc 12 --agent-path szwjf08-utmb/rat-model-lg/oykt6i8g




REM GRU
set PYTHONPATH=%CD%
python scripts\sweep_params\sweep_params_gru.py
wandb agent szwjf08-utmb/rat-model-gru/l95e7qt4


REM kill parallel wandb agents
wmic process where "CommandLine like '%wandb agent%'" delete
taskkill /F /FI "WINDOWTITLE eq wandb_agent_*"

