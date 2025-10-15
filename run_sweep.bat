


REM test single sweep
set PYTHONPATH=%CD%
python scripts\sweep_params\sweep_params_peak.py
wandb agent szwjf08-utmb/rat-model-peaks/i8ofx30o

REM run multiple wandb agents in parallel
set PYTHONPATH=%CD%
call scripts/sweep_params/parallel_sweep.bat 12 szwjf08-utmb/rat-model-peaks/n74fgezn




REM Logistic Regression sweep
set PYTHONPATH=%CD%
python scripts\sweep_params\sweep_params_lg.py
wandb agent szwjf08-utmb/rat-model-lg/7l1eacb5

REM run multiple wandb agents in parallel
set PYTHONPATH=%CD%
call scripts/sweep_params/parallel_sweep.bat 12 szwjf08-utmb/rat-model-lg/7l1eacb5




REM GRU
set PYTHONPATH=%CD%
python scripts\sweep_params\sweep_params_gru.py
wandb agent szwjf08-utmb/rat-model-gru/l95e7qt4


REM kill parallel wandb agents
wmic process where "CommandLine like '%wandb agent%'" delete
taskkill /F /FI "WINDOWTITLE eq wandb_agent_*"

