


REM test single sweep
set PYTHONPATH=%CD%
python scripts\sweep_params\sweep_params_peak.py
wandb agent szwjf08-utmb/rat-model-peaks/4thhqz0z

REM run multiple wandb agents in parallel
set PYTHONPATH=%CD%
call scripts/sweep_params/parallel_sweep.bat 12 szwjf08-utmb/rat-model-peaks/e6uoif2h


REM kill parallel wandb agents
wmic process where "CommandLine like '%wandb agent%'" delete
taskkill /F /FI "WINDOWTITLE eq wandb_agent_*"
