


REM test single sweep
wandb sweep scripts/sweep/sweep_config_lg.yaml

wandb sweep scripts/sweep/sweep_config_gb.yaml

wandb sweep scripts/sweep/sweep_config_gru.yaml

REM run multiple wandb agents in parallel
call scripts\sweep\parallel_sweep.bat 8 szwjf08-utmb/rat-model-gb/qtta5o13


REM kill parallel wandb agents
wmic process where "CommandLine like '%wandb agent%'" delete
taskkill /F /FI "WINDOWTITLE eq wandb_agent_*"
