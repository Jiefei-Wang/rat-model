REM peak parameter
set PYTHONPATH=%CD%
python scripts/sweep_params/sweep_params_peak.py

python scripts/sweep_params/parallel_sweep.py --num-proc 12 --agent-path szwjf08-utmb/rat-model-peaks/dgrdvshl


REM Logistic Regression sweep
set PYTHONPATH=%CD%
python scripts\sweep_params\sweep_params_lg.py
wandb agent szwjf08-utmb/rat-model-lg/9vjahnk9

python scripts/sweep_params/parallel_sweep.py --num-proc 12 --agent-path szwjf08-utmb/rat-model-lg/9vjahnk9

REM https://wandb.ai/szwjf08-utmb/rat-model-lg/sweeps/9vjahnk9?nw=nwuserszwjf08


REM GB
set PYTHONPATH=%CD%
python scripts\sweep_params\sweep_params_gb.py
wandb agent szwjf08-utmb/rat-model-gb/jdjtzr4w

python scripts/sweep_params/parallel_sweep.py --num-proc 12 --agent-path szwjf08-utmb/rat-model-gb/jdjtzr4w



REM GRU
set PYTHONPATH=%CD%
python scripts\sweep_params\sweep_params_gru.py
wandb agent szwjf08-utmb/rat-model-gru/ddkuefen

python scripts/sweep_params/parallel_sweep.py --num-proc 12 --agent-path szwjf08-utmb/rat-model-gru/ddkuefen



