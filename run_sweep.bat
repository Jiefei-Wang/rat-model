

REM Logistic Regression sweep
set PYTHONPATH=%CD%
python scripts\sweep_params\sweep_params_lg.py
wandb agent szwjf08-utmb/rat-model-lg/q2bfohl8

python scripts/sweep_params/parallel_sweep.py --num-proc 12 --agent-path szwjf08-utmb/rat-model-lg/q2bfohl8

REM https://wandb.ai/szwjf08-utmb/rat-model-lg/sweeps/obw5t7md?nw=nwuserszwjf08


REM GB
set PYTHONPATH=%CD%
python scripts\sweep_params\sweep_params_gb.py
wandb agent szwjf08-utmb/rat-model-gb/j2ui9jfj

python scripts/sweep_params/parallel_sweep.py --num-proc 12 --agent-path szwjf08-utmb/rat-model-gb/j2ui9jfj





REM GRU
set PYTHONPATH=%CD%
python scripts\sweep_params\sweep_params_gru.py
wandb agent szwjf08-utmb/rat-model-gru/ddkuefen

python scripts/sweep_params/parallel_sweep.py --num-proc 12 --agent-path szwjf08-utmb/rat-model-gru/ddkuefen