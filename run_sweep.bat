REM peak parameter
set PYTHONPATH=%CD%
python scripts\sweep_models\sweep_peak.py

python scripts/sweep_params/parallel_sweep.py --num-proc 12 --agent-path szwjf08-utmb/rat-model-peaks/y003h4s8


REM Logistic Regression sweep
set PYTHONPATH=%CD%
python scripts\sweep_params\sweep_params_lg.py
wandb agent szwjf08-utmb/rat-model-lg/y003h4s8

python scripts/sweep_params/parallel_sweep.py --num-proc 12 --agent-path szwjf08-utmb/rat-model-lg/y003h4s8

REM https://wandb.ai/szwjf08-utmb/rat-model-lg/sweeps/y003h4s8?nw=nwuserszwjf08


REM GB
set PYTHONPATH=%CD%
python scripts\sweep_params\sweep_params_gb.py
wandb agent szwjf08-utmb/rat-model-gb/qfoq9rc6

python scripts/sweep_params/parallel_sweep.py --num-proc 12 --agent-path szwjf08-utmb/rat-model-gb/qfoq9rc6



REM GRU
set PYTHONPATH=%CD%
python scripts\sweep_params\sweep_params_gru.py
wandb agent szwjf08-utmb/rat-model-gru/ddkuefen

python scripts/sweep_params/parallel_sweep.py --num-proc 12 --agent-path szwjf08-utmb/rat-model-gru/ddkuefen



