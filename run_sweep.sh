# test single sweep
export PYTHONPATH="$PWD"
python scripts/sweep_params/sweep_params_lg.py
wandb agent szwjf08-utmb/rat-model-lg/q2bfohl8


export PYTHONPATH="$PWD"
python scripts/sweep_params/parallel_sweep.py --num-proc 8 --agent-path szwjf08-utmb/rat-model-lg/q2bfohl8


# GB
export PYTHONPATH="$PWD"
python scripts/sweep_params/sweep_params_gb.py
wandb agent szwjf08-utmb/rat-model-gb/d0xgnoni

export PYTHONPATH="$PWD"
python scripts/sweep_params/parallel_sweep.py --num-proc 8 --agent-path szwjf08-utmb/rat-model-gb/d0xgnoni


