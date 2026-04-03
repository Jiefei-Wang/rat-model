# test single sweep
export PYTHONPATH="$PWD"
python scripts/sweep_params/sweep_params_peak.py
wandb agent szwjf08-utmb/rat-model-peaks/7hd2ctwm

python scripts/sweep_params/parallel_sweep.py --num-proc 8 --agent-path szwjf08-utmb/rat-model-peaks/7hd2ctwm



# random forest
python scripts/sweep_params/sweep_params_RF.py
