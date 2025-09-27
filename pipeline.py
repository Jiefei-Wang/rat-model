
################
##  Load data
################
with open('scripts/data/01_raw_data.py') as f:
    exec(f.read())

################
## Train models
## This is for generating the sweep files
## To run the sweeps, use train_mode.bat
################

with open('scripts/sweep_params/sweep_params_peak.py') as f:
    exec(f.read())


with open('scripts/models/sweep_RF.py') as f:
    exec(f.read())

with open('scripts/models/sweep_GB.py') as f:
    exec(f.read())



