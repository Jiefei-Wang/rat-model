@echo off
REM Usage: gradient_boosting_model.bat [script_module_name] [num_processes]

REM Set default script and number of processes
set SCRIPT_MODULE=scripts.sweep.gradient_boosting_model
set NUM_PROC=6

REM If a script name is provided, use it
if not "%1"=="" set SCRIPT_MODULE=%1

REM If a process count is provided, use it
if not "%2"=="" set NUM_PROC=%2

set /a NUM_PROC_MINUS1=%NUM_PROC%-1

for /L %%i in (1,1,%NUM_PROC_MINUS1%) do (
    echo Starting background process %%i
    start /B python -m %SCRIPT_MODULE% >nul 2>&1
)

python -m %SCRIPT_MODULE%