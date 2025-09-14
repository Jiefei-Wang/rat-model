@echo off
REM Usage: parallel_sweep.bat [num_processes] [agent_path]

REM Set default agent path to empty and number of processes
set AGENT_PATH=
set NUM_PROC=6

REM If a process count is provided, use it
if not "%1"=="" set NUM_PROC=%1

REM If an agent path is provided, use it
if not "%2"=="" set AGENT_PATH=%2

REM Check if AGENT_PATH is set
if "%AGENT_PATH%"=="" (
    echo ERROR: Please provide the agent_path as the first argument.
    exit /b 1
)

set /a NUM_PROC_MINUS1=%NUM_PROC%-1

for /L %%i in (1,1,%NUM_PROC_MINUS1%) do (
    echo Starting wandb agent process %%i
    start "wandb_agent_%%i" cmd /c "wandb agent %AGENT_PATH%"
)

REM Start the last process in the foreground
wandb agent %AGENT_PATH%