#!/usr/bin/env python3
"""Run multiple wandb agents and tie child lifecycle to this parent process."""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys
import threading
import time
from typing import List


IS_WINDOWS = os.name == "nt"
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

children: List[subprocess.Popen] = []
shutdown_lock = threading.Lock()
shutting_down = False


def _spawn_agent(agent_path: str, quiet: bool) -> subprocess.Popen:
    kwargs = {}
    if IS_WINDOWS:
        kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    return subprocess.Popen(["wandb", "agent", agent_path], **kwargs)


def _kill_process_tree(proc: subprocess.Popen, force: bool) -> None:
    if proc.poll() is not None:
        return

    if IS_WINDOWS:
        cmd = ["taskkill", "/PID", str(proc.pid), "/T"]
        if force:
            cmd.append("/F")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return

    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except ProcessLookupError:
        pass


def shutdown_children(force: bool = False) -> None:
    global shutting_down
    with shutdown_lock:
        if shutting_down:
            return
        shutting_down = True

    for proc in children:
        _kill_process_tree(proc, force=force)

    deadline = time.time() + 5
    while time.time() < deadline:
        if all(proc.poll() is not None for proc in children):
            return
        time.sleep(0.1)

    for proc in children:
        _kill_process_tree(proc, force=True)


def _handle_sigint(_signum: int, _frame) -> None:
    print("Ctrl+C received: force killing all launched wandb agents...", flush=True)
    shutdown_children(force=True)
    raise SystemExit(130)


def _handle_terminate(signum: int, _frame) -> None:
    print(f"Received signal {signum}, terminating all launched wandb agents...", flush=True)
    shutdown_children(force=True)
    raise SystemExit(128 + signum)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-proc", type=int, default=6)
    parser.add_argument("--agent-path", type=str, required=True)
    args = parser.parse_args()

    if args.num_proc <= 0:
        print("ERROR: --num-proc must be a positive integer.", file=sys.stderr)
        return 1

    atexit.register(shutdown_children)
    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_terminate)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_terminate)

    for i in range(1, args.num_proc + 1):
        quiet = i != 1
        mode = "quiet" if quiet else "verbose"
        print(f"Starting wandb agent process {i} ({mode})", flush=True)
        children.append(_spawn_agent(args.agent_path, quiet=quiet))

    exit_code = 0
    while True:
        alive = [p for p in children if p.poll() is None]
        if not alive:
            break
        try:
            time.sleep(0.2)
        except KeyboardInterrupt:
            # Keep running; SIGINT behavior is handled in _handle_sigint.
            continue

    for proc in children:
        rc = proc.returncode if proc.returncode is not None else proc.wait()
        if exit_code == 0 and rc != 0:
            exit_code = rc

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
