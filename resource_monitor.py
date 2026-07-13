#!/usr/bin/env python3
"""Run a command and record elapsed time plus per-GPU peak memory."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time


def query_gpu_memory():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    memory = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        index, used = [value.strip() for value in line.split(",", 1)]
        memory[index] = int(used)
    return memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("a command is required after --")

    peaks = {}
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            for index, used in query_gpu_memory().items():
                peaks[index] = max(peaks.get(index, 0), used)
            stop.wait(args.interval)

    started = time.time()
    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    process = subprocess.run(command)
    stop.set()
    worker.join(timeout=args.interval + 1.0)
    elapsed = time.time() - started

    report = {
        "command": command,
        "return_code": process.returncode,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_mib": peaks,
        "max_peak_gpu_memory_mib": max(peaks.values()) if peaks else None,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Resource report: {args.output}")
    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
