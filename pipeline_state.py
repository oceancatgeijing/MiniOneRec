#!/usr/bin/env python3
"""Atomic state and checkpoint helpers for the experiment pipeline."""

import argparse
import hashlib
import json
import os
import re
import tempfile


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def config_fingerprint(values):
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path, payload):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".state-", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def latest_hf_checkpoint(output_dir):
    candidates = []
    if not os.path.isdir(output_dir):
        return None
    for name in os.listdir(output_dir):
        match = CHECKPOINT_RE.fullmatch(name)
        if not match:
            continue
        path = os.path.join(output_dir, name)
        required = ["trainer_state.json"]
        if os.path.isdir(path) and all(os.path.isfile(os.path.join(path, item)) for item in required):
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(None, None))[1]


def state_matches(path, stage, values, required_files):
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if state.get("status") != "completed" or state.get("stage") != stage:
        return False
    if state.get("fingerprint") != config_fingerprint(values):
        return False
    return all(os.path.isfile(item) and os.path.getsize(item) > 0 for item in required_files)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    latest = subparsers.add_parser("latest-hf-checkpoint")
    latest.add_argument("--output-dir", required=True)

    check = subparsers.add_parser("check")
    check.add_argument("--state", required=True)
    check.add_argument("--stage", required=True)
    check.add_argument("--config", nargs="*", default=[])
    check.add_argument("--required", nargs="*", default=[])

    complete = subparsers.add_parser("complete")
    complete.add_argument("--state", required=True)
    complete.add_argument("--stage", required=True)
    complete.add_argument("--config", nargs="*", default=[])
    complete.add_argument("--required", nargs="*", default=[])

    args = parser.parse_args()
    if args.command == "latest-hf-checkpoint":
        checkpoint = latest_hf_checkpoint(args.output_dir)
        if checkpoint:
            print(checkpoint)
        return

    if args.command == "check":
        raise SystemExit(
            0 if state_matches(args.state, args.stage, args.config, args.required) else 1
        )

    missing = [path for path in args.required if not os.path.isfile(path) or os.path.getsize(path) == 0]
    if missing:
        raise FileNotFoundError(f"Cannot complete {args.stage}; missing outputs: {missing}")
    atomic_write_json(args.state, {
        "status": "completed",
        "stage": args.stage,
        "fingerprint": config_fingerprint(args.config),
        "config": args.config,
        "outputs": args.required,
    })


if __name__ == "__main__":
    main()
