#!/usr/bin/env python3
"""Atomic ZMQ port-allocation oracle for R15 vLLM PR #44495."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import zmq


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    if not re.search(r"\.bind\(f?[\"']tcp://.*:0[\"']\)", source):
        raise AssertionError("candidate does not bind the communication socket to port 0")
    if "zmq.LAST_ENDPOINT" not in source:
        raise AssertionError("candidate does not recover the atomically bound endpoint")
    tree = ast.parse(source)
    if any(isinstance(node, ast.Name) and node.id == "get_open_port" for node in ast.walk(tree)):
        raise AssertionError("candidate retains the racy probe-then-bind helper")

    context = zmq.Context()
    sockets: list[zmq.Socket] = []
    endpoints: list[str] = []
    try:
        for _ in range(args.count):
            socket = context.socket(zmq.REP)
            socket.bind("tcp://127.0.0.1:0")
            endpoint = socket.getsockopt(zmq.LAST_ENDPOINT).decode()
            sockets.append(socket)
            endpoints.append(endpoint)
        if len(set(endpoints)) != args.count:
            raise AssertionError("port-0 binds did not produce unique live endpoints")
        print(
            "R15_VLLM_ZMQ_ATOMIC_BIND="
            + json.dumps(
                {
                    "bind_count": args.count,
                    "unique_endpoint_count": len(set(endpoints)),
                    "source_uses_port_zero": True,
                    "source_reads_last_endpoint": True,
                    "source_uses_get_open_port": False,
                },
                sort_keys=True,
            )
        )
    finally:
        for socket in sockets:
            socket.close(linger=0)
        context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
