#!/usr/bin/env python3
"""Collect a compact, non-secret Jetson benchmark environment manifest."""

from __future__ import print_function

import argparse
import json
import os
import platform
import subprocess


def command(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, universal_newlines=True).strip()
    except Exception as error:
        return "unavailable: %s" % error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        import torch
        torch_version = torch.__version__
    except Exception as error:
        torch_version = "unavailable: %s" % error
    try:
        import tensorrt
        tensorrt_version = tensorrt.__version__
    except Exception as error:
        tensorrt_version = "unavailable: %s" % error
    release = ""
    if os.path.isfile("/etc/nv_tegra_release"):
        with open("/etc/nv_tegra_release") as handle:
            release = handle.read().strip()
    manifest = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "l4t_release": release,
        "torch": torch_version,
        "tensorrt": tensorrt_version,
        "cuda": command(["nvcc", "--version"]),
        "power_mode": command(["nvpmodel", "-q"]),
        "memory": command(["free", "-h"]),
        "disk": command(["df", "-h", "/"]),
    }
    with open(args.output, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
