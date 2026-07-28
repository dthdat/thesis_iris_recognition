#!/usr/bin/env python3
"""Preprocess the existing Jetson deployment image set once for fair backend timing."""

from __future__ import print_function

import argparse
import importlib.machinery
import json
import os
import time

import numpy as np


EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-module", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    module = importlib.machinery.SourceFileLoader("iris_deployment_backend", args.backend_module).load_module()
    config = module.load_meta(args.metadata)
    paths = []
    for root, _, names in os.walk(args.images):
        for name in sorted(names):
            if name.lower().endswith(EXTENSIONS):
                paths.append(os.path.join(root, name))
    paths.sort()
    values = []
    latencies = []
    failures = []
    kept_paths = []
    for index, path in enumerate(paths):
        started = time.perf_counter()
        polar, metadata = module.preprocess_image(path, config)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if polar is None:
            failures.append({"path": path, "reason": metadata.get("reason", "unknown")})
            continue
        tensor = module.polar_to_model_input(polar, config, np.float32)
        values.append(np.asarray(tensor, dtype=np.float32).reshape(1, 64, 512))
        latencies.append(elapsed_ms)
        kept_paths.append(path)
        if (index + 1) % 20 == 0:
            print("preprocessed %d/%d" % (index + 1, len(paths)), flush=True)
    if not values:
        raise RuntimeError("No deployment images were successfully preprocessed")
    array = np.stack(values, axis=0)
    np.save(args.output, array)
    report = {
        "source_images": len(paths),
        "valid_images": len(values),
        "failed_images": len(failures),
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_median_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "tensor_shape": list(array.shape),
        "paths": kept_paths,
        "failures": failures,
    }
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("valid=%d failed=%d tensor_shape=%r" % (len(values), len(failures), array.shape))


if __name__ == "__main__":
    main()
