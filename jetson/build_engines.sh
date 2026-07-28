#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:?usage: build_engines.sh ONNX_PATH OUTPUT_DIR}"
OUTPUT_DIR="${2:?usage: build_engines.sh ONNX_PATH OUTPUT_DIR}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$ONNX_PATH" ]]; then
  echo "Missing ONNX: $ONNX_PATH" >&2
  exit 2
fi
if [[ ! -x "$TRTEXEC" ]]; then
  echo "Missing trtexec: $TRTEXEC" >&2
  exit 2
fi

COMMON=(
  "--onnx=$ONNX_PATH"
  "--explicitBatch"
  "--minShapes=iris_polar:1x1x64x512"
  "--optShapes=iris_polar:1x1x64x512"
  "--maxShapes=iris_polar:1x1x64x512"
  "--workspace=512"
  "--buildOnly"
  "--verbose"
)

build_one() {
  local precision="$1"
  local engine="$OUTPUT_DIR/iris_iresnet50_msff_${precision}.engine"
  local log="$OUTPUT_DIR/build_${precision}.log"
  if [[ -s "$engine" ]]; then
    echo "Preserving existing versioned engine: $engine"
    return
  fi
  local extra=()
  if [[ "$precision" == "fp16" ]]; then
    extra+=("--fp16")
  fi
  echo "Building $precision engine: $engine"
  local started finished
  started="$(date +%s)"
  "$TRTEXEC" "${COMMON[@]}" "${extra[@]}" "--saveEngine=$engine" 2>&1 | tee "$log"
  finished="$(date +%s)"
  echo "build_seconds=$((finished - started))" | tee -a "$log"
  test -s "$engine"
}

build_one fp32
build_one fp16
sha256sum "$ONNX_PATH" "$OUTPUT_DIR"/*.engine > "$OUTPUT_DIR/artifact_sha256.txt"
df -h "$(dirname "$OUTPUT_DIR")"
