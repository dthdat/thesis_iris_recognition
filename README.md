# Iris Recognition System — Source Release

This folder contains the source code, experiment entry points, notebooks, and
Jetson inference application for the iris-recognition thesis project. It is
prepared for source review and collaboration. It intentionally does **not**
contain standalone benchmark reports, trained weights, ONNX/TensorRT binaries,
datasets, Kaggle outputs, screenshots, participant data, thesis drafts, or
private credentials. The one explicitly labelled full-run notebook retains its
own historical cell outputs for provenance.

## What is included

```text
src/                         Shared data, preprocessing, models, losses, metrics
experiments/                 Training, evaluation, classical evaluation, ONNX export
experiments/configs/         Reproducible model configuration files
notebooks/                   Source notebooks plus one labelled full-run copy
inference/                   Jetson Flask web application and camera UI
jetson/                      Deployment-input, engine-build, and runtime scripts
baselines/arciris_reference/ Reference ArcIris implementation used for compatibility
tests/                       Unit and wiring tests
```

The ArcIris reference implementation is retained with its original academic
attribution. Review its upstream licence and citation requirements before
redistributing it separately.

## Reproducible setup

The project expects Python 3.10 or newer for training and evaluation. Install
the PyTorch/TorchVision build appropriate for the host first, then install the
remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install torch and torchvision for your CPU/CUDA platform.
python -m pip install -r requirements.txt
```

The CASIA-Iris-Thousand images are not included. Set `IRIS_DATASET_ROOT` to a
directory whose immediate children are subject folders such as `000/`, `001/`,
and so on. Do not commit the dataset or any biometric samples.

To run a local smoke test with an available dataset:

```bash
IRIS_DATASET_ROOT=/path/to/CASIA-Iris-Thousand \
python experiments/train.py \
  --config experiments/configs/ours_iresnet_msff_softmask.yaml \
  --max-epochs 1
```

For a normal evaluation or ONNX export, point the command at a locally
generated run directory and checkpoint. Binary model artifacts are deliberately
kept outside this release folder.

## Jetson Nano inference

The application in `inference/iris-recognition-system/` is the browser-based
enrollment and recognition demo used on the Jetson Nano. It requires, on the
Jetson, the compatible TensorRT/PyCUDA runtime adapter, an engine, and its
metadata file. These deployment binaries are not part of the public source
release.

```bash
cd inference/iris-recognition-system
python3 -m pip install -r requirements_jetson.txt
IRIS_ENGINE=/path/to/model_fp16.engine \
IRIS_META=/path/to/model.metadata.json \
bash scripts/run_server.sh
```

The default camera source is the Jetson CSI camera. For a USB camera, set
`IRIS_CAMERA=0`. Host, port, camera dimensions, database path, engine path,
metadata path, and decision threshold are configurable through environment
variables; keep local database files and captures out of Git.

The `jetson/` directory contains source-only helpers for preprocessing
deployment inputs, building engines from a local ONNX file, collecting
read-only environment information, and running backend timing experiments.
Those scripts write their outputs to a user-selected local directory and do
not ship any measurements here.

## Notebook policy

The regular notebooks are executable research starting points with outputs
removed so that old measurements cannot be mistaken for a fresh run. Execute
them only after supplying the dataset and local model artifacts.

`notebooks/iris_baseline_full_trained.ipynb` is the explicitly requested fully
executed copy. Its historical cell outputs are retained for provenance and
teaching review; they are not a replacement for a machine-readable benchmark
report or a newly reproduced run.

## Data, model, and privacy boundaries

Do not commit:

- CASIA or camera images;
- participant or enrollment databases;
- `.pth`, `.onnx`, `.engine`, or other model binaries;
- Kaggle credentials, API tokens, or generated Kaggle bundles;
- benchmark tables, logs, figures, or thesis documents.

Before publication, add a project licence and cite the upstream ArcIris
implementation and the datasets used by the experiment.

## Suggested review order

1. Read `src/preprocessing.py`, `src/masks.py`, and `src/models.py`.
2. Inspect the three YAML configurations in `experiments/configs/`.
3. Follow `experiments/train.py`, `experiments/evaluate.py`, and
   `experiments/export_onnx.py`.
4. Review the Jetson application under `inference/`.
5. Run the unit tests after installing the development dependencies:

   ```bash
   pytest -q tests
   ```

This release is source code only. Any scientific claim must be regenerated
from an explicitly documented run and reviewed against its untouched raw
artifacts.
