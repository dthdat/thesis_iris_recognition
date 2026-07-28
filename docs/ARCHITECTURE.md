# Source architecture

The system has four layers:

1. **Data and preprocessing** — `src/data.py`, `src/preprocessing.py`, and
   `src/masks.py` load eye images, localize and normalize the iris, and create
   model-ready polar strips.
2. **Representation learning** — `src/models.py`, `src/losses.py`, and
   `experiments/train.py` define the IResNet/ArcFace training path.
3. **Evaluation and export** — `experiments/evaluate.py`,
   `experiments/evaluate_classical.py`, and `experiments/export_onnx.py`
   generate embeddings, verification scores, and an ONNX graph from local
   artifacts.
4. **Embedded inference** — `inference/iris-recognition-system/` provides the
   Flask browser application. `jetson/` contains the runtime preparation and
   TensorRT integration helpers used on the Jetson Nano.

The public folder contains no dataset, checkpoint, engine, database, or
measurement output. Those must be supplied locally and kept out of version
control.
