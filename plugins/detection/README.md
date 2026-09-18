# Detection plugin (deploy)

> MediaBus.

Independent detection process. Course lesson stays in
[`topics/02-inference-runtime`](topics/02-inference-runtime/README.md).
Default weights: `models/detection/yolo26s.pt`.

## Infer (read this)

| Word | What | Who | Config |
|------|------|-----|--------|
| **Infer** | Object detector — boxes (`person`, score, xyxy) | **IPC:** topic 02 (`InferSlot`). **Standalone:** this edition’s YOLO / ORT / TensorRT | `run.model`, `run.device`, `run.confidence` |

No MOT — for stable IDs use [`plugins/tracking`](../tracking/README.md).

**IPC on** — attach topic 01 + 02; this process only filters and draws shared boxes.

**IPC off** — open `--source`, run local detector, draw.

## MediaBus connection (IPC)

```bash
# Terminal A — frames
python topics/01-capture-runtime/run.py --source 0 --no-consumer

# Terminal B — infer → InferSlot
python topics/02-inference-runtime/run.py --ipc cam0 --device 0

# Terminal C — draw InferSlot boxes
python plugins/detection/ultralytics/run.py --ipc cam0 --imshow
```

| Symptom | Fix |
|---------|-----|
| `IPC bus 'cam0' is not running` | Start topic 01 with `--no-consumer` |
| `Infer bus 'cam0' is not running` | Start topic 02 with `--ipc cam0` |
| No boxes | Lower `--confidence` or widen `classes` on 02 and the plugin |

## Standalone (local YOLO)

```bash
python plugins/detection/ultralytics/run.py \
  --source data/videos/trackdm1.mp4 --device 0 --imshow --imshow-scale 0.5
```

## Editions

| Edition | Path | License | Standalone model |
|---------|------|---------|------------------|
| Ultralytics | `ultralytics/` | free | `.pt` (default `yolo26s.pt`) |
| ORT | `ort/` | paid | `.onnx` |
| TensorRT | `tensorrt/` | paid | `.engine` |

```bash
python plugins/detection/export_weights.py --format onnx --imgsz 640
python plugins/detection/export_weights.py --format engine --device 0 --imgsz 640
```

## `run.classes` (detection filter)

| Value | Meaning |
|-------|---------|
| `null` | Keep all classes |
| `[person]` / `[car, truck]` | Keep these names |
| `[0, 2]` | Keep these class ids |
| CLI `--classes person car` | Override YAML for one run |

```yaml
run:
  model: models/detection/yolo26s.pt
  classes: [person]
  # classes: null
```

Custom non-COCO models: point `run.model` at your weights and set `classes` /
ORT `names` to that model’s labels (see [plugins README](../README.md#custom-models-non-coco-classes)).

## Environment

### Linux (Ubuntu)

```bash
cd /path/to/computer-vis
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[ml]"   # Ultralytics + Torch
# GPU notes: docs/installation/gpu-ubuntu.md
```

### Windows

```powershell
cd C:\path\to\computer-vis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e ".[ml]"
# GPU notes: docs/installation/windows.md
```

## YAML knobs

| Key | Meaning |
|-----|---------|
| `run.model` | Weights (standalone); ignored with `--ipc` |
| `run.confidence` | Score threshold |
| `run.classes` | Class filter (`null` = all) |
| `run.imgsz` | Infer size (match export) |
| `run.device` | `cpu` \| `0` \| … |
| `run.imshow` / `imshow_scale` | Preview window |
