# plugin-detection

Deployable detection plugin (IPC attach + YOLO boxes; no MOT).

## Setup

```bash
unzip plugin-detection.zip
cd computer-vis
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
# Ultralytics / TensorRT editions — matched CUDA pair first (see requirements.txt):
# pip install torch==2.12.1 torchvision==0.27.1 \
#   --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
pip install -e .
```

Model weights are included under `models/`.

Sample video (when packed): `data/videos/trackdm1.mp4`.

## Run

```bash
# With live MediaBus (capture + infer services already publishing cam0):
python plugins/detection/ultralytics/run.py --ipc cam0 --imshow

# Standalone (webcam):
python plugins/detection/ultralytics/run.py --source 0 \
  --model models/detection/yolo26s.pt --imshow

# Standalone (sample clip):
python plugins/detection/ultralytics/run.py --source data/videos/trackdm1.mp4 \
  --model models/detection/yolo26s.pt --imshow
```

Editions in this pack: **tensorrt, ultralytics, ort**.
Default weights: `models/detection/yolo26s.pt` (standalone).
