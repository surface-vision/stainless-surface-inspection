#!/bin/bash
# imgsz=320 chosen empirically: source images are natively 200x200, median box min-side is 70px
# at 320, and 640 measured >11 min/epoch on MPS versus 35 s/epoch at 320 for no added information.
cd /Users/prathmeshwalimbe/Downloads/JSW-PS1
PY=.venv/bin/python
$PY -u src/train_detector.py --model yolov8s.pt --epochs 150 --imgsz 320 --batch 32 --patience 50 --name yolov8s_neudet
$PY -u src/train_detector.py --model yolov8n.pt --epochs 150 --imgsz 320 --batch 32 --patience 50 --name yolov8n_neudet
echo "ALL_TRAINING_DONE"
