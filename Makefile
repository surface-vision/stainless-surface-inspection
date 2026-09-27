# Jindal Stainless surface-defect detection -- task entry points.
#
# Every target runs through the project venv, so nothing depends on which
# interpreter happens to be first on PATH. Override any variable on the command
# line, e.g.  make train EPOCHS=50 IMGSZ=640

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit

# Training / evaluation knobs.
# Nano is the served checkpoint (see inference.resolve_weights), so it is also
# the default here; pass MODEL=yolov8s.pt NAME=yolov8s_neudet for the comparison.
MODEL   ?= yolov8n.pt
NAME    ?= yolov8n_neudet
EPOCHS  ?= 150
# Training input size. Inference uses INFER_IMGSZ, which is what src/model_study.py
# selected on val -- it is not the same number and conflating them cost ~55% of
# mAP50 the last time the console defaulted to 640.
IMGSZ   ?= 320
INFER_IMGSZ ?= 256
SHUFFLE_SEED ?= 20260909
# Paired-bootstrap resamples for the studies. Below ~1000 the interval is too coarse.
RESAMPLES ?= 1000
COIL_ID ?= JS-HR-2609-0417
BATCH   ?= 32
DEVICE  ?= auto
WEIGHTS ?=
SPLIT   ?= test
TAG     ?= baseline
PORT    ?= 8501

# Passed through only when set, so the scripts keep their own defaults.
WEIGHTS_ARG := $(if $(WEIGHTS),--weights $(WEIGHTS),)

.DEFAULT_GOAL := help
.PHONY: help setup data train eval false-alarm study input-study calibrate cross-domain \
        severstal-data gc10-data bench demo test export report clean-reports

help:  ## show this list
	@grep -hE '^[a-z][a-z0-9-]*:.*?## ' $(MAKEFILE_LIST) \
	  | awk -F':.*?## ' '{printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- environment

setup: requirements.txt  ## create .venv and install the pinned dependencies
	@test -d $(VENV) || python3.11 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@$(PY) -c "import torch; print('torch', torch.__version__, '| mps', torch.backends.mps.is_available())"

# ---------------------------------------------------------------------- data

data:  ## rebuild the stratified 1440/180/180 NEU-DET split from data/NEU-DET-raw
	$(PY) src/prepare_data.py --raw data/NEU-DET-raw --out data/neu-det

# ------------------------------------------------------------------ training

train:  ## fine-tune a detector (writes models/<NAME>/)
	$(PY) src/train_detector.py \
	  --model $(MODEL) --name $(NAME) --epochs $(EPOCHS) \
	  --imgsz $(IMGSZ) --batch $(BATCH) --device $(DEVICE)

# ---------------------------------------------------------------- assessment

eval:  ## mAP, image-level confusion and error gallery -> reports/evaluation.md
# --imgsz is the *served* size, not the training size: an evaluation report describes
# the deployed detector. This target does NOT write reports/operating_point.json --
# that file belongs to false_alarm.py (see the comment on the false-alarm target).
	$(PY) src/evaluate.py --split $(SPLIT) --imgsz $(INFER_IMGSZ) --device $(DEVICE) $(WEIGHTS_ARG)

false-alarm:  ## clean-steel false alarm proxy + the SHIPPED operating point -> reports/false_alarm.md
# This is the module that owns reports/operating_point.json, which the console reads
# for its default confidence. It prices false alarms on crops carrying no annotated
# defect -- genuinely clean steel -- rather than on spurious extra boxes over frames
# that were already defective, which is what evaluate.py can see and is a different
# event entirely. Add --update-operating-point to actually rewrite the demo default.
	$(PY) src/false_alarm.py --device $(DEVICE) $(WEIGHTS_ARG)

study:  ## model / input-size / TTA selection study with bootstrap CIs -> reports/model_study.md
# Owns two decisions the rest of the tree just inherits: which checkpoint ships and
# what INFER_IMGSZ is. Re-run it before changing either. --resamples is the paired
# bootstrap over the 180 test images; below ~1000 the interval is too coarse to read.
	$(PY) src/model_study.py --device $(DEVICE)

input-study:  ## pad-vs-scale and the full resolution grid, both checkpoints -> reports/input_study.md
# Answers a question `study` cannot: whether the 200x200 source is better preserved
# (padded into a 224/256 canvas with no resampling at all) than interpolated up to
# the network size. Writes the losslessly padded splits to data/input_study/ and
# verifies they are byte-exact before scoring anything.
	$(PY) src/input_study.py --device $(DEVICE) --resamples $(RESAMPLES)

calibrate:  ## fit the confidence calibrator on val, report on test -> reports/calibration.md
# Fits score -> P(true positive) on the VALIDATION split only and writes
# reports/calibration.json for the console to apply. Rank-preserving, so mAP is
# unchanged; the report proves that rather than asserting it. Re-fit on any new
# checkpoint -- a calibrator belongs to one set of weights.
	$(PY) src/calibrate.py --imgsz $(INFER_IMGSZ) --device $(DEVICE) $(WEIGHTS_ARG)

cross-domain:  ## score the shipped detector on non-NEU-DET steel -> reports/cross_domain*
# Needs the Severstal frames on disk first: make severstal-data. TAG names the run so
# a baseline and a retrained checkpoint can be compared without overwriting each other.
	$(PY) src/cross_domain.py --imgsz $(INFER_IMGSZ) --device $(DEVICE) \
	  --tag $(TAG) $(WEIGHTS_ARG)

severstal-data:  ## build the Severstal + joint YOLO datasets under data/
	$(PY) src/prepare_severstal.py

gc10-data:  ## ingest GC10-DET (roll marks, edge defects) into YOLO format under data/
	$(PY) src/prepare_gc10.py

bench:  ## latency/throughput sweep and mill capacity plan -> reports/benchmark.json
# --repeats 3 is the floor for a number anyone sizes hardware against, and even
# then the between-invocation spread is ~25% (protocol.repeatability_note).
	$(PY) src/benchmark.py --runs yolov8n_neudet yolov8s_neudet --files best.pt \
	  --iters 100 --budget-s 10 --repeats 3

test:  ## run the smoke suite
	$(PY) -m pytest tests/ -v

# -------------------------------------------------------------------- serving

demo:  ## launch the operator console on http://localhost:8501
	$(STREAMLIT) run demo/app.py --server.port $(PORT)

# ------------------------------------------------------- deployment artefacts

export:  ## export ONNX (+ CoreML) and verify against PyTorch -> export/
	$(PY) src/export_model.py --imgsz $(INFER_IMGSZ) $(WEIGHTS_ARG)

report:  ## per-coil inspection report over the test split -> reports/coil_report.html
# --shuffle-seed is not optional for a report anyone will read: a directory listing
# of NEU-DET is sorted by class, so an unshuffled coil position map shows six clean
# blocks that read as sustained process upsets. The seed makes the shuffle
# reproducible. --imgsz is the selected inference size, not the training size.
	$(PY) src/report.py --images data/neu-det/$(SPLIT)/images \
	  --coil-id $(COIL_ID) --shuffle-seed $(SHUFFLE_SEED) --imgsz $(INFER_IMGSZ)

clean-reports:  ## delete generated reports and figures (never touches models/)
# reports/operating_point.json is deliberately NOT on this list. It is the shipped
# threshold the console boots with, it costs a full false_alarm.py run to rebuild,
# and deleting it silently drops the demo back to its hard-coded fallback. Remove it
# by hand if you mean to.
	rm -rf reports/error_gallery reports/ultralytics
	rm -f reports/evaluation.* reports/operating_point.evaluate.json \
	      reports/benchmark*.json \
	      reports/benchmark_*.png reports/confusion_matrix.png reports/pr_curves.png \
	      reports/f1_vs_threshold.png reports/false_alarms_vs_recall.png \
	      reports/calibration.json reports/calibration.png reports/calibration.md \
	      reports/coil_report.* reports/export_summary*.json
