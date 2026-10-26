# PhD Autonomous Driving Scripts & Pipeline Structure

This directory contains only the Python and hardware source code for the PhD research pipeline, separated from large datasets (`../data/`) and model checkpoints (`../models/`).

## Repository Layout

```text
2025-phd-daniel-guerrero/
├── docs/                          # Weekly research progress blog (Jekyll)
├── PHD Proposal/                  # Doctoral proposal documents
├── Model Wheel Angle Sensor/      # CAD / Hardware files for the steering sensor
├── data/                          # Datasets, BEV frames, CARLA logs, and .h5 files (git-ignored)
├── models/                        # Trained PyTorch checkpoints (.pth / .pt)
│   ├── bev/                       # Active Homographic BEV models (FastViT, ResNet-18, PilotNet)
│   ├── frontal/                   # Active Frontal Perspective models (PilotNet)
│   └── legacy/                    # Earlier checkpoints (DualInput, Canny, MobileNet, etc.)
└── scripts/                       # Source code organized by pipeline stage
    ├── 01_data_collection/        # CARLA data extraction, log replay, DAgger, and watchdogs
    ├── 02_preprocessing/          # Homography (IPM/BEV) and dataset curve balancing
    │   ├── homography/            # H-matrix calibration (calculo_matriz.py) & batch BEV generator
    │   └── balancing/             # Statistical curve balancing scripts
    ├── 03_training/               # PyTorch model training scripts
    │   ├── bev/                   # Homographic BEV training (FastViT, ResNet-18, PilotNet)
    │   ├── frontal/               # Normal frontal perspective training (PilotNet)
    │   └── archive/               # Historical / DualInput training scripts
    ├── 04_inference/              # Standalone CARLA closed-loop inference scripts
    │   ├── bev/                   # Homographic BEV real-time inference
    │   ├── frontal/               # Frontal perspective real-time inference
    │   └── archive/               # Historical inference & control nodes
    ├── 05_hardware_esp32/         # ESP32 firmware for the Wheel Angle Sensor (WAS_ESP32)
    └── utils/                     # Plotting (graph_tools) and CARLA inspection utilities (tools)
```
