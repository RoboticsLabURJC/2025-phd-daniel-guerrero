# PhD Research Repository – Universal Neural Network for Autonomous Driving via Homographic (BEV) Vision

**Author:** Luis Daniel Guerrero  
**Institution:** RoboticsLab URJC / JdeRobot  

This repository contains the source code, trained deep learning models, hardware designs, and weekly progress logbook for the doctoral thesis on **Designing a Universal Neural Network for End-to-End Autonomous Driving using Homographic Bird's-Eye View (BEV) and Frontal Perspective Vision in CARLA**, as well as hardware-in-the-loop integration.

---

## 1. Repository Architecture

```text
2025-phd-daniel-guerrero/
├── docs/                          # GitHub Pages (Jekyll) site for the weekly PhD progress blog
│   ├── _posts/                    # Weekly technical reports (Week 1 to Week 59+)
│   └── assets/images/             # Figures, plots, and hardware photos embedded in the blog
│
├── PHD Proposal/                  # Doctoral research proposal documents
├── Model Wheel Angle Sensor/      # 3D CAD & mechanical parts for the physical steering sensor
│
├── data/                          # (Git-ignored) Large CARLA datasets, BEV frames, and .log/.h5 files
│   ├── dataset_masivo_continuo/   # High-definition frontal (frames/) & homographic BEV (frames_homographic/)
│   ├── datasets_collection/       # Historical datasets
│   └── logs_dagger_expert/        # Expert DAgger CARLA recorder logs
│
├── models/                        # Trained PyTorch checkpoints (.pth / .pt)
│   ├── bev/                       # Active Homographic BEV models (FastViT, ResNet-18, PilotNet)
│   ├── frontal/                   # Active Frontal Perspective models (PilotNet)
│   └── legacy/                    # Earlier experimental checkpoints (DualInput, Canny, MobileNet, TreNet)
│
└── scripts/                       # Python & microcontroller source code organized by pipeline stage
    ├── 01_data_collection/        # CARLA data extraction, log replay, DAgger, and resilient watchdogs
    ├── 02_preprocessing/          # Homography (IPM/BEV) transformation & curve statistical balancing
    │   ├── homography/            # Interactive 4-point H-matrix calibration & batch BEV generator
    │   └── balancing/             # Dataset curve balancing utilities
    ├── 03_training/               # PyTorch training pipelines (Weighted Huber Loss / Hot Balancing)
    │   ├── bev/                   # Homographic BEV training (FastViT, ResNet-18, PilotNet)
    │   ├── frontal/               # Normal frontal perspective training (PilotNet)
    │   └── archive/               # Historical / DualInput training scripts
    ├── 04_inference/              # Standalone closed-loop CARLA inference scripts
    │   ├── bev/                   # Real-time Homographic BEV inference in CARLA
    │   ├── frontal/               # Real-time Frontal perspective inference in CARLA
    │   └── archive/               # Historical inference & control nodes
    ├── 05_hardware_esp32/         # ESP32 firmware for the physical Wheel Angle Sensor (WAS_ESP32)
    └── utils/                     # Telemetry plotting (graph_tools) and CARLA inspection utilities (tools)
```

---

## 2. Core Research Pipelines

### A. Homographic Bird's-Eye View (IPM / BEV) Pipeline
1. **Data Extraction (`scripts/01_data_collection/dataset_gen_hd_260912.py`):** Replays expert CARLA logs in `Town04` using a `1920x1080` RGB camera (`FOV=110°`, `Pitch=-15°`, `x=2.0, z=1.6`) with automatic watchdog recovery.
2. **Homography Calibration & Batch Warping (`scripts/02_preprocessing/homography/`):**
   - `calculo_matriz.py`: Interactive 4-point coplanar lane calibration to compute the $3 \times 3$ homography matrix $H$.
   - `homografic_maker.py`: Masks the upper horizon (`y < 450`) to prevent sky/building distortion and warps frames to `600x800` Bird's-Eye View (`frames_homographic/`).
3. **Model Training (`scripts/03_training/bev/`):**
   - Architectures evaluated under identical image-only conditions (`240x320` input, ImageNet normalization, Weighted Huber Loss with `curve_multiplier=12.0`):
     - **FastViT (`fastvit_t8`):** `train_fastvit_onlyimg_261014.py` $\rightarrow$ `models/bev/modelo_fastvit_steering_only.pth`
     - **ResNet-18:** `train_resn18_noimg_261014.py` $\rightarrow$ `models/bev/modelo_resnet18_steering_only.pth`
     - **PilotNet:** `train_pilotnet_onlyimg_bev.py` $\rightarrow$ `models/bev/modelo_pilotnet_bev_steering_only.pth`

### B. Frontal Perspective Pipeline (`scripts/03_training/frontal/`)
- **PilotNet Frontal (`train_pilotnet_onlyimg_normal.py`):** Trained on the unwarped `1920x1080` frontal perspective frames (`data/dataset_masivo_continuo/frames`) under the exact same hyperparameters as the BEV models $\rightarrow$ `models/frontal/modelo_pilotnet_normal_steering_only.pth`.

---

## 3. Closed-Loop Evaluation with Behavior Metrics

Quantitative closed-loop evaluations in CARLA (`Town04_Opt`) are executed using the **Behavior Metrics** framework (CARLA Python API mode, no ROS required):

```bash
cd ~/code/BehaviorMetrics/behavior_metrics
source ../bmetrics_env/bin/activate
source var_entorno.sh

# Evaluate FastViT (BEV Homographic)
python driver_carla.py -c configs/CARLA/eval_fastvit.yml -g

# Evaluate ResNet-18 (BEV Homographic)
python driver_carla.py -c configs/CARLA/eval_resnet18.yml -g

# Evaluate PilotNet (BEV Homographic)
python driver_carla.py -c configs/CARLA/eval_pilotnet_bev.yml -g

# Evaluate PilotNet (Normal Frontal)
python driver_carla.py -c configs/CARLA/eval_pilotnet_normal.yml -g
```

---

## 4. Weekly Progress Blog (`docs/`)

The `docs/` folder is a **Jekyll (Minimal Mistakes)** static website published automatically via **GitHub Pages**:
- New weekly entries are added as Markdown files in `docs/_posts/YYYY-MM-DD-week-XX.md`.
- Embedded figures and plots are stored in `docs/assets/images/`.
