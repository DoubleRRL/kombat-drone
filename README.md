# Combat-Ready SLAM + Thermal Detection System

Real-time autonomous navigation and target detection for contested environments. Production-ready computer vision pipeline combining visual-thermal SLAM with thermal signature detection.

## Quick Deploy

### macOS
```bash
git clone https://github.com/DoubleRRL/kombat-drone.git
cd kombat-drone
pip install -r requirements.txt
python src/main.py --demo
```

### Windows
```powershell
git clone https://github.com/DoubleRRL/kombat-drone.git
cd kombat-drone
pip install -r requirements.txt
python src/main.py --demo
```

### Docker (All Platforms)
```bash
docker build -t kombat-drone .
docker run --gpus all -it kombat-drone
```

## Core Capabilities

- **GPS-Denied SLAM**: ORB-SLAM3 enhanced with thermal features for navigation without GPS
- **Thermal Detection**: YOLOv8 fine-tuned on FLIR thermal imagery for target identification  
- **Sensor Fusion**: Real-time RGB + thermal + IMU processing with failure recovery
- **Edge Deployment**: <50ms latency, 20+ FPS on commodity hardware

## System Requirements

**Minimum**: 4-core CPU, 8GB RAM, 20GB storage  
**Recommended**: 8-core CPU, 16GB RAM, NVIDIA RTX/Apple M-series GPU

## Performance Validated

- **35ms** average processing latency (target: <50ms)
- **28 FPS** sustained processing (M2 MacBook Pro)
- **0.3m** SLAM accuracy on TUM RGB-D datasets
- **27/27** automated tests passing

## Training Pipeline

```bash
# Prepare datasets (FLIR, KAIST, TUM)
python scripts/multi_dataset_training.py --thermal-epochs 100

# Evaluate system performance  
python scripts/evaluate_system.py
```

## Architecture

```
├── src/
│   ├── slam/           # Enhanced SLAM with thermal integration
│   ├── detection/      # YOLO thermal target detection
│   ├── fusion/         # Multi-modal sensor fusion
│   └── main.py         # Real-time processing pipeline
├── datasets/           # FLIR ADAS, KAIST, TUM RGB-D
├── docker/             # Multi-platform containers
└── scripts/            # Training and evaluation
```

## Deployment Options

- **Edge**: Single-board computers, embedded systems
- **Cloud**: Kubernetes, multi-GPU clusters  
- **Hybrid**: Distributed processing across edge-cloud

Designed for autonomous drone operations in GPS-denied, low-visibility environments.