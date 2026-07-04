# qJointDetect — ML-Powered Rock Joint Detection Plugin for CloudCompare

> **Goal:** A CloudCompare plugin that uses machine learning to automatically identify rock joint surfaces from 3D point clouds — trained on geologist-labeled data from **qCompass** — and outputs `ccFitPlane` objects with dip angle & dip direction measurements.

---

## 📊 Progress Tracker

| Phase | Description | Status | Started | Done |
|---|---|---|---|---|
| **0 — Data Pipeline** | Python scripts: Compass CSV → features → .npz | 🟡 **In Progress** | ✅ | ⬜ |
| **1 — Classic ML** | Train Random Forest, export ONNX | ⬜ | ⬜ | ⬜ |
| **2 — Plugin Skeleton** | C++ qJointDetect loads ONNX, runs inference | ⬜ | ⬜ | ⬜ |
| **3 — UI Polish** | Confidence slider, visualization controls | ⬜ | ⬜ | ⬜ |
| **4 — Self-Training** | User corrections → retrain loop | ⬜ | ⬜ | ⬜ |
| **5 — Deep Learning** | PointNet++ upgrade if needed | ⬜ | ⬜ | ⬜ |

**Legend:** ✅ = Complete &nbsp; 🟡 = In Progress &nbsp; ⬜ = Not Started

---

## 🔧 Files Created So Far

| File | Purpose | Phase |
|---|---|---|
| `PLAN.md` | This document | All |
| `CSV_EXPORT_GUIDE.md` | Guide for geologists to export training data | 0 |
| `scripts/extract_training_data.py` | Feature extraction from Compass CSVs | 0 |
| `scripts/train_joint_model.py` | Train RF → export ONNX | 1 |
| `plugins/core/Standard/qJointDetect/CMakeLists.txt` | Plugin build system | 2 |
| `plugins/core/Standard/qJointDetect/info.json` | Plugin metadata | 2 |
| `training_data/README.md` | Data directory structure | 0 |

---

## 1. Why This Matters

Geologists currently identify rock joints **manually** in CloudCompare:
- Draw traces along joint surface boundaries (qCompass Trace Tool)
- Fit planes to those traces (qCompass Fit Plane Tool)
- Record dip angle (α), dip direction (β), strike
- Export CSV for structural analysis

**Problem:** Manual interpretation of a single slope can take **hours to days**. Results vary between geologists.

**Solution:** Use those manual labels as **training data** for an ML model that learns what a joint looks like in 3D space, then let the model propose joints automatically — with a geologist in the loop to verify/correct.

---

## 2. Compass Data — Our Training Fuel

The existing **qCompass** plugin produces rich labeled data through its CSV export:

### `*_planes.csv` (the critical one)
```
Name,Strike,Dip,Dip_Dir,Cx,Cy,Cz,Nx,Ny,Nz,Sample_Radius,RMS,Gx,Gy,Gz,Length
```

| Column | What It Stores | Training Use |
|---|---|---|
| `Strike`, `Dip`, `Dip_Dir` | Joint plane orientation | ✅ **Regression targets** |
| `Cx`, `Cy`, `Cz` | Plane center (local coords) | ✅ **Where the joint is** |
| `Nx`, `Ny`, `Nz` | Plane normal vector | ✅ **Orientation features ground truth** |
| `Sample_Radius` | Search radius for fitting | Signals scale of joint surface |
| `RMS` | Plane fit error | Signal quality — low RMS = high confidence label |
| `Gx`, `Gy`, `Gz` | Global coordinates | Cross-referencing across scans |

### `*_traces.csv`
```
Name,Trace_id,Point_id,Start_x,Start_y,Start_z,End_x,End_y,End_z,Cost,Cost_Mode
```
Defines the **boundary line** of each joint trace on the rock face — used to extract which cloud points belong to a joint surface vs. the background rock.

### `*_lineations.csv`
Linear features (striations, slickenlines) — secondary validation data.

---

## 3. Input Data → Training Outcomes

### Training Inputs (what the model sees)

For each labeled joint plane in a Compass export:

```
Compass plane (center, normal, radius)
        │
        ▼
 Extract all cloud points within 1.2× radius of plane center
        │
        ▼
 Per-point features:
  ┌── Planarity  = (λ₂ - λ₃) / λ₁     [eigenvalue ratios]
  ├── Anisotropy = (λ₁ - λ₃) / λ₁
  ├── Curvature  = λ₃ / (λ₁+λ₂+λ₃)
  ├── Roughness  = RMS distance to local fit plane
  ├── Normal deviation from joint plane normal
  ├── Local point density (points per m²)
  ├── Vertical gradient (z-range / horizontal span)
  └── Surface normal consistency (k-NN normal dot products)
        │
        ▼
 Training sample: [planarity, anisotropy, curvature, roughness,
                  normal_dev, density, vert_grad, normal_consistency]
        │
 Labels: [is_joint (1.0), dip_target (α), dipdir_target (β)]
```

### Training Outcomes (what the model outputs)

| Output | Format | How We Use It |
|---|---|---|
| **Jointness score** | 0.0 – 1.0 (float) | Threshold → binary joint/not-joint mask |
| **Dip angle α** | 0–90° (regression) | Direct measurement output |
| **Dip direction β** | 0–360° (regression) | Direct measurement output |
| **Surface normal (nx,ny,nz)** | Unit vector (regression) | Alternative: compute dip from normal |

### Minimum Dataset Requirements

| Metric | Classical ML (RF) | Deep Learning (PointNet++) |
|---|---|---|
| Labeled planes | 200–500 | 1,000+ |
| Scans (different slopes) | 5–10 | 20+ |
| Expected accuracy (F1) | ~0.75–0.85 | ~0.85–0.95 |
| Training time | 2 minutes | 4–8 hours on GPU |

---

## 4. ML Strategy — Progressive Approach

### Phase 1: Classical Random Forest Baseline
```
Compass CSV labels ──► Feature extraction ──► Random Forest / XGBoost
                                                    │
                                              Binary classifier:
                                                "is joint surface?"
                                                    │
                                              Orientation regression:
                                        separate RF for dip α, dip dir β
                                                    │
                                              Output: {joint_mask, α, β}
```
**Why start here:** Works with as few as 200 labels, trains in seconds, feature importance tells us which 3D properties matter most. Likely reaches ~80% accuracy.

### Phase 2: PointNet++ Segmentation (if needed)
```
Raw XYZ + normals ──► PointNet++ segmentation head
                              │
                   ┌─────────┴──────────┐
                   │                    │
            joint/not-joint       plane normal
            (classification)      (regression: nx,ny,nz)
```
**When to upgrade:** If RF accuracy < 85% on held-out scans. PointNet++ learns directly from points — no hand-crafted features.

### Phase 3: Self-Training Loop (always-on)
```
User accepts/rejects in CC ──► Correction goes back to training set
                                        │
                                  Retrain model weekly
                                        │
                              Accuracy improves on future scans
```

---

## 5. Plugin Architecture — `qJointDetect`

### Position in the codebase
```
CloudCompare/
└── plugins/core/Standard/
    ├── qCompass/                          ← Existing: manual joint labeling
    └── qJointDetect/                      ← NEW: ML-powered detection
        ├── CMakeLists.txt
        ├── info.json
        ├── include/
        │   ├── ccJointDetect.h            ← Main plugin class
        │   ├── ccJointDetectDlg.h         ← Settings dialog
        │   ├── ccJointDetectTool.h        ← Tool that runs inference
        │   ├── JointFeatureExtractor.h    ← Eigenvalue feature extraction
        │   ├── JointMLClient.h            ← ONNX model inference wrapper
        │   └── JointOutputManager.h       ← Creates ccFitPlane results
        ├── src/
        │   ├── ccJointDetect.cpp
        │   ├── ccJointDetectDlg.cpp
        │   ├── ccJointDetectTool.cpp
        │   ├── JointFeatureExtractor.cpp
        │   ├── JointMLClient.cpp
        │   └── JointOutputManager.cpp
        ├── ui/
        │   └── jointDetectDlg.ui
        └── models/
            └── joint_rf_v1.onnx           ← Bundled trained model
```

### How it integrates with qCompass

The critical design decision: **qJointDetect outputs the SAME `ccFitPlane` objects that qCompass produces.**

This means:
- Detected joints appear in the **same DB tree** as manual Compass joints
- They are **stored with the same metadata** (Strike, Dip, DipDir, Nx, Ny, Nz, RMS, etc.)
- The **existing CSV export in qCompass automatically writes our ML results** without any modification
- Users can **visually compare** ML-detected planes next to their own manual planes
- A geologist can **accept, reject, or adjust** ML-detected joints using familiar Compass tools

### Workflow
```
1. Load point cloud in CloudCompare
2. Open qJointDetect panel
3. Press "Detect Joints"
       │
       ▼
4. Plugin runs inference:
   a. Extract local features per point
   b. Run ONNX model → joint probability per point
   c. Region-grow on high-probability points
   d. RANSAC fit plane per region
   e. Compute dip α, dip dir β, strike
       │
       ▼
5. Results appear in DB tree as ccFitPlane objects
   (same type as qCompass fit planes!)
       │
       ▼
6. Geologist:
   ├── Accepts planes → keeps as-is
   ├── Rejects false positives → removes from tree
   ├── Adjusts orientation using Compass tools
   └── Exports CSV (uses existing qCompass Export!)
       │
       ▼
7. (Optional) Corrections exported → training pipeline
   → retrain model → accuracy improves next run
```

---

## 6. Development Roadmap

| Phase | Duration | Tasks | Output | Progress |
|---|---|---|---|---|
| **Phase 0: Data Pipeline** 🟡 | Week 1 | Python script reads Compass CSVs, extracts labeled patches, saves NumPy arrays | `training_data/dataset.npz`, feature analysis notebook | 🟡 Scripts written, waiting for CSV exports |
| **Phase 1: Classic ML** ⬜ | Weeks 2–3 | Train RF, evaluate, export to ONNX | `joint_rf_v1.onnx`, accuracy report | ⬜ Ready to run when data arrives |
| **Phase 2: Plugin Skeleton** ⬜ | Weeks 4–6 | C++ plugin loads ONNX, runs inference, creates `ccFitPlane` in DB tree | Working qJointDetect plugin | ⬜ CMake + info.json scaffolded |
| **Phase 3: UI Polish** ⬜ | Week 7 | Dialog for confidence threshold, region-growing params, visual feedback | Release v0.1 | ⬜ |
| **Phase 4: Self-Training** ⬜ | Week 8 | Export corrected CC projects back to training pipeline, automatic retraining | Self-improving system | ⬜ |
| **Phase 5: Deep Learning** ⬜ | Weeks 9–12 | PointNet++ integration if needed, multi-scan generalization | Release v1.0 | ⬜ |

### Phase 0 — Data Pipeline (🟡 In Progress)

**What's done:**
- [x] `scripts/extract_training_data.py` — reads Compass `*_planes.csv`, extracts geometric features per point (planarity, anisotropy, curvature, roughness, normal consistency, vertical gradient, density), saves as `.npz`
- [x] `scripts/train_joint_model.py` — trains Random Forest classifier + orientation regressors, exports to ONNX for plugin use
- [x] `CSV_EXPORT_GUIDE.md` — step-by-step guide for geologists
- [x] `training_data/README.md` — data directory structure docs

**What's next:**
- [ ] Get Compass CSV exports from a real rock slope scan
- [ ] Run `extract_training_data.py` on real data
- [ ] Visualize feature distributions
- [ ] Tune feature extraction parameters (k-neighbors, radius multiplier)

### Phase 1 — Classic ML (⬜ Waiting for Data)

- [ ] Run `train_joint_model.py` on extracted features
- [ ] Evaluate classification accuracy (F1 score)
- [ ] Evaluate orientation regression (MAE dip, MAE dip direction)
- [ ] Check feature importance — which 3D properties matter most?
- [ ] Export ONNX model to `plugins/.../models/joint_rf_v1.onnx`

### Phase 2 — Plugin Skeleton (⬜ Waiting for Model)

- [ ] Implement `JointFeatureExtractor.cpp` — C++ port of the feature extraction
- [ ] Implement `JointMLClient.cpp` — ONNX Runtime inference wrapper
- [ ] Implement `JointOutputManager.cpp` — region growing + RANSAC on predictions
- [ ] Implement `ccJointDetectTool.cpp` — main tool UI
- [ ] Wire up CMake with ONNX Runtime
- [ ] Build and test in CloudCompare

### Phase 3 — UI Polish (⬜)

### Phase 4 — Self-Training (⬜)

### Phase 5 — Deep Learning (⬜)

---

## 7. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Training data source** | Compass CSV Export | No new labeling mechanism needed — leverages existing workflow |
| **Model format** | ONNX (Open Neural Network Exchange) | CloudCompare can load ONNX via onnxruntime — no Python dependency in plugin |
| **Result format** | `ccFitPlane` (same as Compass) | Zero friction: existing tools, export, visualization all work with detected joints |
| **ML client** | onnxruntime C++ | Lightweight, cross-platform, no Python in production |
| **Data extraction** | Python offline script | Training is a data-scientist task, not done inside CC |
| **Plugin language** | C++ (Qt + CC API) | Native plugin performance, full CC integration |

---

## 8. Files & Directories

- `PLAN.md` — This document
- `CSV_EXPORT_GUIDE.md` — Guide for geologists to export training data from CloudCompare
- `scripts/train_joint_model.py` — Python training pipeline (Phase 0–1)
- `scripts/extract_training_data.py` — Reads Compass CSVs, extracts features, saves `.npz`
- `plugins/core/Standard/qJointDetect/` — The CloudCompare plugin source
- `plugins/core/Standard/qJointDetect/models/` — Bundled ONNX models

---

## 9. Future Work (after Phase 5)

- **Multi-joint-set clustering** — automatically group detected joints into JS-1, JS-2, JS-3 based on orientation
- **Stereonet export** — generate equal-area lower-hemisphere projection directly in CC
- **Trajectory prediction** — extrapolate joint planes into the rock mass for structural modeling
- **Confidence visualization** — color-code points by jointness score for interpretability

---

*Built with qCompass — thanks to Sam Thiele for the excellent foundation in structural mapping tools.*
