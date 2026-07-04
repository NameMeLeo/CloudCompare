# CSV Export Guide for qCompass — Preparing Training Data

> *For geologists: how to export your Compass-labeled rock joints as CSV files that the ML training pipeline can use.*

---

## Quick Start

1. Open your CloudCompare project with Compass labels
2. **File → Export → Compass → CSV**
3. Choose a save location
4. Four CSV files are created:
   - `*_planes.csv` ← **This is the most important one**
   - `*_traces.csv`
   - `*_lineations.csv`
   - `*_thickness.csv`

---

## Step-by-Step Guide

### 1. Prepare Your CloudCompare Project

Before exporting, make sure:

- ✅ Your point cloud is loaded and visible
- ✅ You have **qCompass** traces drawn along joint surfaces
- ✅ You've used the **Fit Plane** tool on each trace to create `ccFitPlane` objects
- ✅ Your project tree is organized — good naming helps later

**Good:** A project labeled joint by joint where each trace has a fitted plane.
**Even better:** Multiple projects covering different rock slopes (gives the ML model more variety to learn from).

### 2. Open the Export Dialog

```
CloudCompare Menu Bar
  │
  └─ File → Export →
       └─ Compass → CSV
```

*(You can also right-click the root of your DB Tree and choose Export → Compass → CSV)*

### 3. Check the Output Files

The export creates four files:

#### ✅ `*_planes.csv` — Your Joint Labels (THE GOLD)

This is the file the ML training pipeline reads. Each row is one joint plane.

```
Name,Strike,Dip,Dip_Dir,Cx,Cy,Cz,Nx,Ny,Nz,Sample_Radius,RMS,Gx,Gy,Gz,Length
Slope1.JS-1,045,65,315,12.34,56.78,90.12,0.34,-0.56,0.75,1.50,0.023,412345,2345678,900,3.2
```

| Column | Meaning | Why ML needs it |
|---|---|---|
| `Name` | Your label for this joint | Helps track which project/scan it came from |
| `Strike` | Strike direction (°) | Ground truth for training |
| `Dip` | Dip angle (°) | **Training target: dip angle** |
| `Dip_Dir` | Dip direction azimuth (°) | **Training target: dip direction** |
| `Cx`, `Cy`, `Cz` | Plane center (local coords) | **Where the joint is in 3D space** |
| `Nx`, `Ny`, `Nz` | Plane normal unit vector | Alternative training target |
| `Sample_Radius` | Search radius used | Tells ML the scale of this joint |
| `RMS` | Fit error | Low RMS = high quality label (ML uses this as a confidence weight) |
| `Gx`, `Gy`, `Gz` | Global coordinates | For cross-referencing across scans |
| `Length` | Trace length | Secondary feature |

#### ✅ `*_traces.csv` — Joint Traces

Each row is a segment of a trace line along the joint surface.

```
Name,Trace_id,Point_id,Start_x,Start_y,Start_z,End_x,End_y,End_z,Cost,Cost_Mode
```

Used to extract **which points in the point cloud belong to a joint surface** — critical for building the per-point feature training dataset.

#### `*_lineations.csv` — Optional

```
Name,Sx,Sy,Sz,Ex,Ey,Ez,Trend,Plunge,Length
```

Additional linear features on the rock surface. Used for secondary validation.

#### `*_thickness.csv` — Optional

Like lineations but thickness is stored instead of length.

### 4. What to Export for Best ML Results

| Priority | What to Include | Why |
|---|---|---|
| ⭐ **Essential** | At least one scan with ~20+ fitted joint planes | Minimum for training a baseline model |
| ⭐ **More data** | 5–10 different rock slopes, ideally different rock types | Helps the model generalize — joints in granite look different from joints in sandstone |
| ⭐ **Variety** | Joints of different sizes (small 0.5m patches to large 10m+ surfaces) | Teaches the model multi-scale detection |
| ✅ **Good variety** | Different orientations (steep, shallow, N-facing, S-facing, etc.) | Prevents orientation bias |
| ✅ **Clean labels** | Remove any mis-fitted planes before export (check RMS — high RMS = poor fit = bad training label) | Garbage in = garbage out |
| ❌ **Don't export** | Uninterpreted scans (no traces/planes yet) | No labels = no training signal |

### 5. Organizing Multiple Scans

For best results, export each scan **separately** and keep files organized:

```bash
training_data/
├── scan_001_granite/
│   ├── scan_001_granite.bin                 # CC project
│   └── export_2026-07-04/
│       ├── scan_001_granite_planes.csv       # ← One per scan
│       ├── scan_001_granite_traces.csv
│       ├── scan_001_granite_lineations.csv
│       └── scan_001_granite_thickness.csv
├── scan_002_sandstone/
│   └── export_2026-07-04/
│       ├── scan_002_sandstone_planes.csv
│       └── ...
└── scan_003_limestone/
    └── export_2026-07-04/
        └── ...
```

### 6. Quality Tips

| ✅ Do | ❌ Don't |
|---|---|
| Include joints of varying sizes | Don't export low-confidence planes (RMS > 0.1 or whatever is excessive for your scan resolution) |
| Include scans with different rock types | Don't over-label one small section (ML needs spatial variety) |
| Ensure traces follow visible joint boundaries | Don't use traces that cut across rubble or vegetation |
| Fit planes on traces with ≥ 10 points | Don't fit planes on very short traces (< 5 points) |
| Name joints descriptively (`SlopeA_JS1_steep`, `SlopeB_bedding`) | Don't use auto-generated names only (makes tracking harder) |

### 7. After Export — What Happens Next

Once you have CSV files:

1. Copy them to the **training data folder** in the project repo:
   ```
   qJointDetect/training_data/raw_exports/
   ```

2. The **data pipeline script** extracts features from your labeled points:
   ```bash
   python scripts/extract_training_data.py \
       --clouds training_data/raw_exports/*_planes.csv \
       --output training_data/dataset.npz
   ```

3. Train the model:
   ```bash
   python scripts/train_joint_model.py \
       --dataset training_data/dataset.npz \
       --output plugins/core/Standard/qJointDetect/models/joint_rf_v1.onnx
   ```

4. The resulting ONNX model gets bundled with the CloudCompare plugin
   — then back in CC, you can click "Detect Joints" and it will find
   surfaces similar to the ones you labeled! ⛰️

---

## Notes for Collaboration With Data Scientists

- **The CSV files are human-readable** — open them in any text editor or Excel
- **Each row is one independent label** — data scientists need quantity AND quality
- **Include brief notes** on rock type, scan quality, and any interpretation challenges
- **Flag uncertain labels** — the ML model can use weight=0.5 for uncertain joints vs weight=1.0 for confident ones
- **10 good labels are worth 100 rushed ones** — quality over quantity for the first iteration

---

*Questions? Ask the development team — we can adjust the pipeline based on how your data looks!*
