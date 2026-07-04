This directory holds the training datasets extracted from Compass CSV exports.

## Structure
```
training_data/
├── raw_exports/           # Copy your Compass CSV exports here
│   └── scan_001_*.*.csv   # *_planes.csv, *_traces.csv, *_lineations.csv
├── dataset.npz           # Feature matrix and labels (generated)
└── README.md              # This file
```

## Workflow
1. Export CSVs from CloudCompare Compass → copy to `raw_exports/`
2. Run `python scripts/extract_training_data.py` to build `dataset.npz`
3. Run `python scripts/train_joint_model.py` to train model

These files are gitignored — training data is not committed to the repo.
