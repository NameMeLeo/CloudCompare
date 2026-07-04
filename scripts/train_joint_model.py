#!/usr/bin/env python3
"""
train_joint_model.py — Train ML model on Compass-labeled joint data

Trains a Random Forest classifier + regressor on geometric features extracted
from point cloud patches around labeled joint planes. Exports to ONNX for use
inside the CloudCompare plugin.

Usage:
    # 1. Extract features from Compass exports
    python scripts/extract_training_data.py \
        --cloud scans/slope_a.las \
        --planes scans/export/slope_a_planes.csv \
        --output training_data/dataset.npz

    # 2. Train model
    python scripts/train_joint_model.py \
        --dataset training_data/dataset.npz \
        --output plugins/core/Standard/qJointDetect/models/joint_rf_v1.onnx
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import json

try:
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


def load_dataset(path: str) -> tuple:
    """Load .npz dataset from extract_training_data.py."""
    data = np.load(path)
    features = data["features"]
    labels = data["labels"]
    feature_names = data["feature_names"].tolist()
    label_names = data["label_names"].tolist()
    return features, labels, feature_names, label_names


def evaluate_model(
    y_true_class: np.ndarray,
    y_pred_class: np.ndarray,
    y_true_dip: np.ndarray,
    y_pred_dip: np.ndarray,
    y_true_dipdir: np.ndarray,
    y_pred_dipdir: np.ndarray,
):
    """Print evaluation metrics."""
    print("\n📊 === Classification Report (Joint vs. Background) ===")
    print(classification_report(y_true_class, y_pred_class,
                                target_names=["Background", "Joint"]))

    # Only evaluate orientation on joint points
    joint_mask = y_true_class == 1
    if joint_mask.sum() > 0:
        mae_dip = mean_absolute_error(y_true_dip[joint_mask], y_pred_dip[joint_mask])
        mae_dipdir = mean_absolute_error(y_true_dipdir[joint_mask], y_pred_dipdir[joint_mask])
        print(f"\n📐 === Orientation Regression (on joint points only) ===")
        print(f"   Dip angle MAE:      {mae_dip:.2f}°")
        print(f"   Dip direction MAE:  {mae_dipdir:.2f}°")

        # Azimuth has wrap-around. For dip direction (0-360), also compute
        # circular MAE
        diff = np.abs(y_true_dipdir[joint_mask] - y_pred_dipdir[joint_mask])
        circular_diff = np.minimum(diff, 360 - diff)
        print(f"   Dip dir circular MAE: {circular_diff.mean():.2f}°")


def export_to_onnx(
    classifier: RandomForestClassifier,
    dip_regressor: RandomForestRegressor,
    dipdir_regressor: RandomForestRegressor,
    scaler: StandardScaler,
    n_features: int,
    output_path: str,
):
    """Export trained models to a single ONNX file (classifier only for v1)."""
    # Export classifier as ONNX (v1: classification only)
    # Orientation regression runs as a post-processing step in CC
    onnx_model = to_onnx(
        classifier,
        X=np.zeros((1, n_features), dtype=np.float32),
        target_opset=18,
    )
    onnx_path = Path(output_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # Save ONNX
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Also save scaler + regressors as joblib for the Python inference path
    aux_path = onnx_path.with_suffix(".aux.joblib")
    joblib.dump({
        "scaler": scaler,
        "dip_regressor": dip_regressor,
        "dipdir_regressor": dipdir_regressor,
        "feature_names": [
            "planarity", "anisotropy", "curvature",
            "roughness", "normal_consistency", "vert_grad", "density",
        ],
    }, aux_path)

    # Write metadata
    meta_path = onnx_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump({
            "model_type": "RandomForest",
            "n_estimators": classifier.n_estimators,
            "n_features": n_features,
            "feature_names": [
                "planarity", "anisotropy", "curvature",
                "roughness", "normal_consistency", "vert_grad", "density",
            ],
            "label_names": ["background", "joint"],
        }, f, indent=2)

    print(f"   ONNX:     {onnx_path}")
    print(f"   Auxiliary: {aux_path}")
    print(f"   Metadata:  {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Train joint detection model")
    parser.add_argument("--dataset", required=True, help="Path to .npz dataset")
    parser.add_argument("--output", default="plugins/core/Standard/qJointDetect/models/joint_rf_v1.onnx")
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=15)
    args = parser.parse_args()

    # Load data
    print(f"📂 Loading dataset: {args.dataset}")
    X, y, feature_names, label_names = load_dataset(args.dataset)
    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")

    # Split labels
    y_class = y[:, 0]  # is_joint (binary)
    y_dip = y[:, 1]    # dip angle
    y_dipdir = y[:, 2]  # dip direction

    print(f"   Joint points:  {int(y_class.sum()):,}")
    print(f"   Background:    {int((1 - y_class).sum()):,}")

    # Train/test split
    X_train, X_test, yc_train, yc_test, yd_train, yd_test, ydd_train, ydd_test = \
        train_test_split(X, y_class, y_dip, y_dipdir,
                         test_size=args.test_split,
                         stratify=y_class,
                         random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier
    print(f"\n🌲 Training Random Forest classifier ({args.n_estimators} trees)...")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train_scaled, yc_train)
    yc_pred = clf.predict(X_test_scaled)

    # Train dip regressor (only on joint points)
    print(f"📐 Training dip angle regressor...")
    joint_train_mask = yc_train == 1
    dip_reg = RandomForestRegressor(
        n_estimators=min(100, args.n_estimators),
        max_depth=args.max_depth,
        n_jobs=-1,
        random_state=42,
    )
    if joint_train_mask.sum() >= 10:
        dip_reg.fit(X_train_scaled[joint_train_mask], yd_train[joint_train_mask])
    else:
        print("⚠️  Not enough joint samples for regression training")
        dip_reg = None

    # Train dip direction regressor
    print(f"🧭 Training dip direction regressor...")
    dipdir_reg = RandomForestRegressor(
        n_estimators=min(100, args.n_estimators),
        max_depth=args.max_depth,
        n_jobs=-1,
        random_state=42,
    )
    if joint_train_mask.sum() >= 10:
        dipdir_reg.fit(
            X_train_scaled[joint_train_mask], ydd_train[joint_train_mask]
        )
    else:
        print("⚠️  Not enough joint samples for regression training")
        dipdir_reg = None

    # Evaluate
    yd_pred = dip_reg.predict(X_test_scaled) if dip_reg else np.zeros_like(yc_pred)
    ydd_pred = dipdir_reg.predict(X_test_scaled) if dipdir_reg else np.zeros_like(yc_pred)
    evaluate_model(yc_test, yc_pred, yd_test, yd_pred, ydd_test, ydd_pred)

    # Feature importance
    print(f"\n🔍 Feature Importance:")
    for name, imp in sorted(
        zip(feature_names, clf.feature_importances_),
        key=lambda x: x[1], reverse=True,
    ):
        print(f"   {name}: {imp:.3f}")

    # Export
    if dip_reg is None or dipdir_reg is None:
        print("⚠️  Skipping ONNX export: insufficient training data")
        return

    print(f"\n💾 Exporting model...")
    if HAS_ONNX:
        export_to_onnx(clf, dip_reg, dipdir_reg, scaler, X.shape[1], args.output)
    else:
        # Fallback: save as joblib pickle
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "classifier": clf,
            "dip_regressor": dip_reg,
            "dipdir_regressor": dipdir_reg,
            "scaler": scaler,
            "feature_names": feature_names,
        }, output.with_suffix(".joblib"))
        print(f"   skl2onnx not installed. Saved as {output.with_suffix('.joblib')}")
        print(f"   Install: pip install skl2onnx")

    print(f"\n✅ Done! Model ready for CloudCompare plugin.")


if __name__ == "__main__":
    main()
