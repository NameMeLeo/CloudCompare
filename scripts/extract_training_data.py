#!/usr/bin/env python3
"""
extract_training_data.py — Extract labeled point patches from Compass exports

Reads CloudCompare Compass CSV exports (*_planes.csv, *_traces.csv) and the
original point cloud (.las/.ply/.bin), then extracts point neighborhoods
around each labeled joint plane with geometric features.

Usage:
    python extract_training_data.py \\
        --cloud scans/scan_001.las \\
        --planes scans/export/scan_001_planes.csv \\
        --traces scans/export/scan_001_traces.csv \\
        --output training_data/dataset.npz
"""

import argparse
import numpy as np
import pandas as pd
import laspy
from pathlib import Path
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors


def load_point_cloud(path: str) -> np.ndarray:
    """Load XYZ from .las/.laz or .ply file."""
    ext = Path(path).suffix.lower()
    if ext in (".las", ".laz"):
        las = laspy.read(path)
        return np.column_stack([las.x, las.y, las.z])
    else:
        raise ValueError(f"Unsupported format: {ext}")


def compute_eigenvalues(xyz: np.ndarray) -> dict:
    """Compute geometric features from XYZ neighborhood."""
    cov = np.cov(xyz.T)
    eigenvalues = np.linalg.eigvalsh(cov)  # ascending: λ₀ ≤ λ₁ ≤ λ₂
    l0, l1, l2 = eigenvalues
    total = l0 + l1 + l2
    if total == 0:
        return {"planarity": 0, "anisotropy": 0, "curvature": 0}
    return {
        "planarity": (l1 - l0) / l2,    # (λ₂ - λ₃) / λ₁ → 1 = perfect plane
        "anisotropy": (l2 - l0) / l2,    # (λ₁ - λ₃) / λ₁
        "curvature": l0 / total,          # λ₃ / (λ₁+λ₂+λ₃)
    }


def extract_features(
    cloud: np.ndarray,
    labels: pd.DataFrame,
    trace_points: set[int] | None = None,
    k_neighbors: int = 30,
    radius_mult: float = 1.2,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    For each labeled plane in labels, extract the point neighborhood
    and compute features. Returns (features, is_joint_mask, metadata).
    """
    tree = KDTree(cloud)
    features_list = []
    labels_list = []
    meta_list = []

    for _, row in labels.iterrows():
        cx, cy, cz = row["Cx"], row["Cy"], row["Cz"]
        r = row["Sample_Radius"] * radius_mult
        dip = row["Dip"]
        dip_dir = row["Dip_Dir"]
        nx_, ny_, nz_ = row["Nx"], row["Ny"], row["Nz"]
        normal = np.array([nx_, ny_, nz_])

        # Find points within radius of plane center
        indices = tree.query_ball_point([cx, cy, cz], r)
        if len(indices) < 10:
            continue  # skip undersampled planes

        patch = cloud[indices]

        # Per-point features
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(patch)))
        nbrs.fit(patch)
        distances, neighbor_indices = nbrs.kneighbors(patch)

        for i, pt_idx in enumerate(indices):
            # Local neighborhood
            n_ids = neighbor_indices[i]
            local_pts = patch[n_ids]
            center = patch[i]

            # Eigenvalue features
            ev = compute_eigenvalues(local_pts - center)

            # Roughness: RMS of distances to local plane
            centroid = local_pts.mean(axis=0)
            centered = local_pts - centroid
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            local_normal = vh[2]
            distances_to_plane = np.abs((local_pts - centroid) @ local_normal)
            roughness = np.sqrt(np.mean(distances_to_plane ** 2))

            # Normal deviation
            normal_consistency = np.abs(np.dot(local_normal, normal))

            # Vertical gradient (z-range / horizontal span)
            z_range = local_pts[:, 2].max() - local_pts[:, 2].min()
            xy_span = np.linalg.norm(
                local_pts[:, :2].max(axis=0) - local_pts[:, :2].min(axis=0)
            )
            vert_grad = z_range / (xy_span + 1e-8)

            feat = [
                ev["planarity"],
                ev["anisotropy"],
                ev["curvature"],
                roughness,
                normal_consistency,
                vert_grad,
                len(local_pts),  # density
            ]
            features_list.append(feat)

            # Is this point on a joint surface?
            is_joint = 1.0 if pt_idx in (trace_points or set()) else 1.0
            labels_list.append([is_joint, dip, dip_dir, nx_, ny_, nz_])
            meta_list.append({"plane_name": row["Name"], "rms": row["RMS"]})

    return np.array(features_list), np.array(labels_list), meta_list


def load_traces(traces_path: str) -> set[int]:
    """Load trace segment endpoints to identify joint-surface points."""
    if not traces_path:
        return set()
    df = pd.read_csv(traces_path)
    # We'd need to map segment endpoints back to the point cloud indices.
    # For now: return empty set (Phase 0 placeholder).
    return set()


def main():
    parser = argparse.ArgumentParser(description="Extract training data from Compass exports")
    parser.add_argument("--cloud", required=True, help="Point cloud file (.las)")
    parser.add_argument("--planes", required=True, help="Compass planes CSV")
    parser.add_argument("--traces", default="", help="Compass traces CSV (optional)")
    parser.add_argument("--output", default="training_data/dataset.npz", help="Output .npz path")
    args = parser.parse_args()

    print(f"📂 Loading cloud: {args.cloud}")
    cloud = load_point_cloud(args.cloud)
    print(f"   {len(cloud):,} points loaded")

    print(f"📄 Loading planes: {args.planes}")
    planes_df = pd.read_csv(args.planes)
    print(f"   {len(planes_df)} joint planes found")

    trace_points = load_traces(args.traces) if args.traces else set()
    print(f"🔄 Extracting features...")
    features, labels, meta = extract_features(cloud, planes_df, trace_points)

    print(f"✅ Extracted {len(features)} training samples")
    print(f"   Feature shape: {features.shape}")
    print(f"   Label shape:   {labels.shape}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output,
        features=features,
        labels=labels,
        feature_names=[
            "planarity", "anisotropy", "curvature",
            "roughness", "normal_consistency",
            "vert_grad", "density",
        ],
        label_names=["is_joint", "dip", "dip_dir", "nx", "ny", "nz"],
    )
    print(f"💾 Saved to {args.output}")


if __name__ == "__main__":
    main()
