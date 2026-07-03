#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import argparse, os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from scipy.optimize import leastsq
from pathlib import Path
from typing import List, Dict, Tuple

### functions

def read_tiff(path: str | Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF (Z,Y,X). Got shape {arr.shape}.")
    return arr


def find_labels_and_counts(labels_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return labels and voxel counts, skipping zero-voxel labels and printing a summary."""
    labels, counts = np.unique(labels_arr, return_counts=True)
    zero_mask = counts == 0
    n_total = len(labels)
    n_zero = int(zero_mask.sum())
    n_kept = n_total - n_zero
    if n_zero > 0:
        print(f"  Found {n_total} total labels, {n_zero} with zero voxels → keeping {n_kept}")
    else:
        print(f"  Found {n_total} labels (all nonzero)")
    # Keep only those with nonzero voxels
    mask = counts > 0
    return labels[mask], counts[mask]

def fit_sphere(points_zyx: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Least-squares sphere fit on voxel coords (Z,Y,X)."""
    if points_zyx.shape[0] < 4:
        raise ValueError("Not enough points for sphere fit (need >=4)")
    x, y, z = points_zyx[:, 2], points_zyx[:, 1], points_zyx[:, 0]

    def residuals(params):
        x0, y0, z0, r = params
        return np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2) - r

    x0, y0, z0 = np.mean(x), np.mean(y), np.mean(z)
    r0 = np.mean(np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2))
    res, _ = leastsq(residuals, [x0, y0, z0, r0], maxfev=2000)
    x0, y0, z0, r = res
    rms = float(np.sqrt(np.mean(residuals(res)**2)))
    return np.array([z0, y0, x0]), float(r), rms

def measure_components(labels: np.ndarray,
                       voxel_size_xyz: Tuple[float, float, float],
                       file_id: str) -> pd.DataFrame:

    rows: List[Dict] = []
    uniq_labels, counts = find_labels_and_counts(labels)
    shape_zyx = np.array(labels.shape)

    for labv, count in zip(uniq_labels, counts):
        if labv == 0:
            continue
        pts_zyx = np.argwhere(labels == labv)
        if pts_zyx.shape[0] < 4:
            continue

        try:
            center_zyx, r_vox, rms_vox = fit_sphere(pts_zyx)
        except Exception as e:
            print(f"  Label {labv}: sphere fit failed ({e})")
            continue

        # Convert to physical units
        cen_xyz = np.array([
            center_zyx[2] * voxel_size_xyz[0],
            center_zyx[1] * voxel_size_xyz[1],
            center_zyx[0] * voxel_size_xyz[2]
        ])
        r_phys = r_vox * np.mean(voxel_size_xyz)
        diam_phys = 2 * r_vox * np.mean(voxel_size_xyz)
        print(f"  Label {labv:>4d} fitted {diam_phys:6.2f} nm")
        rows.append(dict(
            file=file_id,
            label=int(labv),
            centroid_x=cen_xyz[0],
            centroid_y=cen_xyz[1],
            centroid_z=cen_xyz[2],
            radius=r_phys,
            fit_error=rms_vox,
            fit_error_rel=rms_vox / (r_vox + 1e-12),
            diameter=diam_phys,
            n_vox=int(count)
        ))

    if not rows:
        print("  No valid vesicles after fitting.")
        return pd.DataFrame()

    # --- summary dataframe ---
    df = pd.DataFrame(rows)
    df["sphericity"] = 1.0 / (1.0 + (df["fit_error_rel"] ** 2))

    # --- classify vesicles (inner / outer_with_inner / empty) ---
    centers = df[["centroid_x", "centroid_y", "centroid_z"]].values
    radii = df["radius"].values
    inner, multi = [], []
    for i, (ci, ri) in enumerate(zip(centers, radii)):
        is_inner, contains = False, 0
        for j, (cj, rj) in enumerate(zip(centers, radii)):
            if i == j:
                continue
            d = np.linalg.norm(ci - cj)
            if d + ri < rj:
                is_inner = True
            if d + rj < ri:
                contains += 1
        inner.append(is_inner)
        multi.append(contains)

    df["is_inner"] = inner
    df["contains"] = multi
    df["class"] = np.where(df["is_inner"], "inner",
                    np.where(df["contains"] > 0, "outer_with_inner", "empty"))

    return df

def export_cmm(df: pd.DataFrame, path: str, scale: float = 10.0):
    """
    Write UCSF Chimera/ChimeraX .cmm marker file scaled by a given factor.
    Default scale=10.0 converts nm → Å.
    """
    import xml.etree.ElementTree as ET

    colors = {
        "inner": (1.0, 0.2, 0.2),            # red
        "outer_with_inner": (0.2, 0.4, 1.0), # blue
        "empty": (0.2, 0.8, 0.3),            # green
    }

    # ---------------- FITTED ----------------
    root_fit = ET.Element("marker_set", name="fitted_spheres")
    for i, r in df.iterrows():
        cls = r.get("class", "empty")
        color = colors.get(cls, (0.8, 0.8, 0.8))
        ET.SubElement(root_fit, "marker", {
            "id": str(i + 1),
            "x": f"{r['centroid_x'] * scale:.4f}",
            "y": f"{r['centroid_y'] * scale:.4f}",
            "z": f"{r['centroid_z'] * scale:.4f}",
            "r": f"{color[0]:.3f}", "g": f"{color[1]:.3f}", "b": f"{color[2]:.3f}",
            "radius": f"{r['radius'] * scale:.4f}",
            "note": f"{cls}; fitted_diam={r['diameter']:.2f} nm"
        })

    fit_path = path.replace(".cmm", "_fit.cmm")
    ET.ElementTree(root_fit).write(fit_path, encoding="utf-8", xml_declaration=True)
    print(f"Exported fitted sphere markers to {fit_path}")

def make_plots(df: pd.DataFrame, plots_dir: str):
    import matplotlib.pyplot as plt
    os.makedirs(plots_dir, exist_ok=True)

    # --- Histograms ---
    plt.figure()
    plt.hist(df["diameter"], bins=30, alpha=0.6, label="Fitted sphere", edgecolor="black")
    plt.xlabel("Diameter (nm)"); plt.ylabel("Count")
    plt.legend(); plt.title("Diameter distributions")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "diameter_fit_hist.svg"))
    plt.close()

    # --- Grouped histograms for classes ---
    groups = {
        "inner": df.loc[df["class"] == "inner"],
        "outer_with_inner": df.loc[df["class"] == "outer_with_inner"],
        "empty": df.loc[df["class"] == "empty"],
    }

    plt.figure()
    plt.hist([groups["inner"]["diameter"],
              groups["outer_with_inner"]["diameter"],
              groups["empty"]["diameter"]],
             bins=20, label=["Inner", "Outer_with_inner", "Empty"], edgecolor="black")
    plt.xlabel("Fitted diameter (nm)"); plt.ylabel("Count")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "diameter_fit_groups_hist.svg"))
    plt.close()

    # --- Pie charts (with fixed bins) ---
    bins = [0, 60, 75, 90, 105, 120, 135, np.inf]
    labels_bins = ["<60", "60-75", "75-90", "90-105", "105-120", "120-135", ">135"]

    def plot_pie_for_group(df_grp, title, outname):
        if df_grp.empty:
            return
        counts, _ = np.histogram(df_grp["diameter"].values, bins=bins)
        total = int(counts.sum())
        plt.figure(figsize=(5,5))
        plt.pie(counts, labels=labels_bins,
                autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
                startangle=90, counterclock=False)
        plt.title(f"{title} (Diameter bins, n={total})")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, outname))
        plt.close()

    plot_pie_for_group(groups["inner"], "Vesicles within another", "diameter_pie_inner.svg")
    plot_pie_for_group(groups["outer_with_inner"], "Vesicles containing others", "diameter_pie_outer_with_inner.svg")
    plot_pie_for_group(groups["empty"], "Empty (isolated) vesicles", "diameter_pie_empty.svg")

def main():
    ap = argparse.ArgumentParser(description="Fit spheres to segmented vesicle TIFFs")
    ap.add_argument("--input_dir", required=True, help="Directory with 3D TIFF segmentations")
    ap.add_argument("--pixel-size", type=str, default="1,1,1", help="Voxel size (x,y,z) in nm, comma-separated")
    ap.add_argument("--output", default="vesicles.csv", help="Output CSV summary")
    ap.add_argument("--plots", default="plots", help="Output directory for plots")
    ap.add_argument("--scale", type=float, default=10.0, help="Scaling factor for CMM export (1 = nm to Å)")
    args = ap.parse_args()

    pix_vals = [float(v) for v in args.pixel_size.split(",")]
    if len(pix_vals) == 1:
        voxel_size = (pix_vals[0], pix_vals[0], pix_vals[0])
    elif len(pix_vals) == 3:
        voxel_size = tuple(pix_vals)
    else:
        raise ValueError("Pixel size must be 1 value (isotropic) or 3 comma-separated (x,y,z).")
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.tif*")))
    if not files:
        print("No TIFF files found.")
        return

    all_df = []
    for f in files:
        name = os.path.basename(f)
        print(f"\nProcessing {name} ...")
        seg = read_tiff(f)
        print(f"The dimension of the segmentations are: {seg.shape}")
        df = measure_components(seg, voxel_size, name)
        if df.empty:
            print("No valid vesicles found.")
            continue
        base = os.path.splitext(name)[0]
        export_cmm(df, os.path.join(args.plots, f"{base}.cmm"), scale=args.scale)
        all_df.append(df)
        print(f"Found: {len(df)} vesicles")

    if not all_df:
        print("No vesicles found in any file.")
        return

    df_all = pd.concat(all_df, ignore_index=True)
    df_all.to_csv(args.output, index=False)
    print(f"\nSaved results to {args.output}")
    make_plots(df_all, args.plots)
    print(f"Plots saved to {args.plots}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
