#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python wrapper to project the central X slices of real subtomograms extracted by RELION5.0 to generate 2D particle stacks
# Useful for filaments or membranes where the X-Y box size needs to be large enough to contain the feature
# AIBVK, Bharat-Lab, MRC-LMB, Cambridge, Sep 2025

import argparse
import os
import numpy as np
import starfile
import mrcfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation as R

def get_angpix_from_optics(star_dict) -> float:
    try:
        optics_df = star_dict['optics']
    except Exception as e:
        raise KeyError("No 'data_optics' block found in STAR.") from e
    if 'rlnImagePixelSize' not in optics_df.columns:
        raise KeyError("'rlnImagePixelSize' missing in data_optics.")
    angpix = float(optics_df['rlnImagePixelSize'].iloc[0])
    if not np.isfinite(angpix) or angpix <= 0:
        raise ValueError(f"Bad rlnImagePixelSize value: {angpix}")
    return angpix


def read_volume_3d(image_name: str) -> np.ndarray:
    path = image_name.split('@')[-1]
    if not os.path.exists(path):
        raise FileNotFoundError(f"MRC not found: {path}")
    with mrcfile.open(path, permissive=True) as mrc:
        vol = np.asarray(mrc.data)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume in {path}, got {vol.shape}")
    # mrcfile gives (Z, Y, X)
    return vol.astype(np.float32)

def central_indices(z: int, n_slices: int) -> slice:
    n = max(1, min(int(n_slices), z))
    start = (z - n) // 2
    stop = start + n
    return slice(start, stop)

def projection_of_subtomogram(volume_zyx: np.ndarray, n_slices: int) -> np.ndarray:
    z, _, _ = volume_zyx.shape
    s = central_indices(z, n_slices)
    slab = volume_zyx[s]                 # (n, Y, X)
    proj = slab.sum(axis=0).astype(np.float32)

    # ZMUV like the C++ code
    m = float(proj.mean())
    v = float((proj**2).mean()) - m*m
    sdev = np.sqrt(v) if v > 0 else 0.0
    if sdev > 0.0:
        proj = (proj - m) / sdev
    else:
        proj[:] = 0.0
    return proj

def relion_euler_to_rotation(rot_deg: float, tilt_deg: float, psi_deg: float) -> np.ndarray:
    # Using 'ZYZ' with degrees=True to match RELION's convention.
    r = R.from_euler('ZYZ', [rot_deg, tilt_deg, psi_deg], degrees=True)
    return r.as_matrix()

def extra_reorient_rotation(angle_tilt_prior_deg: float) -> np.ndarray:
    r = R.from_euler('Y', angle_tilt_prior_deg, degrees=True)
    return r.as_matrix()

def rotate_volume(volume_zyx: np.ndarray, Rmat: np.ndarray, order: int = 1) -> np.ndarray:
    if volume_zyx.ndim != 3:
        raise ValueError("rotate_volume expects a 3D array (Z,Y,X)")

    # Center coordinates
    z, y, x = volume_zyx.shape
    center = np.array([(z - 1) / 2.0, (y - 1) / 2.0, (x - 1) / 2.0], dtype=np.float64)

    # Inverse rotation for affine_transform
    Rinv = Rmat.T

    # Compute offset so rotation is around the center
    # For an output voxel p_out, input voxel p_in = Rinv @ (p_out - center) + center
    # => p_in = Rinv @ p_out + (center - Rinv @ center)
    offset = center - Rinv @ center

    rotated = affine_transform(
        volume_zyx,
        matrix=Rinv,
        offset=offset,
        order=order,
        mode='constant',
        cval=0.0,
        prefilter=(order > 1),
    ).astype(np.float32)

    return rotated


def _process_one_tomo(tomo: str,
                      group_rows: list[tuple[int, str]],
                      name_to_image: dict[str, str],
                      angles_map: dict[str, tuple[float, float, float, float]],  # rot, tilt, psi, tilt_prior
                      n_slices: int,
                      out_dir: str,
                      angpix: float,
                      do_reorient: bool,
                      interp_order: int = 1) -> list[tuple[int, str]]:
    projs = []
    
    for _, pname in group_rows:
        vol = read_volume_3d(name_to_image[pname])  # (Z,Y,X)
        
        # Optional reorientation: align to Z-axis using rlnTomo Angles and then re-orient into XY plane along +X using angle prior
        if do_reorient:

            # Apply RELION Euler orientation to align helical axis along Z
            rot, tilt, psi, tilt_prior = angles_map[pname]
            R_euler = relion_euler_to_rotation(rot, tilt, psi)
            R_extra = extra_reorient_rotation(tilt_prior)
            R_total = R_extra @ R_euler
            #R_total = R_euler

            vol = rotate_volume(vol, R_total, order=interp_order)

        img = projection_of_subtomogram(vol, n_slices)
        projs.append(img)
    
    stack = np.stack(projs, axis=0).astype(np.float32)

    # Write per-tomo stack
    os.makedirs(out_dir, exist_ok=True)
    out_mrcs_file = os.path.join(out_dir, f"projections_{tomo}.mrcs")
    abs_path = os.path.abspath(out_mrcs_file)
    with mrcfile.new(abs_path, overwrite=True) as mrc_out:
        mrc_out.set_data(stack)
        mrc_out.voxel_size = (angpix, angpix, angpix)

    # Build (row_idx, rlnImageName) mapping using order within this group
    imgnames = []
    for j, (row_idx, _) in enumerate(group_rows):
        imname = f"{j+1:06d}@{abs_path}"   # 1-based index per per-tomo stack
        imgnames.append((row_idx, imname))

    return imgnames

def main():
    ap = argparse.ArgumentParser(description="Central-slab Z-projection for RELION subtomograms")
    ap.add_argument("--particles-star", required=True)
    ap.add_argument("--class2d-star", required=True)
    ap.add_argument("--angstroms", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-star", required=True)
    ap.add_argument("--n-cpus", type=int, default=None)
    ap.add_argument("--reorient", action="store_true",
                    help="Apply extra rotation using rlnAngleTiltPrior to put helical axis into XY along +X")
    ap.add_argument("--interp", type=int, default=1, choices=[0,1,2,3,4,5],
                    help="Interpolation order for 3D rotation (0=nearest,1=linear,3=cubic)")

    args = ap.parse_args()

    particles_star = starfile.read(args.particles_star)
    class2d_star = starfile.read(args.class2d_star)

    angpix = get_angpix_from_optics(particles_star)
    n_slices = max(1, int(round(args.angstroms / angpix)))
    print(f"Å/px: {angpix:.4f} -> central slices: {n_slices} (~{n_slices*angpix:.1f} Å)")


    # Group class2d rows by rlnTomoName
    cdf = class2d_star['particles'].copy()
    pdf = particles_star['particles']
    name_to_image = dict(zip(pdf['rlnTomoParticleName'].astype(str),
                         pdf['rlnImageName'].astype(str)))

    # Only read/construct angles if reorienting
    angles_map: Optional[Dict[str, Tuple[float, float, float, float]]] = None
    if args.reorient:
        required_cols = ['rlnTomoSubtomogramRot', 'rlnTomoSubtomogramTilt',
                         'rlnTomoSubtomogramPsi', 'rlnAngleTiltPrior', 'rlnTomoParticleName']
        missing = [c for c in required_cols if c not in pdf.columns]
        if missing:
            raise KeyError(f"--reorient requested but STAR is missing columns: {missing}")
        angles_map = {}
        for _, row in pdf.iterrows():
            pname = str(row['rlnTomoParticleName'])
            rot = float(row['rlnTomoSubtomogramRot'])
            tilt = float(row['rlnTomoSubtomogramTilt'])
            psi = float(row['rlnTomoSubtomogramPsi'])
            tilt_prior = float(row['rlnAngleTiltPrior'])
            angles_map[pname] = (rot, tilt, psi, tilt_prior)

    groups = []
    for tomo, group in cdf.groupby('rlnTomoName', sort=False):
        rows = [(idx, str(group.at[idx, 'rlnTomoParticleName'])) for idx in group.index]
        groups.append((tomo, rows))

    # Parallel per-tomogram processing
    n_workers = args.n_cpus or os.cpu_count() or 1
    updates: List[Tuple[int, str]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(
                _process_one_tomo,
                tomo, rows, name_to_image, angles_map,
                n_slices, args.out_dir, angpix,
                args.reorient, args.interp
            )
            for tomo, rows in groups
        ]
        for fut in as_completed(futures):
            updates.extend(fut.result())

    # Apply updates to star (preserving original row indices)
    for idx, imgname in updates:
        cdf.at[idx, 'rlnImageName'] = imgname

    class2d_star['particles'] = cdf
    os.makedirs(os.path.dirname(os.path.abspath(args.out_star)) or ".", exist_ok=True)
    starfile.write(class2d_star, args.out_star, overwrite=True)

    print(f"Wrote stacks to: {os.path.abspath(args.out_dir)}")
    print(f"Wrote updated star: {os.path.abspath(args.out_star)}")

if __name__ == "__main__":
    main()
