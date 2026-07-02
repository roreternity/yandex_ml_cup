#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


CAMERA_NAMES = ("front", "left_fwd", "left_bwd", "right_fwd", "right_bwd", "rear")


@dataclass
class RunConfig:
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    temporal_strength: float = 0.72
    rife_mix: float = 0.42
    use_rife: bool = True
    lidar_mix: float = 0.30
    lidar_conf_norm: float = 0.65
    jpeg_quality: int = 95
    rife_root: Optional[Path] = None
    target_z_tol: float = 0.35
    source_z_tol: float = 0.45
    lidar_inpaint: bool = False
    lidar_fill_strength: float = 0.65
    apply_distortion: bool = True
    dense_ibr: bool = False
    dense_mix: float = 0.35
    dense_conf_norm: float = 0.85
    dense_max_fill_dist: float = 36.0
    dense_source_z_tol: float = 0.22


def imread_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def imwrite_rgb(path: Path, arr: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(
        path, quality=quality, subsampling=0
    )


def load_json(path: Path) -> dict:
    with path.open("r") as fh:
        return json.load(fh)


def target_fraction(meta: dict) -> float:
    ts = meta.get("timestamps_ns")
    if not ts:
        return 0.5
    span = ts["t1"] - ts["t0"]
    if span == 0:
        return 0.5
    return float((ts["target"] - ts["t0"]) / span)


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    delta = pred.astype(np.float32) - gt.astype(np.float32)
    mse = float(np.mean(delta * delta))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def score_from_psnr(value: float) -> float:
    return (min(30.0, max(10.0, value)) - 10.0) * 5.0


def remap_image(img: np.ndarray, flow_xy: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xx + scale * flow_xy[..., 0].astype(np.float32)
    map_y = yy + scale * flow_xy[..., 1].astype(np.float32)
    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def flow_dis(a: np.ndarray, b: np.ndarray, preset: int) -> np.ndarray:
    gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    engine = cv2.DISOpticalFlow_create(preset)
    return engine.calc(gray_a, gray_b, None)


def normalized_gray_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    err = cv2.GaussianBlur(np.abs(ga - gb), (0, 0), 2.0)
    return np.clip(1.0 - err / 42.0, 0.0, 1.0)


def temporal_with_masks(img0: np.ndarray, img1: np.ndarray, alpha: float, cfg: RunConfig) -> np.ndarray:
    mean_img = (1.0 - alpha) * img0.astype(np.float32) + alpha * img1.astype(np.float32)
    f01 = flow_dis(img0, img1, cfg.dis_preset)
    f10 = flow_dis(img1, img0, cfg.dis_preset)

    mid_from_0 = remap_image(img0, f01, alpha)
    mid_from_1 = remap_image(img1, f10, 1.0 - alpha)

    # Forward/backward agreement mask: unreliable near disocclusions and fast objects.
    back_to_0 = remap_image(img1, f01, 1.0)
    back_to_1 = remap_image(img0, f10, 1.0)
    conf0 = normalized_gray_error(img0, back_to_0)
    conf1 = normalized_gray_error(img1, back_to_1)

    temporal = ((1.0 - alpha) * mid_from_0.astype(np.float32) * conf0[..., None])
    temporal += alpha * mid_from_1.astype(np.float32) * conf1[..., None]
    denom = (1.0 - alpha) * conf0 + alpha * conf1
    temporal = temporal / np.maximum(denom[..., None], 1e-3)

    confidence = np.clip(denom, 0.0, 1.0)[..., None]
    out = cfg.temporal_strength * temporal + (1.0 - cfg.temporal_strength) * mean_img
    out = confidence * out + (1.0 - confidence) * mean_img
    return np.clip(out, 0, 255).astype(np.uint8)


_RIFE_MODEL = None
_RIFE_DEVICE = None
DEFAULT_RIFE_ROOT = Path(os.environ.get(
    "RIFE_ROOT",
    Path(__file__).resolve().parents[1] / "task_b_baseline" / "baseline_files" / "baseline_ensemble",
))


def try_rife(img0: np.ndarray, img1: np.ndarray, alpha: float, rife_root: Optional[Path] = None) -> Optional[np.ndarray]:
    """Run RIFE on the same device as the loaded RIFE weights.

    The baseline RIFE file may choose MPS on Mac by itself. If inputs are left on CPU,
    PyTorch crashes with: input(device='cpu') and weight(device='mps:0').
    """
    global _RIFE_MODEL, _RIFE_DEVICE

    root = Path(rife_root) if rife_root is not None else DEFAULT_RIFE_ROOT
    rife_src = root / "ECCV2022-RIFE"

    if not root.exists():
        print(f"[WARN] RIFE root not found: {root}", file=sys.stderr)
        return None

    try:
        import torch
        import torch.nn.functional as F

        sys.path.insert(0, str(root))
        if rife_src.exists():
            sys.path.insert(0, str(rife_src))

        from train_log.RIFE_HDv3 import Model
    except Exception as exc:
        print(f"[WARN] RIFE import failed: {exc}", file=sys.stderr)
        return None

    try:
        if _RIFE_MODEL is None:
            model = Model()
            model.load_model(str(root / "train_log"), -1)
            model.eval()
            _RIFE_MODEL = model

            # Detect the real device of RIFE weights. On your Mac it is usually mps:0.
            try:
                _RIFE_DEVICE = next(model.flownet.parameters()).device
            except Exception:
                _RIFE_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

            print(f"[INFO] RIFE loaded on {_RIFE_DEVICE}", flush=True)

        model = _RIFE_MODEL
        model_device = _RIFE_DEVICE
    except Exception as exc:
        print(f"[WARN] RIFE load failed: {exc}", file=sys.stderr)
        return None

    h, w = img0.shape[:2]
    ph = ((h + 63) // 64) * 64
    pw = ((w + 63) // 64) * 64

    def pack(x: np.ndarray):
        ten = torch.from_numpy(np.ascontiguousarray(x).copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        ten = F.pad(ten, (0, pw - w, 0, ph - h))
        return ten.to(model_device)

    try:
        with torch.no_grad():
            pred = model.inference(pack(img0), pack(img1), timestep=float(alpha))
        arr = pred[0, :, :h, :w].permute(1, 2, 0).detach().cpu().numpy()
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    except RuntimeError as exc:
        print(f"[WARN] RIFE inference failed, fallback to temporal only: {exc}", file=sys.stderr)
        return None

def world_to_camera(xyz: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    w2c = np.linalg.inv(c2w.astype(np.float64))
    return xyz.astype(np.float64) @ w2c[:3, :3].T + w2c[:3, 3]


def _camera_matrix(intr: dict) -> np.ndarray:
    return np.array(
        [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _distortion_coeffs(intr: dict) -> np.ndarray:
    coeffs = intr.get("distortion_coeffs") or []
    if len(coeffs) == 0:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(coeffs, dtype=np.float64).reshape(-1)


def project_camera_points(xyz_cam: np.ndarray, intr: dict, apply_distortion: bool = True):
    """Project camera-frame OpenCV points to image pixels.

    If distortion_coeffs are present, use cv2.projectPoints. For PINHOLE / empty
    distortion this falls back to the faster direct pinhole formula.
    """
    xyz_cam = xyz_cam.astype(np.float64, copy=False)
    z = xyz_cam[:, 2]
    coeffs = _distortion_coeffs(intr)

    if apply_distortion and coeffs.size > 0 and np.any(np.abs(coeffs) > 1e-12):
        K = _camera_matrix(intr)
        pts2d, _ = cv2.projectPoints(
            xyz_cam.reshape(-1, 1, 3),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            K,
            coeffs,
        )
        u = pts2d[:, 0, 0]
        v = pts2d[:, 0, 1]
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            u = intr["fx"] * xyz_cam[:, 0] / z + intr["cx"]
            v = intr["fy"] * xyz_cam[:, 1] / z + intr["cy"]
    return u, v, z


def project(xyz: np.ndarray, c2w: np.ndarray, intr: dict, apply_distortion: bool = True):
    return project_camera_points(world_to_camera(xyz, c2w), intr, apply_distortion=apply_distortion)


def sample_rgb_bilinear(rgb: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized bilinear RGB sampling at floating point pixel coordinates.

    cv2.remap cannot handle maps with more than 32767 rows/cols on some builds.
    LiDAR can easily produce hundreds of thousands of samples, so do the
    interpolation manually instead of using a tall Nx1 remap map.
    """
    h, w = rgb.shape[:2]
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    # Border-replicate behavior for coordinates very close to the image edge.
    u = np.clip(u, 0.0, float(w - 1))
    v = np.clip(v, 0.0, float(h - 1))

    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    wx = (u - x0).astype(np.float32)
    wy = (v - y0).astype(np.float32)

    rgb_f = rgb.astype(np.float32, copy=False)
    c00 = rgb_f[y0, x0]
    c10 = rgb_f[y0, x1]
    c01 = rgb_f[y1, x0]
    c11 = rgb_f[y1, x1]

    top = c00 * (1.0 - wx)[:, None] + c10 * wx[:, None]
    bottom = c01 * (1.0 - wx)[:, None] + c11 * wx[:, None]
    return top * (1.0 - wy)[:, None] + bottom * wy[:, None]



def backproject_pixels_to_world(xx: np.ndarray, yy: np.ndarray, depth: np.ndarray,
                                c2w: np.ndarray, intr: dict,
                                apply_distortion: bool = True) -> np.ndarray:
    """Backproject target pixels with depth to world coordinates.

    For distorted cameras this uses cv2.undistortPoints to invert the distortion
    model before scaling rays by depth. For PINHOLE / empty coefficients it uses
    the standard OpenCV camera coordinates: x-right, y-down, z-forward.
    """
    z = depth.reshape(-1).astype(np.float64)
    x_pix = xx.reshape(-1).astype(np.float64)
    y_pix = yy.reshape(-1).astype(np.float64)
    coeffs = _distortion_coeffs(intr)

    if apply_distortion and coeffs.size > 0 and np.any(np.abs(coeffs) > 1e-12):
        K = _camera_matrix(intr)
        pts = np.stack([x_pix, y_pix], axis=1).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, K, coeffs).reshape(-1, 2)
        x_cam = und[:, 0] * z
        y_cam = und[:, 1] * z
    else:
        x_cam = (x_pix - intr["cx"]) / intr["fx"] * z
        y_cam = (y_pix - intr["cy"]) / intr["fy"] * z

    xyz_cam = np.stack([x_cam, y_cam, z], axis=1)
    R = c2w[:3, :3].astype(np.float64)
    t = c2w[:3, 3].astype(np.float64)
    return xyz_cam @ R.T + t


def densify_lidar_depth(lidar_xyz: np.ndarray, target_pose: np.ndarray, target_intr: dict,
                        cfg: RunConfig) -> tuple[np.ndarray, np.ndarray]:
    """Project lidar to target view and make an approximate dense depth map.

    The output confidence is high at real LiDAR pixels and fades with the 2D
    distance to the nearest LiDAR pixel. This avoids blindly filling sky/far
    holes with arbitrary nearest-neighbor depths.
    """
    width = int(target_intr["width"])
    height = int(target_intr["height"])
    sparse = np.zeros((height, width), np.float32)

    u, v, z = project(lidar_xyz, target_pose, target_intr, apply_distortion=cfg.apply_distortion)
    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(z) & (z > 0.5)
    if not np.any(finite):
        return sparse, np.zeros_like(sparse)

    uu = np.rint(u[finite]).astype(np.int32)
    vv = np.rint(v[finite]).astype(np.int32)
    zz = z[finite].astype(np.float32)
    inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
    if not np.any(inside):
        return sparse, np.zeros_like(sparse)

    pix = vv[inside] * width + uu[inside]
    zvals = zz[inside]
    zbuf = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(zbuf, pix, zvals)
    sparse_flat = sparse.reshape(-1)
    hit = np.isfinite(zbuf)
    sparse_flat[hit] = zbuf[hit]

    mask = sparse > 0
    if not np.any(mask):
        return sparse, np.zeros_like(sparse)

    # Nearest-neighbor fill. Prefer SciPy when available; otherwise use OpenCV
    # distanceTransformWithLabels, which is available in normal opencv-python.
    try:
        from scipy.ndimage import distance_transform_edt
        dist, inds = distance_transform_edt(~mask, return_indices=True)
        dense = sparse[tuple(inds)].astype(np.float32)
    except Exception:
        src = np.where(mask, 0, 255).astype(np.uint8)
        dist, labels = cv2.distanceTransformWithLabels(
            src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
        )
        ys, xs = np.where(mask)
        # OpenCV labels are 1-based; for DIST_LABEL_PIXEL they follow zero-pixel order.
        idx = np.clip(labels.astype(np.int64) - 1, 0, len(xs) - 1)
        dense = sparse[ys[idx], xs[idx]].astype(np.float32)

    # Keep real LiDAR pixels at full confidence. Filled pixels decay by distance.
    conf = np.exp(-dist.astype(np.float32) / max(cfg.dense_max_fill_dist, 1e-3)).astype(np.float32)
    conf[dist > cfg.dense_max_fill_dist] = 0.0
    conf[mask] = 1.0

    # Gentle smoothing of depth only where it was filled, not at raw LiDAR hits.
    dense_blur = cv2.GaussianBlur(dense, (5, 5), 0)
    dense = np.where(mask, sparse, dense_blur).astype(np.float32)
    return dense, conf


def render_dense_ibr(sample_dir: Path, meta: dict, temporal: np.ndarray, cfg: RunConfig) -> tuple[np.ndarray, np.ndarray]:
    """Dense image-based rendering from LiDAR-filled target depth.

    For target pixels with plausible depth, backproject to 3D, project to all
    source cameras at t0/t1, bilinearly sample colors, and fuse them with
    time/view/crop/depth confidence. Returns RGB and confidence map.
    """
    target_cam = meta["target_camera"]
    target_intr = meta["intrinsics"][target_cam]
    width = int(target_intr["width"])
    height = int(target_intr["height"])
    alpha = target_fraction(meta)

    lidar_xyz = np.load(sample_dir / "input" / "lidar.npz")["xyz"]
    target_pose = np.asarray(meta["poses_c2w"]["target"][target_cam], dtype=np.float64)
    depth, depth_conf = densify_lidar_depth(lidar_xyz, target_pose, target_intr, cfg)

    has_depth = (depth > 0.5) & (depth_conf > 1e-4)
    if not np.any(has_depth):
        return np.zeros((height, width, 3), np.uint8), np.zeros((height, width), np.float32)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    flat_valid = np.flatnonzero(has_depth.reshape(-1))
    xyz_world_all = backproject_pixels_to_world(xx, yy, depth, target_pose, target_intr, cfg.apply_distortion)
    xyz_world = xyz_world_all[flat_valid]
    depth_conf_flat = depth_conf.reshape(-1)[flat_valid].astype(np.float32)

    target_center = target_pose[:3, 3]
    target_ray = xyz_world.astype(np.float64) - target_center
    target_ray /= np.linalg.norm(target_ray, axis=1, keepdims=True) + 1e-9

    color_acc = np.zeros((height * width, 3), np.float32)
    weight_acc = np.zeros(height * width, np.float32)
    time_weights = {"t0": 1.0 - alpha, "t1": alpha}

    for moment, tw in time_weights.items():
        if tw <= 1e-6:
            continue
        for cam in CAMERA_NAMES:
            intr = meta["intrinsics"][cam]
            pose = np.asarray(meta["poses_c2w"][moment][cam], dtype=np.float64)
            u, v, src_depth = project(xyz_world, pose, intr, apply_distortion=cfg.apply_distortion)
            sw, sh = int(intr["width"]), int(intr["height"])

            finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(src_depth) & (src_depth > 0.5)
            if not np.any(finite):
                continue
            uu = np.rint(u[finite]).astype(np.int32)
            vv = np.rint(v[finite]).astype(np.int32)
            inside0 = (uu >= 0) & (uu < sw) & (vv >= 0) & (vv < sh)
            if not np.any(inside0):
                continue

            finite_idx = np.flatnonzero(finite)
            cand = finite_idx[inside0]
            uu = uu[inside0]
            vv = vv[inside0]
            dz = src_depth[cand]
            pp = vv * sw + uu

            # Source-view z-buffer based on the dense target points projected into this source.
            src_zbuf = np.full(sh * sw, np.inf, dtype=np.float64)
            np.minimum.at(src_zbuf, pp, dz)
            visible = dz <= src_zbuf[pp] + cfg.dense_source_z_tol
            if not np.any(visible):
                continue
            local = cand[visible]

            rgb = imread_rgb(sample_dir / "input" / moment / f"{cam}.jpg")
            sampled = sample_rgb_bilinear(rgb, u[local], v[local])

            source_center = pose[:3, 3]
            source_ray = xyz_world[local].astype(np.float64) - source_center
            source_ray /= np.linalg.norm(source_ray, axis=1, keepdims=True) + 1e-9
            ray_agreement = np.clip(np.sum(source_ray * target_ray[local], axis=1), 0.0, 1.0)

            same_bonus = 1.80 if cam == target_cam else 0.90
            center_u = np.abs((u[local] - intr["cx"]) / max(intr["cx"], 1.0))
            center_v = np.abs((v[local] - intr["cy"]) / max(intr["cy"], 1.0))
            crop_quality = np.clip(1.0 - 0.35 * np.maximum(center_u, center_v), 0.35, 1.0)
            dist_weight = 1.0 / np.sqrt(np.maximum(src_depth[local], 1.0))
            w_arr = (tw * (0.15 + ray_agreement * ray_agreement) * same_bonus * crop_quality * dist_weight * depth_conf_flat[local]).astype(np.float32)

            dst = flat_valid[local]
            np.add.at(color_acc, dst, sampled * w_arr[:, None])
            np.add.at(weight_acc, dst, w_arr)

    out = np.zeros((height * width, 3), np.uint8)
    good = weight_acc > 1e-5
    out[good] = np.clip(color_acc[good] / weight_acc[good, None], 0, 255).astype(np.uint8)

    # Confidence combines geometry support and view-fusion support. Normalize softly.
    conf = np.zeros(height * width, np.float32)
    conf[good] = np.clip(weight_acc[good] / max(cfg.dense_conf_norm, 1e-6), 0.0, 1.0)
    return out.reshape(height, width, 3), conf.reshape(height, width)

def lidar_paint(sample_dir: Path, meta: dict, cfg: RunConfig) -> tuple[np.ndarray, np.ndarray]:
    target_cam = meta["target_camera"]
    target_intr = meta["intrinsics"][target_cam]
    width = int(target_intr["width"])
    height = int(target_intr["height"])
    xyz = np.load(sample_dir / "input" / "lidar.npz")["xyz"]

    target_pose = np.asarray(meta["poses_c2w"]["target"][target_cam], dtype=np.float64)
    ut, vt, zt = project(xyz, target_pose, target_intr, apply_distortion=cfg.apply_distortion)

    # Rounding can push coordinates just outside the image, so validate after rounding.
    finite = np.isfinite(ut) & np.isfinite(vt) & np.isfinite(zt) & (zt > 0.5)
    ui = np.full_like(zt, -1, dtype=np.int32)
    vi = np.full_like(zt, -1, dtype=np.int32)
    safe = finite & (ut > -1e6) & (ut < 1e6) & (vt > -1e6) & (vt < 1e6)
    ui[safe] = np.rint(ut[safe]).astype(np.int32)
    vi[safe] = np.rint(vt[safe]).astype(np.int32)
    valid = safe & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    if not np.any(valid):
        return np.zeros((height, width, 3), np.uint8), np.zeros((height, width), np.float32)

    pix = vi[valid] * width + ui[valid]
    z = zt[valid]
    zbuf = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(zbuf, pix, z)
    visible = z <= zbuf[pix] + cfg.target_z_tol
    chosen = np.flatnonzero(valid)[visible]
    if chosen.size == 0:
        return np.zeros((height, width, 3), np.uint8), np.zeros((height, width), np.float32)

    xyz_front = xyz[chosen]
    target_pix = vi[chosen] * width + ui[chosen]
    target_center = target_pose[:3, 3]
    target_ray = xyz_front.astype(np.float64) - target_center
    target_ray /= np.linalg.norm(target_ray, axis=1, keepdims=True) + 1e-9

    color_acc = np.zeros((height * width, 3), np.float32)
    weight_acc = np.zeros(height * width, np.float32)

    for moment in ("t0", "t1"):
        for cam in CAMERA_NAMES:
            intr = meta["intrinsics"][cam]
            pose = np.asarray(meta["poses_c2w"][moment][cam], dtype=np.float64)
            u, v, depth = project(xyz_front, pose, intr, apply_distortion=cfg.apply_distortion)
            sw, sh = int(intr["width"]), int(intr["height"])
            finite_src = np.isfinite(u) & np.isfinite(v) & np.isfinite(depth) & (depth > 0.5)
            uu_all = np.full_like(depth, -1, dtype=np.int32)
            vv_all = np.full_like(depth, -1, dtype=np.int32)
            safe_src = finite_src & (u > -1e6) & (u < 1e6) & (v > -1e6) & (v < 1e6)
            uu_all[safe_src] = np.rint(u[safe_src]).astype(np.int32)
            vv_all[safe_src] = np.rint(v[safe_src]).astype(np.int32)
            inside = safe_src & (uu_all >= 0) & (uu_all < sw) & (vv_all >= 0) & (vv_all < sh)
            if not np.any(inside):
                continue

            uu = uu_all[inside]
            vv = vv_all[inside]
            pp = vv * sw + uu
            dz = depth[inside]
            src_zbuf = np.full(sh * sw, np.inf, np.float64)
            np.minimum.at(src_zbuf, pp, dz)
            local_inside = np.flatnonzero(inside)
            good = dz <= src_zbuf[pp] + cfg.source_z_tol
            local = local_inside[good]
            if local.size == 0:
                continue

            rgb = imread_rgb(sample_dir / "input" / moment / f"{cam}.jpg")
            sampled = sample_rgb_bilinear(rgb, u[local], v[local])

            source_center = pose[:3, 3]
            source_ray = xyz_front[local].astype(np.float64) - source_center
            source_ray /= np.linalg.norm(source_ray, axis=1, keepdims=True) + 1e-9
            ray_agreement = np.clip(np.sum(source_ray * target_ray[local], axis=1), 0.0, 1.0)
            same_camera_bonus = 2.00 if cam == target_cam else 0.80
            center_u = np.abs((u[local] - intr["cx"]) / max(intr["cx"], 1.0))
            center_v = np.abs((v[local] - intr["cy"]) / max(intr["cy"], 1.0))
            crop_quality = np.clip(1.0 - 0.30 * np.maximum(center_u, center_v), 0.40, 1.0)
            dist_weight = 1.0 / np.sqrt(np.maximum(zt[chosen][local], 1.0))
            weights = ((0.10 + ray_agreement * ray_agreement) * same_camera_bonus * crop_quality * dist_weight).astype(np.float32)

            dst = target_pix[local]
            np.add.at(color_acc, dst, sampled * weights[:, None])
            np.add.at(weight_acc, dst, weights)

    mask = weight_acc > 1e-5
    painted = np.zeros((height * width, 3), np.uint8)
    painted[mask] = np.clip(color_acc[mask] / weight_acc[mask, None], 0, 255).astype(np.uint8)
    return painted.reshape(height, width, 3), weight_acc.reshape(height, width)


def fill_lidar_holes(lidar_rgb: np.ndarray, lidar_conf: np.ndarray, strength: float) -> tuple[np.ndarray, np.ndarray]:
    """Softly fill small holes in lidar painting.

    Disabled by default because it changes more pixels. Enable with --lidar-inpaint
    only after local validation.
    """
    mask = (lidar_conf > 1e-5).astype(np.uint8)
    if not np.any(mask):
        return lidar_rgb, lidar_conf

    mask255 = mask * 255
    inv_mask = (1 - mask) * 255
    filled = cv2.inpaint(lidar_rgb.astype(np.uint8), inv_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)

    kernel = np.ones((3, 3), np.uint8)
    near = cv2.dilate(mask, kernel, iterations=1).astype(bool)
    out_rgb = lidar_rgb.astype(np.float32)
    use = near & (mask == 0)
    out_rgb[use] = (1.0 - strength) * out_rgb[use] + strength * filled.astype(np.float32)[use]

    conf_blur = cv2.GaussianBlur(lidar_conf.astype(np.float32), (3, 3), 0)
    out_conf = np.maximum(lidar_conf.astype(np.float32), conf_blur * strength)
    return np.clip(out_rgb, 0, 255).astype(np.uint8), out_conf


def camera_lidar_mix(camera: str, base: float) -> float:
    return {
        "front": base * 1.00,
        "left_fwd": base * 1.05,
        "left_bwd": base * 1.35,
        "right_fwd": base * 1.10,
        "right_bwd": base * 1.35,
        "rear": base * 0.35,
    }.get(camera, base)

def render_one(sample_dir: Path, cfg: RunConfig) -> np.ndarray:
    meta = load_json(sample_dir / "meta.json")
    cam = meta["target_camera"]
    alpha = target_fraction(meta)
    img0 = imread_rgb(sample_dir / "input" / "t0" / f"{cam}.jpg")
    img1 = imread_rgb(sample_dir / "input" / "t1" / f"{cam}.jpg")

    temporal = temporal_with_masks(img0, img1, alpha, cfg)
    if cfg.use_rife:
        rife = try_rife(img0, img1, alpha, cfg.rife_root)
        if rife is not None:
            temporal = np.clip(
                (1.0 - cfg.rife_mix) * temporal.astype(np.float32) + cfg.rife_mix * rife.astype(np.float32),
                0,
                255,
            ).astype(np.uint8)

    base = temporal.astype(np.float32)

    if cfg.dense_ibr:
        dense_rgb, dense_conf = render_dense_ibr(sample_dir, meta, temporal, cfg)
        alpha_dense = cfg.dense_mix * np.clip(dense_conf / max(cfg.dense_conf_norm, 1e-6), 0.0, 1.0)
        base = (1.0 - alpha_dense[..., None]) * base + alpha_dense[..., None] * dense_rgb.astype(np.float32)

    lidar_rgb, lidar_conf = lidar_paint(sample_dir, meta, cfg)
    if cfg.lidar_inpaint:
        lidar_rgb, lidar_conf = fill_lidar_holes(lidar_rgb, lidar_conf, cfg.lidar_fill_strength)
    mix = camera_lidar_mix(cam, cfg.lidar_mix)
    alpha_lidar = mix * np.clip(lidar_conf / cfg.lidar_conf_norm, 0.0, 1.0)
    out = (1.0 - alpha_lidar[..., None]) * base
    out += alpha_lidar[..., None] * lidar_rgb.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def validate(args: argparse.Namespace) -> None:
    cfg = make_config(args)
    samples = sorted((args.dataset / "train").iterdir())
    samples = [p for p in samples if p.is_dir()]
    if args.limit:
        samples = samples[: args.limit]

    per_cam = {c: [] for c in CAMERA_NAMES}
    scores = []
    for sample in tqdm(samples, desc="validate"):
        pred = render_one(sample, cfg)
        meta = load_json(sample / "meta.json")
        cam = meta["target_camera"]
        gt = imread_rgb(sample / "target" / f"{cam}.jpg")
        value = psnr(pred, gt)
        per_cam[cam].append(value)
        scores.append(score_from_psnr(value))

    all_psnr = [v for vals in per_cam.values() for v in vals]
    print(f"samples={len(all_psnr)} mean_psnr={np.mean(all_psnr):.4f} mean_score={np.mean(scores):.4f}")
    for cam, vals in per_cam.items():
        if vals:
            print(f"{cam:9s} n={len(vals):3d} psnr={np.mean(vals):.4f}")


def submit(args: argparse.Namespace) -> None:
    cfg = make_config(args)
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    samples = sorted(p for p in (args.dataset / "test").iterdir() if p.is_dir())
    for sample in tqdm(samples, desc="submit"):
        pred = render_one(sample, cfg)
        imwrite_rgb(args.output / sample.name / "pred.jpg", pred, cfg.jpeg_quality)

    if args.zip_path.exists():
        args.zip_path.unlink()
    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for pred_path in sorted(args.output.rglob("pred.jpg")):
            zf.write(pred_path, pred_path.relative_to(args.output.parent))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.zip_path}")


def make_config(args: argparse.Namespace) -> RunConfig:
    preset = {
        "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }[args.preset]
    return RunConfig(
        dis_preset=preset,
        temporal_strength=args.temporal_strength,
        rife_mix=args.rife_mix,
        use_rife=not args.no_rife,
        lidar_mix=args.lidar_mix,
        lidar_conf_norm=args.lidar_conf_norm,
        jpeg_quality=args.quality,
        rife_root=args.rife_root,
        target_z_tol=args.target_z_tol,
        source_z_tol=args.source_z_tol,
        lidar_inpaint=args.lidar_inpaint,
        lidar_fill_strength=args.lidar_fill_strength,
        apply_distortion=not args.no_distortion,
        dense_ibr=args.dense_ibr,
        dense_mix=args.dense_mix,
        dense_conf_norm=args.dense_conf_norm,
        dense_max_fill_dist=args.dense_max_fill_dist,
        dense_source_z_tol=args.dense_source_z_tol,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--preset", choices=("ultrafast", "fast", "medium"), default="medium")
    parser.add_argument("--temporal-strength", type=float, default=0.72)
    parser.add_argument("--rife-mix", type=float, default=0.42)
    parser.add_argument("--no-rife", action="store_true")
    parser.add_argument("--lidar-mix", type=float, default=0.30)
    parser.add_argument("--lidar-conf-norm", type=float, default=0.65)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--rife-root", type=Path, default=None,
                        help="Path to baseline_ensemble. Defaults to RIFE_ROOT env or ../task_b_baseline/baseline_files/baseline_ensemble")
    parser.add_argument("--target-z-tol", type=float, default=0.35,
                        help="Target-view z-buffer tolerance in meters for lidar visibility")
    parser.add_argument("--source-z-tol", type=float, default=0.45,
                        help="Source-view z-buffer tolerance in meters for lidar color sampling")
    parser.add_argument("--lidar-inpaint", action="store_true",
                        help="Softly inpaint tiny holes in lidar painting; validate before using for submit")
    parser.add_argument("--lidar-fill-strength", type=float, default=0.65)
    parser.add_argument("--no-distortion", action="store_true",
                        help="Ignore distortion_coeffs and use pinhole projection only; useful if validation shows distortion is already baked into images")
    parser.add_argument("--dense-ibr", action="store_true",
                        help="Experimental: dense LiDAR depth IBR layer before sparse lidar_paint")
    parser.add_argument("--dense-mix", type=float, default=0.35,
                        help="Strength of dense IBR layer when --dense-ibr is enabled")
    parser.add_argument("--dense-conf-norm", type=float, default=0.85,
                        help="Confidence normalization for dense IBR fusion")
    parser.add_argument("--dense-max-fill-dist", type=float, default=36.0,
                        help="Max pixel distance from sparse LiDAR for dense depth filling")
    parser.add_argument("--dense-source-z-tol", type=float, default=0.22,
                        help="Source z-buffer tolerance for dense IBR")

    sub = parser.add_subparsers(dest="cmd", required=True)
    p_val = sub.add_parser("validate")
    p_val.add_argument("--limit", type=int, default=0)
    p_val.set_defaults(func=validate)

    p_sub = sub.add_parser("submit")
    p_sub.add_argument("--output", type=Path, default=Path("submission"))
    p_sub.add_argument("--zip-path", type=Path, default=Path("submission.zip"))
    p_sub.set_defaults(func=submit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
