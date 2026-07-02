import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RIFE_DIR = ROOT / "task_b_baseline" / "baseline_files" / "baseline_ensemble"
sys.path.insert(0, str(RIFE_DIR))

from train_log.RIFE_HDv3 import Model, device  # noqa: E402

DEFAULT_WEIGHTS = (0.55, 0.35)

CAMERA_WEIGHTS = {
    "front": (0.55, 0.35),
    "left_bwd": (0.50, 0.35),
    "left_fwd": (0.65, 0.35),
    "rear": (0.45, 0.35),
    "right_bwd": (0.45, 0.35),
    "right_fwd": (0.50, 0.45),
}


def load_rife() -> Model:
    model = Model()
    model.load_model(str(RIFE_DIR / "train_log"), -1)
    model.eval()
    return model


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def warp(img: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    height, width = img.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    return cv2.remap(
        img,
        (xs + dx).astype(np.float32),
        (ys + dy).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def infer_rife(model: Model, img0: np.ndarray, img1: np.ndarray, alpha: float) -> np.ndarray:
    height, width = img0.shape[:2]
    padded_height = math.ceil(height / 64) * 64
    padded_width = math.ceil(width / 64) * 64

    def to_tensor(img: np.ndarray) -> torch.Tensor:
        return (
            torch.from_numpy(img)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )

    t0 = F.pad(to_tensor(img0), (0, padded_width - width, 0, padded_height - height))
    t1 = F.pad(to_tensor(img1), (0, padded_width - width, 0, padded_height - height))
    with torch.no_grad():
        out = model.inference(t0, t1, timestep=float(alpha))
    out = out[0, :, :height, :width].permute(1, 2, 0).detach().cpu().numpy()
    return (out * 255.0).clip(0, 255).astype(np.float32)


def render_sample(model: Model, dis, sample_dir: Path) -> np.ndarray:
    meta = json.loads((sample_dir / "meta.json").read_text())
    camera = meta["target_camera"]
    timestamps = meta["timestamps_ns"]
    alpha = (timestamps["target"] - timestamps["t0"]) / (timestamps["t1"] - timestamps["t0"])

    img0 = read_rgb(sample_dir / "input" / "t0" / f"{camera}.jpg")
    img1 = read_rgb(sample_dir / "input" / "t1" / f"{camera}.jpg")

    mean = (1.0 - alpha) * img0.astype(np.float32) + alpha * img1.astype(np.float32)

    gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

    # flow из t0 в t1
    flow01 = dis.calc(gray0, gray1, None)
    dx01 = flow01[..., 0]
    dy01 = flow01[..., 1]

    # flow из t1 в t0
    flow10 = dis.calc(gray1, gray0, None)
    dx10 = flow10[..., 0]
    dy10 = flow10[..., 1]

    # img0 двигаем к target по flow01
    warped0 = warp(img0, -alpha * dx01, -alpha * dy01).astype(np.float32)

    # img1 двигаем к target по flow10
    warped1 = warp(img1, -(1.0 - alpha) * dx10, -(1.0 - alpha) * dy10).astype(np.float32)

    warped = (1.0 - alpha) * warped0 + alpha * warped1
    warp_weight, rife_weight = CAMERA_WEIGHTS.get(camera, DEFAULT_WEIGHTS)
    dis_blend = warp_weight * warped + (1.0 - warp_weight) * mean

    rife = infer_rife(model, img0, img1, alpha)
    pred = rife_weight * rife + (1.0 - rife_weight) * dis_blend

    # Safety blend: если предсказание слишком далеко ушло от обоих исходных кадров,
    # немного возвращаем его к простому temporal mean.
    img0_f = img0.astype(np.float32)
    img1_f = img1.astype(np.float32)

    d0 = np.mean(np.abs(pred - img0_f), axis=2)
    d1 = np.mean(np.abs(pred - img1_f), axis=2)

    bad = (d0 > 35.0) & (d1 > 35.0)

    if np.any(bad):
        pred[bad] = 0.70 * pred[bad] + 0.30 * mean[bad]

    return pred.clip(0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "task_b_submission" / "submission")
    args = parser.parse_args()

    cv2.setNumThreads(4)
    model = load_rife()
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FINE)

    split_dir = args.data_root / args.split
    samples = sorted(p for p in split_dir.iterdir() if p.is_dir())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    for idx, sample_dir in enumerate(samples, 1):
        pred = render_sample(model, dis, sample_dir)
        out_dir = args.output_dir / sample_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred).save(out_dir / "pred.jpg", quality=95, subsampling=0)
        if device.type == "mps":
            torch.mps.empty_cache()
        print(f"[{idx:03d}/{len(samples)}] {sample_dir.name}", flush=True)

    elapsed = time.time() - start
    print(f"Done: {len(samples)} samples in {elapsed:.1f}s -> {args.output_dir}")


if __name__ == "__main__":
    main()
