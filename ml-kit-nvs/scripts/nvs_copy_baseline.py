import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


FALLBACK_CAMERAS = ["front", "left_fwd", "right_fwd", "left_bwd", "right_bwd", "rear"]


def load_meta(sample_dir: Path) -> dict:
    return json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))


def target_size(meta: dict, camera: str):
    intr = meta.get("intrinsics", {}).get(camera, {})
    return int(intr.get("width", 1920)), int(intr.get("height", 1200))


def find_image(sample_dir: Path, time_name: str, camera: str) -> Path | None:
    path = sample_dir / "input" / time_name / f"{camera}.jpg"
    return path if path.exists() else None


def first_available(sample_dir: Path, time_name: str) -> Path:
    for camera in FALLBACK_CAMERAS:
        path = find_image(sample_dir, time_name, camera)
        if path:
            return path
    raise FileNotFoundError(f"no input images found in {sample_dir / 'input' / time_name}")


def read_rgb(path: Path, size):
    return Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR)


def make_prediction(sample_dir: Path, mode: str) -> Image.Image:
    meta = load_meta(sample_dir)
    camera = meta.get("target_camera") or "front"
    size = target_size(meta, camera)
    t0 = find_image(sample_dir, "t0", camera) or first_available(sample_dir, "t0")
    t1 = find_image(sample_dir, "t1", camera) or first_available(sample_dir, "t1")

    if mode == "same_camera_t0":
        return read_rgb(t0, size)
    if mode == "same_camera_t1":
        return read_rgb(t1, size)
    if mode == "front_fallback":
        return read_rgb(find_image(sample_dir, "t0", "front") or first_available(sample_dir, "t0"), size)
    if mode == "same_camera_blend":
        a = np.asarray(read_rgb(t0, size), dtype=np.float32)
        b = np.asarray(read_rgb(t1, size), dtype=np.float32)
        return Image.fromarray(np.clip((a + b) * 0.5, 0, 255).astype(np.uint8))
    raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output", default="submission")
    parser.add_argument("--mode", default="same_camera_blend", choices=["same_camera_t0", "same_camera_t1", "same_camera_blend", "front_fallback"])
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = sorted([p for p in test_dir.iterdir() if p.is_dir()])
    for i, sample in enumerate(samples, start=1):
        pred = make_prediction(sample, args.mode)
        sample_out = out_dir / sample.name
        sample_out.mkdir(parents=True, exist_ok=True)
        pred.save(sample_out / "pred.jpg", quality=args.quality)
        if i % 25 == 0 or i == len(samples):
            print(f"{i}/{len(samples)}")
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()

