import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse == 0:
        return 99.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--pred", required=True)
    args = parser.parse_args()

    truth_root = Path(args.truth)
    pred_root = Path(args.pred)
    scores = []
    for sample in sorted(p for p in truth_root.iterdir() if p.is_dir()):
        target_images = list((sample / "target").glob("*.jpg"))
        pred_path = pred_root / sample.name / "pred.jpg"
        if not target_images or not pred_path.exists():
            continue
        gt = Image.open(target_images[0]).convert("RGB")
        pred = Image.open(pred_path).convert("RGB").resize(gt.size, Image.Resampling.BILINEAR)
        scores.append(psnr(np.asarray(gt), np.asarray(pred)))
    if not scores:
        raise SystemExit("no comparable samples found")
    print(f"n={len(scores)} psnr_mean={np.mean(scores):.6f} psnr_median={np.median(scores):.6f}")


if __name__ == "__main__":
    main()

