import argparse
import json
from pathlib import Path


CAMERAS = ["front", "left_fwd", "left_bwd", "right_fwd", "right_bwd", "rear"]


def load_meta(sample_dir: Path):
    path = sample_dir / "meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_split(split_dir: Path) -> None:
    samples = sorted([p for p in split_dir.iterdir() if p.is_dir()]) if split_dir.exists() else []
    print(f"{split_dir}: {len(samples)} samples")
    for sample in samples[:3]:
        meta = load_meta(sample)
        print(f"\nSample: {sample.name}")
        if meta:
            print("  target_camera:", meta.get("target_camera"))
            print("  delta_s:", meta.get("delta_s"))
            intr = meta.get("intrinsics", {})
            cam = meta.get("target_camera")
            if cam in intr:
                print("  target size:", intr[cam].get("width"), "x", intr[cam].get("height"))
        for t in ["t0", "t1"]:
            existing = [cam for cam in CAMERAS if (sample / "input" / t / f"{cam}.jpg").exists()]
            print(f"  input/{t}: {existing}")
        targets = list((sample / "target").glob("*.jpg")) if (sample / "target").exists() else []
        print("  target images:", [p.name for p in targets[:6]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    root = Path(args.dataset)
    inspect_split(root / "train")
    inspect_split(root / "test")


if __name__ == "__main__":
    main()

