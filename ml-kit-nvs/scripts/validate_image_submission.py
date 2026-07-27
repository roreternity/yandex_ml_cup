import argparse
import json
from pathlib import Path

from PIL import Image


def expected_size(sample_dir: Path):
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    camera = meta.get("target_camera")
    intr = meta.get("intrinsics", {}).get(camera, {})
    return int(intr.get("width", 1920)), int(intr.get("height", 1200))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--strict-size", action="store_true")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    sub_dir = Path(args.submission)
    test_ids = sorted(p.name for p in test_dir.iterdir() if p.is_dir())
    sub_ids = sorted(p.name for p in sub_dir.iterdir() if p.is_dir())
    errors = []

    missing = sorted(set(test_ids) - set(sub_ids))
    extra = sorted(set(sub_ids) - set(test_ids))
    if missing:
        errors.append(f"missing sample folders: {missing[:10]} total={len(missing)}")
    if extra:
        errors.append(f"extra sample folders: {extra[:10]} total={len(extra)}")

    for sample_id in test_ids:
        pred_path = sub_dir / sample_id / "pred.jpg"
        if not pred_path.exists():
            errors.append(f"{sample_id}: missing pred.jpg")
            continue
        try:
            with Image.open(pred_path) as img:
                img.verify()
            with Image.open(pred_path) as img:
                if img.mode != "RGB":
                    errors.append(f"{sample_id}: image mode is {img.mode}, expected RGB")
                if args.strict_size and img.size != expected_size(test_dir / sample_id):
                    errors.append(f"{sample_id}: size {img.size}, expected {expected_size(test_dir / sample_id)}")
        except Exception as exc:
            errors.append(f"{sample_id}: unreadable image: {exc}")

    if errors:
        for error in errors[:50]:
            print("FAIL:", error)
        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more")
        return 1
    print(f"OK: {len(test_ids)} predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

