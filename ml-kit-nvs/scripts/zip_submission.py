import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="submission")
    parser.add_argument("--output", default="submission.zip")
    args = parser.parse_args()

    sub_dir = Path(args.submission)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(sub_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(sub_dir.parent))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

