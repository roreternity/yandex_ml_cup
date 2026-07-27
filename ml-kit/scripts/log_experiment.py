import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--score", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--params", default="{}")
    parser.add_argument("--file", default="experiments.jsonl")
    args = parser.parse_args()

    row = {
        "time": datetime.now(timezone.utc).isoformat(),
        "name": args.name,
        "metric": args.metric,
        "score": float(args.score),
        "params": json.loads(args.params),
        "notes": args.notes,
    }
    path = Path(args.file)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    print(f"logged: {path}")


if __name__ == "__main__":
    main()

