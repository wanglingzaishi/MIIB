"""Aggregate one final test result per seed with an explicit standard deviation convention."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="result.json files from independent mode=test runs")
    parser.add_argument("--metric", default="acc1")
    parser.add_argument("--scale", type=float, default=100.0)
    parser.add_argument("--ddof", type=int, choices=[0, 1], default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(path.read_text()) for path in args.files]
    seeds = [row["seed"] for row in rows]
    if len(set(seeds)) != len(seeds):
        parser.error("Duplicate seeds: provide one final test result per seed")
    if any(row["mode"] != "test" for row in rows):
        parser.error("Training/validation records cannot be aggregated as final test results")
    if len({row["experiment"] for row in rows}) != 1:
        parser.error("All files must be from the same experiment")
    values = [row["metrics"][args.metric] * args.scale for row in rows]
    if args.ddof == 1 and len(values) < 2:
        parser.error("Sample standard deviation requires at least two independent seeds")
    output = {
        "experiment": rows[0]["experiment"],
        "metric": args.metric,
        "seeds": seeds,
        "values": values,
        "scale": args.scale,
        "ddof": args.ddof,
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if args.ddof else statistics.pstdev(values),
    }
    print(json.dumps(output, indent=2))
    if args.output:
        args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
