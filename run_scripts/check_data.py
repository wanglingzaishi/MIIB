"""Resolve and verify every external dataset path without downloading anything.

Usage:
    python run_scripts/check_data.py            # check the catalog resolved paths
    python run_scripts/check_data.py --json     # machine-readable output

Environment:
    MIIB_DATA_ROOT   root that relative catalog entries resolve below
                     (default: <repo>/data)
    MIIB_CATALOG     alternative catalog JSON file
                     (default: <repo>/dataset/catalog.json)
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.catalog import load_catalog  # noqa: E402

# Extra requirements per dataset, beyond the single file/directory in the catalog.
EXTRA_REQUIREMENTS = {
    "visionandtouch": ["*.h5"],
    "mmimdb": ["split.json", "dataset/"],
    "trifeatures": ["train/", "test/"],
}


def check_entry(name, entry):
    path = Path(entry["path"])
    checks = [(path, path.exists())]
    for pattern in EXTRA_REQUIREMENTS.get(name, []):
        child = path / pattern
        if pattern.endswith("/"):
            checks.append((child, child.is_dir()))
        elif pattern.endswith(".h5"):
            checks.append((child, any(path.glob("*.h5"))))
        else:
            checks.append((child, child.exists()))
    metadata = entry.get("metadata")
    if metadata:
        mp = Path(metadata)
        checks.append((mp / "split.json", (mp / "split.json").exists()))
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args()

    root = os.environ.get("MIIB_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data"))
    catalog = load_catalog()
    report = {name: [{"path": str(p), "ok": ok} for p, ok in check_entry(name, entry)]
              for name, entry in catalog.items()}

    if args.json:
        print(json.dumps({"data_root": root, "datasets": report}, indent=2))
        return

    print("MIIB_DATA_ROOT = %s" % root)
    print("MIIB_CATALOG   = %s" % os.environ.get("MIIB_CATALOG", "<default dataset/catalog.json>"))
    print()
    missing_any = False
    for name, checks in report.items():
        status = "OK" if all(c["ok"] for c in checks) else "MISSING"
        missing_any = missing_any or status == "MISSING"
        print("[%s] %s" % (status, name))
        for check in checks:
            print("        %s  %s" % ("found  " if check["ok"] else "missing", check["path"]))
    print()
    print("Some datasets are missing. Copy the files into the paths above, then re-run.")
    print("Datasets with an existing file are never downloaded again: the loader only")
    print("falls back to Google Drive when the resolved path does not exist.")


if __name__ == "__main__":
    main()
