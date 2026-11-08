"""Resolve external datasets without copying or changing their contents."""
import json
import os
from pathlib import Path


def load_catalog():
    catalog_file = Path(os.environ.get("MIIB_CATALOG", Path(__file__).with_name("catalog.json")))
    with catalog_file.open() as stream:
        catalog = json.load(stream)
    root = Path(os.environ.get("MIIB_DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
    for entry in catalog.values():
        for key in ("path", "metadata"):
            if key in entry:
                path = Path(entry[key]).expanduser()
                entry[key] = str(path if path.is_absolute() else root / path)
    return catalog
