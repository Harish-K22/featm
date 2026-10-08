from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import featm


FIELDS = [
    "feature", "modality", "family", "subfamily", "cost", "complexity",
    "memory_profile", "batch_semantics", "stateful", "gpu_available",
    "output_kind", "approximate_available", "requires", "tags",
]


def rows():
    for s in featm.REGISTRY.all():
        yield {
            "feature": s.name,
            "modality": s.modality,
            "family": s.family,
            "subfamily": s.subfamily,
            "cost": s.cost.value,
            "complexity": s.complexity.value,
            "memory_profile": s.memory_profile.value,
            "batch_semantics": s.batch_semantics.value,
            "stateful": bool(s.stateful),
            "gpu_available": bool(s.gpu_available),
            "output_kind": s.output_kind,
            "approximate_available": bool(s.approximate_available),
            "requires": ";".join(s.requires),
            "tags": ";".join(s.tags),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="step5_results/step5_feature_catalog.csv")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = list(rows())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(data)
    families = Counter((r["modality"], r["family"]) for r in data)
    print(f"features={len(data)} modalities={len(set(r['modality'] for r in data))} families={len(families)}")
    print(f"stateful={sum(r['stateful'] for r in data)} gpu_available={sum(r['gpu_available'] for r in data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
