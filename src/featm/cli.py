from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .engine import feature_info, list_families, list_features, list_modalities, search


def _spec_dict(spec):
    data = asdict(spec)
    data.pop("implementation", None)
    cost = data.get("cost")
    if hasattr(cost, "value"):
        data["cost"] = cost.value
    return data


def _print_specs(specs) -> None:
    for spec in specs:
        suffix = f"  [{', '.join(spec.requires)}]" if spec.requires else ""
        print(f"{spec.name}{suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="featm",
        description="Browse the featm feature registry.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List registered feature names")
    p_list.add_argument("pattern", nargs="?", default=None)
    p_list.add_argument("--modality", default=None)
    p_list.add_argument("--family", default=None)

    p_search = sub.add_parser("search", help="Search feature names/descriptions/tags")
    p_search.add_argument("query")

    p_info = sub.add_parser("info", help="Show metadata for one feature")
    p_info.add_argument("name")
    p_info.add_argument("--json", action="store_true")

    sub.add_parser("modalities", help="List available modalities")

    p_families = sub.add_parser("families", help="List feature families")
    p_families.add_argument("--modality", default=None)

    p_run = sub.add_parser("run", help="Run an out-of-core extraction against a file source")
    p_run.add_argument("source", help="Path to a CSV/TSV/JSONL/Parquet/Arrow file")
    p_run.add_argument("--features", nargs="+", required=True, help="Explicit feature names to extract")
    p_run.add_argument(
        "--column", action="append", default=[], metavar="FEATURE=COLUMN",
        help="Explicit input column for one tabular feature, e.g. tabular.scaling.zscore=price. Repeatable.",
    )
    p_run.add_argument("--batch-rows", type=int, default=None)
    p_run.add_argument("--memory-budget", default=None, help="e.g. 2GB")
    p_run.add_argument("--disk-budget", default=None, help="e.g. 50GB")
    p_run.add_argument("--output", choices=["auto", "memory", "disk"], default="auto")
    p_run.add_argument("--output-path", default=None, help="Write the result here (.parquet or .csv)")
    p_run.add_argument("--mode", choices=["exact", "approximate", "auto"], default="exact")
    p_run.add_argument("--errors", choices=["raise", "warn", "skip"], default="skip")
    p_run.add_argument("--checkpoint", default=None, help="Checkpoint directory for resumable runs")
    p_run.add_argument("--json", action="store_true", help="Print the execution report as JSON")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list":
        for name in list_features(args.pattern, modality=args.modality, family=args.family):
            print(name)
        return 0

    if args.command == "search":
        _print_specs(search(args.query))
        return 0

    if args.command == "info":
        spec = feature_info(args.name)
        data = _spec_dict(spec)
        if args.json:
            print(json.dumps(data, indent=2, default=str))
        else:
            for key, value in data.items():
                print(f"{key}: {value}")
        return 0

    if args.command == "modalities":
        for modality in list_modalities():
            print(modality)
        return 0

    if args.command == "families":
        for family in list_families(modality=args.modality):
            print(family)
        return 0

    if args.command == "run":
        from .engine import extract as _extract

        columns = {}
        for item in args.column:
            key, _, value = item.partition("=")
            if not _:
                print(f"--column expects FEATURE=COLUMN, got: {item!r}")
                return 2
            columns[key] = value

        result = _extract(
            args.source,
            features=args.features,
            columns=columns or None,
            batch_rows=args.batch_rows,
            memory_budget=args.memory_budget,
            disk_budget=args.disk_budget,
            output=args.output,
            mode=args.mode,
            errors=args.errors,
            checkpoint=args.checkpoint,
        )
        if args.json:
            print(json.dumps(result.report.to_dict(), indent=2, default=str))
        else:
            print(f"shape: {result.shape}  backend: {'disk' if result.is_disk_backed else 'memory'}")
            print(result.report.summary())
        if args.output_path:
            if str(args.output_path).endswith(".csv"):
                result.to_csv(args.output_path)
            else:
                result.to_parquet(args.output_path)
            print(f"wrote {result.shape[0]} rows x {result.shape[1]} columns to {args.output_path}")
        return 0 if not result.report.failed_features else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
