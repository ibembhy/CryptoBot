from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.loaders import (
    DEFAULT_DB_PATH,
    DEFAULT_OUTPUT_DIR,
    build_context,
    load_all_settlements,
    load_live_ledger,
    load_snapshots,
)
from scripts.analysis.cleaning import (
    add_market_derived_columns,
    assign_series_buckets,
    build_global_spot_features,
    deduplicate_snapshots,
)
from scripts.analysis.analysis_a import run_analysis_a
from scripts.analysis.analysis_b import run_analysis_b
from scripts.analysis.analysis_c import run_analysis_c
from scripts.analysis.analysis_d import run_analysis_d
from scripts.analysis.profiler import run_data_quality_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared analysis entrypoint scaffold.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--series", action="append", choices=["KXBTCD", "KXBTC15M"])
    parser.add_argument("--profile-data-quality", action="store_true")
    parser.add_argument("--prepare-clean-dataset", action="store_true")
    parser.add_argument("--analysis-a", action="store_true")
    parser.add_argument("--analysis-b", action="store_true")
    parser.add_argument("--analysis-b-full", action="store_true")
    parser.add_argument("--analysis-c", action="store_true")
    parser.add_argument("--analysis-d", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = build_context(args.db_path, args.output_dir, args.config_path)
    series_list = args.series or ["KXBTCD", "KXBTC15M"]

    print(f"DB: {context.db_path}")
    print(f"Output: {context.output_dir}")
    print(f"Series: {', '.join(series_list)}")

    settlements = load_all_settlements()
    print(f"Loaded settlements: {len(settlements):,}")

    if args.profile_data_quality:
        outputs = run_data_quality_profile(context, series_list)
        print("Data-quality profiler outputs:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")

    if args.analysis_a:
        outputs = run_analysis_a(context, series_list)
        print("Analysis A outputs:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")

    if args.analysis_b or args.analysis_b_full:
        outputs = run_analysis_b(context, series_list, focused=not args.analysis_b_full)
        print("Analysis B outputs:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")

    if args.analysis_c:
        outputs = run_analysis_c(context)
        print("Analysis C outputs:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")

    if args.analysis_d:
        outputs = run_analysis_d(context)
        print("Analysis D outputs:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")

    for series in series_list:
        snapshots = load_snapshots(
            context.db_path,
            series=series,
            columns=[
                "market_ticker",
                "series_ticker",
                "observed_at",
                "expiry",
                "spot_price",
                "threshold",
                "yes_bid",
                "yes_ask",
                "no_bid",
                "no_ask",
                "implied_probability",
                "volume",
                "open_interest",
                "source",
                "metadata_json",
            ],
        )
        ledger = load_live_ledger(series)
        print(f"{series}: snapshots={len(snapshots):,} ledger_rows={len(ledger):,}")

        if args.prepare_clean_dataset:
            deduped, duplicate_audit = deduplicate_snapshots(snapshots)
            deduped, global_spot = build_global_spot_features(deduped)
            deduped = assign_series_buckets(add_market_derived_columns(deduped))
            clean_path = context.output_dir / f"{series.lower()}_clean_snapshots.csv"
            dupes_path = context.output_dir / f"{series.lower()}_dedup_audit.csv"
            spot_path = context.output_dir / f"{series.lower()}_global_spot_features.csv"
            deduped.to_csv(clean_path, index=False)
            duplicate_audit.to_csv(dupes_path, index=False)
            global_spot.to_csv(spot_path, index=False)
            print(
                f"{series}: deduped={len(deduped):,} removed={len(snapshots) - len(deduped):,} "
                f"global_spot_points={len(global_spot):,}"
            )
            print(f"  clean={clean_path}")
            print(f"  dedup_audit={dupes_path}")
            print(f"  global_spot={spot_path}")


if __name__ == "__main__":
    main()
