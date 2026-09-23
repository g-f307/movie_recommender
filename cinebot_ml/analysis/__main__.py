"""CLI da análise estatística confirmatória."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cinebot_ml.config import PROJECT_ROOT

from .statistical import analyze_official_results, write_outputs


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "results/raw/official_v1_1/2942e51456add29e4c999307/experiment.manifest.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "results/reports/statistical_analysis.json",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=PROJECT_ROOT / "results/tables/b5_vs_b4_ndcg_at_5.csv",
    )
    args = parser.parse_args()
    report, rows = analyze_official_results(args.manifest)
    write_outputs(report, rows, args.json_output, args.csv_output)
    result = report["paired_difference_b5_minus_b4"]
    print(json.dumps({
        "matrix_id": report["matrix_id"],
        "valid_pairs": len(rows),
        "mean_difference": result["mean_difference"],
        "json_output": str(args.json_output),
        "csv_output": str(args.csv_output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
