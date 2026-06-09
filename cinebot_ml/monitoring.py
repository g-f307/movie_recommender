from pathlib import Path

import pandas as pd

from cinebot_ml.config import DRIFT_REPORT_PATH


def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: Path = DRIFT_REPORT_PATH,
) -> str | None:
    if reference_data.empty or current_data.empty:
        return None

    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return None

    common_columns = [column for column in reference_data.columns if column in current_data.columns]
    if not common_columns:
        return None

    reference_subset = reference_data[common_columns].copy()
    current_subset = current_data[common_columns].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=reference_subset, current_data=current_subset)
        snapshot.save_html(str(output_path))
        return str(output_path)
    except Exception:
        return None
