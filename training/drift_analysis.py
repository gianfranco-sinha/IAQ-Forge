"""Offline low-pass filters for sensor drift and seasonal variation analysis.

Computes multi-timescale moving averages on historical sensor data to
characterize long-term drift (weeks/months) and seasonal variation (annual
cycles).  Pure pandas transforms — no CLI or I/O logic.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.exceptions import SchemaMismatchError

logger = logging.getLogger("training.drift_analysis")


@dataclass
class DriftMetrics:
    """Per-feature drift and seasonal analysis results."""

    feature: str
    trend_slope_per_day: float
    trend_r2: float
    trend_pct_per_year: float
    seasonal_amplitude: float
    coverage_pct: float
    gap_count: int
    longest_gap_hours: float


@dataclass
class DriftReport:
    """Complete drift analysis report."""

    date_start: pd.Timestamp
    date_end: pd.Timestamp
    sample_count: int
    median_interval_seconds: float
    feature_metrics: Dict[str, DriftMetrics]
    moving_averages: Dict[str, pd.DataFrame]
    warnings: List[str] = field(default_factory=list)


def _detect_gaps(
    index: pd.DatetimeIndex, threshold_hours: float
) -> List[Dict]:
    """Find gaps in a DatetimeIndex exceeding the threshold."""
    if len(index) < 2:
        return []
    deltas = pd.Series(index[1:]) - pd.Series(index[:-1])
    threshold = pd.Timedelta(hours=threshold_hours)
    gaps = []
    for i, dt in enumerate(deltas):
        if dt > threshold:
            gaps.append({
                "start": index[i],
                "end": index[i + 1],
                "duration_hours": dt.total_seconds() / 3600,
            })
    return gaps


def _ols_trend(series: pd.Series):
    """Compute OLS linear trend on a series with datetime index.

    Returns (slope_per_day, r_squared, mean_value).
    """
    clean = series.dropna()
    if len(clean) < 2:
        return 0.0, 0.0, clean.mean() if len(clean) else 0.0

    # Convert timestamps to days since start
    t0 = clean.index[0]
    x = np.array([(t - t0).total_seconds() / 86400 for t in clean.index])
    y = clean.values.astype(float)

    # OLS via normal equations
    n = len(x)
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = (x * y).sum()
    sum_x2 = (x * x).sum()

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 0.0, 0.0, float(y.mean())

    slope = (n * sum_xy - sum_x * sum_y) / denom

    # R^2
    y_mean = y.mean()
    ss_tot = ((y - y_mean) ** 2).sum()
    y_pred = slope * x + (sum_y - slope * sum_x) / n
    ss_res = ((y - y_pred) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return float(slope), float(r2), float(y_mean)


def compute_moving_averages(
    df: pd.DataFrame,
    features: List[str],
    gap_threshold_hours: float = 1.0,
) -> DriftReport:
    """Compute weekly/monthly/yearly moving averages and drift metrics.

    Args:
        df: DataFrame with DatetimeIndex and feature columns.
        features: List of column names to analyze.
        gap_threshold_hours: Gaps longer than this are cataloged.

    Returns:
        DriftReport with per-feature metrics and moving average DataFrames.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise SchemaMismatchError(
            "DataFrame must have a DatetimeIndex",
            suggestion="Ensure the DataFrame index is a pd.DatetimeIndex before calling analyze_drift().",
        )

    df = df.sort_index()
    warnings = []

    # Filter to requested features that exist
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        warnings.append(f"Features not found in data: {missing}")

    if not available:
        raise SchemaMismatchError(
            f"No requested features found. Available: {list(df.columns)}",
            suggestion="Check that the feature names match the DataFrame columns.",
        )

    # Global gap detection
    gaps = _detect_gaps(df.index, gap_threshold_hours)
    if gaps:
        warnings.append(
            f"Found {len(gaps)} gaps > {gap_threshold_hours}h "
            f"(longest: {max(g['duration_hours'] for g in gaps):.1f}h)"
        )

    # Median sample interval
    if len(df) >= 2:
        deltas = pd.Series(df.index[1:]) - pd.Series(df.index[:-1])
        median_interval = deltas.median().total_seconds()
    else:
        median_interval = 0.0

    # Resample to 1-minute intervals (regularize without interpolating across gaps)
    df_1min = df[available].resample("1min").mean()

    # Expected total minutes in the date range
    total_span_minutes = (df.index[-1] - df.index[0]).total_seconds() / 60

    feature_metrics = {}
    moving_averages = {}

    for feat in available:
        series = df_1min[feat]

        # Rolling windows with min_periods fractions
        expected_7d = 7 * 24 * 60  # minutes in 7 days
        expected_30d = 30 * 24 * 60
        expected_365d = 365 * 24 * 60

        weekly_ma = series.rolling(
            window=expected_7d, min_periods=int(expected_7d * 0.5)
        ).mean()
        monthly_ma = series.rolling(
            window=expected_30d, min_periods=int(expected_30d * 0.3)
        ).mean()
        yearly_ma = series.rolling(
            window=expected_365d, min_periods=int(expected_365d * 0.2)
        ).mean()

        # Store moving averages as a DataFrame
        ma_df = pd.DataFrame({
            "daily": series.resample("1D").mean(),
        })
        # Resample the rolling results to daily for storage
        ma_df["weekly"] = weekly_ma.resample("1D").mean()
        ma_df["monthly"] = monthly_ma.resample("1D").mean()
        ma_df["yearly"] = yearly_ma.resample("1D").mean()
        moving_averages[feat] = ma_df

        # OLS trend on daily-resampled data
        daily = series.resample("1D").mean().dropna()
        slope_per_day, r2, mean_val = _ols_trend(daily)

        # Percent per year
        pct_per_year = (slope_per_day * 365 / mean_val * 100) if abs(mean_val) > 1e-12 else 0.0

        # Seasonal amplitude: range of (monthly_ma - yearly_ma)
        seasonal_diff = monthly_ma - yearly_ma
        seasonal_diff_clean = seasonal_diff.dropna()
        if len(seasonal_diff_clean) > 0:
            seasonal_amplitude = float(
                seasonal_diff_clean.max() - seasonal_diff_clean.min()
            )
        else:
            seasonal_amplitude = 0.0

        # Per-feature gaps
        feat_gaps = _detect_gaps(
            series.dropna().index, gap_threshold_hours
        )
        longest_gap = (
            max(g["duration_hours"] for g in feat_gaps) if feat_gaps else 0.0
        )

        # Coverage
        non_null = series.notna().sum()
        expected_count = total_span_minutes + 1 if total_span_minutes > 0 else 1
        coverage = min(100.0, non_null / expected_count * 100)

        feature_metrics[feat] = DriftMetrics(
            feature=feat,
            trend_slope_per_day=slope_per_day,
            trend_r2=r2,
            trend_pct_per_year=pct_per_year,
            seasonal_amplitude=seasonal_amplitude,
            coverage_pct=coverage,
            gap_count=len(feat_gaps),
            longest_gap_hours=longest_gap,
        )

    # ── VOC compensation: regress out temperature/humidity effects ────────
    # MOX sensors (BME680) have strong T/H cross-sensitivity.  A raw
    # voc_resistance trend could be seasonal (summer = hotter/more humid →
    # lower resistance) rather than true sensor degradation.  We fit
    # log(voc_resistance) ~ temperature + rel_humidity via OLS on
    # daily-resampled data, then run the same drift analysis on the
    # residuals.  The compensated metric isolates true sensor drift.
    voc_key = "voc_resistance"
    temp_key = "temperature"
    hum_key = "rel_humidity"
    compensated_key = "voc_resistance_compensated"

    if all(k in available for k in (voc_key, temp_key, hum_key)):
        daily_voc = df_1min[voc_key].resample("1D").mean().dropna()
        daily_temp = df_1min[temp_key].resample("1D").mean().dropna()
        daily_hum = df_1min[hum_key].resample("1D").mean().dropna()

        # Align on common dates
        common_idx = daily_voc.index.intersection(
            daily_temp.index
        ).intersection(daily_hum.index)

        if len(common_idx) >= 10:
            y = np.log(daily_voc.loc[common_idx].values.astype(float))
            X = np.column_stack([
                np.ones(len(common_idx)),
                daily_temp.loc[common_idx].values.astype(float),
                daily_hum.loc[common_idx].values.astype(float),
            ])

            # OLS: beta = (X'X)^-1 X'y
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                y_pred = X @ beta
                residuals = y - y_pred

                # R^2 of the compensation model (how much T/H explains)
                ss_tot = ((y - y.mean()) ** 2).sum()
                ss_res = (residuals ** 2).sum()
                compensation_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

                warnings.append(
                    f"VOC compensation model: log(voc_resistance) ~ "
                    f"{beta[1]:+.4f}*temperature {beta[2]:+.4f}*rel_humidity "
                    f"(R2={compensation_r2:.3f}, "
                    f"T/H explains {compensation_r2*100:.1f}% of VOC variance)"
                )

                # Build a daily series of residuals and run drift analysis on it
                residual_series = pd.Series(residuals, index=common_idx)
                slope_per_day, r2, mean_val = _ols_trend(residual_series)
                # For pct_per_year, use the raw VOC mean (residuals are zero-centered)
                raw_voc_mean = daily_voc.loc[common_idx].mean()
                pct_per_year = (
                    (slope_per_day * 365 / np.log(raw_voc_mean) * 100)
                    if raw_voc_mean > 1e-12
                    else 0.0
                )

                feat_gaps = _detect_gaps(
                    residual_series.dropna().index, gap_threshold_hours
                )
                longest_gap = (
                    max(g["duration_hours"] for g in feat_gaps) if feat_gaps else 0.0
                )

                # Seasonal amplitude on compensated residuals
                # Resample residuals to 1-min (already daily, just use as-is)
                seasonal_amplitude = float(
                    residual_series.max() - residual_series.min()
                ) if len(residual_series) > 0 else 0.0

                feature_metrics[compensated_key] = DriftMetrics(
                    feature=compensated_key,
                    trend_slope_per_day=slope_per_day,
                    trend_r2=r2,
                    trend_pct_per_year=pct_per_year,
                    seasonal_amplitude=seasonal_amplitude,
                    coverage_pct=feature_metrics[voc_key].coverage_pct,
                    gap_count=len(feat_gaps),
                    longest_gap_hours=longest_gap,
                )

                # Store compensated MA for plotting
                moving_averages[compensated_key] = pd.DataFrame({
                    "daily": residual_series,
                    "weekly": residual_series.rolling(7, min_periods=4).mean(),
                    "monthly": residual_series.rolling(30, min_periods=10).mean(),
                    "yearly": residual_series.rolling(365, min_periods=73).mean(),
                })

            except np.linalg.LinAlgError:
                warnings.append(
                    "VOC compensation failed: singular matrix in OLS fit"
                )
        else:
            warnings.append(
                f"VOC compensation skipped: only {len(common_idx)} common "
                f"daily samples (need >= 10)"
            )

    return DriftReport(
        date_start=df.index[0],
        date_end=df.index[-1],
        sample_count=len(df),
        median_interval_seconds=median_interval,
        feature_metrics=feature_metrics,
        moving_averages=moving_averages,
        warnings=warnings,
    )


def _drift_status(metrics: DriftMetrics) -> str:
    """Classify drift severity."""
    if abs(metrics.trend_pct_per_year) > 10 and metrics.trend_r2 > 0.1:
        return "DRIFT"
    elif abs(metrics.trend_pct_per_year) > 5:
        return "WARNING"
    return "OK"


def format_console_report(report: DriftReport) -> str:
    """Format a DriftReport as a tabular console summary."""
    lines = []
    lines.append(f"Drift Analysis Report")
    lines.append(f"{'=' * 90}")
    lines.append(
        f"Date range: {report.date_start} to {report.date_end}"
    )
    lines.append(f"Samples: {report.sample_count:,}")
    lines.append(f"Median interval: {report.median_interval_seconds:.1f}s")
    lines.append("")

    if report.warnings:
        for w in report.warnings:
            lines.append(f"  WARNING: {w}")
        lines.append("")

    # Table header
    header = (
        f"{'Feature':<18} {'Status':<8} {'Trend %/yr':>10} "
        f"{'R2':>6} {'Seasonal':>10} {'Coverage':>8} "
        f"{'Gaps':>5} {'Max Gap h':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for feat, m in sorted(report.feature_metrics.items()):
        status = _drift_status(m)
        lines.append(
            f"{feat:<18} {status:<8} {m.trend_pct_per_year:>+10.2f} "
            f"{m.trend_r2:>6.3f} {m.seasonal_amplitude:>10.2f} "
            f"{m.coverage_pct:>7.1f}% "
            f"{m.gap_count:>5} {m.longest_gap_hours:>9.1f}"
        )

    lines.append("")
    return "\n".join(lines)


def save_drift_output(report: DriftReport, output_dir: str) -> None:
    """Write drift_summary.json and per-feature moving_averages CSVs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    summary = {
        "date_start": str(report.date_start),
        "date_end": str(report.date_end),
        "sample_count": report.sample_count,
        "median_interval_seconds": report.median_interval_seconds,
        "warnings": report.warnings,
        "features": {},
    }
    for feat, m in report.feature_metrics.items():
        summary["features"][feat] = {
            "status": _drift_status(m),
            "trend_slope_per_day": m.trend_slope_per_day,
            "trend_r2": m.trend_r2,
            "trend_pct_per_year": m.trend_pct_per_year,
            "seasonal_amplitude": m.seasonal_amplitude,
            "coverage_pct": m.coverage_pct,
            "gap_count": m.gap_count,
            "longest_gap_hours": m.longest_gap_hours,
        }

    summary_path = out / "drift_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved drift summary to %s", summary_path)

    # Per-feature CSVs
    for feat, ma_df in report.moving_averages.items():
        csv_path = out / f"{feat}_moving_averages.csv"
        ma_df.to_csv(csv_path)
        logger.info("Saved %s moving averages to %s", feat, csv_path)


def plot_drift_charts(report: DriftReport, output_dir: str) -> None:
    """Generate one PNG per feature: daily raw, weekly/monthly/yearly MA, trend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for feat, ma_df in report.moving_averages.items():
        fig, ax = plt.subplots(figsize=(14, 5))

        # Daily raw (light)
        if "daily" in ma_df.columns:
            daily = ma_df["daily"].dropna()
            if len(daily) > 0:
                ax.plot(
                    daily.index, daily.values,
                    color="#CCCCCC", linewidth=0.5, label="Daily avg",
                )

        # Weekly MA
        if "weekly" in ma_df.columns:
            weekly = ma_df["weekly"].dropna()
            if len(weekly) > 0:
                ax.plot(
                    weekly.index, weekly.values,
                    color="#4A90D9", linewidth=1.0, label="Weekly MA",
                )

        # Monthly MA
        if "monthly" in ma_df.columns:
            monthly = ma_df["monthly"].dropna()
            if len(monthly) > 0:
                ax.plot(
                    monthly.index, monthly.values,
                    color="#2ECC71", linewidth=1.5, label="Monthly MA",
                )

        # Yearly MA
        if "yearly" in ma_df.columns:
            yearly = ma_df["yearly"].dropna()
            if len(yearly) > 0:
                ax.plot(
                    yearly.index, yearly.values,
                    color="#E67E22", linewidth=2.0, label="Yearly MA",
                )

        # Linear trend (red dashed)
        metrics = report.feature_metrics.get(feat)
        if metrics and "daily" in ma_df.columns:
            daily = ma_df["daily"].dropna()
            if len(daily) >= 2:
                t0 = daily.index[0]
                x_days = np.array([
                    (t - t0).total_seconds() / 86400 for t in daily.index
                ])
                intercept = daily.values[0]
                trend_line = intercept + metrics.trend_slope_per_day * x_days
                ax.plot(
                    daily.index, trend_line,
                    color="red", linewidth=1.0, linestyle="--",
                    label=f"Trend ({metrics.trend_pct_per_year:+.1f}%/yr)",
                )

        ax.set_title(f"{feat} — Drift Analysis")
        ax.set_xlabel("Date")
        ax.set_ylabel(feat)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        png_path = out / f"{feat}_drift.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        logger.info("Saved %s drift chart to %s", feat, png_path)
