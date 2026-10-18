#!/usr/bin/env python3
"""Analyze temporal cryptocurrency arbitrage spreads from a scanner CSV.

The scanner writes one row per venue quote per scan. A spread seen in a single
scan is usually noise because quotes are taken about a second apart, so what
matters is which spreads persist across consecutive scans. Some venues also
return frozen quotes, which fabricate enormous spreads, so they are detected and
excluded before any spread is computed.

Usage:
    python analyze_spreads.py spreads.csv [--threshold 30] [--min-scans 3]
        [--max-gap 300] [--stale-scans 5] [--out report.md] [--runs-csv runs.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

BEST = "Jupiter (best)"

REQUIRED_COLUMNS = [
    "timestamp",
    "slot",
    "pair",
    "amount",
    "venue",
    "out_amount",
    "price",
    "vs_best_bps",
    "impact_pct",
    "route",
    "cross_venue_spread_bps",
    "round_trip_bps",
]

NUMERIC_COLUMNS = ["out_amount", "price", "vs_best_bps", "round_trip_bps", "cross_venue_spread_bps"]


class SpreadAnalysisError(RuntimeError):
    """Raised when the input cannot be analyzed at all."""


def load_spreads(path: str | Path) -> tuple[pd.DataFrame, int]:
    """Load and validate the CSV.

    Returns the cleaned frame and the number of rows that had to be skipped.
    A row is skipped when its timestamp cannot be parsed, when the pair or venue
    is missing, or when out_amount is not a number. Missing optional columns are
    added as empty so the rest of the code has a stable shape.
    """
    path = Path(path)
    if not path.is_file():
        raise SpreadAnalysisError(f"input file not found: {path}")

    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise SpreadAnalysisError(f"input file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise SpreadAnalysisError(f"could not parse CSV: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise SpreadAnalysisError(
            "missing required column(s): " + ", ".join(missing)
            + f" (found: {', '.join(raw.columns)})"
        )
    if "stale" not in raw.columns:
        raw["stale"] = ""

    original_rows = len(raw)

    frame = raw.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["pair"] = frame["pair"].astype(str).str.strip()
    frame["venue"] = frame["venue"].astype(str).str.strip()

    valid = (
        frame["timestamp"].notna()
        & frame["out_amount"].notna()
        & frame["pair"].ne("")
        & frame["pair"].ne("nan")
        & frame["venue"].ne("")
        & frame["venue"].ne("nan")
    )
    skipped = int((~valid).sum())
    frame = frame.loc[valid].reset_index(drop=True)

    if frame.empty:
        raise SpreadAnalysisError(
            f"no usable rows in {path} ({original_rows} read, {skipped} skipped)"
        )
    return frame, skipped


def scan_order(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (pair, timestamp) giving the aggregated route's output."""
    best = frame.loc[frame["venue"] == BEST, ["pair", "timestamp", "out_amount"]]
    best = best.drop_duplicates(subset=["pair", "timestamp"], keep="first")
    return best.sort_values(["pair", "timestamp"]).reset_index(drop=True)


def detect_frozen(frame: pd.DataFrame, stale_scans: int) -> pd.Series:
    """Flag frozen quotes.

    Per (pair, venue) in time order a quote is a *repeat* when its out_amount is
    exactly equal to that venue's previous quote for the pair AND the aggregated
    best route moved between the two scans. A changed venue quote resets the
    counter; a market where nothing moved leaves it unchanged. A quote is frozen
    once it has ``stale_scans - 1`` consecutive repeats.
    """
    if stale_scans < 2:
        raise SpreadAnalysisError("--stale-scans must be at least 2")

    best_by_scan = scan_order(frame).set_index(["pair", "timestamp"])["out_amount"]
    previous_best: dict[tuple[str, pd.Timestamp], float] = {}
    for pair, group in scan_order(frame).groupby("pair", sort=False):
        timestamps = list(group["timestamp"])
        values = list(group["out_amount"])
        for index in range(1, len(timestamps)):
            previous_best[(pair, timestamps[index])] = values[index - 1]

    frozen = pd.Series(False, index=frame.index)
    ordered = frame.sort_values(["pair", "venue", "timestamp"], kind="mergesort")
    pair_venue = [ordered["pair"], ordered["venue"]]

    previous_out = ordered.groupby(pair_venue, sort=False)["out_amount"].shift(1)
    same_quote = ordered["out_amount"].eq(previous_out)

    scans = scan_order(frame)
    scans["best_before"] = scans.groupby("pair", sort=False)["out_amount"].shift(1)
    moved = scans.rename(columns={"out_amount": "best_now"})[
        ["pair", "timestamp", "best_now", "best_before"]
    ]
    joined = ordered.merge(moved, on=["pair", "timestamp"], how="left")
    market_moved = (
        joined["best_now"].notna()
        & joined["best_before"].notna()
        & joined["best_now"].ne(joined["best_before"])
    )

    increment = same_quote.to_numpy() & market_moved.to_numpy()
    run_id = (~same_quote).groupby(pair_venue, sort=False).cumsum()
    counts = pd.Series(increment, index=ordered.index).groupby(
        [ordered["pair"], ordered["venue"], run_id], sort=False
    ).cumsum()

    reached = counts >= stale_scans - 1
    # Freeze the whole run of identical quotes, not just the scan where the
    # threshold was reached: the earlier copies are the same stuck quote and would
    # otherwise keep fabricating a spread in those scans.
    whole_run = reached.groupby(
        [ordered["pair"], ordered["venue"], run_id], sort=False
    ).transform("max").fillna(False).astype(bool)

    frozen.loc[ordered.index] = whole_run.to_numpy()
    return frozen


def collapse_scans(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per (pair, timestamp) with the spread recomputed from live venues.

    The spread is (max price / min price - 1) * 10,000 over the non-frozen single
    venues, replacing the scanner's own ``cross_venue_spread_bps``. With fewer
    than two live venues the spread is undefined and the scan is not anomalous.
    """
    single = frame.loc[(frame["venue"] != BEST) & (~frame["frozen"])]
    grouped = single.groupby(["pair", "timestamp"], sort=True)

    aggregated = grouped.agg(
        live_venues=("venue", "nunique"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        min_vs_best=("vs_best_bps", "min"),
    ).reset_index()

    # The outlier is the live venue with the lowest vs_best_bps. Doing this with a
    # python-level lambda per scan costs tens of seconds on a 500k-row file, so the
    # index of the minimum is taken vectorised and mapped back.
    outliers = (
        single.loc[single.groupby(["pair", "timestamp"], sort=True)["vs_best_bps"].idxmin().dropna()]
        .drop_duplicates(subset=["pair", "timestamp"])
        .set_index(["pair", "timestamp"])["venue"]
    )
    aggregated["outlier_venue"] = (
        aggregated.set_index(["pair", "timestamp"]).index.map(outliers).to_numpy()
    )

    spread = pd.Series(float("nan"), index=aggregated.index)
    enough = aggregated["live_venues"] >= 2
    spread.loc[enough] = (
        aggregated.loc[enough, "max_price"] / aggregated.loc[enough, "min_price"] - 1.0
    ) * 10_000.0
    aggregated["spread_bps"] = spread
    aggregated["anomalous"] = aggregated["spread_bps"] >= threshold

    round_trip = frame.groupby(["pair", "timestamp"], sort=True)["round_trip_bps"].first()
    aggregated["round_trip_bps"] = aggregated.set_index(["pair", "timestamp"]).index.map(round_trip)

    return aggregated.sort_values(["pair", "timestamp"]).reset_index(drop=True)


def _outlier_venue(scan_rows: pd.DataFrame) -> str | None:
    """The live venue with the lowest vs_best_bps creates the spread."""
    candidates = scan_rows.loc[
        (scan_rows["venue"] != BEST) & scan_rows["vs_best_bps"].notna()
    ]
    if candidates.empty:
        return None
    return str(candidates.loc[candidates["vs_best_bps"].idxmin(), "venue"])


def find_runs(scans: pd.DataFrame, min_scans: int, max_gap: float) -> pd.DataFrame:
    """Persistent anomaly runs per pair, in time order."""
    records: list[dict] = []

    for pair, group in scans.groupby("pair", sort=True):
        group = group.sort_values("timestamp")
        run: list = []
        previous_ts = None

        def close(current: list) -> None:
            if len(current) >= min_scans:
                spreads = [s["spread_bps"] for s in current]
                trips = [s["round_trip_bps"] for s in current if s["round_trip_bps"] is not None]
                venues = [s["outlier_venue"] for s in current if s["outlier_venue"]]
                dominant = pd.Series(venues).mode().iloc[0] if venues else None
                records.append(
                    {
                        "pair": pair,
                        "start": current[0]["timestamp"],
                        "end": current[-1]["timestamp"],
                        "duration_s": (current[-1]["timestamp"] - current[0]["timestamp"]).total_seconds(),
                        "scans": len(current),
                        "mean_spread_bps": sum(spreads) / len(spreads),
                        "max_spread_bps": max(spreads),
                        "dominant_outlier": dominant,
                        "mean_round_trip_bps": (sum(trips) / len(trips)) if trips else None,
                    }
                )

        for row in group.itertuples(index=False):
            if not row.anomalous:
                close(run)
                run = []
                previous_ts = row.timestamp
                continue
            if previous_ts is not None and (row.timestamp - previous_ts).total_seconds() > max_gap:
                close(run)
                run = []
            run.append(
                {
                    "timestamp": row.timestamp,
                    "spread_bps": row.spread_bps,
                    "round_trip_bps": row.round_trip_bps,
                    "outlier_venue": row.outlier_venue,
                }
            )
            previous_ts = row.timestamp
        close(run)

    columns = [
        "pair", "start", "end", "duration_s", "scans",
        "mean_spread_bps", "max_spread_bps", "dominant_outlier", "mean_round_trip_bps",
    ]
    return pd.DataFrame(records, columns=columns)


def frozen_periods(frame: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive frozen quotes into periods."""
    periods: list[dict] = []
    frozen_rows = frame.loc[frame["frozen"]].sort_values(["pair", "venue", "timestamp"])

    for (pair, venue), group in frozen_rows.groupby(["pair", "venue"], sort=True):
        start = previous = None
        for row in group.itertuples(index=False):
            if start is None:
                start = previous = row
                continue
            if row.timestamp - previous.timestamp > pd.Timedelta(seconds=600):
                periods.append(_period(pair, venue, start, previous))
                start = row
            previous = row
        if start is not None:
            periods.append(_period(pair, venue, start, previous))

    return pd.DataFrame(
        periods,
        columns=["pair", "venue", "start", "end", "duration_s", "out_amount", "distance_bps"],
    )


def _period(pair: str, venue: str, start, end) -> dict:
    return {
        "pair": pair,
        "venue": venue,
        "start": start.timestamp,
        "end": end.timestamp,
        "duration_s": (end.timestamp - start.timestamp).total_seconds(),
        "out_amount": start.out_amount,
        "distance_bps": start.vs_best_bps,
    }


def per_pair_stats(scans: pd.DataFrame, runs: pd.DataFrame, frozen_count: pd.Series) -> pd.DataFrame:
    rows = []
    for pair, group in scans.groupby("pair", sort=True):
        spreads = group["spread_bps"].dropna()
        trips = group["round_trip_bps"].dropna()
        pair_runs = runs.loc[runs["pair"] == pair] if not runs.empty else runs
        longest = pair_runs.sort_values("duration_s", ascending=False).head(1)
        rows.append(
            {
                "pair": pair,
                "scans": len(group),
                "window_h": (group["timestamp"].max() - group["timestamp"].min()).total_seconds() / 3600.0,
                "pct_at_threshold": float(group["anomalous"].mean() * 100.0),
                "median_spread": float(spreads.median()) if len(spreads) else None,
                "p95_spread": float(spreads.quantile(0.95)) if len(spreads) else None,
                "max_spread": float(spreads.max()) if len(spreads) else None,
                "persistent_runs": len(pair_runs),
                "longest_run_scans": int(longest["scans"].iloc[0]) if len(longest) else 0,
                "longest_run_s": float(longest["duration_s"].iloc[0]) if len(longest) else 0.0,
                "median_round_trip": float(trips.median()) if len(trips) else None,
                "pct_positive_round_trip": float((trips > 0).mean() * 100.0) if len(trips) else None,
                "frozen_excluded": int(frozen_count.get(pair, 0)),
            }
        )
    return pd.DataFrame(rows)


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def _fmt(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:.{digits}f}"


def build_report(
    frame: pd.DataFrame,
    scans: pd.DataFrame,
    runs: pd.DataFrame,
    periods: pd.DataFrame,
    args: argparse.Namespace,
    skipped: int,
) -> str:
    start = frame["timestamp"].min()
    end = frame["timestamp"].max()
    lines: list[str] = []

    lines.append("# Spread Persistence Report")
    lines.append("")
    lines.append("## Input and parameters")
    lines.append("")
    lines.append(f"- Input file: `{args.csv}`")
    lines.append(f"- Time range: {start.strftime('%Y-%m-%d %H:%M:%SZ')} → {end.strftime('%Y-%m-%d %H:%M:%SZ')}")
    lines.append(f"- Rows used: {len(frame)}")
    lines.append(f"- Rows skipped as malformed: {skipped}")
    lines.append(
        f"- Parameters: threshold={args.threshold} bps, min-scans={args.min_scans}, "
        f"max-gap={args.max_gap}s, stale-scans={args.stale_scans}"
    )
    lines.append("")

    frozen_count = frame.loc[frame["frozen"]].groupby("pair").size()
    stats = per_pair_stats(scans, runs, frozen_count)

    lines.append("## Per-pair summary")
    lines.append("")
    lines.append(
        "| Pair | Scans | Window | ≥ threshold | Median | p95 | Max | Persistent runs | Longest run | Median round trip | Positive round trip | Frozen excluded |"
    )
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for row in stats.itertuples(index=False):
        lines.append(
            f"| {row.pair} | {row.scans} | {format_duration(row.window_h * 3600)} | "
            f"{_fmt(row.pct_at_threshold)}% | {_fmt(row.median_spread)} | {_fmt(row.p95_spread)} | "
            f"{_fmt(row.max_spread)} | {row.persistent_runs} | "
            f"{row.longest_run_scans} scans / {format_duration(row.longest_run_s)} | "
            f"{_fmt(row.median_round_trip)} | {_fmt(row.pct_positive_round_trip)}% | {row.frozen_excluded} |"
        )
    lines.append("")

    lines.append("## Persistent runs")
    lines.append("")
    if runs.empty:
        lines.append("No run met the persistence criteria.")
    else:
        lines.append(
            "| Pair | Start | End | Duration | Scans | Mean spread | Max spread | Dominant outlier | Mean round trip |"
        )
        lines.append("|---|---|--:|--:|--:|--:|--:|---|---:|")
        for row in runs.sort_values("duration_s", ascending=False).itertuples(index=False):
            lines.append(
                f"| {row.pair} | {row.start.strftime('%H:%M:%S')} | {row.end.strftime('%H:%M:%S')} | "
                f"{format_duration(row.duration_s)} | {row.scans} | {_fmt(row.mean_spread_bps)} | "
                f"{_fmt(row.max_spread_bps)} | {row.dominant_outlier or '—'} | {_fmt(row.mean_round_trip_bps)} |"
            )
    lines.append("")

    lines.append("## Outlier venues during anomalous scans")
    lines.append("")
    anomalous = scans.loc[scans["anomalous"] & scans["outlier_venue"].notna()]
    if anomalous.empty:
        lines.append("No anomalous scans.")
    else:
        counts = anomalous.groupby(["pair", "outlier_venue"]).size().reset_index(name="count")
        totals = anomalous.groupby("pair").size().to_dict()
        lines.append("| Pair | Venue | Anomalous scans as outlier | Share |")
        lines.append("|---|---|--:|--:|")
        for row in counts.sort_values(["pair", "count"], ascending=[True, False]).itertuples(index=False):
            share = 100.0 * row.count / totals.get(row.pair, row.count)
            lines.append(f"| {row.pair} | {row.outlier_venue} | {row.count} | {share:.2f}% |")
    lines.append("")

    lines.append("## Frozen quotes")
    lines.append("")
    if periods.empty:
        lines.append("No frozen quotes detected.")
    else:
        lines.append("| Pair | Venue | Start | End | Duration | Frozen out_amount | Distance from best |")
        lines.append("|---|---|--:|--:|--:|--:|--:|")
        for row in periods.sort_values("duration_s", ascending=False).itertuples(index=False):
            lines.append(
                f"| {row.pair} | {row.venue} | {row.start.strftime('%H:%M:%S')} | "
                f"{row.end.strftime('%H:%M:%S')} | {format_duration(row.duration_s)} | "
                f"{row.out_amount:.10g} | {_fmt(row.distance_bps)} bps |"
            )
    disagreements = int((frame["stale_flag"].astype("boolean").fillna(False) != frame["frozen"]).sum())
    lines.append("")
    lines.append(f"Disagreements with the scanner's own `stale` flag: **{disagreements}**.")
    lines.append("")

    lines.append("## Findings")
    lines.append("")
    for finding in build_findings(stats, runs, periods):
        lines.append(f"- {finding}")
    lines.append("")

    return "\n".join(lines)


def build_findings(stats: pd.DataFrame, runs: pd.DataFrame, periods: pd.DataFrame) -> list[str]:
    findings: list[str] = []

    for row in stats.sort_values("pct_at_threshold", ascending=False).itertuples(index=False):
        if row.pct_at_threshold >= 50 and row.persistent_runs:
            pair_runs = runs.loc[runs["pair"] == row.pair]
            dominant = pair_runs["dominant_outlier"].mode()
            who = dominant.iloc[0] if len(dominant) else "an unnamed venue"
            findings.append(
                f"{row.pair}: spread ≥ threshold in {row.pct_at_threshold:.2f}% of scans, "
                f"{row.persistent_runs} persistent run(s), and {who} was the dominant outlier."
            )
        elif row.persistent_runs == 0:
            findings.append(
                f"{row.pair}: no spread persisted for {row.longest_run_scans or 0} scans — "
                "single-scan spreads there are noise, not opportunity."
            )

    if not periods.empty:
        worst = periods.sort_values("duration_s", ascending=False).iloc[0]
        findings.append(
            f"The longest frozen quote was {worst['venue']} on {worst['pair']}, "
            f"stuck at {worst['out_amount']:.10g} for {format_duration(worst['duration_s'])}, "
            f"{abs(worst['distance_bps']):.2f} bps away from the best route — exactly the kind of "
            "quote that fabricates a spread if it is not excluded."
        )

    if not runs.empty:
        longest = runs.sort_values("duration_s", ascending=False).iloc[0]
        findings.append(
            f"The longest persistent run was {longest['pair']} lasting {format_duration(longest['duration_s'])} "
            f"across {int(longest['scans'])} scans, mean spread {longest['mean_spread_bps']:.2f} bps."
        )
    return findings[:4]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze temporal arbitrage spreads from a scanner CSV.")
    parser.add_argument("csv", help="path to spreads.csv")
    parser.add_argument("--threshold", type=float, default=30.0, help="spread in bps that counts as anomalous (default 30)")
    parser.add_argument("--min-scans", type=int, default=3, help="consecutive anomalous scans for a persistent run (default 3)")
    parser.add_argument("--max-gap", type=float, default=300.0, help="seconds between scans that breaks a run (default 300)")
    parser.add_argument("--stale-scans", type=int, default=5, help="identical scans that make a quote frozen (default 5)")
    parser.add_argument("--out", default=None, help="write the Markdown report to this file (default stdout)")
    parser.add_argument("--runs-csv", default=None, help="also export the persistent runs table as CSV")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        frame, skipped = load_spreads(args.csv)
        frozen = detect_frozen(frame, args.stale_scans)
        frame = frame.assign(frozen=frozen.to_numpy())
        frame["stale_flag"] = (
            frame["stale"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        )
        scans = collapse_scans(frame, args.threshold)
        runs = find_runs(scans, args.min_scans, args.max_gap)
        periods = frozen_periods(frame)
        report = build_report(frame, scans, runs, periods, args, skipped)
    except SpreadAnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"report written to {args.out}", file=sys.stderr)
    else:
        print(report)

    if args.runs_csv:
        export = runs.copy()
        for column in ("start", "end"):
            if column in export:
                export[column] = pd.to_datetime(export[column]).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        export.to_csv(args.runs_csv, index=False)
        print(f"runs written to {args.runs_csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
