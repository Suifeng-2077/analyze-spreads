"""Tests for analyze_spreads.py, using synthetic CSV fixtures written on the fly."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_spreads as A  # noqa: E402

HEADER = [
    "timestamp", "slot", "pair", "amount", "venue", "out_amount", "price",
    "vs_best_bps", "impact_pct", "route", "cross_venue_spread_bps", "round_trip_bps",
]


def write_csv(path: Path, scans: list[dict], with_stale: bool = True) -> Path:
    """Write a scanner CSV. Each scan dict: ts, pair, best, venues{name: price}."""
    header = HEADER + (["stale"] if with_stale else [])
    rows: list[list[str]] = []
    for scan in scans:
        venues = scan["venues"]
        best_price = scan["best"]
        spread = 0.0
        if len(venues) >= 2:
            spread = (max(venues.values()) / min(venues.values()) - 1.0) * 10_000.0
        trip = scan.get("round_trip", "")
        rows.append([
            scan["ts"], str(scan.get("slot", 1)), scan["pair"], str(scan.get("amount", 1)),
            A.BEST, f"{best_price}", f"{best_price}", "", "0", "Jupiter route",
            f"{spread}", f"{trip}",
        ] + ([""] if with_stale else []))
        for name, price in venues.items():
            vs_best = (price / best_price - 1.0) * 10_000.0
            stale = scan.get("stale", {}).get(name, "")
            rows.append([
                scan["ts"], str(scan.get("slot", 1)), scan["pair"], str(scan.get("amount", 1)),
                name, f"{price}", f"{price}", f"{vs_best}", "0.1",
                f"{name} → {name}",  # UTF-8 arrow
                f"{spread}", f"{trip}",
            ] + ([stale] if with_stale else []))
    path.write_text("\ufeff" + _render(header, rows), encoding="utf-8")
    return path


def _render(header: list[str], rows: list[list[str]]) -> str:
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def ts(minute: int, second: int = 0) -> str:
    return f"2026-10-17T16:{minute:02d}:{second:02d}.000Z"


def run_pipeline(path: Path, **kwargs):
    defaults = dict(threshold=30.0, min_scans=2, max_gap=300.0, stale_scans=5)
    defaults.update(kwargs)
    frame, skipped = A.load_spreads(path)
    frozen = A.detect_frozen(frame, defaults["stale_scans"])
    frame = frame.assign(frozen=frozen.to_numpy())
    frame["stale_flag"] = (
        frame["stale"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
    )
    scans = A.collapse_scans(frame, defaults["threshold"])
    runs = A.find_runs(scans, defaults["min_scans"], defaults["max_gap"])
    periods = A.frozen_periods(frame)
    return frame, scans, runs, periods, skipped


# --------------------------------------------------------------------------- runs

def test_run_of_exactly_min_scans_is_persistent(tmp_path: Path) -> None:
    scans = [
        {"ts": ts(m), "pair": "SOL/USDC", "best": 103.0, "venues": {"A": 100.0, "B": 100.5}}
        for m in (0, 2, 4)
    ]
    path = write_csv(tmp_path / "s.csv", scans)
    _, _, runs, _, _ = run_pipeline(path, min_scans=3, threshold=30)
    assert len(runs) == 1
    assert int(runs.iloc[0]["scans"]) == 3


def test_run_of_min_scans_minus_one_is_not_persistent(tmp_path: Path) -> None:
    scans = [
        {"ts": ts(m), "pair": "SOL/USDC", "best": 103.0, "venues": {"A": 100.0, "B": 100.5}}
        for m in (0, 2)
    ]
    path = write_csv(tmp_path / "s.csv", scans)
    _, _, runs, _, _ = run_pipeline(path, min_scans=3, threshold=30)
    assert runs.empty


def test_gap_larger_than_max_gap_breaks_the_run(tmp_path: Path) -> None:
    scans = [
        {"ts": ts(0), "pair": "SOL/USDC", "best": 103.0, "venues": {"A": 100.0, "B": 100.5}},
        {"ts": ts(1), "pair": "SOL/USDC", "best": 103.0, "venues": {"A": 100.0, "B": 100.5}},
        {"ts": ts(20), "pair": "SOL/USDC", "best": 103.0, "venues": {"A": 100.0, "B": 100.5}},
        {"ts": ts(21), "pair": "SOL/USDC", "best": 103.0, "venues": {"A": 100.0, "B": 100.5}},
    ]
    path = write_csv(tmp_path / "s.csv", scans)
    _, _, runs, _, _ = run_pipeline(path, min_scans=3, max_gap=300, threshold=30)
    assert runs.empty, "two pairs of two scans must not chain across a 19-minute gap"


def test_one_scan_below_threshold_breaks_the_run(tmp_path: Path) -> None:
    narrow = {"A": 100.0, "B": 100.05}  # 5 bps, below threshold
    wide = {"A": 100.0, "B": 100.5}      # 50 bps
    scans = [
        {"ts": ts(0), "pair": "SOL/USDC", "best": 103.0, "venues": wide},
        {"ts": ts(2), "pair": "SOL/USDC", "best": 103.0, "venues": wide},
        {"ts": ts(4), "pair": "SOL/USDC", "best": 103.0, "venues": narrow},
        {"ts": ts(6), "pair": "SOL/USDC", "best": 103.0, "venues": wide},
        {"ts": ts(8), "pair": "SOL/USDC", "best": 103.0, "venues": wide},
    ]
    path = write_csv(tmp_path / "s.csv", scans)
    _, scans_df, runs, _, _ = run_pipeline(path, min_scans=3, threshold=30)
    assert runs.empty, "the widest piece here is 2 scans, below a minimum of 3"
    assert scans_df["anomalous"].sum() == 4


def test_threshold_boundary_is_inclusive(tmp_path: Path) -> None:
    scans = [{"ts": ts(0), "pair": "SOL/USDC", "best": 103.0, "venues": {"A": 100.0, "B": 100.3}}]
    path = write_csv(tmp_path / "s.csv", scans)
    frame, _ = A.load_spreads(path)
    frozen = A.detect_frozen(frame, 5)
    frame = frame.assign(frozen=frozen.to_numpy())
    exact = float(A.collapse_scans(frame, 0.0).iloc[0]["spread_bps"])
    assert A.collapse_scans(frame, exact).iloc[0]["anomalous"], "spread == threshold must count"
    assert not A.collapse_scans(frame, exact + 1e-9).iloc[0]["anomalous"]


def test_several_pairs_do_not_mix(tmp_path: Path) -> None:
    scans = []
    for m in (0, 2, 4):
        scans.append({"ts": ts(m), "pair": "SOL/USDC", "best": 103.0, "venues": {"A": 100.0, "B": 100.5}})
        scans.append({"ts": ts(m), "pair": "JUP/USDC", "best": 0.9, "venues": {"A": 0.90, "B": 0.9001}})
    path = write_csv(tmp_path / "s.csv", scans)
    _, scans_df, runs, _, _ = run_pipeline(path, min_scans=2, threshold=30)
    assert set(scans_df["pair"]) == {"SOL/USDC", "JUP/USDC"}
    assert len(runs) == 1
    assert runs.iloc[0]["pair"] == "SOL/USDC"


# ------------------------------------------------------------------------ parsing

def test_missing_venue_scan_is_not_anomalous(tmp_path: Path) -> None:
    scans = [{"ts": ts(0), "pair": "SOL/USDC", "best": 103.0, "venues": {"A": 100.0}}]
    path = write_csv(tmp_path / "s.csv", scans)
    _, scans_df, _, _, _ = run_pipeline(path)
    assert pd.isna(scans_df.iloc[0]["spread_bps"])
    assert not bool(scans_df.iloc[0]["anomalous"])


def test_empty_round_trip_is_tolerated(tmp_path: Path) -> None:
    scans = [{"ts": ts(0), "pair": "SOL/USDC", "best": 103.0, "venues": {"A": 100.0, "B": 100.5}}]
    path = write_csv(tmp_path / "s.csv", scans)
    _, scans_df, runs, _, _ = run_pipeline(path, min_scans=1)
    assert pd.isna(scans_df.iloc[0]["round_trip_bps"])
    assert pd.isna(runs.iloc[0]["mean_round_trip_bps"])


def test_malformed_row_is_skipped_and_counted(tmp_path: Path) -> None:
    scans = [{"ts": ts(0), "pair": "SOL/USDC", "best": 103.0, "venues": {"A": 100.0, "B": 100.5}}]
    path = write_csv(tmp_path / "s.csv", scans)
    text = path.read_text(encoding="utf-8").splitlines()
    text.append("not-a-timestamp,1,SOL/USDC,1,Raydium,100.0,100.0,-1,0,r,0,1,")
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    _, _, _, _, skipped = run_pipeline(path)
    assert skipped == 1


def test_route_with_comma_and_arrow_parses(tmp_path: Path) -> None:
    path = tmp_path / "s.csv"
    path.write_text(
        "\ufeff" + _render(HEADER + ["stale"], [
            [ts(0), "1", "SOL/USDC", "1", A.BEST, "103.0", "103.0", "", "0", "x", "0", "", ""],
            [ts(0), "1", "SOL/USDC", "1", "Raydium", "100.0", "100.0", "-300", "0.1",
             'Raydium → CLMM, "split"', "0", "", ""],
            [ts(0), "1", "SOL/USDC", "1", "Whirlpool", "100.5", "100.5", "-250", "0.1",
             "Whirlpool", "0", "", ""],
        ]) + "\n",
        encoding="utf-8",
    )
    frame, skipped = A.load_spreads(path)
    assert skipped == 0
    assert 'CLMM, "split"' in frame.loc[frame["venue"] == "Raydium", "route"].iloc[0]


def test_missing_required_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("timestamp,slot,pair\n2026-10-17T16:00:00.000Z,1,SOL/USDC\n", encoding="utf-8")
    with pytest.raises(A.SpreadAnalysisError):
        A.load_spreads(path)


def test_twelve_column_format_without_stale(tmp_path: Path) -> None:
    scans = [{"ts": ts(0), "pair": "SOL/USDC", "best": 103.0, "venues": {"A": 100.0, "B": 100.5}}]
    path = write_csv(tmp_path / "s.csv", scans, with_stale=False)
    frame, skipped = A.load_spreads(path)
    assert skipped == 0
    assert "stale" in frame.columns


# ------------------------------------------------------------------------- frozen

def _frozen_series(tmp_path: Path, repeats: int, best_moves: bool = True) -> Path:
    scans = []
    for minute in range(repeats):
        best = 103.0 + (minute * 0.1 if best_moves else 0.0)
        scans.append({
            "ts": ts(minute),
            "pair": "SOL/USDC",
            "best": best,
            "venues": {"Raydium": best - 0.05, "Phoenix": 89.0},
        })
    return write_csv(tmp_path / "f.csv", scans)


def test_frozen_at_exactly_stale_scans(tmp_path: Path) -> None:
    path = _frozen_series(tmp_path, repeats=5, best_moves=True)
    frame, _, _, periods, _ = run_pipeline(path, stale_scans=5)
    assert frame["frozen"].sum() == 5
    assert not periods.empty
    assert periods.iloc[0]["venue"] == "Phoenix"


def test_one_scan_fewer_is_not_frozen(tmp_path: Path) -> None:
    path = _frozen_series(tmp_path, repeats=4, best_moves=True)
    frame, _, _, periods, _ = run_pipeline(path, stale_scans=5)
    assert frame["frozen"].sum() == 0
    assert periods.empty


def test_flat_market_does_not_freeze(tmp_path: Path) -> None:
    path = _frozen_series(tmp_path, repeats=6, best_moves=False)
    frame, _, _, _, _ = run_pipeline(path, stale_scans=5)
    assert frame["frozen"].sum() == 0, "identical quotes while the best route is also flat are not frozen"


def test_frozen_venue_fake_spread_disappears(tmp_path: Path) -> None:
    scans = []
    for minute in range(5):
        best = 103.0 + minute * 0.1
        scans.append({
            "ts": ts(minute),
            "pair": "SOL/USDC",
            "best": best,
            "venues": {"Raydium": best - 0.05, "Whirlpool": best - 0.02, "Phoenix": 89.0},
        })
    path = write_csv(tmp_path / "f.csv", scans)

    frame, _ = A.load_spreads(path)
    naive = A.collapse_scans(frame.assign(frozen=False), 30.0)
    assert bool(naive.iloc[-1]["anomalous"]), "without frozen detection the spread looks enormous"

    frozen = A.detect_frozen(frame, 5)
    aware = A.collapse_scans(frame.assign(frozen=frozen.to_numpy()), 30.0)
    assert not bool(aware.iloc[-1]["anomalous"]), "excluding Phoenix must remove the fake spread"


def test_stale_flag_disagreement_is_reported(tmp_path: Path) -> None:
    scans = [
        {"ts": ts(0), "pair": "SOL/USDC", "best": 103.0, "venues": {"A": 100.0, "B": 100.5},
         "stale": {"A": "true"}},
    ]
    path = write_csv(tmp_path / "s.csv", scans)
    frame, _, runs, periods, _ = run_pipeline(path, min_scans=1)
    report = A.build_report(
        frame, A.collapse_scans(frame, 30.0), runs, periods,
        argparse_namespace(path), 0,
    )
    assert "Disagreements with the scanner's own `stale` flag: **1**" in report


# ------------------------------------------------------------------------- report

class argparse_namespace:
    def __init__(self, path: Path) -> None:
        self.csv = str(path)
        self.threshold = 30.0
        self.min_scans = 2
        self.max_gap = 300.0
        self.stale_scans = 5
        self.out = None
        self.runs_csv = None


def test_report_has_all_sections_and_formats_durations(tmp_path: Path) -> None:
    scans = [
        {"ts": ts(0), "pair": "SOL/USDC", "best": 103.0, "venues": {"A": 100.0, "B": 100.5}, "round_trip": 12.5},
        {"ts": ts(5), "pair": "SOL/USDC", "best": 103.0, "venues": {"A": 100.0, "B": 100.5}, "round_trip": 8.0},
    ]
    path = write_csv(tmp_path / "s.csv", scans)
    frame, scans_df, runs, periods, skipped = run_pipeline(path, min_scans=2, threshold=30)
    report = A.build_report(frame, scans_df, runs, periods, argparse_namespace(path), skipped)
    for heading in (
        "## Input and parameters",
        "## Per-pair summary",
        "## Persistent runs",
        "## Outlier venues during anomalous scans",
        "## Frozen quotes",
        "## Findings",
    ):
        assert heading in report
    assert "5m" in A.format_duration(300)
    assert A.format_duration(3900) == "1h 05m"


def test_runs_csv_export_round_trip(tmp_path: Path) -> None:
    scans = [
        {"ts": ts(0), "pair": "SOL/USDC", "best": 103.0, "venues": {"A": 100.0, "B": 100.5}},
        {"ts": ts(2), "pair": "SOL/USDC", "best": 103.0, "venues": {"A": 100.0, "B": 100.5}},
    ]
    path = write_csv(tmp_path / "s.csv", scans)
    out = tmp_path / "report.md"
    runs_csv = tmp_path / "runs.csv"
    code = A.main([str(path), "--min-scans", "2", "--out", str(out), "--runs-csv", str(runs_csv)])
    assert code == 0
    assert out.is_file() and "Spread Persistence Report" in out.read_text(encoding="utf-8")
    exported = pd.read_csv(runs_csv)
    assert len(exported) == 1
    assert exported.iloc[0]["pair"] == "SOL/USDC"
