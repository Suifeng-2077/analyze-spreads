# Spread Persistence Report

## Input and parameters

- Input file: `sample_spreads.csv`
- Time range: 2026-10-17 16:03:44Z → 2026-10-17 16:03:58Z
- Rows used: 11
- Rows skipped as malformed: 0
- Parameters: threshold=30.0 bps, min-scans=1, max-gap=300.0s, stale-scans=5

## Per-pair summary

| Pair | Scans | Window | ≥ threshold | Median | p95 | Max | Persistent runs | Longest run | Median round trip | Positive round trip | Frozen excluded |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| BONK/SOL | 1 | 0m | 0.00% | 27.24 | 27.24 | 27.24 | 0 | 0 scans / 0m | -11.08 | 0.00% | 0 |
| SOL/USDC | 1 | 0m | 0.00% | 29.01 | 29.01 | 29.01 | 0 | 0 scans / 0m | 1.89 | 100.00% | 0 |

## Persistent runs

No run met the persistence criteria.

## Outlier venues during anomalous scans

No anomalous scans.

## Frozen quotes

No frozen quotes detected.

Disagreements with the scanner's own `stale` flag: **0**.

## Findings

- BONK/SOL: no spread persisted for 0 scans — single-scan spreads there are noise, not opportunity.
- SOL/USDC: no spread persisted for 0 scans — single-scan spreads there are noise, not opportunity.
