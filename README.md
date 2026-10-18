# analyze_spreads

Reads the scanner's `spreads.csv` and reports which arbitrage spreads **persist** across
consecutive scans, after removing frozen quotes that fabricate spreads that are not real.

A spread seen in a single scan is usually noise: the scanner takes its quotes about a second apart.
What matters is the spread that survives several scans. Separately, some venues return the same
output for many minutes while the rest of the market moves — one stuck Phoenix order book quoted
`89.018192` USDC per SOL for over 20 minutes while the market sat near 103, a fake spread of roughly
1,500 bps. Those quotes are detected and excluded before any spread is computed.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+ and pandas. Nothing else, no network access, no API keys.

## Usage

```bash
python analyze_spreads.py spreads.csv [--threshold 30] [--min-scans 3] [--max-gap 300] \
    [--stale-scans 5] [--out report.md] [--runs-csv runs.csv]
```

| Option | Default | Meaning |
|---|---|---|
| `--threshold` | 30 | spread in bps at or above which a scan counts as anomalous |
| `--min-scans` | 3 | consecutive anomalous scans needed for a run to count as persistent |
| `--max-gap` | 300 | seconds between two scans of the same pair after which a run is broken |
| `--stale-scans` | 5 | identical quotes, while the market moves, that make a quote frozen |
| `--out` | stdout | write the Markdown report here |
| `--runs-csv` | — | also export the persistent runs table as CSV |

Exit codes: `0` success, `2` the input could not be analyzed (missing file, missing column, no
usable rows). Malformed rows inside the file never abort the run — they are skipped and counted in
the report.

## What it computes

1. **Load and validate.** Timestamps are parsed as UTC. A missing required column is a clear error;
   unparseable rows are skipped and counted.
2. **Frozen quotes**, per pair and venue in time order, independent of the scanner's own `stale`
   column. A quote is a *repeat* when `out_amount` equals that venue's previous quote for the pair
   **and** the `Jupiter (best)` output changed between the two scans. A changed venue quote resets
   the counter; a flat market where neither moved leaves the counter unchanged. Reaching
   `--stale-scans - 1` repeats freezes the quote, and the whole run of identical quotes is excluded
   so the fake spread disappears from the earlier scans too. Disagreements with `stale` are counted.
3. **One row per scan.** The spread is recomputed from the non-frozen single venues:
   `(max price / min price - 1) × 10,000`. Fewer than two live venues means the spread is undefined
   and the scan is not anomalous.
4. **Outlier venue**: the live single venue with the lowest `vs_best_bps`.
5. **Persistent runs** per pair: consecutive anomalous scans with no gap above `--max-gap`, kept only
   when they contain at least `--min-scans` scans.
6. **Per-pair statistics** and a six-section Markdown report, with 2–4 findings derived from the data.

Both the 13-column format (with `stale`) and the older 12-column format are accepted.

## Tests

```bash
pytest -q
```

19 tests with synthetic fixtures built inside the tests, covering the run boundary (`min-scans`
exactly counts, `min-scans − 1` does not), gaps, a single scan below the threshold, the threshold
boundary being inclusive, several pairs in one file, missing venues, empty `round_trip_bps`, a
malformed row, a `route` containing a quoted comma and a UTF-8 arrow, every frozen-quote case
(exactly `--stale-scans` freezes, one fewer does not, a flat market does not, a frozen venue's fake
spread disappears after exclusion), the 12-column format, and a `stale` flag that disagrees with the
computed result.

## Performance

A synthetic 500,000-row file (66 MB, six venues, three pairs) runs in **3.9 s** on a laptop CPU
(13th Gen Intel i7-13620H). The hot paths are vectorised: the per-scan outlier is taken with
`groupby().idxmin()` rather than a Python callback, and frozen detection is a chain of grouped shifts
and cumulative sums rather than a row loop.

## Example

`sample_spreads.csv` is the two-scan sample from the brief. Running

```bash
python analyze_spreads.py sample_spreads.csv --min-scans 1 --out example_report.md
```

produces `example_report.md`. Excerpt:

```
## Per-pair summary

| Pair | Scans | Window | ≥ threshold | Median | p95 | Max | Persistent runs | ... |
|---|--:|--:|--:|--:|--:|--:|--:|
| BONK/SOL | 1 | 0m | 0.00% | 27.24 | 27.24 | 27.24 | 0 | ... |
| SOL/USDC | 1 | 0m | 0.00% | 29.01 | 29.01 | 29.01 | 0 | ... |

## Persistent runs

No run met the persistence criteria.
```

With `--min-scans 1` the two scans each become a one-scan run, which is the point: the sample is two
scans forty seconds apart, so nothing in it can be persistent yet, and the report says so instead of
inventing a finding.

## Notes

The script only reads a CSV. It never connects to a wallet or a blockchain, never places a trade, and
makes no network calls.
