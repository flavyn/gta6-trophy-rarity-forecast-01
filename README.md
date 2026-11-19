# RarityForecaster — `gta6-trophy-rarity-forecast-01`

`main.ts` (Node 22, standard library only, no network) reads every `*.in` case in the input directory and writes one
`<id>.out` forecast per case: for each trophy of the case, five quantiles of its Exophase earn rate at the case's
target, at the levels 0.05, 0.25, 0.5, 0.75 and 0.95.

Outside the grader's container:

```
mkdir out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## What it models

The page is a snapshot, not a live number: Exophase rewrites a game's counts on its own schedule, so between refreshes
a read repeats its predecessor byte for byte. On launch night the house saw the page frozen at 638 players from
23:48Z to 04:00Z, then at 3,868 players by 04:30Z, with every rate but the one at 100% lower.

1. **Runs of unchanged reads.** Reads are collapsed into runs; the gaps between run starts give the refresh interval
   `mu`, and `p = 1 - exp(-horizon / mu)` is the chance a refresh lands inside the horizon. Before the history shows
   any refresh, a prior interval stands in.
2. **Drift of the fresh level.** Each run's level is a point on the logit scale against `u = ln(1 + hours since
   release)`. The slope is fitted by least squares over the last 30 hours of runs, shrunk toward zero, and shrunk
   55/45 toward the cross-trophy median slope, because the dilution is one process for every trophy. It is then
   damped by 0.85: a chord slope read off a saturating curve keeps overshooting past the saturation.
3. **The mixture the forecast is.** No refresh in the horizon leaves the level exactly where the last read put it;
   at least one refresh leaves it at the fitted fresh level. The five quantiles are read off that mixture's CDF, so a
   5% quantile drops below the no-refresh level only when a refresh is likely, and a 95% one does not pay for a drop
   nobody expects. With no observed refresh at all, a prior drop of 1.45 logit (638 -> 3,868 tracked cut the launch
   rates about 4.5x) stands in.
4. **Width.** The band combines the measured spread of the trophies' own drifts (a robust MAD net of fit noise), the
   fit's noise, a prior that the regime changes, a random walk of 0.45 logit per square-root day, and a floor. A
   rate under 0.05% — one or two players on a small page — is widened further, since its count noise dominates.

## Measured

See the builder's report. Nothing here is tuned to the tournament's own answer, which is not available offline.
