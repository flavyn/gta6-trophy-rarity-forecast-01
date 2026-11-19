// RarityForecaster — gta6-trophy-rarity-forecast-01 (Node 22, standard library only, no network).
//
// Forecasts, for every trophy of a case, five quantiles of its Exophase earn rate at the case's target, scored by the
// mean pinball loss on the logit scale over every trophy and level. Three things drive the model:
//
// 1. The page is a stale snapshot, not a live number. Exophase rewrites a game's counts on its own schedule (the house
//    saw launch night frozen at 638 players from 23:48Z to 04:00Z, then 3,868 players by 04:30Z, every rate but the
//    100% one lower). So a read repeats its predecessor byte for byte between refreshes, and a forecast that only
//    reads "the last rate" is forecasting a page that may be hours old. The model therefore tracks the runs of
//    unchanged reads, measures the refresh interval mu from the observed run starts, and treats "a refresh lands inside
//    the horizon" as a real event with probability p = 1 - exp(-horizon / mu).
//
// 2. Between refreshes the rate only moves at a refresh, and each refresh dilutes the rates (the newly tracked players
//    hold fewer trophies). The level of the freshest run is therefore modelled on the logit scale against log-time,
//    u = ln(1 + hours since release), which decays the drift instead of running a straight line for days. The slope is
//    fitted by least squares over the recent runs, shrunk toward zero, and then shrunk toward the cross-trophy median
//    slope: the dilution is one process for all trophies, so a rare trophy's noisy slope borrows from it.
//
// 3. The forecast is the mixture of the two outcomes that actually exist: no refresh in the horizon (the level stays
//    where the last read put it) and at least one refresh (the level is the fitted fresh level at the target). Its
//    quantiles are read off that mixture's CDF, so the 5% quantile sits below the no-refresh level only when a refresh
//    is likely, and the 95% does not pay for a drop nobody expects. A condition with no observed refresh at all (the
//    launch-night practice) falls back to a prior drop, so an unseen refresh still lands inside the band.
//
// logit(x) = ln(c / (100 - c)) with c = x clamped to [0.01, 99.99], digit for digit the scorer's clamp.
//
// Outside the grader's container: MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const DEBUG = process.env.MAKEFUN_DEBUG === '1';

/** The five levels, in the order every row lists its quantiles. */
const LEVELS: readonly number[] = [0.05, 0.25, 0.5, 0.75, 0.95];

// ------------------------------------------------------------------ tunables
/** Refresh interval prior, hours, used until the history shows one (launch night froze 4.7 h, so a few hours). */
const PRIOR_MU_HOURS = 1.0;
/** Logit drop of one refresh, launch-night prior: 638 -> 3,868 tracked cut the rates about 4.5x. */
const REFRESH_DROP_PRIOR = -1.45;
/** Standard deviation of that prior drop, logit units. */
const REFRESH_DROP_SD = 0.55;
/** Random-walk standard deviation of the logit level, per square-root day, beyond the fitted drift. */
const RW_DAILY = 0.45;
/** Never forecast a tighter band than this, logit units. */
const SIGMA_FLOOR = 0.3;
/** The fitted slope's own shrinkage toward zero (fit noise). */
const SLOPE_SHRINK = 0.7;
/** How much of the cross-trophy median slope every trophy borrows. */
const SLOPE_POOL = 0.55;
/** Hard cap on |slope|, logit units per unit of ln(1 + hours). */
const SLOPE_MAX = 1.2;
/** The prior that the drift itself changes after the checkpoint, logit units per unit of ln(1 + hours). */
const REGIME_SD = 0.18;
/** Floor and cap of the measured cross-trophy drift spread, logit units per unit of ln(1 + hours). */
const DRIFT_SPREAD_FLOOR = 0.08;
const DRIFT_SPREAD_CAP = 0.6;
/** The most logit units the drift may move the median away from the last observed level. */
const MAX_CHANGE = 2.5;
/** How much of the extrapolated drift is taken: a chord slope overshoots where the curve saturates. */
const DRIFT_DAMP = 0.85;
/** The no-refresh arm's width: the page is literally unchanged, only its 2-decimal rounding moves. */
const FLAT_SIGMA = 0.03;
/** The fit window, hours before the checkpoint. */
const FIT_WINDOW_HOURS = 30;
/** A rate under this (percent) is one or two players on a small page: widen its band. */
const RARE_RATE = 0.05;
/** The most refreshes a no-history fallback assumes. */
const MAX_PRIOR_REFRESHES = 2;

// ------------------------------------------------------------------ numerics
const clamp = (x: number, lo: number, hi: number): number => (x < lo ? lo : x > hi ? hi : x);
const clampRate = (x: number): number => clamp(x, 0.01, 99.99);
const logit = (pct: number): number => {
  const c = clampRate(pct);
  return Math.log(c / (100 - c));
};
const invLogitPct = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** The standard normal's CDF (Abramowitz & Stegun 26.2.17), to about 1e-7. */
function normCdf(z: number): number {
  const s = z < 0 ? -1 : 1;
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const erf = s * (1 - poly * Math.exp(-x * x));
  return 0.5 * (1 + erf);
}

function medianOf(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? (s[m] as number) : ((s[m - 1] as number) + (s[m] as number)) / 2;
}

/** One arm of the mixture: its centre, its width, and its share of the mass. */
interface Arm {
  l: number;
  s: number;
  p: number;
}

/** The quantile at `tau` of the mixture of normal arms (p summing to 1), by bisection on the logit axis. */
function mixtureQuantile(tau: number, arms: Arm[]): number {
  const cdf = (u: number): number => {
    let total = 0;
    for (const a of arms) total += a.p * normCdf((u - a.l) / a.s);
    return total;
  };
  let lo = -20;
  let hi = 20;
  for (let i = 0; i < 90; i++) {
    const mid = (lo + hi) / 2;
    if (cdf(mid) < tau) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

// ------------------------------------------------------------------ the case
interface Trophy {
  id: number;
}
interface Read {
  slot?: string;
  at: string;
  players_tracked?: number;
  trophies_earned?: number;
  rates: Array<number | null>;
}
interface Case {
  schema?: string;
  id?: string;
  at: string;
  target?: string;
  horizon_minutes?: number;
  levels?: number[];
  game?: { release_date?: string; name?: string; platform?: string };
  trophies: Trophy[];
  history?: Read[];
}

/** A run of reads that the page did not change through: the level it shows, and when that run began. */
interface Run {
  tMs: number;
  u: number;
  key: string;
  rates: Array<number | null>;
}

interface Fit {
  b: number;
  uBar: number;
  lBar: number;
  sxx: number;
  residVar: number;
  n: number;
}

/** Least squares of a level on u, with the residual variance for the extrapolation's own width. */
function fitLine(points: Array<{ u: number; l: number }>): Fit | null {
  const n = points.length;
  if (n < 3) return null;
  let su = 0;
  let sl = 0;
  for (const p of points) {
    su += p.u;
    sl += p.l;
  }
  const uBar = su / n;
  const lBar = sl / n;
  let sxx = 0;
  let sxl = 0;
  for (const p of points) {
    const du = p.u - uBar;
    sxx += du * du;
    sxl += du * (p.l - lBar);
  }
  if (!(sxx > 1e-9)) return null;
  const b = sxl / sxx;
  let rss = 0;
  for (const p of points) {
    const r = p.l - (lBar + b * (p.u - uBar));
    rss += r * r;
  }
  const residVar = n > 2 ? rss / (n - 2) : 0.16;
  return { b, uBar, lBar, sxx, residVar, n };
}

/** The house baseline's numbers for one trophy: persistence with a 0.5 x sqrt(days) logit spread. */
function persistenceRow(id: number, lastRate: number | null, horizonMinutes: number): { id: number; q: number[] } {
  const l = logit(lastRate ?? 0);
  const spread = 0.5 * Math.sqrt(Math.max(0, horizonMinutes) / 1440);
  const Z = [-1.645, -0.674, 0, 0.674, 1.645];
  return { id, q: Z.map((z) => round4(invLogitPct(l + z * spread))) };
}

/** One case's rows. */
function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const history = Array.isArray(c.history) ? c.history : [];
  const trophies = Array.isArray(c.trophies) ? c.trophies : [];
  const atMs = Date.parse(c.at);
  const atSafe = Number.isFinite(atMs) ? atMs : Date.now();
  const horizonMinutes = Number.isFinite(c.horizon_minutes) ? (c.horizon_minutes as number) : 90;
  const targetMs = c.target !== undefined && Number.isFinite(Date.parse(c.target)) ? Date.parse(c.target) : atSafe + horizonMinutes * 60_000;
  const hHours = Math.max(1 / 60, (targetMs - atSafe) / 3_600_000);
  const hDays = hHours / 24;

  if (history.length === 0) {
    return trophies.map((t) => persistenceRow(t.id, null, horizonMinutes));
  }

  const releaseText = c.game?.release_date ?? c.at.slice(0, 10);
  const releaseMs = Number.isFinite(Date.parse(`${releaseText}T00:00:00Z`)) ? Date.parse(`${releaseText}T00:00:00Z`) : atSafe;
  const uOf = (ms: number): number => Math.log(1 + Math.max(0, (ms - releaseMs) / 3_600_000));
  const uAt = uOf(atSafe);
  const uT = uOf(targetMs);

  // The runs of unchanged reads: the page only moves at a refresh.
  const runs: Run[] = [];
  for (const r of history) {
    if (!r || !Array.isArray(r.rates)) continue;
    const at = Number.isFinite(Date.parse(r.at)) ? Date.parse(r.at) : atSafe;
    const key = `${r.players_tracked ?? -1}|${r.rates.join(',')}`;
    if (runs.length === 0 || key !== (runs[runs.length - 1] as Run).key) {
      runs.push({ tMs: at, u: uOf(at), key, rates: r.rates });
    }
  }
  if (runs.length === 0) return trophies.map((t) => persistenceRow(t.id, null, horizonMinutes));

  // The refresh interval: the gaps between run starts, or a prior while the page has not moved yet.
  const gaps: number[] = [];
  for (let i = 1; i < runs.length; i++) {
    const gap = ((runs[i] as Run).tMs - (runs[i - 1] as Run).tMs) / 3_600_000;
    if (gap >= 0.08) gaps.push(gap);
  }
  let mu: number;
  if (gaps.length >= 3) mu = medianOf(gaps);
  else if (gaps.length >= 1) mu = (medianOf(gaps) + PRIOR_MU_HOURS) / 2;
  else mu = PRIOR_MU_HOURS;
  mu = clamp(mu, 0.25, 12);
  const pRefresh = 1 - Math.exp(-hHours / mu);
  const expectedRefreshes = hHours / mu;

  const lastRead = history[history.length - 1] as Read;
  const lastRates = Array.isArray(lastRead.rates) ? lastRead.rates : [];

  // A trophy the last read lacks (or every read lacks) borrows the cross-trophy level: the shape prior for a trophy
  // that has no reading of its own yet.
  const seenLevels: number[] = [];
  for (const r of runs) for (const rate of r.rates) if (typeof rate === 'number' && Number.isFinite(rate)) seenLevels.push(logit(rate));
  const commonLevel = seenLevels.length > 0 ? medianOf(seenLevels) : -4.6;
  const levelOf = (i: number): number => {
    for (let k = runs.length - 1; k >= 0; k--) {
      const rate = (runs[k] as Run).rates[i];
      if (typeof rate === 'number' && Number.isFinite(rate)) return logit(rate);
    }
    return commonLevel;
  };
  const rateNowOf = (i: number): number | null => {
    const rate = lastRates[i];
    return typeof rate === 'number' && Number.isFinite(rate) ? rate : null;
  };

  // Fit one slope per trophy over the recent runs, then pool them.
  const windowStart = atSafe - FIT_WINDOW_HOURS * 3_600_000;
  const recent = runs.filter((r, i) => i === runs.length - 1 || r.tMs >= windowStart);
  const ownSlopes: Array<number | null> = [];
  const fits: Array<Fit | null> = [];
  const slopeSeSq: Array<number | null> = [];
  for (let i = 0; i < trophies.length; i++) {
    const points: Array<{ u: number; l: number }> = [];
    for (const run of recent) {
      const rate = run.rates[i];
      if (typeof rate === 'number' && Number.isFinite(rate)) points.push({ u: run.u, l: logit(rate) });
    }
    const fit = fitLine(points);
    fits.push(fit);
    ownSlopes.push(fit === null ? null : clamp(fit.b, -SLOPE_MAX, SLOPE_MAX));
    slopeSeSq.push(fit === null || !(fit.sxx > 0) ? null : fit.residVar / fit.sxx);
  }
  const defined = ownSlopes.filter((b): b is number => b !== null);
  const pooledSlope = defined.length > 0 ? medianOf(defined) : 0;

  // How much the trophies' drifts really differ, net of the fit noise: a robust MAD around the pooled slope, minus the
  // median estimation variance. A page of trophies that all drift together gets a narrow band from this; one where
  // they scatter gets a wide one.
  const noiseVar = medianOf(slopeSeSq.filter((v): v is number => v !== null));
  const mad = medianOf(defined.map((b) => Math.abs(b - pooledSlope))) * 1.4826;
  const driftSpread = clamp(Math.sqrt(Math.max(0, mad * mad - Math.max(0, noiseVar))), DRIFT_SPREAD_FLOOR, DRIFT_SPREAD_CAP);

  const rows: Array<{ id: number; q: number[] }> = [];
  for (let i = 0; i < trophies.length; i++) {
    const trophy = trophies[i] as Trophy;
    const rateNow = rateNowOf(i);
    const lNow = levelOf(i);
    const fit = fits[i] as Fit | null;
    const own = ownSlopes[i] ?? null;

    let lRef: number;
    let sRef: number;
    if (fit !== null && own !== null) {
      const w = SLOPE_SHRINK * (1 - SLOPE_POOL);
      const b = w * own + SLOPE_POOL * pooledSlope;
      const du = uT - fit.uBar;
      // Two unbiased reads of where a refreshed level would sit at the target: the fitted line read at uT, and the
      // last level the page actually showed carried forward by the same drift. Averaging them keeps the latest
      // observation when the line's centroid sits a refresh behind it.
      const lineAt = fit.lBar + b * du;
      const carried = lNow + b * (uT - uAt);
      lRef = 0.5 * lineAt + 0.5 * carried;
      // A chord slope read off a saturating curve keeps overshooting past the saturation, so only part of the
      // extrapolated drift is taken, and a single regime's drift is not worth extrapolating without bound.
      lRef = lNow + clamp((lRef - lNow) * DRIFT_DAMP, -MAX_CHANGE, MAX_CHANGE);
      // The truth drifts at a_i ~ (pooled, driftSpread); the forecast drifts at w*own + (1-w)*pooled. What is left,
      // plus the fit's own noise, plus a possible regime change, times the extrapolation distance.
      const ownSeSq = slopeSeSq[i] ?? 0;
      const varSlope = (1 - w) * (1 - w) * driftSpread * driftSpread + w * w * ownSeSq + REGIME_SD * REGIME_SD;
      const fitVar = fit.residVar / fit.n;
      sRef = Math.sqrt(fitVar + varSlope * du * du + RW_DAILY * RW_DAILY * hDays + SIGMA_FLOOR * SIGMA_FLOOR);
    } else {
      // No observed refresh: a prior drop, one refresh while the horizon is short, more if it is not.
      // No observed refresh: a prior drop, one refresh while the horizon is short, more if it is not. Each later
      // refresh dilutes less than the first, so the prior total stays bounded like any other drift.
      const assumed = expectedRefreshes < 2 ? 1 : Math.min(expectedRefreshes, MAX_PRIOR_REFRESHES);
      lRef = lNow + clamp(REFRESH_DROP_PRIOR * assumed, -MAX_CHANGE, MAX_CHANGE);
      sRef = Math.sqrt(
        REFRESH_DROP_SD * REFRESH_DROP_SD * assumed + RW_DAILY * RW_DAILY * hDays + SIGMA_FLOOR * SIGMA_FLOOR,
      );
    }

    // One or two players on a small page: the count's own noise dominates.
    if (typeof rateNow === 'number' && rateNow > 0 && rateNow < RARE_RATE) sRef = Math.sqrt(sRef * sRef + 0.35 * 0.35);

    lRef = clamp(lRef, -9.2, 9.2);
    const arms: Arm[] =
      pRefresh >= 0.995
        ? [{ l: lRef, s: sRef, p: 1 }]
        : [
            { l: lNow, s: FLAT_SIGMA, p: 1 - pRefresh },
            { l: lRef, s: sRef, p: pRefresh },
          ];

    const q = LEVELS.map((tau) => round4(clampRate(invLogitPct(mixtureQuantile(tau, arms)))));
    for (let k = 1; k < q.length; k++) if (q[k] < q[k - 1]) q[k] = q[k - 1];
    rows.push({ id: trophy.id, q });

    if (DEBUG) {
      process.stderr.write(
        `trophy ${trophy.id}: now=${typeof rateNow === 'number' ? rateNow.toFixed(2) : '-'} fit=${fit === null ? 'none' : fit.b.toFixed(3)} pooled=${pooledSlope.toFixed(3)} p=${pRefresh.toFixed(3)} mu=${mu.toFixed(2)} lRef=${lRef.toFixed(3)} sRef=${sRef.toFixed(3)} q=${q.join(',')}\n`,
      );
    }
  }
  return rows;
}

// ------------------------------------------------------------------ the run
mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  let rows: Array<{ id: number; q: number[] }>;
  let c: Case | null = null;
  try {
    c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
    rows = forecastCase(c);
  } catch (err) {
    // A case always gets a valid forecast: never let a parse or model error void the entry.
    process.stderr.write(`rarity: ${file}: ${err instanceof Error ? err.message : String(err)}\n`);
    const trophies = c !== null && Array.isArray(c.trophies) ? c.trophies : [];
    const hz = c !== null && Number.isFinite(c.horizon_minutes) ? (c.horizon_minutes as number) : 90;
    rows = trophies.map((t) => persistenceRow(t.id, null, hz));
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: rows })}\n`);
}
if (DEBUG) process.stderr.write(`rarity: wrote ${inputs.length} forecast(s) to ${OUTPUT_DIR}\n`);
