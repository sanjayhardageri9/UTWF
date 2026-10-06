"""
UTWF: Urban Traffic Waveform Fingerprinting -- simulation study.

Generates synthetic traffic for three city topologies (grid, radial, organic),
fits the per-junction sinusoidal (harmonic) model and the city waveform
fingerprint on clean training weeks, injects congestion events into later
weeks, and evaluates congestion detection against baselines.

Everything is synthetic. The numbers this script prints are real outputs of
the experiment, but they describe the simulator, not any real city.

Usage:
    python utwf_sim.py                    # full run, 5 seeds per topology
    python utwf_sim.py --seeds 2 --quick  # fast smoke test

Outputs (in --out, default ./results):
    metrics_by_run.csv        one row per (topology, seed, method)
    metrics_summary.csv       mean and std over seeds
    fingerprint_classification.csv
    summary.md                tables ready to paste into the paper
    config.json               every parameter used
    figures/*.png             paper figures (300 dpi)
"""

import argparse
import json
import os
from collections import deque
from dataclasses import dataclass, asdict, field

import numpy as np

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

@dataclass
class Config:
    dt_min: int = 5                  # sampling interval (minutes)
    train_days: int = 28             # clean data used to fit the model
    val_days: int = 7                # events injected; used to pick thresholds
    test_days: int = 14              # events injected; reported results
    n_harmonics: int = 6             # daily harmonics in the junction model
    fp_harmonics: int = 8            # harmonics kept in the city fingerprint
    weekend_scale: float = 0.65      # weekend demand relative to weekday
    process_noise: float = 0.08      # AR(1) multiplicative demand noise (std)
    process_rho: float = 0.7         # AR(1) coefficient of demand noise
    day_noise: float = 0.05          # day-to-day amplitude variation (std)
    meas_noise: float = 0.03         # sensor noise (std, multiplicative)
    events_per_week: int = 20        # congestion events per week (whole city)
    sev_min: float = 0.15            # min flow reduction at incident junction
    sev_max: float = 0.60            # max flow reduction
    dur_min_steps: int = 6           # 30 min
    dur_max_steps: int = 36          # 180 min
    label_min_reduction: float = 0.10  # (junction, t) labelled congested above this
    ewma_alpha: float = 0.5          # temporal smoothing of z-scores
    spatial_weight: float = 0.3      # weight of neighbour residuals in the score
    common_alpha: float = 0.1        # smoothing of the city-wide common-mode factor
    persistence: int = 3             # consecutive steps above threshold to alarm
    cities_per_topology_fp: int = 10 # cities per topology for fingerprint test
    seeds: int = 5
    topologies: tuple = ("grid", "radial", "organic")

    @property
    def steps_per_day(self):
        return 24 * 60 // self.dt_min


# ----------------------------------------------------------------------------
# City topologies
# ----------------------------------------------------------------------------

@dataclass
class City:
    name: str
    xy: np.ndarray                  # (n, 2) coordinates in km
    edges: list                     # undirected (i, j)
    centre: int
    adj: list = field(default_factory=list)

    def __post_init__(self):
        n = len(self.xy)
        self.adj = [[] for _ in range(n)]
        for i, j in self.edges:
            self.adj[i].append(j)
            self.adj[j].append(i)

    @property
    def n(self):
        return len(self.xy)


def make_grid(rng, side=6, spacing=0.8):
    xy, idx = [], {}
    for r in range(side):
        for c in range(side):
            idx[(r, c)] = len(xy)
            xy.append((c * spacing, r * spacing))
    edges = []
    for r in range(side):
        for c in range(side):
            if c + 1 < side:
                edges.append((idx[(r, c)], idx[(r, c + 1)]))
            if r + 1 < side:
                edges.append((idx[(r, c)], idx[(r + 1, c)]))
    xy = np.array(xy) + rng.normal(0, 0.03, (len(xy), 2))
    centre = int(np.argmin(np.linalg.norm(xy - xy.mean(0), axis=1)))
    return City("grid", xy, edges, centre)


def make_radial(rng, spokes=7, rings=5, ring_gap=0.9):
    xy, edges = [(0.0, 0.0)], []
    idx = {}
    for r in range(1, rings + 1):
        for s in range(spokes):
            ang = 2 * np.pi * s / spokes + rng.normal(0, 0.05)
            idx[(r, s)] = len(xy)
            xy.append((r * ring_gap * np.cos(ang), r * ring_gap * np.sin(ang)))
    for s in range(spokes):
        edges.append((0, idx[(1, s)]))
        for r in range(1, rings):
            edges.append((idx[(r, s)], idx[(r + 1, s)]))
    for r in (2, 4):  # two ring roads
        for s in range(spokes):
            edges.append((idx[(r, s)], idx[(r, (s + 1) % spokes)]))
    return City("radial", np.array(xy), edges, 0)


def make_organic(rng, n=36, radius=1.25):
    while True:
        xy = rng.uniform(0, 5, (n, 2))
        d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if d[i, j] < radius and rng.random() < 0.8]
        city = City("organic", xy, edges, 0)
        if _connected(city):
            city.centre = int(np.argmin(np.linalg.norm(xy - xy.mean(0), axis=1)))
            return city


def _connected(city):
    seen, q = {0}, deque([0])
    while q:
        u = q.popleft()
        for v in city.adj[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return len(seen) == city.n


def make_city(kind, rng):
    return {"grid": make_grid, "radial": make_radial, "organic": make_organic}[kind](rng)


def bfs_depth(city):
    depth = np.full(city.n, -1)
    depth[city.centre] = 0
    q = deque([city.centre])
    while q:
        u = q.popleft()
        for v in city.adj[u]:
            if depth[v] < 0:
                depth[v] = depth[u] + 1
                q.append(v)
    return depth


# ----------------------------------------------------------------------------
# Traffic generation: local sinusoidal demand + linear inter-junction transfer
# ----------------------------------------------------------------------------

def periodic_bump(hours, peak_h, kappa):
    """Smooth 24 h periodic bump (von Mises shape), normalised to max 1."""
    x = np.exp(kappa * np.cos(2 * np.pi * (hours - peak_h) / 24))
    return x / np.exp(kappa)


def demand_profiles(hours, is_weekend):
    """Inbound (AM-peaked) and outbound (PM-peaked) demand shapes."""
    wd_in = 0.15 + periodic_bump(hours, 8.0, 3.0) + 0.35 * periodic_bump(hours, 13.0, 2.0)
    wd_out = 0.15 + periodic_bump(hours, 17.5, 3.0) + 0.35 * periodic_bump(hours, 13.0, 2.0)
    we = 0.15 + 0.8 * periodic_bump(hours, 13.5, 1.5)
    inbound = np.where(is_weekend, we, wd_in)
    outbound = np.where(is_weekend, we, wd_out)
    return inbound, outbound


def shift(x, k):
    """Delay series by k steps (pad start with first value)."""
    if k <= 0:
        return x
    return np.concatenate([np.full(k, x[0]), x[:-k]])


def ar1_noise(rng, T, std, rho):
    e = rng.normal(0, std * np.sqrt(1 - rho ** 2), T)
    out = np.empty(T)
    out[0] = rng.normal(0, std)
    for t in range(1, T):
        out[t] = rho * out[t - 1] + e[t]
    return out


def simulate_flows(city, cfg, rng, n_days):
    """Return clean flows q (n, T) in vehicles per interval, plus time info."""
    spd = cfg.steps_per_day
    T = n_days * spd
    t = np.arange(T)
    hours = (t % spd) * cfg.dt_min / 60.0
    day = t // spd
    is_weekend = (day % 7) >= 5                       # day 0 is a Monday
    inbound, outbound = demand_profiles(hours, is_weekend)
    day_factor = np.repeat(1 + rng.normal(0, cfg.day_noise, n_days), spd)
    scale = np.where(is_weekend, cfg.weekend_scale, 1.0) * day_factor

    depth = bfs_depth(city)
    dist = np.linalg.norm(city.xy - city.xy[city.centre], axis=1)
    base = rng.uniform(10, 40, city.n) * (1 + 0.6 * dist / dist.max())  # more trips start at the edge

    # Directed links: inbound flows from higher depth to lower depth (towards
    # the centre); outbound is the reverse. Each link has a travel lag and a
    # transfer coefficient a_ij (share of upstream flow passed on).
    def links(direction):
        out = {i: [] for i in range(city.n)}
        for i, j in city.edges:
            for u, v in ((i, j), (j, i)):
                go = depth[u] > depth[v] if direction == "in" else depth[u] < depth[v]
                if go:
                    out[u].append(v)
        lk = {}
        for u, vs in out.items():
            if not vs:
                continue
            keep = rng.uniform(0.55, 0.8)        # rest exits / parks locally
            w = rng.dirichlet(np.ones(len(vs)))
            for v, wv in zip(vs, w):
                length = np.linalg.norm(city.xy[u] - city.xy[v])
                lag = max(1, int(round(length / 0.4)))   # ~0.4 km per 5 min in traffic
                lk[(u, v)] = (keep * wv, lag)
        return lk

    flows = np.zeros((city.n, T))
    truth_links = {}
    for direction, profile, order in (
        ("in", inbound, np.argsort(-depth)),     # process edge -> centre
        ("out", outbound, np.argsort(depth)),    # process centre -> edge
    ):
        lk = links(direction)
        truth_links[direction] = lk
        q = np.zeros((city.n, T))
        for v in order:
            noise = 1 + ar1_noise(rng, T, cfg.process_noise, cfg.process_rho)
            local = base[v] * profile * scale * np.clip(noise, 0.5, 1.5)
            if direction == "out":
                local = local * 0.9
            upstream = sum(a * shift(q[u], lag) for (u, w), (a, lag) in lk.items() if w == v)
            q[v] = local + (upstream if not np.isscalar(upstream) else 0.0)
        flows += q

    info = {"hours": hours, "day": day, "is_weekend": is_weekend, "links": truth_links}
    return flows, info


def inject_events(city, flows, cfg, rng, start_step, end_step):
    """Multiply flows by congestion reductions; return observed flows + truth."""
    spd = cfg.steps_per_day
    n, T = flows.shape
    reduction = np.zeros((n, T))
    events = []
    n_events = int(round(cfg.events_per_week * (end_step - start_step) / (7 * spd)))
    for _ in range(n_events):
        j = int(rng.integers(n))
        d0 = int(rng.integers(start_step // spd, end_step // spd))
        s = d0 * spd + int(rng.integers(6 * 60 // cfg.dt_min, 21 * 60 // cfg.dt_min))
        dur = int(rng.integers(cfg.dur_min_steps, cfg.dur_max_steps + 1))
        sev = float(rng.uniform(cfg.sev_min, cfg.sev_max))
        e = min(s + dur, end_step)
        events.append({"junction": j, "start": s, "end": e, "severity": sev})
        ramp = np.ones(e - s)
        r = min(2, len(ramp) // 2)
        if r:
            ramp[:r] = np.linspace(0.5, 1, r)
            ramp[-r:] = np.linspace(1, 0.5, r)
        _apply(reduction, j, s, e, sev * ramp)
        for u in city.adj[j]:                         # spillback / starvation
            lag = int(rng.integers(1, 3))
            _apply(reduction, u, s + lag, e + lag, 0.5 * sev * ramp)
            for w in city.adj[u]:
                if w != j:
                    _apply(reduction, w, s + 2 * lag, e + 2 * lag, 0.25 * sev * ramp)
    reduction = np.clip(reduction, 0, 0.9)
    labels = reduction >= cfg.label_min_reduction
    return flows * (1 - reduction), labels, events


def _apply(red, j, s, e, val):
    T = red.shape[1]
    s2, e2 = max(0, s), min(T, e)
    if e2 <= s2:
        return
    v = val[(s2 - s):(s2 - s) + (e2 - s2)]
    red[j, s2:e2] = 1 - (1 - red[j, s2:e2]) * (1 - v)  # combine overlapping events


# ----------------------------------------------------------------------------
# UTWF model: per-junction harmonic fit, residual scoring, city fingerprint
# ----------------------------------------------------------------------------

def harmonic_design(hours, is_weekend, K):
    cols = []
    for flag in (~is_weekend, is_weekend):
        f = flag.astype(float)
        cols.append(f)
        for k in range(1, K + 1):
            w = 2 * np.pi * k * hours / 24
            cols += [f * np.cos(w), f * np.sin(w)]
    return np.column_stack(cols)


class HarmonicModel:
    """Junction signal model: q_j(t) ~ sum_k A_jk cos(k w t - phi_jk), per day type."""

    def __init__(self, K):
        self.K = K

    def fit(self, q, hours, is_weekend):
        X = harmonic_design(hours, is_weekend, self.K)
        self.beta, *_ = np.linalg.lstsq(X, q.T, rcond=None)
        return self

    def predict(self, hours, is_weekend):
        return (harmonic_design(hours, is_weekend, self.K) @ self.beta).T


class HistoricalAverage:
    """Baseline: mean flow for each time-of-week slot in training."""

    def fit(self, q, cfg):
        spd = cfg.steps_per_day
        weeks = q.shape[1] // (7 * spd)
        self.profile = q[:, : weeks * 7 * spd].reshape(q.shape[0], weeks, 7 * spd).mean(1)
        return self

    def predict(self, T, offset):
        idx = (np.arange(T) + offset) % self.profile.shape[1]
        return self.profile[:, idx]


def robust_scale(resid, hours):
    """Residual spread per junction per hour of day (MAD * 1.4826)."""
    h = hours.astype(int)
    s = np.zeros((resid.shape[0], 24))
    for hh in range(24):
        r = resid[:, h == hh]
        med = np.median(r, axis=1, keepdims=True)
        s[:, hh] = 1.4826 * np.median(np.abs(r - med), axis=1)
    return np.maximum(s, 1e-3)


def ewma(z, alpha):
    out = np.empty_like(z)
    out[:, 0] = z[:, 0]
    for t in range(1, z.shape[1]):
        out[:, t] = alpha * z[:, t] + (1 - alpha) * out[:, t - 1]
    return out


def common_mode(obs, pred, alpha):
    """City-wide demand factor c(t) = median over junctions of obs/pred,
    smoothed causally. Congestion is local; a demand swing moves every
    junction together, so dividing it out stops whole-city false alarms."""
    ratio = np.median(obs / np.maximum(pred, 1e-6), axis=0, keepdims=True)
    return ewma(ratio, alpha)


def spatial_mix(z, city, w):
    nb = np.array([z[city.adj[j]].mean(0) for j in range(city.n)])
    return (1 - w) * z + w * nb


def rolling_z(q, window):
    """Baseline: deviation from the trailing-window mean, in trailing std units."""
    n, T = q.shape
    c = np.cumsum(np.pad(q, ((0, 0), (1, 0))), axis=1)
    c2 = np.cumsum(np.pad(q ** 2, ((0, 0), (1, 0))), axis=1)
    z = np.zeros_like(q)
    for t in range(window, T):
        m = (c[:, t] - c[:, t - window]) / window
        v = (c2[:, t] - c2[:, t - window]) / window - m ** 2
        z[:, t] = (q[:, t] - m) / np.sqrt(np.maximum(v, 1e-6))
    return z


def city_fingerprint(q, hours, is_weekend, cfg):
    """City waveform fingerprint: normalised harmonic amplitudes of the mean
    weekday aggregate flow, plus the spread of junction peak phases (which
    encodes how flow waves travel across the network)."""
    spd = cfg.steps_per_day
    Q = q.sum(0)
    wd_days = [d for d in range(q.shape[1] // spd) if d % 7 < 5]
    prof = np.mean([Q[d * spd:(d + 1) * spd] for d in wd_days], axis=0)
    F = np.fft.rfft(prof)
    amp = np.abs(F[1:cfg.fp_harmonics + 1]) / np.abs(F[0])
    # Phase of the 2nd harmonic (captures the twin-peak timing) per junction
    jprof = np.mean([q[:, d * spd:(d + 1) * spd] for d in wd_days], axis=0)
    ph = np.angle(np.fft.rfft(jprof, axis=1)[:, 2])
    R = np.abs(np.mean(np.exp(1j * ph)))          # phase coherence across junctions
    return np.concatenate([amp, [1 - R]]), prof


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def roc_curve(score, label):
    order = np.argsort(-score)
    y = label[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    P, N = y.sum(), (~y).sum()
    tpr = np.concatenate([[0], tp / max(P, 1)])
    fpr = np.concatenate([[0], fp / max(N, 1)])
    prec = tp / np.maximum(tp + fp, 1)
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))  # trapezoid rule (works on any numpy)
    ap = np.sum(np.diff(np.concatenate([[0], tp / max(P, 1)])) * prec)
    return fpr, tpr, auc, ap


def alarm_runs(score, thr, m):
    """Boolean alarm mask: score > thr for >= m consecutive steps."""
    above = score > thr
    alarm = np.zeros_like(above)
    for j in range(above.shape[0]):
        run = 0
        for t in range(above.shape[1]):
            run = run + 1 if above[j, t] else 0
            if run >= m:
                alarm[j, t - m + 1:t + 1] = True
    return alarm


def runs_of(mask_row):
    d = np.diff(np.concatenate([[0], mask_row.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def event_metrics(score, labels, events, offset, thr, cfg, n_days):
    alarm = alarm_runs(score, thr, cfg.persistence)
    detected, delays = 0, []
    for ev in events:
        j, s, e = ev["junction"], ev["start"] - offset, ev["end"] - offset
        hit = np.where(alarm[ev["junction"], s:e])[0]
        if hit.size:
            detected += 1
            # an alarm is only raised once the run reaches `persistence` steps
            delays.append((hit[0] + cfg.persistence - 1) * cfg.dt_min)
    true_runs = false_runs = 0
    for j in range(alarm.shape[0]):
        for a, b in runs_of(alarm[j]):
            if labels[j, max(0, a - 3):b + 3].any():
                true_runs += 1
            else:
                false_runs += 1
    recall = detected / max(len(events), 1)
    precision = true_runs / max(true_runs + false_runs, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "false_alarms_per_junction_day": false_runs / (alarm.shape[0] * n_days),
        "median_delay_min": float(np.median(delays)) if delays else float("nan"),
    }


def choose_threshold(score, labels, events, offset, cfg, n_days):
    best = (-1, None)
    for thr in np.quantile(score, np.linspace(0.90, 0.9995, 40)):
        f1 = event_metrics(score, labels, events, offset, thr, cfg, n_days)["f1"]
        if f1 > best[0]:
            best = (f1, thr)
    return best[1]


# ----------------------------------------------------------------------------
# One experiment
# ----------------------------------------------------------------------------

TOPO_ID = {"grid": 11, "radial": 23, "organic": 37}

METHODS = ("UTWF", "UTWF (no spatial)", "UTWF (no common-mode)",
           "Historical average + CM", "Historical average", "Rolling z-score")


def run_one(kind, seed, cfg):
    rng = np.random.default_rng(seed * 1000 + TOPO_ID[kind])
    city = make_city(kind, rng)
    spd = cfg.steps_per_day
    n_days = cfg.train_days + cfg.val_days + cfg.test_days
    clean, info = simulate_flows(city, cfg, rng, n_days)
    obs_clean = clean * (1 + rng.normal(0, cfg.meas_noise, clean.shape))

    t_train = cfg.train_days * spd
    t_val = t_train + cfg.val_days * spd
    T = n_days * spd
    obs, labels, events = inject_events(city, obs_clean, cfg, rng, t_train, T)
    hours, wk = info["hours"], info["is_weekend"]

    # Fit on clean training weeks only
    tr = slice(0, t_train)
    hm = HarmonicModel(cfg.n_harmonics).fit(obs[:, tr], hours[tr], wk[tr])
    ha = HistoricalAverage().fit(obs[:, tr], cfg)
    pred_h = hm.predict(hours, wk)
    pred_a = ha.predict(T, 0)
    pred_hc = pred_h * common_mode(obs, pred_h, cfg.common_alpha)
    pred_ac = pred_a * common_mode(obs, pred_a, cfg.common_alpha)

    def z(pred):
        sc = robust_scale(obs[:, tr] - pred[:, tr], hours[tr])
        return (obs - pred) / sc[:, hours.astype(int)]

    def drop(zz, spatial=False):     # higher score = more evidence of a flow drop
        s = ewma(zz, cfg.ewma_alpha)
        return -(spatial_mix(s, city, cfg.spatial_weight) if spatial else s)

    z_hc = z(pred_hc)
    scores = {
        "UTWF": drop(z_hc, spatial=True),
        "UTWF (no spatial)": drop(z_hc),
        "UTWF (no common-mode)": drop(z(pred_h), spatial=True),
        "Historical average + CM": drop(z(pred_ac)),
        "Historical average": drop(z(pred_a)),
        "Rolling z-score": drop(rolling_z(obs, 12)),
    }

    val_ev = [e for e in events if e["start"] < t_val]
    test_ev = [e for e in events if e["start"] >= t_val]
    rows, rocs = [], {}
    for m, s in scores.items():
        thr = choose_threshold(s[:, t_train:t_val], labels[:, t_train:t_val], val_ev,
                               t_train, cfg, cfg.val_days)
        em = event_metrics(s[:, t_val:], labels[:, t_val:], test_ev, t_val, thr, cfg,
                           cfg.test_days)
        fpr, tpr, auc, ap = roc_curve(s[:, t_val:].ravel(), labels[:, t_val:].ravel())
        rocs[m] = (fpr, tpr)
        rows.append({"topology": kind, "seed": seed, "method": m,
                     "roc_auc": auc, "avg_precision": ap, "threshold": float(thr),
                     "n_test_events": len(test_ev), **em})

    # Fit quality of the sinusoidal junction model on clean test data
    te = slice(t_val, T)
    r2 = 1 - ((obs_clean[:, te] - pred_h[:, te]) ** 2).sum(1) / \
        ((obs_clean[:, te] - obs_clean[:, te].mean(1, keepdims=True)) ** 2).sum(1)
    fit = {"topology": kind, "seed": seed, "median_r2": float(np.median(r2)),
           "min_r2": float(r2.min()), "n_junctions": city.n, "n_links": len(city.edges)}

    extras = dict(city=city, obs=obs, clean=obs_clean, pred=pred_h, pred_cm=pred_hc, z=scores["UTWF"],
                  labels=labels, events=test_ev, hours=hours, wk=wk, t_val=t_val,
                  t_train=t_train, rocs=rocs, thr=rows[0]["threshold"])
    return rows, fit, extras


def fingerprint_study(cfg):
    feats, labels, profiles = [], [], {}
    for kind in cfg.topologies:
        for i in range(cfg.cities_per_topology_fp):
            rng = np.random.default_rng(10_000 + i * 31 + TOPO_ID[kind])
            city = make_city(kind, rng)
            q, info = simulate_flows(city, cfg, rng, cfg.train_days)
            q = q * (1 + rng.normal(0, cfg.meas_noise, q.shape))
            f, prof = city_fingerprint(q, info["hours"], info["is_weekend"], cfg)
            feats.append(f)
            labels.append(kind)
            profiles.setdefault(kind, []).append(prof / prof.mean())
    X = np.array(feats)
    X = (X - X.mean(0)) / X.std(0)
    y = np.array(labels)
    correct = 0
    for i in range(len(y)):                      # leave-one-out nearest centroid
        mask = np.arange(len(y)) != i
        cents = {k: X[mask & (y == k)].mean(0) for k in cfg.topologies}
        pred = min(cents, key=lambda k: np.linalg.norm(X[i] - cents[k]))
        correct += pred == y[i]
    return correct / len(y), X, y, profiles


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # validated categorical order
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"


def _style(plt):
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def make_figures(ex_by_topo, fp, outdir, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fd = os.path.join(outdir, "figures")
    os.makedirs(fd, exist_ok=True)
    spd = cfg.steps_per_day

    # Fig 1: city topologies
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    for ax, (kind, ex) in zip(axs, ex_by_topo.items()):
        c = ex["city"]
        for i, j in c.edges:
            ax.plot(*c.xy[[i, j]].T, color=GRID, lw=1.2, zorder=1)
        ax.scatter(*c.xy.T, s=14, color=COLORS[0], zorder=2)
        ax.scatter(*c.xy[c.centre], s=40, color=COLORS[1], zorder=3)
        ax.set_title(f"{kind} ({c.n} junctions)", color=INK)
        ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(os.path.join(fd, "fig1_topologies.png")); plt.close(fig)

    # Fig 2: junction signal vs fitted sinusoidal model (one training week)
    ex = ex_by_topo["grid"]
    j = int(np.argmax(ex["clean"].mean(1)))
    t = np.arange(7 * spd)
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot(t / spd, ex["clean"][j, t], color=COLORS[0], lw=0.8, label="Simulated flow")
    ax.plot(t / spd, ex["pred"][j, t], color=COLORS[1], label=f"Harmonic model (K={cfg.n_harmonics})")
    ax.set_xlabel("Day (Mon = 0)"); ax.set_ylabel("Vehicles / 5 min")
    ax.legend(loc="upper right", ncol=2)
    fig.savefig(os.path.join(fd, "fig2_junction_signal.png")); plt.close(fig)

    # Fig 3: city waveform fingerprints
    acc, X, y, profiles = fp
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    h = np.arange(spd) * cfg.dt_min / 60
    for k, (kind, profs) in enumerate(profiles.items()):
        P = np.array(profs)
        axs[0].plot(h, P.mean(0), color=COLORS[k], label=kind)
        axs[0].fill_between(h, P.min(0), P.max(0), color=COLORS[k], alpha=0.15, lw=0)
    axs[0].set_xlabel("Hour of weekday"); axs[0].set_ylabel("Aggregate flow / daily mean")
    axs[0].set_xticks(range(0, 25, 4)); axs[0].legend()
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P2 = Xc @ Vt[:2].T
    for k, kind in enumerate(cfg.topologies):
        m = y == kind
        axs[1].scatter(P2[m, 0], P2[m, 1], s=22, color=COLORS[k], label=kind,
                       edgecolor="white", linewidth=1)
    axs[1].set_xlabel("Fingerprint PC1"); axs[1].set_ylabel("Fingerprint PC2")
    axs[1].set_title(f"Leave-one-out topology accuracy: {acc:.0%}", color=INK)
    axs[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig3_city_fingerprints.png")); plt.close(fig)

    # Fig 4: residual trace around the largest test event
    ex = ex_by_topo["grid"]
    ev = max(ex["events"], key=lambda e: e["severity"])
    j, s, e = ev["junction"], ev["start"], ev["end"]
    w = slice(s - spd // 4, e + spd // 4)
    tt = (np.arange(w.start, w.stop) - (s // spd) * spd) * cfg.dt_min / 60  # hours since midnight of event day
    fig, axs = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
    axs[0].plot(tt, ex["obs"][j, w], color=COLORS[0], label="Observed")
    axs[0].plot(tt, ex["pred_cm"][j, w], color=COLORS[1], label="Fingerprint expectation")
    axs[1].plot(tt, ex["z"][j, w], color=COLORS[0], label="UTWF score")
    axs[1].axhline(ex["thr"], color=MUTED, ls="--", lw=1, label="Threshold")
    for a in axs:
        a.axvspan(tt[s - w.start], tt[min(e - w.start, len(tt) - 1)], color=GRID, alpha=0.7, lw=0)
    axs[0].set_ylabel("Vehicles / 5 min"); axs[1].set_ylabel("Score")
    axs[1].set_xlabel("Hour of day (shaded: injected congestion, "
                      f"{ev['severity']:.0%} flow reduction)")
    axs[0].legend(loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0))
    axs[1].legend(loc="upper left", ncol=2)
    fig.savefig(os.path.join(fd, "fig4_residual_event.png")); plt.close(fig)

    # Fig 5: ROC per topology
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    for ax, (kind, ex) in zip(axs, ex_by_topo.items()):
        for k, m in enumerate(METHODS):
            fpr, tpr = ex["rocs"][m]
            step = max(1, len(fpr) // 2000)
            ax.plot(fpr[::step], tpr[::step], color=COLORS[k], label=m)
        ax.plot([0, 1], [0, 1], color=GRID, lw=1)
        ax.set_title(kind, color=INK); ax.set_xlabel("False positive rate")
    axs[0].set_ylabel("True positive rate")
    axs[-1].legend(loc="lower right", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig5_roc.png")); plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k])
                             for k in keys) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="shorter run for testing")
    args = ap.parse_args()

    cfg = Config(seeds=args.seeds)
    if args.quick:
        cfg.train_days, cfg.val_days, cfg.test_days, cfg.cities_per_topology_fp = 14, 7, 7, 4
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    all_rows, fits, ex_by_topo = [], [], {}
    for kind in cfg.topologies:
        for seed in range(cfg.seeds):
            rows, fit, ex = run_one(kind, seed, cfg)
            all_rows += rows
            fits.append(fit)
            if seed == 0:
                ex_by_topo[kind] = ex
            u = rows[0]
            print(f"{kind:8s} seed {seed}: UTWF F1={u['f1']:.3f} AUC={u['roc_auc']:.3f} "
                  f"R2={fit['median_r2']:.3f}")

    write_csv(os.path.join(args.out, "metrics_by_run.csv"), all_rows)
    write_csv(os.path.join(args.out, "model_fit.csv"), fits)

    metrics = ["f1", "precision", "recall", "false_alarms_per_junction_day",
               "median_delay_min", "roc_auc", "avg_precision"]
    summary = []
    for kind in cfg.topologies:
        for m in METHODS:
            rs = [r for r in all_rows if r["topology"] == kind and r["method"] == m]
            row = {"topology": kind, "method": m}
            for k in metrics:
                v = np.array([r[k] for r in rs], dtype=float)
                row[k + "_mean"] = float(np.nanmean(v))
                row[k + "_std"] = float(np.nanstd(v))
            summary.append(row)
    write_csv(os.path.join(args.out, "metrics_summary.csv"), summary)

    print("Fingerprint study ...")
    fp = fingerprint_study(cfg)
    with open(os.path.join(args.out, "fingerprint_classification.csv"), "w") as f:
        f.write("cities_per_topology,leave_one_out_accuracy\n")
        f.write(f"{cfg.cities_per_topology_fp},{fp[0]:.4f}\n")

    make_figures(ex_by_topo, fp, args.out, cfg)

    # Markdown summary for the paper
    L = ["# UTWF simulation results", "",
         f"Synthetic data. {cfg.seeds} seeds per topology; {cfg.train_days} training days, "
         f"{cfg.val_days} validation days (threshold selection), {cfg.test_days} test days. "
         "Mean ± std over seeds.", "",
         "## Detection performance (test period)", "",
         "| Topology | Method | F1 | Precision | Recall | False alarms / junction-day | "
         "Median delay (min) | ROC-AUC |",
         "|---|---|---|---|---|---|---|---|"]
    for r in summary:
        L.append(f"| {r['topology']} | {r['method']} | "
                 f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f} | {r['precision_mean']:.3f} | "
                 f"{r['recall_mean']:.3f} | {r['false_alarms_per_junction_day_mean']:.3f} | "
                 f"{r['median_delay_min_mean']:.1f} | {r['roc_auc_mean']:.3f} |")
    L += ["", "## Junction model fit (clean test data)", "",
          "| Topology | Median R² | Min R² |", "|---|---|---|"]
    for kind in cfg.topologies:
        fs = [f for f in fits if f["topology"] == kind]
        L.append(f"| {kind} | {np.mean([f['median_r2'] for f in fs]):.3f} | "
                 f"{np.min([f['min_r2'] for f in fs]):.3f} |")
    L += ["", "## City fingerprint", "",
          f"Leave-one-out nearest-centroid topology classification over "
          f"{cfg.cities_per_topology_fp * len(cfg.topologies)} simulated cities: "
          f"**{fp[0]:.1%}**.", "",
          "Note: topology differences in the fingerprint come from the simulator's "
          "assumptions (link lengths, network depth, demand placement), so this shows the "
          "fingerprint *can* separate structurally different networks, not that real "
          "cities differ this way."]
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
