"""Bayesian optimisation of the layout generator parameters on a low-fidelity formal value (improvement 3).

The layout search has few parameters (radial / azimuthal spacing, radial growth, optionally four sector
thinning factors: 3-7 dimensions) and an expensive objective. The formal value V(L) on a few scenarios with
fast inner settings ("low fidelity") is used directly as the objective, so the search is not steered by the
centre-aim pre-screen that disagreed with the formal ranking (Hami: the three pre-screen picks ranked below
the initial field). A Gaussian process (Matern 5/2, one length scale per dimension, maximum marginal
likelihood) models V over the unit box; batches of q points maximising expected improvement (constant-liar
batching) are evaluated together, every (layout, scenario) in one pool. No scikit-learn needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import qmc

_SQ5 = math.sqrt(5.0)


def _matern52(a: np.ndarray, b: np.ndarray, ls: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) / ls) ** 2, 0.0).sum(axis=-1))
    return (1.0 + _SQ5 * d + 5.0 / 3.0 * d * d) * np.exp(-_SQ5 * d)


@dataclass
class GP:
    """Zero-mean GP on standardised targets; hyperparameters (log length scales, log noise) by max likelihood."""

    x: np.ndarray
    y: np.ndarray
    ls: np.ndarray = dc_field(default=None)
    noise: float = 1e-4
    _mu: float = 0.0
    _sd: float = 1.0
    _chol: tuple | None = None
    _alpha: np.ndarray | None = None

    def fit(self, restarts: int = 4, seed: int = 0) -> "GP":
        x, y = np.asarray(self.x, float), np.asarray(self.y, float)
        self._mu, self._sd = float(y.mean()), float(y.std()) or 1.0
        z = (y - self._mu) / self._sd
        d = x.shape[1]

        def nll(theta):
            ls, noise = np.exp(theta[:d]), math.exp(theta[d])
            k = _matern52(x, x, ls) + (noise + 1e-9) * np.eye(len(x))
            try:
                c = cho_factor(k, lower=True)
            except np.linalg.LinAlgError:
                return 1e10
            a = cho_solve(c, z)
            return float(0.5 * z @ a + np.log(np.diag(c[0])).sum())

        rng = np.random.default_rng(seed)
        bounds = [(math.log(0.03), math.log(5.0))] * d + [(math.log(1e-6), math.log(0.5))]
        best = None
        for r in range(restarts):
            th0 = np.r_[rng.uniform(math.log(0.1), math.log(1.0), d), math.log(1e-3)] if r else np.r_[np.full(d, math.log(0.3)), math.log(1e-3)]
            res = minimize(nll, th0, method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        self.ls, self.noise = np.exp(best.x[:d]), math.exp(best.x[d])
        k = _matern52(x, x, self.ls) + (self.noise + 1e-9) * np.eye(len(x))
        self._chol = cho_factor(k, lower=True)
        self._alpha = cho_solve(self._chol, z)
        return self

    def predict(self, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ks = _matern52(np.atleast_2d(xs), self.x, self.ls)
        mu = ks @ self._alpha
        v = cho_solve(self._chol, ks.T)
        var = np.maximum(1.0 - np.einsum("ij,ji->i", ks, v), 1e-12)
        return self._mu + self._sd * mu, self._sd * np.sqrt(var)


def expected_improvement(mu, sd, best: float, xi: float = 0.0) -> np.ndarray:
    imp = mu - best - xi
    z = imp / np.maximum(sd, 1e-12)
    return imp * ndtr(z) + sd * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def suggest(x: np.ndarray, y: np.ndarray, q: int = 1, seed: int = 0, n_candidates: int = 4096) -> np.ndarray:
    """q new points in the unit box maximising EI, batch by the constant liar (pending points get the
    current best value, so the batch spreads out)."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x.shape[1]
    out = []
    xx, yy = x.copy(), y.copy()
    for j in range(q):
        gp = GP(xx, yy).fit(seed=seed + j)
        best = float(yy.max())
        cand = qmc.Sobol(d, seed=int(rng.integers(1 << 31))).random(n_candidates)
        mu, sd = gp.predict(cand)
        ei = expected_improvement(mu, sd, best)
        starts = cand[np.argsort(-ei)[:5]]
        top, top_ei = starts[0], float(ei.max())

        def neg(p):
            m, s = gp.predict(p[None, :])
            return -float(expected_improvement(m, s, best)[0])

        for s0 in starts:
            r = minimize(neg, s0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * d)
            if -r.fun > top_ei:
                top, top_ei = np.clip(r.x, 0, 1), -r.fun
        # never repeat a point already evaluated or proposed
        if len(xx) and np.min(np.linalg.norm(xx - top, axis=1)) < 1e-6:
            top = cand[int(np.argmax(ei * (np.min(np.linalg.norm(cand[:, None] - xx[None], axis=2), axis=1) > 1e-3)))]
        out.append(top)
        xx = np.vstack([xx, top])
        yy = np.r_[yy, best]  # constant liar
    return np.array(out)


@dataclass
class BayesTrace:
    x: list = dc_field(default_factory=list)  # unit-box points
    y: list = dc_field(default_factory=list)  # objective (V, MWh)
    stage: list = dc_field(default_factory=list)  # "initial" | "bayes"


def optimise(f_batch, d: int, n_init: int, n_total: int, q: int = 1, seed: int = 0, stop=None, say=None) -> BayesTrace:
    """Maximise f over [0, 1]^d. ``f_batch(points) -> values`` evaluates a batch (in parallel)."""
    say = say or (lambda m: None)
    stop = stop or (lambda: False)
    tr = BayesTrace()
    init = qmc.Sobol(d, seed=seed).random(int(2 ** math.ceil(math.log2(max(n_init, 1)))))[:n_init]
    vals = f_batch(init)
    tr.x += list(init)
    tr.y += list(map(float, vals))
    tr.stage += ["initial"] * len(init)
    say(f"Bayesian search: {len(init)} initial layouts, best V {max(tr.y):.1f} MWh")
    k = 0
    while len(tr.y) < n_total and not stop():
        qq = min(q, n_total - len(tr.y))
        pts = suggest(np.array(tr.x), np.array(tr.y), qq, seed=seed + 101 * (k + 1))
        vals = f_batch(pts)
        tr.x += list(pts)
        tr.y += list(map(float, vals))
        tr.stage += ["bayes"] * len(pts)
        k += 1
        say(f"Bayesian search: {len(tr.y)}/{n_total} layouts, best V {max(tr.y):.1f} MWh")
    return tr
