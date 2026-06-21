# Simulation 4 — going BIG (the feasibility frontier)

**This is the headline.** Simulations 1–3 establish that ZQE is consistent,
low-variance, fast, and family-agnostic. Simulation 4 makes the point the whole
estimator was *designed* for: it scales to problem sizes where likelihood-based
GLLVMs (`gllvm`) become **infeasible** — out of memory, or out of time — while
ZQE keeps recovering the loadings at near-linear cost.

The metric is no longer only "who is more accurate"; it is **"who can fit it at
all, and how fast."** An out-of-memory or timed-out `gllvm` run is a legitimate
data point: `DNF` ("did not finish").

## Why ZQE scales (the proposition to defend)

One ZQE step needs only: (i) sampling fantasy data `Y_q ~ p_θ` — `O(n·p)`;
(ii) the analytic encoder, a ridge solve `z = (σ²I + WᵀW)⁻¹ Wᵀ(T(y) − b)` —
`O(n·p·q)` to form, `O(q³)` to solve; (iii) the estimating-equation contraction
`T(y)·η` — `O(n·p·q)`. No encoder is trained, nothing of size `n×n` or `p×p` is
inverted. Cost is therefore **linear in `p` and in `n`** (and cubic only in the
small `q`).

**Double mini-batching.** When `n·p` does not fit in memory, subsample *both*
observations (`S_n`) and responses (`S_p`) per step. This stays consistent: the
`m₁ − m₂` centering cancels the encoder's bias for **any** encoder map, *provided
the identical response-subset and encoder are applied to the real term `m₁` and
the fantasy term `m₂` within a step* (and the response-sum is reweighted by
`p/|S_p|`). The only cost is efficiency, not validity, and it has one practical
guard: keep `|S_p| ≫ q` so the analytic `z` stays well-determined (a MAP from
fewer than ~`q` responses is high-variance). With this, fits like `p = 50 000,
q = 50, n = 10 000` run comfortably on a single GPU.

## Design

Identifiable dense low-rank truth (so Procrustes remains a valid recovery metric):
Poisson GLLVM, true rank `q`, dense loadings, intercepts as in simulations 1–3.

**Frontier sweep** — push the dimension `gllvm` chokes on, holding the others fixed:

| axis | grid | fixed |
|------|------|-------|
| responses `p` | `{100, 500, 2 000, 10 000, 50 000}` | `n = 2 000`, `q = 10` |
| latent dim `q` | `{5, 20, 50}` | `p = 2 000`, `n = 2 000` |
| observations `n` | `{1 000, 10 000, 100 000}` | `p = 1 000`, `q = 10` |

(`gllvm` is expected to `DNF` early on every axis: large `p`/`n` → memory & time;
large `q` → its starting-value factor analysis fails, as already seen.)

**Methods**: `zqe` (with double mini-batching; a fixed wall-clock/iteration budget)
vs `gllvm` (VA, with a memory cap and a timeout → `DNF` flag). A modest `H`
(e.g. 5–10) suffices: timing is stable and the claim is feasibility, not a fine
Procrustes distribution.

**Reported per (size, rep, method)**: wall-clock time, peak memory, `DNF` flag,
and — wherever the method finished — the Procrustes error vs the true `W`. Headline
figures: (1) time vs size on log–log axes, with `gllvm`'s `DNF` wall marked;
(2) Procrustes vs size, showing ZQE holds accuracy across the whole range.

## Scope note

The **overcomplete** regime (`q ≫ p`, e.g. `p = 100, q = 10 000`) is *not* part of
this simulation. With `q < p` the model stays identifiable and Procrustes-scored,
which keeps the scaling claim airtight ("same recovery, feasible where `gllvm`
DNFs"). Large-`q` *pattern discovery* — fit dense with generous `q`, then **varimax**
to read off sparse structure (no L0 needed; rotation is free under the `O(q)`
gauge) — is a separate study with a different metric (subspace/support recovery),
deliberately kept out of this one.

## Reproducibility / layout

Same conventions as simulations 1–3: `seed == rep`; one CSV per (size, rep);
resumable; `DNF`/failures flagged, never fatal. Driver: `sweep.py` (to be written).
Experiment: `experiment.ipynb`. Analysis: `analysis.ipynb`. Columns:
`results/DATA_DICTIONARY.md`.
