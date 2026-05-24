# Overleaf figure export

Upload `figures/` to Overleaf, reference as `figures/<name>` from .tex. PNG at 300 dpi.

## Figure index

### Headline

- `payoff-matrix.png` — heatmap of `R / R*` for every (trained model × test regime) cell. Headline result.

### Trajectories (V3 §6.3)

- `trajectories-disc-logu-zoom.png` — focal contrast: D_disc_cv vs. D_logu_cv vs. oracle on each test regime, with per-panel y-limits (0–10, 0–40, 150–300) so the gap on each regime is readable. The §6.3 visualization.
- `trajectories-d-disc-cv.png` — D_disc_cv as the focal model, ±1 std band, with the full §4 portfolio overlaid: per-regime oracle, random-ADP oracles for D_disc and D_logu, plug-in, prior-only, myopic, MAP-σ at both priors, MLE-σ, secretary, and the offline-hindsight horizontal line.
- `trajectories-d-logu-cv.png` — D_logu_cv as the focal model, same full portfolio.
- `trajectories-d-1-cv.png`, `trajectories-d-2-cv.png`, `trajectories-d-3-cv.png` — focal-model versions for the static-σ runs (D_1, D_2, D_3), full §4 portfolio overlaid. Static-σ runs collapse to constants on their training regime; useful for the appendix.
- `trajectories-all-cv.png` — synthesis figure: all 5 cv-trained models overlaid on each test regime. Shows the static-vs-multi-regime contrast at a glance — the three single-regime models plateau at their training scale across all three test regimes while D_disc_cv (emphasized) tracks the per-regime oracle on each panel.

### Per-σ payoff (V3 §6.3 / §6.4)

- `per-sigma-d-disc.png` — grouped bars D_disc_cv vs D_disc_act, R/R\* per σ-bin (σ ∈ {1, 10, 100}). Baseline R\* = per-distribution random-ADP oracle ("honest gap"). Shows uniform competence across all bins.
- `per-sigma-d-logu.png` — grouped bars D_logu_cv vs D_logu_act, R/R\* per log-σ bin. The cv-vs-act asymmetry visualization for §6.3: act ≈ 1.0 across all bins; cv climbs from −0.04 to 0.72.

### Diagnostics (V3 §6.3)

- `diagnostic-logu-abs-error.png` — absolute error of D_logu_cv's cv-head output vs. the bin-σ static oracle, by log-σ bin (raw threshold units).
- `diagnostic-logu-loss-unit-error.png` — same comparison, but in the cv loss's natural units `|Ĉ − C\*| / σ_i`.

### Action-agreement (V3 §6.5)

- `agreement-oracle.png` — standalone 10×3 panel: per-step action agreement between each trained model and the per-regime oracle on each test regime. Extracted from the larger 7-baseline grid because the full grid is too dense for the paper.

### Training appendix

- `training-curves.png` — full sweep training curves grid (10 runs, train + val loss vs. step, best step marked).
- `training-curves-per-sigma.png` — per-σ-group val loss for the four random-variance runs (D_disc_cv, D_disc_act, D_logu_cv, D_logu_act). Tells the σ=100 catch-up story for D_disc.

### Probing analysis (V3 §7.2)

- `probe-curves-logu-act.png` — 11-panel grid (3×4 layout, 12th cell holds the legend), normalized test MSE vs.\ residual-stream layer $\ell \in \{0, \ldots, 8\}$ for the frozen $\mathcal{D}_{\mathrm{logu}}$\_act model. Trained: linear solid blue / MLP dashed red. Random-init noise floor: dotted pale blue / pale red. Constant-label control: dot-dashed lighter shades. Panels (in order): $\bar X_t$, $\hat\sigma_t^2$, $\hat\sigma_t^{\mathrm{MAP}}$, $\hat\sigma_t^{\mathrm{MLE}}$, $\widehat C_t^\star$ (oracle threshold), $\widehat C_t^{\mathrm{plug\text{-}in,MAP}}$, $\widehat C_t^{\mathrm{plug\text{-}in,MLE}}$, then the four log-likelihood targets $\log p(X_{1:t} \mid \sigma)$ at $\sigma \in \{1, 10, 100\}$ and at the true latent $\sigma_i$. Headline threshold panel: $\widehat C_t^\star$ MLP $\mathrm{nMSE} = 0.0042$ at $\ell = 6$ ($32\times$ below random-init, $16\times$ below control). Headline likelihood panel: $\log p(X_{1:t} \mid \sigma_i)$ MLP $\mathrm{nMSE} = 0.008$ at $\ell = 5$ ($13\times$ below random-init, $24\times$ below control) --- evidence of a likelihood-as-a-function-of-$\sigma$ representation.

### Offline (hindsight-prophet) supervision appendix

- `payoff-matrix-extended-offline.png` — 14×7 R/R\* heatmap (pastel pink → pastel green). Six static single-regime models in the top block; the eight random-variance rows below pair each oracle-supervised model with its offline-supervised counterpart. Light horizontal lines mark the pair boundaries; the bold line separates static from random.
- `bars-oracle-vs-offline.png` — two-panel grouped bar chart (D_disc on top, D_logu on bottom). Pastel bars: oracle cv (blue), offline cv (pale blue), oracle act (green), offline act (pale green) per test regime, with σ printed under each group.
- `trajectories-d-disc-cv-offline.png` — focal `D_disc_cv_offline` (blue) overlaying the oracle-cv counterpart `D_disc_cv_sig2` (pale blue) and the per-regime static-σ oracle `C*_t` (dashed black) on all seven test regimes (2×4 panels, one panel hidden).
- `trajectories-d-logu-cv-offline.png` — same layout, focal `D_logu_cv_offline` (green) overlaying `D_logu_cv_sig2` (pale green) and `C*_t`. The "offline ≥ oracle by construction" gap is visible on the in-prior σ regimes; on `D_disc-σ ∈ {3, 30}` both models share the same regime-misidentification pattern.

## D_logu_cv diagnostic — interpretation

The two diagnostic figures disambiguate two competing hypotheses for D_logu_cv's low-σ payoff failure (R/R\* = −0.04 at σ ∈ [1, 2.5]).

- **H1 — gradient imbalance.** The cv loss is `MSE / σ_i`; per-sequence squared errors scale as σ², so loss contributions per sequence still scale as σ. The optimizer would then fit high-σ sequences preferentially and bias low-σ thresholds toward the geometric-mean σ. **Predicts loss-unit error grows monotonically toward low σ.**
- **H2 — cv-head linear bottleneck.** The cv head is a linear map h_t → Ĉ_t. Under D_logu, Ĉ ranges over two decades. The model's representation noise is roughly σ-invariant in σ-units, but the linear head amplifies it absolutely at high σ and *relatively* at low σ. **Predicts loss-unit error roughly flat across bins.**

Bin values measured on the seed-44 D_logu per-σ test cache (5 bins, ~2k sequences each), comparing Ĉ_t to the bin's static-ADP oracle at the bin's geometric-mean σ over the cv-mask range t ∈ {2, …, n−1}:

| log_10 σ bin | abs error | loss-unit error | n     |
|---           |---        |---              |---    |
| [0.0, 0.4)   | 0.871     | 0.539           | 2048  |
| [0.4, 0.8)   | 2.245     | 0.562           | 1993  |
| [0.8, 1.2)   | 5.513     | 0.544           | 1956  |
| [1.2, 1.6)   | 13.683    | 0.547           | 2004  |
| [1.6, 2.0]   | 32.296    | 0.525           | 1999  |

**Result: H2 supported (loss-unit error flat at 0.54 across bins, 1.07× ratio; H1 rejected).** Loss-unit error is essentially flat (0.525–0.562) while absolute error grows ~37× across the σ range. The cv loss is regime-balanced as designed; the downstream payoff at low σ collapses because a constant σ-relative threshold error easily flips the stopping rule when threshold magnitudes are ≈2.5, whereas the same relative error on threshold values that are ≈260 (high σ) leaves the stopping rule mostly intact.

## Provenance

| Filename                                    | Source                                                   | How                  |
|---                                          |---                                                       |---                   |
| payoff-matrix.png                           | results/phase5/payoff_matrix.png                         | re-rendered, copied  |
| trajectories-disc-logu-zoom.png             | results/phase5/trajectories_disc_logu_zoom.png           | re-rendered, copied  |
| trajectories-d-disc-cv.png                  | results/phase5/trajectories_D_disc_cv.png                | re-rendered, copied  |
| trajectories-d-logu-cv.png                  | results/phase5/trajectories_D_logu_cv.png                | re-rendered, copied  |
| trajectories-all-cv.png                     | results/phase5/trajectories_all_cv.png                   | re-rendered, copied  |
| trajectories-d-1-cv.png                     | results/phase5/trajectories_D_1_cv.png                   | re-rendered, copied  |
| trajectories-d-2-cv.png                     | results/phase5/trajectories_D_2_cv.png                   | re-rendered, copied  |
| trajectories-d-3-cv.png                     | results/phase5/trajectories_D_3_cv.png                   | re-rendered, copied  |
| per-sigma-d-disc.png                        | results/phase5/per_sigma_D_disc.png                      | re-rendered, copied  |
| per-sigma-d-logu.png                        | results/phase5/per_sigma_D_logu.png                      | re-rendered, copied  |
| agreement-oracle.png                        | results/phase5/agreement_tensor.npz (oracle column)      | freshly rendered     |
| training-curves.png                         | results/phase4/training_curves.png                       | re-rendered, copied  |
| training-curves-per-sigma.png               | results/phase4/training_curves_per_sigma.png             | re-rendered, copied  |
| diagnostic-logu-abs-error.png               | seed-44 D_logu per-σ test cache + D_logu_cv checkpoint   | freshly rendered     |
| diagnostic-logu-loss-unit-error.png         | seed-44 D_logu per-σ test cache + D_logu_cv checkpoint   | freshly rendered     |
| payoff-matrix-extended-offline.png          | results/offline-supervision/payoff-matrix-extended-offline.png | freshly rendered, copied |
| bars-oracle-vs-offline.png                  | results/offline-supervision/bars-oracle-vs-offline.png   | freshly rendered, copied |
| trajectories-d-disc-cv-offline.png          | results/offline-supervision/trajectories-D_disc_cv_offline.png | freshly rendered, copied |
| trajectories-d-logu-cv-offline.png          | results/offline-supervision/trajectories-D_logu_cv_offline.png | freshly rendered, copied |
| probe-curves-logu-act.png                   | results/probing/logu-act/probe-curves-logu-act.png        | freshly rendered, copied |

All re-renders were produced by the original plotting scripts at dpi=300:

- `eval/render.py` (`write_payoff_matrix`, `render_trajectories`, `render_all_cv_overlaid`, `render_per_sigma`)
- `train/render_curves.py` (`render_grid`, `render_per_sigma`)
- `eval/diagnostic_logu.py` (the new diagnostic; standalone agreement-oracle panel rendered inline from `agreement_tensor.npz`)
