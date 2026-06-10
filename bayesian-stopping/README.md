# Bayesian optimal stopping

A GPT-2 decoder trained on the Bayesian optimal-stopping problem with Normal–Normal conjugate priors. Five training distributions $\{\mathcal{D}_1, \mathcal{D}_2, \mathcal{D}_3, \mathcal{D}_{\mathrm{disc}}, \mathcal{D}_{\mathrm{logu}}\}$ × two supervisions (continuation-value, action) = ten trained models. Approximate dynamic programming supplies the per-step oracle threshold; the network observes the raw scalar stream and predicts either the continuation value (MSE) or the optimal action (BCE).

## Layout

```
report/         research-notes.tex + figures/ + compiled PDF (the writeup)
code/
  oracle/       static + 2D random-variance ADP, marginal log-likelihood
  model/        transformer, heads, losses
  data/         samplers, labelers, train/val/test stream
  train/        run configs, training loop, sweep driver
  eval/         payoff, agreement, trajectories, baselines, renderers
  interp/       linear probes (per-timestep + attention-pooled)
results/
  phase2/         ADP convergence checks (report Appendix A)
  phase4/         per-model training curves + log.jsonl (ten trained models)
  phase5/         cross-regime evaluation: payoff matrix + agreement tensor
                  → report fig:payoff-and-agreement
  ood/            OOD test results at σ ∈ {0.3, 3, 30, 300}
                  → report §Results: OOD
  offline-supervision/  hindsight-prophet baseline experiments
                  → report fig:bars-oracle-vs-offline
  regime-shift/   mid-sequence σ-switch experiments B.1 and B.2
                  → report §Regime-shift adaptation
  probing/        legacy probing campaigns (logu-act, disc-act) — superseded by probe/full100
  sweep_logs/     raw stdout/stderr from each sweep run
probe/
  full100/        n=100 ensemble mechanistic-probing sweep — the active probing data.
                  Contains members/000..099/{probe_runs,probe_runs_attn}/,
                  stats/, stats_init/, stats_attn/, stats_init_attn/,
                  figures/, comparison/, logs/, manifest.json.
                  Powers fig:main-svsa, fig:main-paired, fig:main-paired-init,
                  fig:main-budget-* in the report.
                  See manifest.json inside for reuse / provenance metadata.
checkpoints/    trained-model configs + training-curve logs (one folder per of the ten runs);
                model weights (*.pt) removed in cleanup — see Reproduction below
```

## Writeup

[`report/research-notes.tex`](report/research-notes.tex) is the authoritative document — setup, oracle, model, full experimental results (in-distribution, OOD, regime-shift), mechanistic probing, and appendices with all proofs. Compiled PDF lives next to it.

## Running

```bash
cd code
bash train/run_sweep.sh full           # train all ten models
python -m eval.run_eval                # score every model on every regime
python -m eval.render_ood              # rebuild the OOD figures
python -m interp.probe_run             # run the legacy probing pipeline (results/probing/)
python -m interp.prelim_v2.run_member  # n=100 mechanistic probing (probe/full100/)
```

Each rendering script writes to [`report/figures/`](report/figures/) (the PNGs the tex includes) and mirrors a copy under `results/` or `probe/` next to the source data.

## Reproduction

This directory has been cleaned: model weights (`*.pt`), cached hidden activations (`hidden_cache/`), the upstream n=20 sweep, and the legacy disc-act probing cache are removed to recover disk. Surviving artifacts are sufficient to **defend** every number and figure in the report (all aggregated stats, plotting CSVs, configs, training logs, and the compiled PDF are intact), but not to extend the sweeps without re-training. To reproduce from scratch:

1. Re-train the ten action/cv models from configs in `checkpoints/*/config.json` (~hours of GPU per model)
2. Re-run eval scripts in `code/eval/` to regenerate `results/phase4`, `results/phase5`, `results/ood`, etc.
3. Re-cache hidden states and re-run probes via `code/interp/prelim_v2/` to regenerate `probe/full100/`

`probe/full100/.../manifest.json` documents the n=100 sweep's provenance: predecessor n=20 commit, reuse policy for members 0–19, ensemble seeds, probe-family configs, and training budgets.
