# Bayesian optimal stopping

A GPT-2 decoder trained on the Bayesian optimal-stopping problem with Normal–Normal conjugate priors. Five training distributions $\{\mathcal{D}_1, \mathcal{D}_2, \mathcal{D}_3, \mathcal{D}_{\mathrm{disc}}, \mathcal{D}_{\mathrm{logu}}\}$ × two supervisions (continuation-value, action) = ten trained models. Approximate dynamic programming supplies the per-step oracle threshold; the network observes the raw scalar stream and predicts either the continuation value (MSE) or the optimal action (BCE).

## Layout

```
report/      research-notes.tex + figures/ + compiled PDF
code/
  oracle/    static + 2D random-variance ADP, marginal log-likelihood
  model/     transformer, heads, losses
  data/      samplers, labelers, train/val/test stream
  train/     run configs, training loop, sweep driver
  eval/      payoff, agreement, trajectories, baselines, renderers
  interp/    linear probes (per-timestep + attention-pooled)
results/     per-run logs, payoff/agreement/trajectory caches, probe runs
checkpoints/ trained models (one folder per run)
```

## Writeup

[`report/research-notes.tex`](report/research-notes.tex) is the authoritative document — setup, oracle, model, full experimental results (in-distribution, OOD, temporal shift), mechanistic probing, and appendices with all proofs.

## Running

```bash
cd code
bash train/run_sweep.sh full           # train all ten models
python -m eval.run_eval                # score every model on every regime
python -m eval.render_ood              # rebuild the OOD figures
python -m interp.probe_run             # run the probing pipeline
```

Each rendering script writes to [`report/figures/`](report/figures/) (the PNGs the tex includes) and mirrors a copy under `results/` next to the source data.
