# Transformers as Online Algorithms

Empirical study of whether transformers, trained on exact oracle labels, can recover the algorithms that solve classical online decision problems — optimal stopping, ski rental, and cache eviction — and whether what they learn generalizes beyond the training distribution.

## Subprojects

### [`bayesian-stopping/`](bayesian-stopping/)

The main, active project. A GPT-2-style decoder is trained on the **Bayesian optimal stopping** problem with Normal–Normal conjugate priors, including settings where the noise level $\sigma$ is itself random across sequences (discrete and log-uniform priors). Approximate dynamic programming supplies the per-step oracle threshold $\widehat C^\star_t$; the network is supervised on either the continuation value (regression) or the optimal action (binary). Headline experiments:

- **In-distribution and OOD competence** at $\sigma \in \{0.3, 1, 3, 10, 30, 100, 300\}$, including interpolation between training $\sigma$ values and extrapolation beyond.
- **In-context regime shifts**: the noise level switches mid-sequence, and the trained models adapt within a few observations, while every i.i.d. baseline (including the ADP oracle that supplied training labels) plots a flat post-shift threshold by construction.
- **Mechanistic analysis**: linear probing of frozen models' hidden states for the sufficient statistics, posterior probabilities, and continuation values that the algorithmic hypothesis predicts, with a planned paired-ensemble permutation test for statistical grounding.

Full writeup in [`bayesian-stopping/report/research-notes.tex`](bayesian-stopping/report/research-notes.tex); training/eval/probing code in [`bayesian-stopping/code/`](bayesian-stopping/code/).

### [`caching/`](caching/)

A transformer that learns the **Belady (furthest-in-future) eviction policy** for caching. Each attention block has two heads — one keyed on the full cache (k=32 slots, always visible), the other keyed on a sliding window of the request sequence — so the model scales to T=16,000-long traces while attending over only $k + w$ tokens per step. Trained on eviction decisions extracted from an oracle pass over the full trace, evaluated on synthetic LRU/LFU/ARC-favoring workloads. Code under [`caching/learned_eviction/`](caching/learned_eviction/).

### [`robust-transformers-with-chain-of-thought/`](robust-transformers-with-chain-of-thought/)

Two-part work on **optimal stopping** and **ski rental**:

1. **Learning the algorithm.** A small causal transformer (~20M params, 2 layers, 2 heads) with a **2D chain-of-thought scratchpad** that performs backward induction in-place, one continuation value per scratchpad token. Trained on exact DP labels; architecture sized to match the theoretical construction.
2. **Robustifying it at deployment.** A wrapper that, at inference time, switches the learned policy to a worst-case-safe baseline whenever its confidence (or chain-consistency) falls below a tunable threshold $\beta$. The transformer keeps the average-case competitive ratio of the DP oracle on in-distribution inputs and inherits a worst-case guarantee on adversarial ones.

Details in [`robust-transformers-with-chain-of-thought/experiment.tex`](robust-transformers-with-chain-of-thought/experiment.tex).

## Shared thesis

Each subproject trains a transformer to imitate an oracle that sees information the model does not — the DP oracle that knows the prior, the prophet that sees the future, or the variance estimator that knows the regime. The recurring question is whether the network, with only the online posterior, learns an **algorithm** (a noise-conditional rule that generalizes to novel inputs) or a **shortcut** (a fixed policy tuned to the training distribution that fails off-distribution). Across the three settings, the answer depends sharply on whether training forces the model to handle a distribution rather than a fixed instance.
