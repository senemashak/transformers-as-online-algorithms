# Tex changes — pre-results → results

## Files changed

- `research-notes_v3.tex` — modified.
- `research-notes_v3_pre_results.tex` — pre-edit backup (do not edit further).

## What was added

- New §7 **Results** (numbered §7 in the file because the existing §6 "In-Context Learning: What and How?" is preserved with its planning-phase subsections §6.1–§6.4 in place; see "Decisions not pinned" below). Six subsections:
    - §7.1 Static-variance models reproduce the V2 shortcut — V3 reproduction of the V2 single-regime shortcut without scale buffers, inline 3×3 payoff sub-matrix, three trajectory figures.
    - §7.2 $\mathcal{D}_{\mathrm{disc}}$ training recovers the regime-invariant algorithm — D_disc payoff numbers, per-σ uniform competence, trajectory adaptation as the algorithm-hypothesis signature.
    - §7.3 $\mathcal{D}_{\mathrm{logu}}$_cv exhibits a supervision-form-dependent failure — central diagnostic subsection: lays out the cv vs act asymmetry, decomposes cv error in raw vs loss units, localizes the bottleneck to the linear cv head.
    - §7.4 Comparison to data-only baselines — table comparing trained random-variance models against MAP-σ and MLE-σ plug-in baselines.
    - §7.5 Agreement-vs-payoff pathology — re-emergence of V2's high-agreement-low-payoff pathology in V3's static-model OOD cells; one figure.
    - §7.6 Findings — five evidence-anchored bullets summarising §7.1–§7.5.
- New Appendix A **Training curves** — train/val curves for all 10 runs and per-σ-bin val curves for the 4 random-variance runs; one paragraph of caption-supporting prose explaining the σ=100 catchup as the basis for the cv loss's $1/\sigma_i$ normalization.

## What was moved

- Existing §6.5–§6.10 (interpretability planning: attention entropy, probing, causal interventions, logit lens, activation patching, head ablation) → new §8 **Future work — interpretability suite**, §8.1–§8.6, condensed from the original multi-paragraph subsections to single-paragraph stubs each consisting of a one-sentence summary plus a one-sentence note that the analysis is deferred and is now scoped against a specific question raised by §7.

## What was modified

- §1 Introduction — abstract paragraph **left as the original** (the question-ending paragraph "Can transformers learn this kind of instance adaptability …?"). One new paragraph appended after the hypothesis paragraph (the one with the itemize on "Learning-algorithm vs. Shortcut hypothesis"): two sentences pointing forward to §7 (the algorithm hypothesis is confirmed under $\mathcal{D}_{\mathrm{disc}}$; $\mathcal{D}_{\mathrm{logu}}$ surfaces the supervision-form-dependent failure that §7.3 localizes).
- Per-model trajectory captions in §7.1, §7.2, §7.3 — updated from "all seven decision-rule baselines" to "all baselines from Section~\ref{sec:baselines}", reflecting the expanded baseline set in the trajectory PNGs (per-regime oracle, both random-ADP oracles, plug-in, prior-only, myopic, both MAP-σ flavors, MLE-σ, secretary, offline-hindsight horizontal line).

## What was untouched

- §1 main body (all original paragraphs, including the abstract / first paragraph).
- §2 (Setup), §3 (Bayes-optimal Oracle), §4 (Portfolio of Baseline Algorithms), §5 (Model and Training).
- §6 (In-Context Learning: What and How?) §6.1 Metrics, §6.2 Static-variance experiment, §6.3 Random-variance experiment, §6.4 Threshold trajectories — preserved per the "don't change §1–§5 prose" instruction (see "Decisions not pinned").
- Existing Appendix B (Derivation of the marginal log-likelihood, was Appendix A in the pre-edit file).
- Bibliography (`\bibliography{refs}`, style `plainnat`).
- Macro / package definitions (no preamble changes).

## Figures introduced

In order of first appearance in the .tex:

1. `\includegraphics[width=\textwidth]{figures/trajectories-d-1-cv.png}` — §7.1, Fig.~\ref{fig:traj-d1cv}.
2. `\includegraphics[width=\textwidth]{figures/trajectories-d-2-cv.png}` — §7.1, Fig.~\ref{fig:traj-d2cv}.
3. `\includegraphics[width=\textwidth]{figures/trajectories-d-3-cv.png}` — §7.1, Fig.~\ref{fig:traj-d3cv}.
4. `\includegraphics[width=0.55\textwidth]{figures/payoff-matrix.png}` — §7.2, Fig.~\ref{fig:payoff-matrix}.
5. `\includegraphics[width=0.85\textwidth]{figures/per-sigma-d-disc.png}` — §7.2, Fig.~\ref{fig:per-sigma-disc}.
6. `\includegraphics[width=\textwidth]{figures/trajectories-d-disc-cv.png}` — §7.2, Fig.~\ref{fig:traj-disc-cv}.
7. `\includegraphics[width=\textwidth]{figures/trajectories-d-logu-cv.png}` — §7.3, Fig.~\ref{fig:traj-logu-cv}.
8. `\includegraphics[width=\textwidth]{figures/trajectories-disc-logu-zoom.png}` — §7.3, Fig.~\ref{fig:traj-zoom}.
9. `\includegraphics[width=0.95\textwidth]{figures/per-sigma-d-logu.png}` — §7.3, Fig.~\ref{fig:per-sigma-logu}.
10. `\includegraphics[width=\textwidth]{figures/diagnostic-logu-abs-error.png}` — §7.3, in Fig.~\ref{fig:diag-abs} (left minipage).
11. `\includegraphics[width=\textwidth]{figures/diagnostic-logu-loss-unit-error.png}` — §7.3, in Fig.~\ref{fig:diag-abs} (right minipage).
12. `\includegraphics[width=0.5\textwidth]{figures/agreement-oracle.png}` — §7.5, Fig.~\ref{fig:agreement-oracle}.
13. `\includegraphics[width=\textwidth]{figures/training-curves.png}` — Appendix A, Fig.~\ref{fig:training-curves}.
14. `\includegraphics[width=\textwidth]{figures/training-curves-per-sigma.png}` — Appendix A, Fig.~\ref{fig:training-curves-per-sigma}.

## Decisions not pinned

- **Section numbering.** Your prompt numbered the new Results section as §6 and the moved Future-work section as §7. The file's existing §6 ("In-Context Learning: What and How?") with its planning-phase subsections §6.1–§6.4 (Metrics, Static-variance experiment, Random-variance experiment, Threshold trajectories) is preserved per the "don't change §1–§5 prose" constraint — so Results becomes §7 in the file and Future work becomes §8. Cross-references in the abstract and §1 use `\ref{sec:results}` and `\ref{sec:res-logu-cv}` so LaTeX auto-numbers correctly. If you want the prompt's exact numbering, the cleanest move is to fold the existing §6.1–§6.4 into §5 as new §5.3–§5.6 (renaming §5 to "Model, Training, and Evaluation") — happy to make that pass.
- **Trajectory triptych in §7.1.** Your prompt suggested a 1×3 layout via subcaption / subfigure if loaded, otherwise three side-by-side mini-figures. The source PNGs are themselves wide 1×3 panels (4500×1350 px), and at $\sim 0.32\textwidth$ side-by-side they would be unreadable. I used three vertically-stacked full-width `\begin{figure}[H]` blocks instead; the preamble was not modified. If you would prefer a single combined figure, the cleanest option is `\usepackage{subcaption}` plus three `\begin{subfigure}{\textwidth}` panels in one `figure` — happy to switch.
- **Diagnostic figure layout in §7.3.** The two diagnostic plots (`diagnostic-logu-abs-error.png` and `diagnostic-logu-loss-unit-error.png`) are side-by-side via two `\begin{minipage}{0.49\textwidth}` blocks inside a single `figure`, with a single shared caption (one `\label{fig:diag-abs}` referenced as such in prose). No subcaption package needed.
- **Baseline table in §7.4 has 7 rows, not 6.** Your prompt specified 6 rows including a single MAP-σ-plug-in entry. There are two MAP-σ flavors in the data (`MAP_sigma_disc` prior and `MAP_sigma_logu` prior) with materially different numbers on $\mathcal{D}_3$ (0.991 vs 0.975). I included both as separate rows so the comparison is honest; reduce to one row by deleting whichever flavor is redundant if you prefer the original 6-row layout.
- **Section 7.6 bullet count.** Your prompt allowed 3–5 findings bullets; I used 5.
- **§7.5 V2 reference.** Your prompt allowed citing V2 if there's a citable chunk, otherwise informal reference. The bibliography (`refs.bib`) wasn't audited for a V2 entry; I used informal reference ("the phenomenon was identified in V2's analysis…") consistent with §7.1's V2 mention.
- **Section title for §8.** "Future work --- interpretability suite" with em-dash via `---`, matching the existing tex's typography (e.g., the title of Section~\ref{sec:setup} uses no em-dash, but the introduction prose uses `---` consistently).
- **Appendix ordering.** Training curves placed as Appendix A (before the existing marginal-log-likelihood derivation, which becomes Appendix B). Alternative ordering (training curves last) would not change content, only label letters.
- **Backup file location.** `research-notes_v3_pre_results.tex` saved alongside `research-notes_v3.tex` in `bayesian-stopping/v3/`. Not in `report/` — Overleaf only consumes the live tex.
