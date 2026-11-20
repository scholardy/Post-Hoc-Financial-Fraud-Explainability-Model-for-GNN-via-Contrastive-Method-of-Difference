# Contrastive method-of-difference explanations for GNN fraud detectors — code and results

Experiments for *"A Post-Hoc Financial Fraud Explainability Model for Graph Neural Networks via Contrastive
Method of Difference"*. The explained models are the GCN, GraphSAGE and GAT transaction classifiers trained in the
LS-GAT study (strictly causal transaction graphs of IEEE-CIS, PaySim and the ULB Credit Card data; checkpoints in
`models/main/`). Every number and figure in the paper is produced from `results/` by `scripts/`.

## Layout
```
configs/experiments.json   graph / model settings of the explained GNNs (from the LS-GAT run)
configs/xai.json           explainability settings (targets, budgets, method hyper-parameters, profiles)
src/lsgat/                 graph construction and models of the LS-GAT study (unchanged copy)
src/xai/core.py            computation subgraphs, masked forward pass (exact w.r.t. the full graph)
src/xai/explainers.py      proposed explainer + GNNExplainer, PGExplainer, SubgraphX, Grad-CAM, random
src/xai/evaluate.py        fidelity, necessity, sparsity, discriminative power, stability
scripts/run_xai.py         runs the stages; one JSON per explained transaction under results/<stage>/
scripts/make_tables_figures.py   results -> tables/*.csv, figures/*.png
scripts/update_paper.py    fills the manuscript with the tables, figures and numbers
kaggle/                    Kaggle notebook (GPU T4 x2) and run script
models/main/               trained GNN checkpoints (10 seeds x GCN/GraphSAGE/GAT x 3 datasets)
```

## Reproduce
- **Kaggle (all results, GPU T4 x2):** `python kaggle/make_upload_zip.py`, upload the zip as a private dataset, open
  `kaggle/xai_kaggle.ipynb`, attach the three Kaggle datasets and the code dataset, run with `MODE="pilot"`, then
  `MODE="full"`. Download `xai_results.zip` and unzip it here.
- **Tables / figures / paper:** `python scripts/make_tables_figures.py` then `python scripts/update_paper.py`.
- **Local smoke test (CPU, minutes):** `python scripts/run_xai.py --profile smoke --data-profile local --datasets creditcard`
  (needs the raw CSVs under `data/raw/` or `../lsgat_experiments/data/raw/`).

## Main results (GraphSAGE detector, equal explanation size; `tables/summary.json`)

| Dataset | Targets | Ours DP | Best baseline DP | Ours flip rate | Ours fidelity |
|---|---|---|---|---|---|
| Credit Card | 55 | 0.327 | 0.272 (gnnexplainer) | 100% | 0.968 |
| IEEE-CIS | 300 | 0.137 | 0.126 (gnnexplainer) | 40% | 0.985 |
| PaySim | 22 | 0.100 | 0.099 (subgraphx) | 100% | 0.980 |

- IEEE-CIS: significantly better than every baseline on fidelity, necessity and discriminative power (Holm-corrected
  Wilcoxon, p < 0.001). Credit Card: significantly better on discriminative power and necessity, slightly lower fidelity.
  PaySim: all methods alike (only 22 detected frauds have any earlier linked transaction).
- Ablation: signed contrast is the decisive component; legitimate references and the ensemble help on IEEE-CIS.
- Time per explanation on a T4: ours 0.40 s, GNNExplainer 0.47 s, SubgraphX 1.00 s.

## Environment of the reported run
Kaggle notebook, 2 x NVIDIA Tesla T4; PyTorch 2.11.0+cu130, PyTorch Geometric 2.8.1. Every stored test metric of the
90 explained checkpoints was reproduced exactly (`logs/full_*.log`, `[perf]` lines).

## Paper
`scripts/make_tables_figures.py` -> `scripts/case_study.py` (needs the local IEEE-CIS data) -> `scripts/update_paper.py`
writes `../For Tosin - Explainability/Explainability manuscript - FINAL.docx` from `paper/Explainability_manuscript_original.docx`.
The script checks that no placeholder text remains, that table / figure numbers run in order and that every
citation has a reference (and vice versa).

## Notes
- The supplied explainer script (`METHOD OF DIFFERENCE EXPLAINABILITY.txt`) computed the gradient of f(G_i) - f(G_j)
  with f(G_j) held constant, so the legitimate reference had no effect on the explanation. `src/xai/explainers.py`
  implements the method of difference properly: every element is compared with its hop-matched legitimate
  counterpart (features and messages), and only evidence for fraud is kept.
- The human-evaluation study of the original draft was not carried out and has been removed from the paper.
