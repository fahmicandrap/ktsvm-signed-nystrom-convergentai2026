# Final study protocol

This document describes the design reported in the paper. All values are the ones used by the 40 cells in `results/`.

## Design
| Item | Setting |
|---|---|
| Exact KTSVM | N = 2,000 and 4,000; seeds 101–105 (10 cells) |
| Signed Nyström KTSVM | m = 250 and 1,000; N = 2,000, 4,000 and 8,000; seeds 101–105 (30 cells) |
| Exact runs at N = 8,000 | excluded before execution because the estimated dense peak memory exceeded the predeclared limit of 60 % of the available RAM; not attempted |
| Total | 40 terminal cells, all converged |

These 40 cells form the complete balanced design of the study. They were executed as part of a broader campaign plan that also scheduled larger N; cells of that plan beyond N = 8,000 were not completed for all seeds and are not part of the study or of this repository.

## Data and preprocessing
Official UNSW-NB15 training (175,341 rows) and testing (82,332 rows) CSV partitions (see `data/README_DATA.md`). Predictors: the 42 columns other than `id`, `attack_cat` and `label` (39 numerical, 3 categorical). Order of operations per cell: select the sample rows by id → one-hot encode `proto`, `service`, `state` with the vocabulary fixed from the complete training partition (`src/g1/categorical_vocabulary_v2.json`; unseen categories → all-zero block; 194 features) → fit `StandardScaler` on the sample only → transform the sample and the evaluation probe.

## Sampling, landmarks and probes
- Training samples: class-stratified nested prefixes of class-wise permutations (`src/phase9b/sampling.py`), attack/normal 1,361/639 (N = 2,000), 2,722/1,278 (N = 4,000), 5,445/2,555 (N = 8,000); manifests in `manifests/sampling/`.
- Landmarks: first m entries of the SHA-256 ordering of the sampled ids (the m = 250 set is contained in the m = 1,000 set); duplicated predictor vectors kept; `manifests/landmarks/`.
- Kernel-pair probe: 200,000 ordered pairs of distinct sample positions per (seed, N), uniform with replacement, diagonal excluded; `manifests/kernel_pairs/`.
- Function probe: 5,000 rows of the testing partition, class-stratified (2,753 attack / 2,247 normal), identical for all cells; `manifests/evaluation_probe.json`.

## Model and solver
Sigmoid kernel K(x, z) = tanh(γ xᵀz + θ) with γ = 10/148 and θ = −1 (fixed before the experiments, not tuned); c = 1, d = 0.001. Both twins are solved in the dual with GPCG; stopping rule r_∞ ≤ 1e-6 and gap_rel ≤ 1e-6, budget 2,001,015 Hessian–vector products per twin. Exact reference: dense K, S, M, H in float64, M solved by LU with two steps of iterative refinement using long-double residuals. Signed Nyström: eigendecomposition of the landmark block W, threshold τ_W = 1e-10 · max|λ(W)|, positive and negative eigenvalues retained, no clipping; matrix-free reduced SPD operator (`src/g3r1/`).

## Measurements
- Kernel-pair fidelity: relative L2 error, 99th-percentile and maximum absolute error over the pair probe.
- Raw-function fidelity (N = 2,000 and 4,000 only): NRMSE (RMSE / population SD of the exact function), Pearson correlation and relative L2 error of each twin's raw outputs on the function probe, against the exact cell with the same N and seed. Twin outputs are never converted to class labels.
- Training time: kernel construction + factorization + operator or dense Hessian construction + both solves (loading, preprocessing, probe evaluation and diagnostics excluded).
- Peak RSS: largest VmHWM over the training phases (reset at each phase start), includes the interpreter baseline.

## Execution
One process per cell, one logical core (Intel(R) Xeon(R) Processor @ 2.10GHz), one BLAS thread, no swap, 3 h wall-clock limit per cell; Linux-6.18.44-fc-v37-x86_64-with-glibc2.39, Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1, scipy-openblas 0.3.31.188.0. A cell interrupted by the execution environment was restarted from the beginning; the first uninterrupted terminal execution is the record in `results/`.
