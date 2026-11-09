# Configuration table

Compact configuration of the experiment reported in the paper (values resolved from the frozen artifacts by `analysis/original_pipeline/generate_all_tables.py`).

| Item | Setting |
|---|---|
| Data | UNSW-NB15 official training partition (175,341 rows) for sampling; official testing partition (82,332 rows) only for the fixed function probe |
| N and seeds | N = 2,000, 4,000, 8,000 (nested, stratified samples); seeds 101–105 |
| Exact KTSVM | N = 2,000 and 4,000 (five seeds each) |
| Signed Nyström KTSVM | m = 250 and 1,000 at N = 2,000, 4,000, 8,000 (five seeds each) |
| Kernel and model | K(x, z) = tanh(γxᵀz + θ), γ = 10/148, θ = −1; c = 1, d = 0.001 |
| Solver | GPCG; stop when r∞ ≤ 10⁻⁶ and gap_rel ≤ 10⁻⁶; HVP budget 2,001,015 per twin |
| Fidelity probes | 200,000 fixed training-row pairs per (seed, N); 5,000 fixed test rows |
| Execution | One process per cell, one BLAS thread; 40 terminal cells (10 exact, 30 Nyström) |
| Preprocessing order | select sample rows by id → one-hot encode `proto`, `service`, `state` with the vocabulary fixed from the complete training partition (`src/g1/categorical_vocabulary_v2.json`; unseen categories → all-zero block) → fit `StandardScaler` on the sample only → transform sample and probe |
| Kernel-pair probe | 200,000 ordered pairs (i, k) of sample positions per (seed, N), uniform with replacement, i ≠ k (diagonal excluded), `SeedSequence([seed, N, 5150])` (`src/phase9b/probe_generator.py`) |
| Function probe | 5,000 rows of the official testing partition, class-stratified (2,753 attack / 2,247 normal), seed 20260922, fixed for all cells |
| Landmarks | first m entries of the SHA-256 ordering of the sampled ids (m = 250 ⊂ m = 1,000); duplicated predictor vectors kept |
| Eigenvalue threshold | τ_W = 1e-10 · max_i |λ_i(W)|; positive and negative eigenvalues retained, no clipping |
| Exact reference | dense K, S, M, H in float64; M solved by LU with two steps of iterative refinement using long-double residuals; not scheduled for N ≥ 8,000 (predeclared resource limit) |
| Timing | training time = kernel + factorization + operator/Hessian construction + both solves (load, preprocessing, probe, diagnostics excluded) |
| Memory | peak RSS = max VmHWM over training phases (reset via /proc/self/clear_refs at each phase start), includes interpreter baseline |
| Environment | Intel(R) Xeon(R) Processor @ 2.10GHz, 1 logical core, no swap, Linux-6.18.44-fc-v37-x86_64-with-glibc2.39, Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1, scipy-openblas 0.3.31.188.0, 1 BLAS thread, float64, 3 h wall-clock limit per cell |
| Interrupted runs | restarted from the beginning; first uninterrupted terminal execution is canonical; partial attempts excluded |

## Sample composition (training samples)

Class counts and unique predictor vectors (42 raw predictor columns) per sample; generated from the official CSV and the sampling manifests (`data/sample_composition.csv`).

| seed | N | attack | normal | unique predictor vectors |
|---|---|---|---|---|
| 101 | 2,000 | 1,361 | 639 | 1,859 |
| 101 | 4,000 | 2,722 | 1,278 | 3,570 |
| 101 | 8,000 | 5,445 | 2,555 | 6,759 |
| 102 | 2,000 | 1,361 | 639 | 1,885 |
| 102 | 4,000 | 2,722 | 1,278 | 3,593 |
| 102 | 8,000 | 5,445 | 2,555 | 6,748 |
| 103 | 2,000 | 1,361 | 639 | 1,882 |
| 103 | 4,000 | 2,722 | 1,278 | 3,592 |
| 103 | 8,000 | 5,445 | 2,555 | 6,743 |
| 104 | 2,000 | 1,361 | 639 | 1,874 |
| 104 | 4,000 | 2,722 | 1,278 | 3,595 |
| 104 | 8,000 | 5,445 | 2,555 | 6,738 |
| 105 | 2,000 | 1,361 | 639 | 1,877 |
| 105 | 4,000 | 2,722 | 1,278 | 3,621 |
| 105 | 8,000 | 5,445 | 2,555 | 6,751 |
