# Matrix-Free Signed Nyström KTSVM — reproducibility package

Supplementary material for: S. Sylviani, F. C. Permana, D. A. Kusuma, “Matrix-Free Signed Nyström KTSVM: Resource-Fidelity Trade-offs in Network Intrusion Analytics”, Convergent AI 2026.

The repository contains everything needed to verify the 40 computational cells reported in the paper (10 exact KTSVM cells at N = 2,000 and 4,000; 30 signed Nyström cells with m = 250 and 1,000 at N = 2,000, 4,000 and 8,000; seeds 101–105). Raw twin functions are compared; no decision rule or classification metric is computed.

| Item | Contents |
|---|---|
| `AUTHORS.md` | Authors, ORCID iDs, affiliations, corresponding author, maintainer, ownership of the contents, funding |
| `FINAL_STUDY_PROTOCOL.md` | Design, data, preprocessing, sampling, model, solver, measurements and execution environment |
| `CONFIGURATION.md` | Configuration table and per-sample composition |
| `REPRODUCE.md` | How to verify every number of the paper from this repository |
| `src/` | Implementation in its original relative layout; the eight frozen files are byte-identical to the validated implementation (`src/PUBLIC_IMPLEMENTATION_MANIFEST.json`) |
| `manifests/` | Identifier manifests for the final scope only: training samples, landmark orderings, kernel-pair positions, function probe, environment record |
| `results/` | The 40 terminal cell records (`result.checked.json`, `spec.json`, worker log) and raw-function outputs on the probe |
| `validation/` | Pre-registered validation of the matrix-free operator, performed before any cell was run |
| `diagnostics/` | Post-hoc operator diagnostics of Sect. 4.2 (κ₂(Φ), thin-QR check; no training, no solver run) |
| `analysis/` | Extracted results, table and figure source data with provenance, evidence ledger, portable verification (`reproduce.py`, `plot_figures.py`) and the original pipeline |
| `per_seed_diagnostics.csv` | Per-seed phase times, peak RSS, HVP counts, ms/HVP, retained rank and condition numbers of all 40 cells |
| `data/` | Official source, SHA-256 of the CSV files and sample composition (raw data not included) |

Quick check: `pip install -r requirements.txt` and `python analysis/reproduce.py`.

Paths stored in `results/*/spec.json` preserve the original execution environment and are retained for provenance; repository-relative locations are given in `FINAL_STUDY_PROTOCOL.md` and used by the scripts in `analysis/` and `diagnostics/`.

## Authors and contact
Sisilia Sylviani (corresponding author, sisilia.sylviani@unpad.ac.id), Fahmi Candra Permana (repository maintainer) and Dianne Amor Kusuma; details in `AUTHORS.md`. The code, manifests, cell records and analyses are the authors' work; the UNSW-NB15 data are third-party data and are not included.

License: MIT for the code (see `LICENSE`); the UNSW-NB15 data are subject to their own terms. Citation: see `CITATION.cff`.
