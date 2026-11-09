# Reproducing the paper's numbers

Steps 1–3 below do not train a model. Step 4 is optional and re-executes a computational cell (both solver runs).

1. Install the requirements: `pip install -r requirements.txt` (Python 3.12).
2. Verify the package and every number of the paper:
   `python analysis/reproduce.py`
   The script checks `SHA256SUMS.txt`, rebuilds the per-cell quantities from `results/`, recomputes every quantitative entry of `analysis/manuscript_evidence_ledger.csv` (all values in Tables 1–2 and in the text) and compares it with the published value, checks the figure source data against the raw records and redraws Figs. 2–3 into `analysis/figures_regenerated/`. It exits with status 0 only if everything matches.
3. Optional, requires the official training CSV (see `data/README_DATA.md`): recompute the post-hoc operator diagnostics (κ₂(Φ) and the thin-QR check of Sect. 4.2):
   `python diagnostics/operator_diagnostics.py --data-dir <folder with UNSW_NB15_training-set.csv>` (add `--limit 1` for a quick check). Results are written to `diagnostics/reproduced/`.
4. Optional, re-running a cell: `src/phase9b/cell_worker.py <spec.json>` executes one cell with the frozen implementation. The `spec.json` files in `results/` keep the absolute paths of the original execution environment for provenance; for a new run, copy a spec and point `paths` to the official CSVs and to the corresponding files in `manifests/`. Timings depend on the hardware.
