# Anonymous Submission Replication Package

This repository is a minimized, anonymized replication package prepared for journal submission. It contains the code, notebook workflow, and precomputed outputs needed to inspect and reproduce the experiments without exposing author-identifying repository metadata.

## Repository Contents

- `sync_exp.ipynb`: Main notebook for reproducing and inspecting the experimental workflow.
- `lab6/exp1.py`: Command-line entry point for the synthetic experiment suite.
- `lab6/utils.py`: Core ranking, evaluation, and strategy implementations.
- `lab6/visualization_helper.py`: Figure and summary-table generation utilities.
- `plot_caggrim_ratio_heatmap.py`: Standalone plotting helper for comparison surfaces.
- `lab6/results/`: Included output directories for the archived `agg` and `card` runs.
- `lab9/data_loader.py`: Optional loaders for external network datasets referenced by exploratory code.

## Environment Setup

The codebase targets Python 3.10+ and depends on a small scientific Python stack.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to execute the notebook locally, install Jupyter in the same environment:

```bash
pip install jupyter
```

## Quick Start

Run the notebook:

```bash
jupyter notebook sync_exp.ipynb
```

Run the main experiment driver from the command line:

```bash
python -m lab6.exp1 --metric agg --score_rule borda --init_score custom
python -m lab6.exp1 --metric card --score_rule borda --init_score custom
```

Generate a comparison plot from an existing results table:

```bash
python plot_caggrim_ratio_heatmap.py \
  --filepath lab6/results/lab6exp1-20250515-222136__agg__borda__custom-init/alpha_beta_sensitivity_results.csv
```

## Included Results

The repository already includes two result directories that document the archived baseline runs:

- `lab6/results/lab6exp1-20250515-222136__agg__borda__custom-init`
- `lab6/results/lab6exp1-20250515-222800__card__borda__custom-init`

Each directory contains:

- `parameters.json`: the exact parameterization used for the run;
- `alpha_beta_sensitivity_results.csv`: the aggregated results table;
- `optimal_beta_summary.csv`: summary statistics over the parameter sweep;
- publication-style PNG figures generated from the sweep.

## Reproducibility Notes

- The default experiments are seeded (`seed=42`) for deterministic synthetic initialization.
- The main experiment code can be run either as a script (`python lab6/exp1.py ...`) or as a module (`python -m lab6.exp1 ...`).
- External network datasets used by `lab9/data_loader.py` are not bundled in this archive; if those exploratory loaders are used, the corresponding raw files must be placed in the expected relative paths documented in that module.
- This submission copy intentionally omits links to the public development repository and does not include author-identifying contact metadata.

## Anonymous Submission Notes

- `CITATION.cff` is intentionally anonymous for the review period.
- `LICENSE` is included as a review-copy notice rather than a public release license.
- The local Git remote configuration has been removed from this clone to avoid carrying public repository links into the submission artifact.

## Verification

The repository includes a lightweight smoke test covering importability and a reduced-size end-to-end experiment:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```
