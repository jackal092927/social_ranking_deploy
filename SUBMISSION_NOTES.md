# Submission Notes

This repository is an anonymized deployment copy prepared for manuscript submission.

## Anonymization Actions Applied

- Removed public Git remote configuration from the local clone.
- Replaced submission-facing README language with anonymous wording.
- Added anonymous `CITATION.cff` metadata for review use.
- Added a review-copy `LICENSE` notice instead of a public release license.
- Normalized notebook kernel metadata to a generic `Python 3` kernel label.
- Removed tracked macOS metadata (`lab6/.DS_Store`) from version control.

## Reviewer-Facing Entry Points

- Notebook workflow: `sync_exp.ipynb`
- Main experiment driver: `python -m lab6.exp1`
- Smoke test: `python -m unittest discover -s tests -p 'test_*.py'`

## After Review

Before public release, replace the anonymous metadata with:

- the final paper title;
- author names and affiliations;
- the public repository URL, if desired;
- the intended open-source or distribution license;
- the final software citation metadata.
