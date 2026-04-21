# Extraction Notes

This package is a deliberate extraction from a research-oriented AS-TNC codebase.

Included:

- tensor-network graph generators
- tensor-network representation and contraction helpers
- blockwise output slicing
- recursive partitioning
- separator-state construction and merging
- exact and adaptive ASTNC contraction paths
- reusable separator-state cache

Excluded:

- Hydra configuration trees
- task/data/runner orchestration
- CSV export and paper reproduction scripts
- paper assets and reporting-only metrics

The main engineering goal is to expose the contraction capability as a clean package API rather than as an experiment pipeline.
