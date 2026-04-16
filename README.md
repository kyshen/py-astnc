# ASTNC

`astnc` is a compact Python toolkit for dense materialization of tensor-network outputs with controllable approximation.

It extracts the reusable core of the AS-TNC research prototype into a package-oriented project:

- `exact` materialization as a baseline
- `fixed_rank` low-rank separator compression
- `astnc` adaptive approximation with workpoints (`l1`, `l2`, `l3`)
- blockwise output emission
- reusable separator-state cache across block calls
- small, direct API for demos, tests, and downstream reuse

This project is intentionally not a generic tensor-network framework and not a paper-reproduction harness. The focus is open-leg dense output materialization.

## Install

```bash
pip install -e .
```

For tests:

```bash
pip install -e .[dev]
pytest
```

## Quick Start

```python
import astnc as at

tn = at.random_connected(
    num_nodes=8,
    phys_dim=3,
    bond_dim=4,
    open_legs_per_node=1,
    edge_prob=0.35,
    seed=0,
)

dense, info = at.materialize(
    tn,
    method="astnc",
    workpoint="l2",
    block_labels=2,
    chunk_size=1,
    return_info=True,
)

print(dense.shape)
print(info["rel_method"])
print(info["meta"]["mean_rank"])
```

## Public API

- `materialize(...)`
- `random_connected(...)`
- `ring(...)`
- `tree(...)`
- `grid2d(...)`
- `create_cache(...)`
- `get_workpoint(...)`
- `available_workpoints()`

## Design Notes

- Paper reproduction code, Hydra configuration, CSV exporters, and experiment runners are deliberately excluded.
- The public layer exposes a small tool-oriented surface, while algorithm details live in `astnc.internal`.
- Workpoints give a stable user-facing knob without forcing callers to manage every compression hyperparameter.

