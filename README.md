# ASTNC

`astnc` is a compact Python toolkit for adaptive sketch tensor network contraction with controllable approximation.

It extracts the reusable core of the AS-TNC research prototype into a package-oriented project:

- `exact` contraction as a baseline
- `astnc` adaptive approximation with configurable `tol`
- blockwise output emission
- reusable separator-state cache across block calls
- small, direct API for demos, tests, and downstream reuse

This project is intentionally not a generic tensor-network framework and not a paper-reproduction harness. The focus is open-leg dense output contraction.

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

dense, info = at.contract_astnc(
    tn,
    tol=1e-3,
    block_spec={0: 1, 1: 1},
    return_info=True,
)

print(dense.shape)
print(info["rel_method"])
print(info["meta"]["mean_internal_rank"])
```

For the exact baseline, use:

```python
exact = at.contract_exact(tn)
```

Typical `tol` values:

- `5e-4`: more accurate
- `1e-3`: balanced default
- `3e-3`: faster, looser approximation

`block_spec` and cache reuse are only supported by `contract_astnc(...)`.

## Public API

- `contract_exact(...)`
- `contract_astnc(...)`
- `random_connected(...)`
- `ring(...)`
- `tree(...)`
- `grid2d(...)`

## Design Notes

- Paper reproduction code, Hydra configuration, CSV exporters, and experiment runners are deliberately excluded.
- The public layer exposes a small tool-oriented surface, while algorithm details live in `astnc.internal`.
