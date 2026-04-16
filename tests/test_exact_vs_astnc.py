import numpy as np

import astnc as at


def test_exact_and_astnc_have_same_shape_and_bounded_error():
    tn = at.ring(
        num_nodes=6,
        phys_dim=3,
        bond_dim=4,
        open_legs_per_node=1,
        seed=0,
    )
    exact = at.materialize(tn, method="exact", block_labels=2, chunk_size=1)
    approx = at.materialize(tn, method="astnc", workpoint="l2", block_labels=2, chunk_size=1)
    rel = np.linalg.norm(exact - approx) / (np.linalg.norm(exact) + 1e-12)
    assert exact.shape == approx.shape
    assert np.isfinite(rel)
    assert rel < 0.5

