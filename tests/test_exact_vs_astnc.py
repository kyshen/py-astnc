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
    exact = at.contract_exact(tn)
    approx = at.contract_astnc(tn, workpoint="l2", block_spec={0: 1, 1: 1})
    rel = np.linalg.norm(exact - approx) / (np.linalg.norm(exact) + 1e-12)
    assert exact.shape == approx.shape
    assert np.isfinite(rel)
    assert rel < 0.5
