import astnc as at


def test_basic_api_roundtrip():
    tn = at.random_connected(
        num_nodes=6,
        phys_dim=3,
        bond_dim=4,
        open_legs_per_node=1,
        edge_prob=0.45,
        seed=0,
    )
    dense, info = at.contract_astnc(
        tn,
        tol=1e-3,
        block_spec={0: 1, 1: 1},
        return_info=True,
    )
    assert dense.shape == tn.output_shape
    assert info["method"] == "astnc"
    assert info["block_spec"] == {0: 1, 1: 1}
    assert info["num_blocks"] > 1
    assert "mean_internal_rank" in info["meta"]


def test_exact_api_roundtrip():
    tn = at.ring(
        num_nodes=4,
        phys_dim=2,
        bond_dim=3,
        open_legs_per_node=1,
        seed=0,
    )
    dense, info = at.contract_exact(tn, return_info=True)
    assert dense.shape == tn.output_shape
    assert info["method"] == "exact"
    assert info["num_blocks"] == 1
    assert info["block_spec"] == {}
