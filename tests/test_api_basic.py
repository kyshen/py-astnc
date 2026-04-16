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
    dense, info = at.materialize(
        tn,
        method="astnc",
        workpoint="l2",
        block_labels=2,
        chunk_size=1,
        return_info=True,
    )
    assert dense.shape == tn.output_shape
    assert info["method"] == "astnc"
    assert info["workpoint"] == "l2"
    assert info["num_blocks"] > 1
    assert "mean_rank" in info["meta"]

