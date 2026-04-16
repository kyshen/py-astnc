import astnc as at


def test_cache_reuse_accumulates_hits():
    tn = at.grid2d(
        rows=3,
        cols=3,
        phys_dim=3,
        bond_dim=4,
        open_legs_per_node=1,
        seed=0,
    )
    cache = at.create_cache()
    _, first_info = at.materialize(
        tn,
        method="astnc",
        workpoint="l2",
        block_labels=2,
        chunk_size=1,
        cache=cache,
        return_info=True,
    )
    _, second_info = at.materialize(
        tn,
        method="astnc",
        workpoint="l2",
        block_labels=2,
        chunk_size=1,
        cache=cache,
        return_info=True,
    )
    assert first_info["meta"]["cache_requests"] > 0
    assert second_info["meta"]["cache_hits"] >= first_info["meta"]["cache_hits"]
    assert second_info["meta"]["cache_hit_rate"] > 0.0
