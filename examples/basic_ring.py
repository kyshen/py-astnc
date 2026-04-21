import astnc as at

tn = at.ring(
    num_nodes=6,
    phys_dim=3,
    bond_dim=4,
    open_legs_per_node=1,
    seed=0,
)

dense, info = at.contract_astnc(
    tn,
    workpoint="l2",
    block_spec={0: 1, 1: 1},
    return_info=True,
)

print("shape:", dense.shape)
print("blocks:", info["num_blocks"])
print("mean rank:", info["meta"].get("mean_rank"))
