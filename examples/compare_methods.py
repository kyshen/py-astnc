import numpy as np

import astnc as at

tn = at.random_connected(
    num_nodes=8,
    phys_dim=3,
    bond_dim=4,
    open_legs_per_node=1,
    edge_prob=0.35,
    seed=0,
)

exact, exact_info = at.contract_exact(tn, return_info=True)
astnc, astnc_info = at.contract_astnc(
    tn,
    workpoint="l2",
    block_spec={0: 1, 1: 1},
    return_info=True,
)


def rel_error(ref, pred):
    return np.linalg.norm(ref - pred) / (np.linalg.norm(ref) + 1e-12)


print("astnc rel error:", rel_error(exact, astnc))
print("exact total time (s):", exact_info["total_time_sec"])
print("astnc total time (s):", astnc_info["total_time_sec"])
