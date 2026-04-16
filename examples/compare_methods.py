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

exact = at.materialize(tn, method="exact", block_labels=2, chunk_size=1)
fixed = at.materialize(tn, method="fixed_rank", target_rank=4, block_labels=2, chunk_size=1)
astnc = at.materialize(tn, method="astnc", workpoint="l2", block_labels=2, chunk_size=1)


def rel_error(ref, pred):
    return np.linalg.norm(ref - pred) / (np.linalg.norm(ref) + 1e-12)


print("fixed-rank rel error:", rel_error(exact, fixed))
print("astnc rel error:", rel_error(exact, astnc))

