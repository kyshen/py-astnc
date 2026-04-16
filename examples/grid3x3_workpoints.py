import numpy as np

import astnc as at

tn = at.grid2d(
    rows=3,
    cols=3,
    phys_dim=3,
    bond_dim=4,
    open_legs_per_node=1,
    seed=0,
)

reference = at.materialize(tn, method="exact", block_labels=2, chunk_size=1)
for name in at.available_workpoints():
    dense, info = at.materialize(
        tn,
        method="astnc",
        workpoint=name,
        block_labels=2,
        chunk_size=1,
        return_info=True,
    )
    rel = np.linalg.norm(reference - dense) / (np.linalg.norm(reference) + 1e-12)
    print(name, "rel_error=", rel, "mean_rank=", info["meta"].get("mean_rank"))

