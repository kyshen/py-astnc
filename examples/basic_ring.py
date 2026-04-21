from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
    tol=1e-3,
    block_spec={0: 1, 1: 1},
    return_info=True,
)

print("shape:", dense.shape)
print("blocks:", info["num_blocks"])
print("mean internal rank:", info["meta"].get("mean_internal_rank"))
