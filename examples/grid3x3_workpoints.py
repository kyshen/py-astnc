from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import astnc as at

tn = at.grid2d(
    rows=3,
    cols=3,
    phys_dim=3,
    bond_dim=4,
    open_legs_per_node=1,
    seed=0,
)

reference = at.contract_exact(tn)
for tol in (5e-4, 3e-3, 4e-3):
    dense, info = at.contract_astnc(
        tn,
        tol=tol,
        block_spec={},
        return_info=True,
    )
    rel = np.linalg.norm(reference - dense) / (np.linalg.norm(reference) + 1e-12)
    print("tol=", tol, "rel_error=", rel)
