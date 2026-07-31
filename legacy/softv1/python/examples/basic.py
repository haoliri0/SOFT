from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import soft

summary = soft.sample_counts_file(
    "../magic_state_cultivation/circuits/circuit_d3_p0.001.stim",
    shots_n=100000,
    entries_m=32,
    seed=42,
    observable=0,
    cuda=True,
)

print(summary)
