"""Utilities for generating tiny synthetic arrays for smoke tests.

The research data are not included in this public repository. This module
creates shape-compatible dummy data so readers can inspect the preprocessing
and training scripts without access to the original data.
"""

from pathlib import Path

import numpy as np


def generate_dummy_dataset(output_dir="data/sample", n_samples=512, seed=42):
    """Create small arrays compatible with the training scripts.

    The values are synthetic and should not be used as experimental evidence.
    """
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_cp_now = rng.uniform(-1.2, 1.2, n_samples)
    y_cp_now = rng.uniform(0.5, 6.5, n_samples)
    target_length = rng.uniform(50.0, 250.0, n_samples)
    own_length = rng.uniform(50.0, 250.0, n_samples)
    own_speed = rng.uniform(8.0, 18.0, n_samples)
    target_cog = rng.uniform(150.0, 210.0, n_samples)

    input_data = np.stack(
        [x_cp_now, y_cp_now, target_length, own_length, own_speed, target_cog],
        axis=1,
    )

    side = rng.choice([-1.0, 1.0], size=n_samples)
    future_x_cp = side * rng.lognormal(mean=-0.8, sigma=0.45, size=n_samples)
    passing_y = np.maximum(0.3, y_cp_now - rng.lognormal(mean=-0.3, sigma=0.35, size=n_samples))
    output_data = np.stack([future_x_cp, passing_y], axis=1)

    np.save(output_dir / "input_data.npy", input_data)
    np.save(output_dir / "output_data.npy", output_data)
    return input_data, output_data


if __name__ == "__main__":
    generate_dummy_dataset()
