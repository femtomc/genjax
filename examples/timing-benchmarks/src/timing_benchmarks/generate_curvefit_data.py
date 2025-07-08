#!/usr/bin/env python
"""Generate polynomial regression data and save to disk."""

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.generation import generate_polynomial_data

def main():
    # Create output directory
    output_dir = Path("data/curvefit")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    print("Generating polynomial regression data...")
    data = generate_polynomial_data(n_points=50, seed=42)
    
    # Convert JAX arrays to numpy for saving
    data_dict = {
        'xs': np.array(data.xs),
        'ys': np.array(data.ys),
        'true_a': data.true_a,
        'true_b': data.true_b,
        'true_c': data.true_c,
        'noise_std': data.noise_std,
        'n_points': data.n_points
    }
    
    # Save data
    output_file = output_dir / "polynomial_data.npz"
    np.savez(output_file, **data_dict)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()