"""Data acquisition and synthetic dataset generator.

Generates a multimodal synthetic dataset containing CSV, JSON and image files
so you can run experiments without large external downloads.
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image


def generate_synthetic_dataset(out_dir, n_samples=10000, img_size=(64, 64), seed=42):
    """Generate synthetic dataset with numeric, categorical, SMILES-like strings, JSON metadata and images.

    Args:
        out_dir (str): output directory (will contain `raw/` with csv/json/images)
        n_samples (int): number of samples to generate
    """
    rng = np.random.RandomState(seed)
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw"
    img_dir = raw_dir / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n_samples):
        mol_len = rng.randint(10, 60)
        # make a fake SMILES-like string (not chemically valid necessarily)
        smiles = ''.join(rng.choice(list('CNOPFClBrI=123456789()[]#'), size=mol_len))
        num_feat = rng.normal(loc=0.0, scale=1.0, size=5)
        cat_feat = rng.choice(['A', 'B', 'C'])
        target = rng.binomial(1, p=0.1 if i % 7 == 0 else 0.5)

        rows.append({
            'id': f'sample_{i}',
            'smiles': smiles,
            'feat1': float(num_feat[0]),
            'feat2': float(num_feat[1]),
            'feat3': float(num_feat[2]),
            'feat4': float(num_feat[3]),
            'feat5': float(num_feat[4]),
            'cat': cat_feat,
            'target': int(target),
            'image_path': str(img_dir / f'image_{i}.png')
        })

        # generate a simple random image
        arr = (rng.rand(*img_size, 3) * 255).astype('uint8')
        img = Image.fromarray(arr)
        img.save(img_dir / f'image_{i}.png')

    df = pd.DataFrame(rows)
    csv_path = raw_dir / 'dataset.csv'
    json_path = raw_dir / 'dataset.json'
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient='records', lines=False)

    return {'csv': str(csv_path), 'json': str(json_path), 'images': str(img_dir)}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data', help='output folder')
    p.add_argument('--n', type=int, default=10000)
    args = p.parse_args()
    res = generate_synthetic_dataset(args.out, n_samples=args.n)
    print('Generated:', res)
