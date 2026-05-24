"""
PyTorch datasets for optimal stopping and ski rental.

Each item is one instance with exact DP labels.
Supports variable-length horizons: each instance samples n uniformly from
[n_min, n_max], sequences are padded to max_n, and n is stored per item
so the model receives it as input via horizon_embed.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from core.sampling import (
    sample_stopping_batch, STOPPING_TRAIN_FAMILIES,
    sample_ski_batch, SKI_TRAIN_FAMILIES,
)
from core.dp import stopping_labels, ski_labels


class StoppingDataset(Dataset):
    """
    Optimal stopping dataset with DP labels and variable horizons.

    Each item:
        values    : (pad_n,) int64 — padded with 0 beyond n
        C         : (pad_n,) float32 — continuation values, 0-padded
        a         : (pad_n,) float32 — optimal stop label, 0-padded
        mask      : (pad_n,) bool — True for real positions (< n)
        n_horizon : int — actual sequence length
    """

    def __init__(self, num_instances, n, M, dist_type=None, seed=0,
                 families=None, n_min=None, n_max=None, pad_n=None):
        """
        Args:
            n      : default horizon (used if n_min/n_max not set)
            n_min  : minimum horizon for uniform sampling (inclusive)
            n_max  : maximum horizon for uniform sampling (inclusive)
            pad_n  : pad all sequences to this length (default: n_max or n)
        """
        rng = np.random.default_rng(seed)
        if families is None:
            families = STOPPING_TRAIN_FAMILIES

        # Determine horizon range
        if n_min is not None and n_max is not None:
            self.n_min, self.n_max = n_min, n_max
        else:
            self.n_min, self.n_max = n, n

        self.pad_n = pad_n if pad_n is not None else self.n_max
        self.data = []

        for _ in range(num_instances):
            # Sample horizon uniformly at random
            h = int(rng.integers(self.n_min, self.n_max + 1))
            inst = sample_stopping_batch(1, h, M, dist_type=dist_type,
                                         rng=rng, families=families)[0]
            lbl = stopping_labels(inst.pmf, inst.values)

            # Pad to pad_n
            p = self.pad_n - h
            values = np.pad(inst.values, (0, p), constant_values=0)
            C = np.pad(lbl["C"], (0, p), constant_values=0.0)
            a = np.pad(lbl["a"], (0, p), constant_values=0.0)
            mask = np.zeros(self.pad_n, dtype=bool)
            mask[:h] = True

            self.data.append({
                "values": torch.tensor(values, dtype=torch.long),
                "C": torch.tensor(C, dtype=torch.float32),
                "a": torch.tensor(a, dtype=torch.float32),
                "mask": torch.tensor(mask, dtype=torch.bool),
                "n_horizon": h,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SkiRentalDataset(Dataset):
    """
    Ski rental dataset with DP labels and variable horizons.

    Each item:
        input_seq : (pad_n,) int64 — 1s for real positions, 0-padded
        J         : (pad_n,) float32 — cost-to-go, 0-padded
        a         : (pad_n,) float32 — optimal buy label, 0-padded
        mask      : (pad_n,) bool — True for real positions (< n)
        n_horizon : int — actual sequence length
        B         : float — buy cost for this instance
        r         : float — rent cost
    """

    def __init__(self, num_instances, n, B, r, dist_type=None, seed=0,
                 families=None, n_min=None, n_max=None, pad_n=None,
                 B_min=None, B_max=None):
        """
        Args:
            B, r     : default cost parameters (used if B_min/B_max not set)
            B_min    : minimum buy cost for uniform sampling (inclusive)
            B_max    : maximum buy cost for uniform sampling (inclusive)
                       If set, each instance samples B ~ Uniform[B_min, B_max]
                       with r fixed. Cost ratio B/r varies per instance.
        """
        rng = np.random.default_rng(seed)
        if families is None:
            families = SKI_TRAIN_FAMILIES

        if n_min is not None and n_max is not None:
            self.n_min, self.n_max = n_min, n_max
        else:
            self.n_min, self.n_max = n, n

        self.pad_n = pad_n if pad_n is not None else self.n_max
        self.data = []

        for _ in range(num_instances):
            h = int(rng.integers(self.n_min, self.n_max + 1))

            # Sample buy cost if range provided
            if B_min is not None and B_max is not None:
                B_inst = float(rng.integers(B_min, B_max + 1))
            else:
                B_inst = B

            inst = sample_ski_batch(1, h, B_inst, r, dist_type=dist_type,
                                    rng=rng, families=families)[0]
            lbl = ski_labels(inst.pmf_T, inst.n, inst.B, inst.r)

            p = self.pad_n - h
            input_seq = np.pad(np.ones(h, dtype=int), (0, p), constant_values=0)
            J = np.pad(lbl["J"], (0, p), constant_values=0.0)
            a = np.pad(lbl["a"], (0, p), constant_values=0.0)
            mask = np.zeros(self.pad_n, dtype=bool)
            mask[:h] = True

            self.data.append({
                "input_seq": torch.tensor(input_seq, dtype=torch.long),
                "J": torch.tensor(J, dtype=torch.float32),
                "a": torch.tensor(a, dtype=torch.float32),
                "mask": torch.tensor(mask, dtype=torch.bool),
                "n_horizon": h,
                "B": B_inst,
                "r": r,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def make_dataloader(dataset, batch_size=128, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # Test with uniform random horizons
    ds = StoppingDataset(8, 20, 1000, seed=0, n_min=20, n_max=200)
    print("=== Stopping (uniform horizon [20, 200]) ===")
    for i in range(4):
        item = ds[i]
        print(f"  item {i}: n_horizon={item['n_horizon']}, "
              f"values.shape={item['values'].shape}, mask.sum={item['mask'].sum().item()}")

    sds = SkiRentalDataset(8, 20, 10, 1, seed=0, n_min=20, n_max=200)
    print("\n=== Ski rental (uniform horizon [20, 200]) ===")
    for i in range(4):
        item = sds[i]
        print(f"  item {i}: n_horizon={item['n_horizon']}, "
              f"input_seq.shape={item['input_seq'].shape}, mask.sum={item['mask'].sum().item()}")
