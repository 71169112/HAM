"""Provide device selection, random seeding, and sample batching for training."""
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def initialize_random_state(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_accelerator(requested):
    if requested < 0 or not torch.cuda.is_available():
        return torch.device('cpu')
    if requested >= torch.cuda.device_count():
        raise ValueError(f'CUDA device {requested} is not available.')
    return torch.device(f'cuda:{requested}')


def partition_pairs(pairs, seed):
    training, reserved = train_test_split(
        np.arange(len(pairs)), test_size=0.2, random_state=seed, stratify=pairs[:, 2])
    validation, testing = train_test_split(
        reserved, test_size=0.5, random_state=seed, stratify=pairs[reserved, 2])
    return {'train': training, 'valid': validation, 'test': testing}


def pair_batches(pairs, batch_size, *, training):
    numeric = torch.as_tensor(pairs, dtype=torch.long)
    samples = TensorDataset(*numeric.unbind(dim=1))
    return DataLoader(samples, batch_size=batch_size, shuffle=training)
