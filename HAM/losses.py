"""Combine three-class supervision with gene-disease matching constraints."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class HybridObjective(nn.Module):
    def __init__(self, class_labels, temperature=0.15, association_weight=0.2,
                 therapeutic_multiplier=1.5):
        super().__init__()
        frequencies = np.bincount(np.asarray(class_labels, dtype=np.int64), minlength=3)
        if np.any(frequencies == 0):
            raise ValueError('The dataset must include all three association classes.')
        reweighting = np.sqrt(frequencies.sum() / (3 * frequencies))
        self.register_buffer('classification_weights', torch.tensor(reweighting, dtype=torch.float32))
        self.register_buffer('association_multipliers', torch.tensor([1.0, 1.0, therapeutic_multiplier]))
        self.temperature = temperature
        self.association_weight = association_weight

    def classification(self, prediction, categories):
        return F.cross_entropy(prediction, categories, weight=self.classification_weights)

    def association(self, gene_vectors, disease_vectors, categories):
        similarity = F.cosine_similarity(gene_vectors, disease_vectors, dim=-1) / self.temperature
        matched = similarity.sigmoid()
        # Encourage low similarity for NA and high similarity for the other classes; average within each class.
        penalties = torch.where(categories == 0, -(1 - matched + 1e-8).log(), -(matched + 1e-8).log())
        averages = torch.stack([
            (penalties * (categories == category)).sum() / ((categories == category).sum() + 1e-8)
            for category in range(3)
        ])
        return (averages * self.association_multipliers).sum()

    def forward(self, prediction, categories):
        supervised = self.classification(prediction.logits, categories)
        matching = self.association(prediction.gene_vectors, prediction.disease_vectors, categories)
        return supervised + self.association_weight * matching
