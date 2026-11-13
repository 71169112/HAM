"""HAM: gene-disease prediction with learnable hyperedge scalars and two-view fusion."""
from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn

from graph_data import GENE


@dataclass(frozen=True)
class PairPrediction:
    logits: torch.Tensor
    gene_vectors: torch.Tensor
    disease_vectors: torch.Tensor


class TypeProjection(nn.Module):
    """Project entities of the same type together instead of calling a linear layer for each entity."""

    def __init__(self, categories, width, projected_width):
        super().__init__()
        self.transforms = nn.ModuleDict({
            str(category): nn.Linear(width, projected_width, bias=False)
            for category in categories
        })
        self.projected_width = projected_width

    def forward(self, vectors, categories):
        projected = vectors.new_empty((len(vectors), self.projected_width))
        for name, transform in self.transforms.items():
            selected = (categories == int(name)).nonzero(as_tuple=True)[0]
            if selected.numel():
                projected.index_copy_(0, selected, transform(vectors.index_select(0, selected)))
        return projected


class MembershipAttention(nn.Module):
    """Read information from context entities through permitted membership relations."""

    def __init__(self, receiver_categories, sender_categories, width, head_width, heads):
        super().__init__()
        self.heads = heads
        self.head_width = head_width
        projected_width = heads * head_width
        self.receiver_projection = TypeProjection(receiver_categories, width, projected_width)
        self.sender_key = TypeProjection(sender_categories, width, projected_width)
        self.sender_value = TypeProjection(sender_categories, width, projected_width)
        self.receiver_norm = nn.LayerNorm(width)
        self.sender_norm = nn.LayerNorm(width)
        self.attention_dropout = nn.Dropout(0.1)
        self.recombine = nn.Linear(width + projected_width, width)

    def _split_heads(self, vectors):
        return vectors.reshape(len(vectors), self.heads, self.head_width).transpose(0, 1)

    def forward(self, receivers, senders, allowed, receiver_categories, sender_categories):
        normalized_receivers = self.receiver_norm(receivers)
        normalized_senders = self.sender_norm(senders)
        queries = self._split_heads(self.receiver_projection(normalized_receivers, receiver_categories))
        keys = self._split_heads(self.sender_key(normalized_senders, sender_categories))
        values = self._split_heads(self.sender_value(normalized_senders, sender_categories))
        affinities = torch.matmul(queries, keys.transpose(-1, -2)) / sqrt(self.head_width)
        affinities = affinities.masked_fill(~allowed.unsqueeze(0), -torch.inf)
        proportions = self.attention_dropout(affinities.softmax(dim=-1))
        message = torch.matmul(proportions, values).transpose(0, 1).reshape(len(receivers), -1)
        return self.recombine(torch.cat((normalized_receivers, nn.functional.silu(message)), dim=-1))


class HypergraphEncoder(nn.Module):
    """Update hyperedges first, then update nodes using the refreshed hyperedges."""

    def __init__(self, structure, width, head_width, heads):
        super().__init__()
        self.register_buffer('vertex_ids', structure.vertex_ids, persistent=False)
        self.register_buffer('group_ids', structure.group_ids, persistent=False)
        self.register_buffer('membership', structure.membership, persistent=False)
        self.register_buffer('vertex_categories', structure.vertex_categories, persistent=False)
        self.register_buffer('group_categories', structure.group_categories, persistent=False)
        self.group_importance = nn.Parameter(torch.empty(len(structure.group_ids), 1))
        nn.init.normal_(self.group_importance, mean=1.0, std=0.01)
        self.node_to_edge = MembershipAttention((0, 2, 3, 4), (0, 1), width, head_width, heads)
        self.edge_to_node = MembershipAttention((0, 1), (0, 2, 3, 4), width, head_width, heads)

    def forward(self, entity_vectors):
        vertices = entity_vectors.index_select(0, self.vertex_ids)
        groups = entity_vectors.index_select(0, self.group_ids) * self.group_importance
        refreshed_groups = self.node_to_edge(groups, vertices, self.membership.T,
                                             self.group_categories, self.vertex_categories)
        refreshed_vertices = self.edge_to_node(vertices, refreshed_groups, self.membership,
                                               self.vertex_categories, self.group_categories)
        return refreshed_groups, refreshed_vertices


class GeneViewFusion(nn.Module):
    """Align the node and gene-hyperedge views to the full gene set before fusion."""

    def __init__(self, catalog, structure, width):
        super().__init__()
        self.gene_count = len(catalog.gene_ids)
        self.register_buffer('node_positions', torch.searchsorted(catalog.gene_ids, structure.vertex_ids), persistent=False)
        self.register_buffer('gene_group_rows', (structure.group_categories == GENE).nonzero(as_tuple=True)[0], persistent=False)
        gene_group_ids = structure.group_ids.index_select(0, self.gene_group_rows)
        self.register_buffer('group_positions', torch.searchsorted(catalog.gene_ids, gene_group_ids), persistent=False)
        self.view_scorer = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.view_scorer, mean=0.0, std=0.01)

    def forward(self, vertex_vectors, group_vectors):
        node_view = vertex_vectors.new_zeros((self.gene_count, vertex_vectors.shape[1]))
        node_view = node_view.index_copy(0, self.node_positions, vertex_vectors)
        edge_view = torch.zeros_like(node_view).index_copy(
            0, self.group_positions, group_vectors.index_select(0, self.gene_group_rows))
        pooled_views = torch.stack((node_view.mean(dim=0), edge_view.mean(dim=0)), dim=1)
        contributions = (self.view_scorer @ pooled_views).softmax(dim=0)
        return contributions[0] * node_view + contributions[1] * edge_view


class HAM(nn.Module):
    """Combine a shared entity table, sequential hypergraph encoding, and three-class gene-disease prediction."""

    def __init__(self, data, vector_size=32, attention_size=64, heads=1, classifier_dropout=0.3):
        super().__init__()
        self.entity_vectors = nn.Embedding(len(data.catalog.categories), vector_size)
        self.register_buffer('entity_ids', torch.arange(len(data.catalog.categories)), persistent=False)
        self.register_buffer('gene_ids', data.catalog.gene_ids, persistent=False)
        self.gene_encoder = HypergraphEncoder(data.gene_graph, vector_size, attention_size, heads)
        self.gene_fusion = GeneViewFusion(data.catalog, data.gene_graph, vector_size)
        self.disease_encoder = HypergraphEncoder(data.disease_graph, vector_size, attention_size, heads)
        self.classifier = nn.Sequential(
            nn.Linear(2 * vector_size, 8 * vector_size),
            nn.BatchNorm1d(8 * vector_size), nn.GELU(), nn.Dropout(classifier_dropout),
            nn.Linear(8 * vector_size, 4 * vector_size),
            nn.BatchNorm1d(4 * vector_size), nn.GELU(), nn.Dropout(classifier_dropout / 2),
            nn.Linear(4 * vector_size, 3),
        )

    def encode_entities(self):
        initial = self.entity_vectors(self.entity_ids)
        gene_groups, gene_nodes = self.gene_encoder(initial)
        merged_genes = self.gene_fusion(gene_nodes, gene_groups) + initial.index_select(0, self.gene_ids)
        # Create a new tensor on each write to preserve gradients across encoders.
        updated = initial.index_copy(0, self.gene_encoder.group_ids, gene_groups)
        updated = updated.index_copy(0, self.gene_ids, merged_genes)
        disease_groups, disease_nodes = self.disease_encoder(updated)
        residual_diseases = disease_nodes + updated.index_select(0, self.disease_encoder.vertex_ids)
        updated = updated.index_copy(0, self.disease_encoder.group_ids, disease_groups)
        return updated.index_copy(0, self.disease_encoder.vertex_ids, residual_diseases)

    def forward(self, gene_ids, disease_ids):
        vectors = self.encode_entities()
        selected_genes = vectors.index_select(0, gene_ids)
        selected_diseases = vectors.index_select(0, disease_ids)
        prediction = self.classifier(torch.cat((selected_genes, selected_diseases), dim=-1))
        return PairPrediction(prediction, selected_genes, selected_diseases)
