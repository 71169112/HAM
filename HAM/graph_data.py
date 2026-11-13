"""Compile biological association tables into two hypergraphs with explicit row and column identities."""
from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from scipy.sparse import coo_matrix
import torch


GENE, DISEASE, GO, HPO, DO = 0, 1, 2, 3, 4
RELATION_LABELS = {'NA': 0, 'biomarker': 1, 'genomic_alterations': 1, 'therapeutic': 2}


@dataclass(frozen=True)
class EntityCatalog:
    """Derive entity IDs and types from metadata instead of hardcoded index boundaries."""

    lookup: dict
    categories: torch.Tensor
    gene_ids: torch.Tensor
    disease_ids: torch.Tensor

    @classmethod
    def from_metadata(cls, metadata):
        lookup, categories = {}, []
        for label, category in (('genes', GENE), ('go', GO), ('disease', DISEASE),
                                ('hpo', HPO), ('do', DO)):
            identifiers = sorted(metadata[f'unique_{label}_set'])
            offset = len(categories)
            lookup[label] = {identifier: offset + pos for pos, identifier in enumerate(identifiers)}
            categories.extend([category] * len(identifiers))
        return cls(
            lookup=lookup,
            categories=torch.tensor(categories, dtype=torch.long),
            gene_ids=torch.tensor(list(lookup['genes'].values()), dtype=torch.long),
            disease_ids=torch.tensor(list(lookup['disease'].values()), dtype=torch.long),
        )


@dataclass(frozen=True)
class MembershipGraph:
    """Rows of membership correspond to vertex_ids; columns correspond to group_ids."""

    vertex_ids: torch.Tensor
    group_ids: torch.Tensor
    membership: torch.Tensor
    vertex_categories: torch.Tensor
    group_categories: torch.Tensor


@dataclass(frozen=True)
class BiologicalData:
    catalog: EntityCatalog
    gene_graph: MembershipGraph
    disease_graph: MembershipGraph
    labelled_pairs: np.ndarray


def _translate_column(column, encoding):
    mapped = column.map(encoding)
    if mapped.isna().any():
        raise ValueError('An association contains an entity absent from the catalog.')
    return mapped.to_numpy(dtype=np.int64)


def _group_members(anchors, members, *, exclude_anchor=False):
    grouped = {int(anchor): set() for anchor in np.unique(anchors)}
    for anchor, member in zip(anchors, members):
        if not exclude_anchor or anchor != member:
            grouped[int(anchor)].add(int(member))
    return grouped


def _compile_membership(grouped, catalog):
    group_ids = sorted(grouped)
    vertex_ids = sorted(set().union(*grouped.values()))
    if not group_ids or not vertex_ids or any(not grouped[key] for key in group_ids):
        raise ValueError('A graph must contain vertices and nonempty groups.')
    row_of = {identifier: row for row, identifier in enumerate(vertex_ids)}
    coordinates = [(row_of[member], col)
                   for col, identifier in enumerate(group_ids) for member in grouped[identifier]]
    rows, cols = np.asarray(coordinates, dtype=np.int64).T
    matrix = coo_matrix((np.ones(len(rows), dtype=np.bool_), (rows, cols)),
                        shape=(len(vertex_ids), len(group_ids))).toarray()
    vertices = torch.tensor(vertex_ids, dtype=torch.long)
    groups = torch.tensor(group_ids, dtype=torch.long)
    return MembershipGraph(vertices, groups, torch.from_numpy(matrix),
                           catalog.categories[vertices], catalog.categories[groups])


def read_biological_data(directory):
    """Read the expected data schema without modifying pandas columns in place."""
    directory = Path(directory)
    with (directory / 'datasets.pkl').open('rb') as stream:
        tables = pickle.load(stream)
    with (directory / 'unique_node_sets.pkl').open('rb') as stream:
        catalog = EntityCatalog.from_metadata(pickle.load(stream))

    interactions = tables['gene_gene']
    anchors = _translate_column(interactions.iloc[:, 0], catalog.lookup['genes'])
    members = _translate_column(interactions.iloc[:, 1], catalog.lookup['genes'])
    gene_groups = _group_members(anchors, members, exclude_anchor=True)

    annotations = tables['gene_go']
    members = _translate_column(annotations.iloc[:, 0], catalog.lookup['genes'])
    anchors = _translate_column(annotations.iloc[:, 1], catalog.lookup['go'])
    gene_groups.update(_group_members(anchors, members))

    associations = tables['tbga']
    classes = associations.iloc[:, 0].map(RELATION_LABELS)
    if classes.isna().any():
        raise ValueError('Unknown association class.')
    gene_ids = _translate_column(associations.iloc[:, 1], catalog.lookup['genes'])
    disease_ids = _translate_column(associations.iloc[:, 2], catalog.lookup['disease'])
    targets = classes.to_numpy(dtype=np.int64)
    positive = targets != 0
    disease_groups = _group_members(gene_ids[positive], disease_ids[positive])
    for table_name, category_name in (('disease_hpo', 'hpo'), ('disease_do', 'do')):
        annotations = tables[table_name]
        members = _translate_column(annotations.iloc[:, 0], catalog.lookup['disease'])
        anchors = _translate_column(annotations.iloc[:, 1], catalog.lookup[category_name])
        disease_groups.update(_group_members(anchors, members))

    return BiologicalData(catalog, _compile_membership(gene_groups, catalog),
                          _compile_membership(disease_groups, catalog),
                          np.column_stack((gene_ids, disease_ids, targets)))
