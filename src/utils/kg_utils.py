# -*- coding: utf-8 -*-

"""
src.utils.kg_utils - Knowledge Graph Utility Functions.

This module provides helper functions for loading, parsing, and preparing
knowledge graph data for the embedding and reasoning modules.
"""

import os
from typing import List, Tuple, Dict, Set

def load_kg_triples(file_path: str) -> List[Tuple[str, str, str]]:
    """
    Loads knowledge graph triples from a text file.

    Each line in the file is expected to be in the format "head\trelation\ttail".

    Args:
        file_path (str): The path to the KG data file (e.g., train.txt).

    Returns:
        List[Tuple[str, str, str]]: A list of triples, where each triple
        is a tuple of (head, relation, tail).

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        ValueError: If a line in the file does not contain exactly three elements.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Knowledge graph file not found at: {file_path}")

    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                triples.append(tuple(parts))
            else:
                print(f"Warning: Skipping malformed line in {file_path}: {line.strip()}")
    
    print(f"Loaded {len(triples)} triples from {file_path}")
    return triples


def build_entity_relation_maps(
    triples: List[Tuple[str, str, str]]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Builds dictionaries mapping entities and relations to unique integer IDs.

    This is a standard preprocessing step for training KG embedding models.

    Args:
        triples (List[Tuple[str, str, str]]): A list of KG triples.

    Returns:
        Tuple[Dict[str, int], Dict[str, int]]: A tuple containing two dictionaries:
            - entity_to_id: Maps entity names (str) to integer IDs.
            - relation_to_id: Maps relation names (str) to integer IDs.
    """
    entities: Set[str] = set()
    relations: Set[str] = set()

    for head, relation, tail in triples:
        entities.add(head)
        entities.add(tail)
        relations.add(relation)
        
    # Sort to ensure deterministic mapping
    sorted_entities = sorted(list(entities))
    sorted_relations = sorted(list(relations))

    entity_to_id = {entity: i for i, entity in enumerate(sorted_entities)}
    relation_to_id = {relation: i for i, relation in enumerate(sorted_relations)}
    
    print(f"Built maps for {len(entity_to_id)} unique entities and {len(relation_to_id)} unique relations.")
    
    return entity_to_id, relation_to_id