# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_triples(file_path):
    """Reads triples from a file, expecting (head, relation, tail) per line."""
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            triples.append((h, r, t))
    return triples

class KGDataset(Dataset):
    """
    A PyTorch Dataset for Knowledge Graph triples.

    This class handles the loading of triples, mapping entities and relations to
    integer IDs, and providing indexed access to the data.
    """
    def __init__(self, triples, entity2id, relation2id):
        self.triples = triples
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.num_entities = len(entity2id)
        self.num_relations = len(relation2id)
        
        self.triple_ids = torch.LongTensor([
            (self.entity2id[h], self.relation2id[r], self.entity2id[t])
            for h, r, t in self.triples
        ])

    def __len__(self):
        return len(self.triple_ids)

    def __getitem__(self, idx):
        return self.triple_ids[idx]

    @staticmethod
    def create_mappings(triples):
        """Creates entity-to-ID and relation-to-ID mappings from a list of triples."""
        entity2id = {}
        relation2id = {}

        def get_id(mapping, item):
            if item not in mapping:
                mapping[item] = len(mapping)
            return mapping[item]

        for h, r, t in triples:
            get_id(entity2id, h)
            get_id(relation2id, r)
            get_id(entity2id, t)
            
        return entity2id, relation2id

def create_dataloaders(data_dir, batch_size):
    """
    Creates train, validation, and test dataloaders for the KG.

    It ensures that validation and test sets use the same entity/relation mappings
    derived from the training set.

    Args:
        data_dir (str): Path to the directory containing train.txt, valid.txt, test.txt.
        batch_size (int): The batch size for the dataloaders.

    Returns:
        tuple: (train_loader, valid_loader, test_loader, entity2id, relation2id)
    """
    logging.info(f"Loading KG data from {data_dir}...")
    
    train_path = os.path.join(data_dir, 'train.txt')
    valid_path = os.path.join(data_dir, 'valid.txt')
    test_path = os.path.join(data_dir, 'test.txt')

    train_triples = read_triples(train_path)
    valid_triples = read_triples(valid_path)
    test_triples = read_triples(test_path)
    
    logging.info(f"Found {len(train_triples)} training, {len(valid_triples)} validation, and {len(test_triples)} test triples.")

    # Create mappings based ONLY on the training set to avoid data leakage
    entity2id, relation2id = KGDataset.create_mappings(train_triples)
    logging.info(f"Number of unique entities: {len(entity2id)}")
    logging.info(f"Number of unique relations: {len(relation2id)}")
    
    # Create datasets
    train_dataset = KGDataset(train_triples, entity2id, relation2id)
    valid_dataset = KGDataset(valid_triples, entity2id, relation2id)
    test_dataset = KGDataset(test_triples, entity2id, relation2id)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, valid_loader, test_loader, entity2id, relation2id