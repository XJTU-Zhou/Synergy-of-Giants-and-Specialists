# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import yaml
import os
import argparse
import logging
from tqdm import tqdm
import numpy as np

from src.knowledge_module.dataset import create_dataloaders
from src.knowledge_module.model import TransEx
from src.knowledge_module.evaluate_link_prediction import evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_epoch(model, dataloader, optimizer, device, config, is_training=True):
    """
    Runs a single epoch of training or evaluation.

    Args:
        model (nn.Module): The TransEx model.
        dataloader (DataLoader): The data loader for the current epoch.
        optimizer (Optimizer): The optimizer for training.
        device (torch.device): The device to run on.
        config (dict): The configuration dictionary.
        is_training (bool): Flag to indicate if in training or evaluation mode.
    """
    if is_training:
        model.train()
    else:
        model.eval()
    
    total_loss = 0
    total_bce_loss = 0
    total_reg_loss = 0
    
    num_samples = config['training']['negative_samples_per_positive']
    
    pbar = tqdm(dataloader, desc="Training" if is_training else "Evaluating")
    for batch in pbar:
        positive_triples = batch.to(device)
        
        # Unpack positive triples
        heads, rels, tails = positive_triples[:, 0], positive_triples[:, 1], positive_triples[:, 2]
        
        # Negative Sampling
        # Corrupt either head or tail, but not both
        corrupt_head_mask = (torch.rand(len(positive_triples)) > 0.5).to(device)
        
        random_entities = torch.randint(0, model.num_entities, (len(positive_triples),)).to(device)
        
        corrupted_heads = torch.where(corrupt_head_mask, random_entities, heads)
        corrupted_tails = torch.where(~corrupt_head_mask, random_entities, tails)
        
        negative_triples_h = torch.stack([corrupted_heads, rels, tails], dim=1)
        negative_triples_t = torch.stack([heads, rels, corrupted_tails], dim=1)
        
        # Combine positive and negative samples for scoring
        all_heads = torch.cat([heads, corrupted_heads, heads])
        all_rels = torch.cat([rels, rels, rels])
        all_tails = torch.cat([tails, tails, corrupted_tails])
        
        labels = torch.cat([
            torch.ones(len(positive_triples)),
            torch.zeros(len(negative_triples_h)),
            torch.zeros(len(negative_triples_t))
        ]).to(device)
        
        if is_training:
            optimizer.zero_grad()

        scores = model.get_scores(all_heads, all_rels, all_tails)
        
        bce_loss = model.compute_loss(scores, labels)
        reg_loss = model.compute_regularization_loss(config['training']['lambda_reg'])
        loss = bce_loss + reg_loss
        
        if is_training:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_bce_loss += bce_loss.item()
        total_reg_loss += reg_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'bce': f'{bce_loss.item():.4f}',
            'reg': f'{reg_loss.item():.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    logging.info(f"{'Epoch' if is_training else 'Validation'} Average Loss: {avg_loss:.4f}")
    return avg_loss

def train(config):
    """
    Main training loop for the TransEx model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Data Loading ---
    data_dir = config['data']['kg_embedding_dataset_path']
    batch_size = config['training']['batch_size']
    train_loader, valid_loader, test_loader, entity2id, relation2id = create_dataloaders(data_dir, batch_size)
    
    num_entities = len(entity2id)
    num_relations = len(relation2id)

    # --- Model Initialization ---
    model_params = config['model']
    model = TransEx(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=model_params['embedding_dim'],
        nhead=model_params['nhead'],
        num_encoder_layers=model_params['num_encoder_layers'],
        dim_feedforward=model_params['dim_feedforward'],
        dropout=model_params['dropout']
    ).to(device)
    
    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # --- Training Loop ---
    best_mrr = 0.0
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        logging.info(f"--- Epoch {epoch}/{config['training']['epochs']} ---")
        run_epoch(model, train_loader, optimizer, device, config, is_training=True)
        
        # --- Validation ---
        if epoch % config['training']['validation_freq'] == 0:
            logging.info("Running validation for link prediction...")
            with torch.no_grad():
                results = evaluate(model, valid_loader.dataset.triple_ids.to(device), entity2id, device)
                mrr = results['mrr']
                logging.info(f"Validation MRR: {mrr:.4f}, Hits@1: {results['hits@1']:.4f}, Hits@10: {results['hits@10']:.4f}")
                
                if mrr > best_mrr:
                    best_mrr = mrr
                    logging.info(f"New best MRR found! Saving model to {output_dir}")
                    torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                    # Save embeddings and mappings for downstream tasks
                    torch.save(model.entity_embeddings.weight.data, os.path.join(output_dir, 'entity_embeddings.pth'))
                    torch.save(model.relation_embeddings.weight.data, os.path.join(output_dir, 'relation_embeddings.pth'))
                    torch.save(entity2id, os.path.join(output_dir, 'entity2id.pth'))
                    torch.save(relation2id, os.path.join(output_dir, 'relation2id.pth'))

    logging.info("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the TransEx Knowledge Graph Embedding model.")
    parser.add_argument('--config', type=str, default='configs/stage2_transex_config.yaml',
                        help='Path to the configuration YAML file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)