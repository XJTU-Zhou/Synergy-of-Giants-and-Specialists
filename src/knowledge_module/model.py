# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class TransEx(nn.Module):
    """
    Implementation of the TransEx model for Knowledge Graph embeddings.

    TransEx combines a Transformer encoder with complex-valued embeddings to model
    the intricate, contextual, and often asymmetric dependencies found in
    manufacturing process KGs.

    The architecture follows the paper's description, particularly the scoring
    function detailed in Equation 7.
    """
    def __init__(self, num_entities, num_relations, embedding_dim,
                 nhead=2, num_encoder_layers=4, dim_feedforward=512, dropout=0.15):
        """
        Args:
            num_entities (int): Total number of unique entities in the KG.
            num_relations (int): Total number of unique relations in the KG.
            embedding_dim (int): The dimension for EACH of the real and imaginary parts.
                                 Total embedding size will be 2 * embedding_dim.
            nhead (int): Number of attention heads in the Transformer.
            num_encoder_layers (int): Number of layers in the Transformer encoder.
            dim_feedforward (int): Dimension of the feedforward network in the Transformer.
            dropout (float): Dropout rate.
        """
        super(TransEx, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.total_embedding_dim = 2 * embedding_dim

        # Embeddings for entities and relations
        # Stored as [real_part, imaginary_part] concatenated
        self.entity_embeddings = nn.Embedding(num_entities, self.total_embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, self.total_embedding_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.total_embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important for batch processing
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, heads, rels):
        """
        Processes head and relation embeddings through the Transformer to get contextual outputs.

        Args:
            heads (torch.LongTensor): A batch of head entity indices. Shape: (batch_size,)
            rels (torch.LongTensor): A batch of relation indices. Shape: (batch_size,)

        Returns:
            tuple: A tuple of four tensors (Out_A, Out_B, Out_C, Out_D) as per Equation 7.
                   Each tensor has shape (batch_size, embedding_dim).
        """
        head_embeds = self.entity_embeddings(heads)
        rel_embeds = self.relation_embeddings(rels)

        # Create a sequence for the Transformer: (batch_size, seq_len, dim)
        # Sequence consists of [head_embedding, relation_embedding]
        seq_input = torch.stack([head_embeds, rel_embeds], dim=1)
        
        # Transformer processing
        transformer_output = self.transformer_encoder(seq_input) # Shape: (batch_size, 2, total_dim)
        
        # Apply dropout
        transformer_output = self.dropout(transformer_output)

        # Extract contextualized head and relation embeddings
        context_head_embed = transformer_output[:, 0, :] # Shape: (batch_size, total_dim)
        context_rel_embed = transformer_output[:, 1, :]  # Shape: (batch_size, total_dim)
        
        # Split the outputs into four components as described in the paper around Eq. 7
        # Out_A, Out_B from head; Out_D, Out_C from relation
        out_a, out_b = torch.chunk(context_head_embed, 2, dim=1)
        out_d, out_c = torch.chunk(context_rel_embed, 2, dim=1)
        
        return out_a, out_b, out_c, out_d

    def score(self, transformer_outputs, tails):
        """
        Calculates the score of triples based on the Transformer outputs and tail embeddings.
        This function implements the scoring mechanism from Equation 7.

        Args:
            transformer_outputs (tuple): The (Out_A, Out_B, Out_C, Out_D) tuple from forward().
            tails (torch.LongTensor): A batch of tail entity indices. Shape: (batch_size,)

        Returns:
            torch.Tensor: The final scores for each triple. Shape: (batch_size,)
        """
        out_a, out_b, out_c, out_d = transformer_outputs
        
        tail_embeds = self.entity_embeddings(tails)
        tail_re, tail_im = torch.chunk(tail_embeds, 2, dim=1)

        # Equation 7: TransEx(h,r,t) = <Out(A), Re(et)> + <Out(B), Im(et)>
        #                             + <Out(C), Im(et)> - <Out(D), Re(et)>
        # Note: The paper has a slight ambiguity in the equation (+/-). We follow the text.
        # Let's re-verify from paper image: it is -(Out(D), Re(e_t)). And +(Out(C), Im(e_t)).
        
        score_part1 = torch.sum(out_a * tail_re, dim=1)
        score_part2 = torch.sum(out_b * tail_im, dim=1)
        score_part3 = torch.sum(out_c * tail_im, dim=1)
        score_part4 = torch.sum(out_d * tail_re, dim=1)
        
        # Final score calculation based on the paper's formula
        # It seems the plus/minus signs might be mixed up in the paper's text.
        # A standard ComplEx-style interaction would be: Re*Re + Re*Im - Im*Re + Im*Im
        # However, to be faithful to the paper, we implement Equation 7 as written:
        scores = score_part1 + score_part2 + score_part3 - score_part4
        
        return scores

    def get_scores(self, heads, rels, tails):
        """A convenience method to get scores for a given set of triples."""
        transformer_outputs = self.forward(heads, rels)
        return self.score(transformer_outputs, tails)

    def compute_loss(self, scores, labels):
        """Computes the Binary Cross-Entropy loss."""
        return self.bce_loss(scores, labels.float())

    def compute_regularization_loss(self, lambda_reg=0.01):
        """
        Computes the L2 regularization loss on embeddings as per Equation 10.
        
        Args:
            lambda_reg (float): The regularization coefficient.
        
        Returns:
            torch.Tensor: The regularization loss.
        """
        l2_reg = (torch.mean(self.entity_embeddings.weight.pow(2)) +
                  torch.mean(self.relation_embeddings.weight.pow(2)))
        
        return lambda_reg * l2_reg