# -*- coding: utf-8 -*-
"""
Backbones of Pre-training Models (from input to last hidden-layer output)
"""

import torch
import torch.nn as nn
import model.embeddings as emb
import model.encoders as enc
from icecream import ic



class Backbone(nn.Module):
    def __init__(self, config):
        super(Backbone, self).__init__()
        self.total_node = sum(config.node_degree)
        self.attn_method = config.attn_method
        self.attn_methods = {"max": self.pos2attn_max, 
                             "add": self.pos2attn_add}

    def unzip_tree_position(self, zipped_position):
        """
        args: zipped_position: [batch_size, seq_len, tree_depth], range: [0, total_node]
        rets: entire_position: [batch_size, seq_len, total_node]
        lower_bound = 0, upper_bound = (total_node-1)
        use one excessive bit to temporarily represent not-applicable nodes
        """
        batch_size, seq_len, _ = zipped_position.size()
        entire_position = torch.zeros(batch_size, seq_len, self.total_node + 1).to(zipped_position.device)
        entire_position = entire_position.scatter_(-1, zipped_position, 1.0).long()
        entire_position = entire_position[:, :, : self.total_node]  # remove last column
        return entire_position

    def get_attention_mask(self, entire_top, entire_left, indicator):
        attention_mask = self.attn_methods[self.attn_method](entire_top, entire_left)
        attention_mask = self.create_post_mask(attention_mask, indicator)
        return attention_mask
    
    def pos2attn_max(self, pos_top, pos_left):   # entire position
        top_attn_mask = self.pos2attn(pos_top)
        left_attn_mask = self.pos2attn(pos_left)
        attn_mask = torch.max(top_attn_mask, left_attn_mask)
        # attn_mask = top_attn_mask + left_attn_mask
        return attn_mask

    def pos2attn_add(self, pos_top, pos_left):   # entire position
        top_attn_mask = self.pos2attn(pos_top)
        left_attn_mask = self.pos2attn(pos_left)
        attn_mask = top_attn_mask + left_attn_mask
        return attn_mask

    def pos2attn(self, position):   # entire position
        """Compute a one-dimension attention distance matrix from a entire-mode tree position. """
        vector_matrix = position.unsqueeze(2).repeat(1, 1, position.size()[1], 1)   # [batch, seq_len, seq_len, total_node]
        attention_mask = torch.abs(vector_matrix - vector_matrix.transpose(1, 2))
        attention_mask = torch.sum(attention_mask, dim=-1)
        return attention_mask

    def create_post_mask(self, attn_dist, indicator, padding_dist=100):
        # slightly different from tuta, removing sep and cls matrix
        pad_matrix = (indicator == 0).long().unsqueeze(-1).repeat(1, 1, attn_dist.size(1))
        pad_matrix = torch.max(pad_matrix, pad_matrix.transpose(-1, -2)) * padding_dist
        attn_dist = attn_dist + pad_matrix
        return attn_dist


class BbForTuta(Backbone):
    def __init__(self, config):
        super(Backbone, self).__init__()
        self.embeddings = emb.EmbeddingForTuta(config)
        self.encoder = enc.Encoder(config)
        self.attn_methods = {"max": self.pos2attn_max, 
                             "add": self.pos2attn_add}
        self.attn_method = config.attn_method
        self.total_node = sum(config.node_degree)

    def forward(self, 
        token_id, num_mag, num_pre, num_top, num_low, 
        token_order, pos_row, pos_col, pos_top, pos_left, 
        format_vec, indicator
    ):
        embedded_states = self.embeddings(
            token_id, num_mag, num_pre, num_top, num_low, 
            token_order, pos_row, pos_col, pos_top, pos_left, format_vec
        )
        entire_pos_top = self.unzip_tree_position(pos_top)  
        entire_pos_left = self.unzip_tree_position(pos_left)
        attn_mask = self.get_attention_mask(entire_pos_top, entire_pos_left, indicator)
        encoded_states = self.encoder(embedded_states, attn_mask)
        return encoded_states




BACKBONES = {
    "formula_prediction": BbForTuta,
    "fortap": BbForTuta
}
