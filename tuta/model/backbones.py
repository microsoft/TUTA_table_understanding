# -*- coding: utf-8 -*-
"""
Backbones of Pre-training Models (from input to last hidden-layer output)
"""

import torch
import torch.nn as nn
import model.embeddings as emb
import model.encoders as enc



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
        """
        [CLS] sees all of the tokens except for the [PAD]s
        [SEP]s in table see each other & their own cells; [SEP]s in clc/tcr choices see as their tokens
        Tokens see their friend and corresponding [SEP]
        """
        cls_matrix = (indicator == -1).long().unsqueeze(-1).repeat(1, 1, attn_dist.size(1))
        cls_matrix = torch.max(cls_matrix, cls_matrix.transpose(-1, -2))
        cls_matrix = -(cls_matrix * attn_dist)
        pad_matrix = (indicator == 0).long().unsqueeze(-1).repeat(1, 1, attn_dist.size(1))
        pad_matrix = torch.max(pad_matrix, pad_matrix.transpose(-1, -2)) * padding_dist
        attn_dist = attn_dist + cls_matrix + pad_matrix

        # only table-[SEP]s and root can see their contexts
        sep_matrix = (indicator > 0).long() * (indicator%2 == 1).long()
        sep_matrix = sep_matrix.unsqueeze(-1).repeat(1, 1, attn_dist.size(1))
        sep_matrix = (1 - sep_matrix * sep_matrix.transpose(1, 2)) * padding_dist
        attn_dist = attn_dist * (sep_matrix + 1)
        return attn_dist
        
    def create_post_mask_padonly(self, attn_dist, indicator, padding_dist=100):
        pad_matrix = (indicator == 0).long().unsqueeze(-1).repeat(1, 1, attn_dist.size(1))
        pad_matrix = torch.max(pad_matrix, pad_matrix.transpose(-1, -2)) * padding_dist
        attn_dist = attn_dist + pad_matrix
        return attn_dist



class BbForBase(Backbone):
    def __init__(self, config):
        super(Backbone, self).__init__()
        self.embeddings = emb.EmbeddingForBase(config)
        self.encoder = enc.Encoder(config)
        self.attn_methods = {"max": self.pos2attn_max, 
                             "add": self.pos2attn_add}
        self.attn_method = config.attn_method
        self.total_node = sum(config.node_degree)

    def forward(self, token_id, num_mag, num_pre, num_top, num_low, token_order, pos_top, pos_left, format_vec, indicator):
        embedded_states = self.embeddings(token_id, num_mag, num_pre, num_top, num_low, token_order, format_vec)
        entire_pos_top = self.unzip_tree_position(pos_top)  
        entire_pos_left = self.unzip_tree_position(pos_left)
        attn_mask = self.get_attention_mask(entire_pos_top, entire_pos_left, indicator)
        encoded_states = self.encoder(embedded_states, attn_mask)
        return encoded_states



class BbForTutaExplicit(Backbone):
    def __init__(self, config):
        super(Backbone, self).__init__()
        self.embeddings = emb.EmbeddingForTutaExplicit(config)
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
        entire_pos_top = self.unzip_tree_position(pos_top)  
        entire_pos_left = self.unzip_tree_position(pos_left)
        embedded_states = self.embeddings(
            token_id, num_mag, num_pre, num_top, num_low, 
            token_order, pos_row, pos_col, entire_pos_top, entire_pos_left, format_vec
        )
        attn_mask = self.get_attention_mask(entire_pos_top, entire_pos_left, indicator)
        encoded_states = self.encoder(embedded_states, attn_mask)
        return encoded_states


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
    "tuta": BbForTuta, 
    "base": BbForBase, 
    "tuta_explicit": BbForTutaExplicit
}
