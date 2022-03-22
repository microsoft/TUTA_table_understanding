# -*- coding: utf-8 -*-
"""
Embedding layers with token / number / order / position / format embeddings
for TUTA.
"""

import torch
import torch.nn as nn


class EmbeddingForTuta(nn.Module):
    def __init__(self, config):
        super(EmbeddingForTuta, self).__init__()
        self.hidden_size = config.hidden_size
        self.quad_embed_size = self.layout_size = self.hidden_size // 4          
        self.tree_size = self.hidden_size - self.layout_size  
        self.uni_layout_size = self.layout_size // 2          
        self.uni_tree_size = self.tree_size // (config.tree_depth * 2) 
        self.total_node = sum(config.node_degree) + 1

        self.token_weight = nn.Embedding(config.vocab_size, config.hidden_size)
        self.magnitude_weight = nn.Embedding(config.magnitude_size + 2, self.quad_embed_size)
        self.precision_weight = nn.Embedding(config.precision_size + 2, self.quad_embed_size)
        self.top_digit_weight = nn.Embedding(config.top_digit_size + 2, self.quad_embed_size)
        self.low_digit_weight = nn.Embedding(config.low_digit_size + 2, self.quad_embed_size)

        self.order_weight = nn.Embedding(config.max_cell_length, self.hidden_size)
        self.row_weight = nn.Embedding(config.row_size + 1, self.uni_layout_size)
        self.column_weight = nn.Embedding(config.column_size + 1, self.uni_layout_size)
        self.top_tree_weight = nn.Embedding(self.total_node, self.uni_tree_size)
        self.left_tree_weight = nn.Embedding(self.total_node, self.uni_tree_size)

        self.format_weight = nn.Linear(config.num_format_feature, config.hidden_size, bias=False)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, 
        token_id, num_mag, num_pre, num_top, num_low, 
        order, pos_row, pos_col, pos_top, pos_left, format_vec
    ):
        # ic(token_id)
        token_states = self.token_weight(token_id)
        # ic(num_mag)
        magnitude_states = self.magnitude_weight(num_mag)
        # ic(num_pre)
        precision_states = self.precision_weight(num_pre)
        # ic(num_top)
        top_digit_states = self.top_digit_weight(num_top)
        # ic(num_low)
        low_digit_states = self.low_digit_weight(num_low)
        numeric_states = torch.cat(
            (magnitude_states, precision_states, top_digit_states, low_digit_states), 
            dim = 2
        )
        string_states = token_states + numeric_states

        # ic(order)
        order_states = self.order_weight(order)
        batch_size, seq_len = order.size()
        # ic(pos_row)
        row_states = self.row_weight(pos_row)
        # ic(pos_left)
        left_tree_states = self.left_tree_weight(pos_left)
        left_tree_states = left_tree_states.contiguous().view(batch_size, seq_len, -1) 
        # horizontal_states = torch.cat((row_states, left_tree_states), -1)
        # ic(pos_col)
        column_states = self.column_weight(pos_col)
        # ic(pos_top)
        top_tree_states = self.top_tree_weight(pos_top)  
        top_tree_states = top_tree_states.contiguous().view(batch_size, seq_len, -1)
        # vertital_states = torch.cat((column_states, top_tree_states), -1)
        position_states = order_states + torch.cat(
            (row_states, left_tree_states, column_states, top_tree_states), 
            dim=-1
        )
        # ic(format_vec)
        format_states = self.format_weight(format_vec)

        embedded_states = string_states + position_states + format_states
        embedded_states = self.LayerNorm(embedded_states)
        embedded_states = self.dropout(embedded_states)
        return embedded_states


# EMBEDS = {
#     "tuta": EmbeddingForTuta,
#     "base": EmbeddingForBase,
#     "tuta_explicit": EmbeddingForTutaExplicit
# }
