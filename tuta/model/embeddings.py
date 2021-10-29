# -*- coding: utf-8 -*-
"""
Embedding layers with token / number / order / position / format embeddings
for TUTA-base, TUTA-explicit, and TUTA (TUTA-implicit)
"""

import torch
import torch.nn as nn



class EmbeddingForBase(nn.Module):
    def __init__(self, config):
        super(EmbeddingForBase, self).__init__()
        self.hidden_size = config.hidden_size
        self.quad_embed_size = config.hidden_size // 4 

        self.token_weight = nn.Embedding(config.vocab_size, config.hidden_size)
        self.magnitude_weight = nn.Embedding(config.magnitude_size + 2, self.quad_embed_size)
        self.precision_weight = nn.Embedding(config.precision_size + 2, self.quad_embed_size)
        self.top_digit_weight = nn.Embedding(config.top_digit_size + 2, self.quad_embed_size)
        self.low_digit_weight = nn.Embedding(config.low_digit_size + 2, self.quad_embed_size)

        self.order_weight = nn.Embedding(config.max_cell_length, self.hidden_size)
        self.format_weight = nn.Linear(config.num_format_feature, config.hidden_size, bias=False)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_id, num_mag, num_pre, num_top, num_low, order, format_vec):
        token_states = self.token_weight(token_id)
        magnitude_states = self.magnitude_weight(num_mag)
        precision_states = self.precision_weight(num_pre)
        top_digit_states = self.top_digit_weight(num_top)
        low_digit_states = self.low_digit_weight(num_low)
        numeric_states = torch.cat((magnitude_states, precision_states, top_digit_states, low_digit_states), dim = 2)
        string_states = token_states + numeric_states

        order_states = self.order_weight(order)
        format_states = self.format_weight(format_vec)

        embedded_states = string_states + order_states + format_states
        embedded_states = self.LayerNorm(embedded_states)
        embedded_states = self.dropout(embedded_states)
        return embedded_states



class EmbeddingForTutaExplicit(nn.Module):
    def __init__(self, config):
        super(EmbeddingForTutaExplicit, self).__init__()
        self.hidden_size = config.hidden_size                                 # 768
        self.layout_embed_size = self.num_embed_size = self.hidden_size // 4  # 768 // 4 = 192
        self.tree_embed_size = self.hidden_size - self.layout_embed_size      # 576
        self.uni_layout_size = self.layout_embed_size // 2                    # 96
        self.uni_tree_size = self.tree_embed_size // 2                        # 288
        self.tree_depth, self.node_degree = config.tree_depth, config.node_degree  # 4, [32, 32, 64, 160]
        self.total_node = sum(self.node_degree)                       # 288
        self.weight_number = self.uni_tree_size // self.total_node  # 1

        self.token_weight = nn.Embedding(config.vocab_size, self.hidden_size)
        self.magnitude_weight = nn.Embedding(config.magnitude_size + 2, self.num_embed_size)
        self.precision_weight = nn.Embedding(config.precision_size + 2, self.num_embed_size)
        self.top_digit_weight = nn.Embedding(config.top_digit_size + 2, self.num_embed_size)
        self.low_digit_weight = nn.Embedding(config.low_digit_size + 2, self.num_embed_size)

        self.order_weight = nn.Embedding(config.max_cell_length, self.hidden_size)
        self.row_weight = nn.Embedding(config.row_size + 1, self.uni_layout_size)
        self.column_weight = nn.Embedding(config.column_size + 1, self.uni_layout_size)
        self.tree_weight = nn.Embedding(2, self.uni_tree_size)
        top, left = self.init_tree_weight(), self.init_tree_weight()
        self.tree_weight.weight.data.copy_(torch.cat((top.unsqueeze(0), left.unsqueeze(0)), dim = 0) )

        self.format_weight = nn.Linear(config.num_format_feature, config.hidden_size, bias=False)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def init_tree_weight(self): 
        """apply decay weight to each of total node, from a p-vector of length node_degree. """
        base_weight = torch.rand(self.weight_number)   # size = [weight_number]
        curr_weight = torch.ones(self.weight_number)
        weight_matrix = torch.ones(self.weight_number, self.total_node)
        start, end = 0, self.node_degree[0]
        for idepth in range(self.tree_depth - 1):
            curr_weight = curr_weight * base_weight
            start = start + self.node_degree[idepth]
            end = end + self.node_degree[idepth+1]
            weight_matrix[:, start: end] = curr_weight
        weight_matrix = weight_matrix.contiguous().view(-1)
        return weight_matrix    

    def compute_tree_position(self, position, index):
        weight = torch.LongTensor([index]).expand(position.size()[:2]).to(position.device) 
        weight_states = self.tree_weight(weight)      
        tree_states = position.unsqueeze(2).repeat(1, 1, self.weight_number, 1).float() 
        tree_states = tree_states.contiguous().view(weight_states.size())  
        weight_states = weight_states * tree_states
        return weight_states

    def forward(self, token_id, num_mag, num_pre, num_top, num_low, order, pos_row, pos_col, pos_top, pos_left, format_vec):
        token_states = self.token_weight(token_id)
        magnitude_states = self.magnitude_weight(num_mag)
        precision_states = self.precision_weight(num_pre)
        top_digit_states = self.top_digit_weight(num_top)
        low_digit_states = self.low_digit_weight(num_low)
        numeric_states = torch.cat(
            (magnitude_states, precision_states, top_digit_states, low_digit_states), 
            dim = 2
        )
        string_states = token_states + numeric_states

        order_states = self.order_weight(order)
        row_states = self.row_weight(pos_row) 
        column_states = self.column_weight(pos_col)
        top_tree_states = self.compute_tree_position(pos_top, 0)
        left_tree_states = self.compute_tree_position(pos_left, 1)
        position_states = order_states + torch.cat(
            (row_states, left_tree_states, column_states, top_tree_states), 
            dim = 2
        )

        format_states = self.format_weight(format_vec)
        embedded_states = string_states + position_states + format_states
        embedded_states = self.LayerNorm(embedded_states)
        embedded_states = self.dropout(embedded_states)
        return embedded_states



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
        token_states = self.token_weight(token_id)
        magnitude_states = self.magnitude_weight(num_mag)
        precision_states = self.precision_weight(num_pre)
        top_digit_states = self.top_digit_weight(num_top)
        low_digit_states = self.low_digit_weight(num_low)
        numeric_states = torch.cat(
            (magnitude_states, precision_states, top_digit_states, low_digit_states), 
            dim = 2
        )
        string_states = token_states + numeric_states

        order_states = self.order_weight(order)
        batch_size, seq_len = order.size()

        row_states = self.row_weight(pos_row)       
        left_tree_states = self.left_tree_weight(pos_left)   
        left_tree_states = left_tree_states.contiguous().view(batch_size, seq_len, -1) 
        # horizontal_states = torch.cat((row_states, left_tree_states), -1)
        column_states = self.column_weight(pos_col)   
        top_tree_states = self.top_tree_weight(pos_top)  
        top_tree_states = top_tree_states.contiguous().view(batch_size, seq_len, -1)
        # vertital_states = torch.cat((column_states, top_tree_states), -1)
        position_states = order_states + torch.cat(
            (row_states, left_tree_states, column_states, top_tree_states), 
            dim=-1
        )
        format_states = self.format_weight(format_vec)

        embedded_states = string_states + position_states + format_states
        embedded_states = self.LayerNorm(embedded_states)
        embedded_states = self.dropout(embedded_states)
        return embedded_states


EMBEDS = {
    "tuta": EmbeddingForTuta, 
    "base": EmbeddingForBase, 
    "tuta_explicit": EmbeddingForTutaExplicit
}
