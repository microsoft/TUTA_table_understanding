# -*- coding: utf-8 -*-
"""
Heads of Pre-training Objectives
    MLM: Masked Language Modeling
    CLC: Cell-Level Cloze
    TCR: Table Context Retrieval

"""

import torch
import torch.nn as nn
import model.act_funcs as act


# %% Pre-training Objective
class MlmHead(nn.Module):
    def __init__(self, config):
        super(MlmHead, self).__init__()
        self.uniform_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = act.ACT_FCN[config.hidden_act]
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.predict_linear = nn.Linear(config.hidden_size, config.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, encoded_states, mlm_label):
        """
        Args:
            encoded_states: [batch_size, max_seq_len, hidden_size]
            mlm_label: [batch_size, max_seq_len]
        Returns:
            token_loss, correct, count: scalar
        """
        _, _, hidden_size = encoded_states.size()
        mlm_label = mlm_label.contiguous().view(-1)
        encoded_states = encoded_states.contiguous().view(-1, hidden_size)
        mlm_logits = encoded_states[mlm_label >= 0, :]

        mlm_logits = self.uniform_linear(mlm_logits)
        mlm_logits = self.act_fn(mlm_logits)
        mlm_logits = self.layer_norm(mlm_logits)  
        mlm_logits = self.predict_linear(mlm_logits)  

        mlm_label = mlm_label[mlm_label >= 0]
        mlm_loss = self.loss(mlm_logits, mlm_label)
        mlm_predict = mlm_logits.argmax(dim=1)
        mlm_correct = torch.sum(mlm_predict.eq(mlm_label).float())
        mlm_count = torch.tensor(mlm_logits.size()[0] + 1e-6)
        return mlm_loss, mlm_correct, mlm_count


class ClcHead(nn.Module):
    def __init__(self, config):
        super(ClcHead, self).__init__()
        self.uniform_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)
        self.tanh = nn.Tanh()

    def forward(self, encoded_states, indicator, clc_label):
        """
        Args: 
            encoded_states: [batch, seq_len, hidden]
            indicator: [batch, seq_len], to guide cell-wise summation
            clc_labels: [batch, seq_len], to mark copied & pasted positions
        Returns:
            sep_triple: loss, correct, and total count of [SEP] predictions
            tok_triple: loss, correct, and total count of token-aggregation predictions
        """
        # perform cell-wise summation
        x_mask = indicator.unsqueeze(1)                    
        y_mask = x_mask.transpose(-1, -2)                  
        mask_matrix = y_mask.eq(x_mask).float()            
        encoded_states = mask_matrix.matmul(encoded_states) 

        # linear transformation for the query and key on encoded states
        encoded_states_Q = self.uniform_linear(encoded_states)
        encoded_states_Q = self.tanh(encoded_states_Q)
        encoded_states_K = self.uniform_linear(encoded_states)
        encoded_states_K = self.tanh(encoded_states_K)

        # attention sequence to matrix
        encoded_matrix = encoded_states_Q.matmul(encoded_states_K.transpose(-1, -2))   
        # apply mask except for blanks (in column), same for any cell in one table
        attn_mask = (clc_label < 0).unsqueeze(1).repeat(1, encoded_states.size()[1], 1).float()  
        attn_mask = -1000000000.0 * (1. - attn_mask).float()     
        encoded_matrix = encoded_matrix + attn_mask             
        encoded_matrix = encoded_matrix.contiguous().view(-1, encoded_states.size()[1])   

        # select out pasted cells (row)
        clc_label = clc_label.contiguous().view(-1)                      
        clc_logits = encoded_matrix[clc_label > 0, :]                  
        clc_label = clc_label[clc_label > 0]                          

        # forward and calculate loss, respectively for SEP and TOKENs
        sep_logits = clc_logits[0: : 2, :]                         
        sep_labels = clc_label[0: : 2]                         
        sep_predict = sep_logits.argmax(dim=-1)
        sep_correct = torch.sum(sep_predict.eq(sep_labels).float())
        sep_loss = self.loss(sep_logits, sep_labels)
        sep_count = torch.tensor(sep_logits.size()[0] + 1e-6)

        tok_logits = clc_logits[1: : 2, :]                        
        tok_labels = clc_label[1: : 2]                            
        tok_predict = tok_logits.argmax(dim=-1)                     
        tok_correct = torch.sum(tok_predict.eq(tok_labels).float())  
        tok_loss = self.loss(tok_logits, tok_labels)                 
        tok_count = torch.tensor(tok_logits.size()[0] + 1e-6) 
        return (sep_loss, sep_correct, sep_count), (tok_loss, tok_correct, tok_count)


class TcrHead(nn.Module):
    def __init__(self, config):
        super(TcrHead, self).__init__()
        self.num_tcr_type = config.num_tcr_type
        self.uniform_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = act.ACT_FCN[config.hidden_act]
        self.tanh = nn.Tanh()
        self.predict_linear = nn.Linear(1, config.num_tcr_type)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, encoded_states, indicator, tcr_label):
        """
        Args: 
            encoded_states: [batch, seq_len, hidden]
            indicator: [batch, seq_len], to guide cell-wise summation
            tcr_label: [batch, seq_len], to mark the heads of context chunks
                       '0' false, '1' true, '-1' padding (ignore_index)
        Returns:
            loss, correct (scalar), and total label count
        """
        # perform cell-wise summation
        x_mask = indicator.unsqueeze(1)                      
        y_mask = x_mask.transpose(-1, -2)                  
        mask_matrix = y_mask.eq(x_mask).float()             
        encoded_states = mask_matrix.matmul(encoded_states)  
        
        tcr_label = tcr_label.contiguous().view(-1)          
        tcr_ignore = (tcr_label < 0).float()
        tcr_count = torch.sum(1. - tcr_ignore) + 1e-6

        tcr_states = self.uniform_linear(encoded_states)
        tcr_states = self.act_fn(tcr_states)
        cls_states = tcr_states[:, 0, :].unsqueeze(2)   
        tcr_logits = tcr_states.matmul(cls_states)    
        tcr_logits = self.predict_linear(self.act_fn(tcr_logits))
        tcr_logits = tcr_logits.contiguous().view(-1, self.num_tcr_type)
        tcr_logit_mask = (-100.0 * tcr_ignore).unsqueeze(1).repeat(1, 2)
        tcr_logits = tcr_logits + tcr_logit_mask
        tcr_loss = self.loss(tcr_logits, tcr_label)
        tcr_predict = torch.argmax(tcr_logits, dim=-1) 
        
        tcr_correct = torch.sum(tcr_predict.eq(tcr_label))
        return tcr_loss, tcr_correct, tcr_count


# %% Downstream Tasks
class CtcHead(nn.Module):
    def __init__(self, config):
        super(CtcHead, self).__init__()
        self.uniform_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = act.ACT_FCN[config.hidden_act]
        self.tanh = nn.Tanh()
        self.predict_linear = nn.Linear(config.hidden_size, config.num_ctc_type)
        self.loss = nn.CrossEntropyLoss()

        self.aggregator = config.aggregator
        self.aggr_funcs = {"sum": self.token_sum, 
                           "avg": self.token_avg}
    
    def token_sum(self, token_states, indicator):
        """take the sum of token encodings (not including [SEP]s) as cell encodings """
        x_mask = indicator.unsqueeze(1)                      # [batch_size, 1, seq_len]
        y_mask = x_mask.transpose(-1, -2)                    # [batch_size, seq_len, 1]
        mask_matrix = y_mask.eq(x_mask).float()              # [batch_size, seq_len, seq_len]
        sum_states = mask_matrix.matmul(token_states)        # [batch_size, seq_len, hidden_size]
        return sum_states
    
    def token_avg(self, token_states, indicator):
        """take the average of token encodings (not including [SEP]s) as cell encodings """
        x_mask = indicator.unsqueeze(1)                      # [batch_size, 1, seq_len]
        y_mask = x_mask.transpose(-1, -2)                    # [batch_size, seq_len, 1]
        mask_matrix = y_mask.eq(x_mask).float()              # [batch_size, seq_len, seq_len]
        sum_matrix = torch.sum(mask_matrix, dim=-1)
        mask_matrix = mask_matrix.true_divide(sum_matrix.unsqueeze(-1))
        cell_states = mask_matrix.matmul(token_states)  # [batch_size, seq_len, hidden_size]
        return cell_states

    def forward(self, encoded_states, indicator, ctc_label):
        # get cell encodings from token sequence
        cell_states = self.aggr_funcs[self.aggregator](encoded_states, indicator)

        ctc_label = ctc_label.contiguous().view(-1)
        cell_states = cell_states.contiguous().view(ctc_label.size()[0], -1)
        ctc_logits = cell_states[ctc_label > -1, :]       # [batch_total_cell_num, hidden_size]
        ctc_label = ctc_label[ctc_label > -1]

        # separator
        sep_logits = self.uniform_linear(ctc_logits[0::2, :])
        sep_logits = self.tanh(sep_logits)
        sep_logits = self.predict_linear(sep_logits)
        sep_predict = sep_logits.argmax(dim=-1)
        sep_labels = ctc_label[0: : 2]
        # sep_correct = torch.sum(sep_predict.eq(sep_labels).float())
        sep_loss = self.loss(sep_logits, sep_labels)
        # sep_count = torch.tensor(sep_logits.size()[0] + 1e-6)

        # token-sum
        tok_logits = self.uniform_linear(ctc_logits[1::2, :])
        tok_logits = self.tanh(tok_logits)
        tok_logits = self.predict_linear(tok_logits)
        tok_predict = tok_logits.argmax(dim=-1)  
        tok_labels = ctc_label[1: : 2]                                  # [batch-variant copied num]
        # tok_correct = torch.sum(tok_predict.eq(tok_labels).float())   # scalar
        tok_loss = self.loss(tok_logits, tok_labels)                    # scalar
        # tok_count = torch.tensor(tok_logits.size()[0] + 1e-6)         # 1d tensor
        # return (sep_loss, sep_correct, sep_count), (tok_loss, tok_correct, tok_count)
        return (sep_loss, sep_predict, sep_labels), (tok_loss, tok_predict, tok_labels)

    
class TtcHead(nn.Module):
    """Fine-tuning head for the task of table type classification."""

    def __init__(self, config):
        super(TtcHead, self).__init__()
        self.uniform_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = act.ACT_FCN[config.hidden_act]
        # self.tanh = nn.Tanh()
        self.predict_linear = nn.Linear(config.hidden_size, config.num_table_types)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, encoded_states, ttc_label, return_prediction=True):
        """Predict table types with the transformed CLS, then compute loss against the ttc_label. 
        
        Args:
            encoded_states <float> [batch-size, seq-len, hidden-size]: representation of the last hidden layer.
            ttc_label <int> [batch-size]: type of the table.
        Returns: 
            loss <float> []: computed cross-entropy loss.
            ttc_logits <float> [batch-size, num_table_types]: logits over table types,
            *prediction <int> [batch-size]: predicted table type.
        """
        transformed_states = self.uniform_linear(encoded_states)   # [batch-size, seq-len, hidden-size]
        table_state = transformed_states[:, 0, :]                  # [batch-size, hidden-size]
        table_state = self.act_fn(table_state)                     # [batch-size, hidden-size]
        ttc_logits = self.predict_linear(table_state)              # [batch-size, hidden-size]
        loss = self.loss(ttc_logits, ttc_label)                    # []
        
        if return_prediction == True:
            prediction = ttc_logits.argmax(dim=-1)                 # [batch-size]
            return loss, prediction
        
        return loss, ttc_logits
