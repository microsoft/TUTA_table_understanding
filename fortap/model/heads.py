# -*- coding: utf-8 -*-
"""
Heads of Pre-training Objectives and Downstream tasks.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

import model.act_funcs as act
from model.generation import *
from constants import NR_AGGR_TO_INDEX, FP_ENCODE_VOCAB

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

        device_id = encoded_states.get_device()

        _, _, hidden_size = encoded_states.size()
        mlm_label = mlm_label.contiguous().view(-1)
        if torch.max(mlm_label) == -1:
            return torch.tensor(0).cuda(device_id), torch.tensor(0).cuda(device_id), torch.tensor(0).cuda(device_id)
        encoded_states = encoded_states.contiguous().view(-1, hidden_size)
        mlm_logits = encoded_states[mlm_label >= 0, :]

        mlm_logits = self.uniform_linear(mlm_logits)
        mlm_logits = self.act_fn(mlm_logits)
        mlm_logits = self.layer_norm(mlm_logits)
        mlm_logits = self.predict_linear(mlm_logits)

        mlm_label = mlm_label[mlm_label >= 0]
        mlm_loss = self.loss(mlm_logits, mlm_label)
        mlm_predict = mlm_logits.argmax(dim=1)
        mlm_correct = torch.sum(mlm_predict.eq(mlm_label))
        mlm_count = torch.tensor(mlm_logits.size()[0] + 1e-6)
        return mlm_loss, mlm_correct, mlm_count


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
        x_mask = indicator.unsqueeze(1)  # [batch_size, 1, seq_len]
        y_mask = x_mask.transpose(-1, -2)  # [batch_size, seq_len, 1]
        mask_matrix = y_mask.eq(x_mask).float()  # [batch_size, seq_len, seq_len]
        sum_states = mask_matrix.matmul(token_states)  # [batch_size, seq_len, hidden_size]
        return sum_states

    def token_avg(self, token_states, indicator):
        """take the average of token encodings (not including [SEP]s) as cell encodings """
        x_mask = indicator.unsqueeze(1)  # [batch_size, 1, seq_len]
        y_mask = x_mask.transpose(-1, -2)  # [batch_size, seq_len, 1]
        mask_matrix = y_mask.eq(x_mask).float()  # [batch_size, seq_len, seq_len]
        sum_matrix = torch.sum(mask_matrix, dim=-1)
        mask_matrix = mask_matrix.true_divide(sum_matrix.unsqueeze(-1))
        cell_states = mask_matrix.matmul(token_states)  # [batch_size, seq_len, hidden_size]
        return cell_states

    def forward(self, encoded_states, indicator, ctc_label):
        # get cell encodings from token sequence
        cell_states = self.aggr_funcs[self.aggregator](encoded_states, indicator)

        ctc_label = ctc_label.contiguous().view(-1)
        cell_states = cell_states.contiguous().view(ctc_label.size()[0], -1)
        ctc_logits = cell_states[ctc_label > -1, :]  # [batch_total_cell_num, hidden_size]
        ctc_label = ctc_label[ctc_label > -1]

        # separator
        sep_logits = self.uniform_linear(ctc_logits[0::2, :])
        sep_logits = self.tanh(sep_logits)
        sep_logits = self.predict_linear(sep_logits)
        sep_predict = sep_logits.argmax(dim=-1)
        sep_labels = ctc_label[0:: 2]
        # sep_correct = torch.sum(sep_predict.eq(sep_labels).float())
        sep_loss = self.loss(sep_logits, sep_labels)
        # sep_count = torch.tensor(sep_logits.size()[0] + 1e-6)

        # token-sum
        tok_logits = self.uniform_linear(ctc_logits[1::2, :])
        tok_logits = self.tanh(tok_logits)
        tok_logits = self.predict_linear(tok_logits)
        tok_predict = tok_logits.argmax(dim=-1)
        tok_labels = ctc_label[1:: 2]  # [batch-variant copied num]
        # tok_correct = torch.sum(tok_predict.eq(tok_labels).float())   # scalar
        tok_loss = self.loss(tok_logits, tok_labels)  # scalar
        # tok_count = torch.tensor(tok_logits.size()[0] + 1e-6)         # 1d tensor
        # return (sep_loss, sep_correct, sep_count), (tok_loss, tok_correct, tok_count)
        return (sep_loss, sep_predict, sep_labels), (tok_loss, tok_predict, tok_labels)


# Formula Pretrain Head
class SRHead(nn.Module):
    """ Semantic reference pre-train head."""

    def __init__(self, config):
        super(SRHead, self).__init__()
        self.linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.act_fn = nn.GELU()
        self.cls = nn.Linear(config.hidden_size, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, encoded_states, sr_label):
        """ Binary classification indicating if a header is semantic referenced by/from formula header.
        Args:
            encoded_states: (b, seq_len, h)
            sr_label: (b, seq_len)
        Return:
            loss: average CELoss over batch
            batch_pred_logits: List of predicted logits
            batch_target_labels: List of target labels
        """
        batch_size = encoded_states.size(0)
        cnt_valid_sample, loss = 0, 0.
        batch_pred_logits, batch_target_labels = [], []
        for i in range(batch_size):
            enc_emb = encoded_states[i]  # (seq_len, h)
            header_emb = enc_emb[(sr_label[i] >= 0) & (sr_label[i] < 2)]  # (num_sr, h)
            if header_emb.size(0) == 0:  # no pos or neg label
                continue
            formula_emb = enc_emb[sr_label[i] == 2]  # (1, h)
            fuse_emb = torch.cat((header_emb, formula_emb.repeat(header_emb.size(0), 1)), dim=-1)  # (num_sr, 2*h)
            pred_logits = self.cls(self.act_fn(self.linear(fuse_emb)))  # (num_sr, 2)
            target_labels = sr_label[i][(sr_label[i] >= 0) & (sr_label[i] < 2)]  # (num_sr,)
            loss += self.loss_fn(pred_logits, target_labels)
            cnt_valid_sample += 1
            batch_pred_logits.append(pred_logits)
            batch_target_labels.append(target_labels)
        loss /= (cnt_valid_sample + 1e-6)
        return loss, batch_pred_logits, batch_target_labels


class NRHead(nn.Module):
    """ Numerical reasoning pre-train head."""

    def __init__(self, config):
        super(NRHead, self).__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = nn.GELU()
        self.cls = nn.Linear(config.hidden_size, len(NR_AGGR_TO_INDEX))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, encoded_states, nr_label, op_appear_flag):
        """ Multi classification indicating which aggr/op should cells with the same label use.
        Args:
            encoded_states: (b, seq_len, h)
            nr_label: (b, seq_len)
            op_appear_flag: (b, len(NR_AGGR_TO_INDEX)), 0 or 1
        Return:
            loss: average CELoss over batch
            batch_pred_logits: List of predicted logits
            batch_target_labels: List of target labels
        """
        batch_size = encoded_states.size(0)
        cnt_valid_sample, loss = 0, 0.
        batch_pred_logits, batch_target_labels = [], []
        device_id = encoded_states.get_device()
        for i in range(batch_size):
            enc_emb = encoded_states[i]  # (seq_len, h)
            op_emb = scatter_mean(src=enc_emb,
                                  index=nr_label[i].unsqueeze(-1).repeat(1, enc_emb.size(-1)),
                                  dim=0)  # (len(NR_AGGR_TO_INDEX)+1, h))
            op_emb = op_emb[:-1]  # (len(NR_AGGR_TO_INDEX), h), the last dimension is for garbage collection
            op_emb = op_emb[op_appear_flag[i] == 1]  # (num_appear_op, h)
            if op_emb.size(0) == 0:  # no op label
                continue
            target_labels = torch.arange(0, len(NR_AGGR_TO_INDEX))[op_appear_flag[i] == 1]  # (num_appear_op, )
            if device_id >= 0:
                target_labels = target_labels.to(f"cuda:{device_id}")
            assert op_emb.size(0) == target_labels.size(0)

            random_indices = torch.randperm(op_emb.size(0))  # (num_appear_op, )
            if device_id >= 0:
                random_indices = random_indices.to(f"cuda:{device_id}")
            op_emb = op_emb[random_indices]
            target_labels = target_labels[random_indices]  # (num_appear_op, )
            pred_logits = self.cls(self.act_fn(self.linear(op_emb)))  # (num_appear_op, len(NR_AGGR_TO_INDEX))
            loss += self.loss_fn(pred_logits, target_labels)
            cnt_valid_sample += 1
            batch_pred_logits.append(pred_logits)
            batch_target_labels.append(target_labels)
        loss /= (cnt_valid_sample + 1e-6)
        return loss, batch_pred_logits, batch_target_labels


class OpMLMHead(nn.Module):
    """ Formula operator recovery head."""

    def __init__(self, config):
        super(OpMLMHead, self).__init__()
        self.uniform_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = act.ACT_FCN[config.hidden_act]
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.predict_linear = nn.Linear(config.hidden_size, len(FP_ENCODE_VOCAB))
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, encoded_states, op_mlm_label):
        """
        Args:
            encoded_states: [batch_size, max_seq_len, hidden_size]
            op_mlm_label: [batch_size, max_seq_len], >=0 for formula token id in vocab, -1 for not mask
        Returns:
            op_mlm_loss, correct, count: scalar
        """
        device_id = encoded_states.get_device()
        _, _, hidden_size = encoded_states.size()
        op_mlm_label = op_mlm_label.contiguous().view(-1)  # (b*seq_len)
        if torch.sum(op_mlm_label >= 0) == 0:
            return torch.tensor(0).cuda(device_id), torch.tensor(0).cuda(device_id), torch.tensor(0).cuda(device_id)

        encoded_states = encoded_states.contiguous().view(-1, hidden_size)  # (b*seq_len, h)
        op_mlm_logits = encoded_states[op_mlm_label >= 0, :]  # (num_masks_in_batch, h)
        op_mlm_logits = self.uniform_linear(op_mlm_logits)
        op_mlm_logits = self.act_fn(op_mlm_logits)
        op_mlm_logits = self.layer_norm(op_mlm_logits)
        op_mlm_logits = self.predict_linear(op_mlm_logits)  # (num_masks_in_batch, len(FP_ENCODE_VOCAB))

        op_mlm_label = op_mlm_label[op_mlm_label >= 0]  # (num_masks_in_batch, )
        op_mlm_loss = self.loss(op_mlm_logits, op_mlm_label)
        op_mlm_predict = op_mlm_logits.argmax(dim=1)
        op_mlm_correct = torch.sum(op_mlm_predict.eq(op_mlm_label))
        op_mlm_count = torch.tensor(op_mlm_logits.size()[0] + 1e-6)
        # ic(op_mlm_predict)
        # ic(op_mlm_label)
        # print()
        return op_mlm_loss, op_mlm_correct, op_mlm_count


class RangeMLMHead(nn.Module):
    """ Formula range recovery head."""
    def __init__(self, config):
        super(RangeMLMHead, self).__init__()
        self.range_affine = nn.Linear(config.hidden_size, config.hidden_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, encoded_states, candi_cell_token_mask, range_mlm_label):
        """
        Args:
            encoded_states: [batch_size, max_seq_len, hidden_size]
            candi_cell_token_mask: [batch_size, max_seq_len]
            range_mlm_label: [batch_size, max_seq_len], >0 for range [sep] token index in seq, 0 for not range
        Returns:
            range_mlm_loss, correct, count: scalar
        """
        device_id = encoded_states.get_device()
        batch_size, _, hidden_size = encoded_states.size()
        if torch.max(range_mlm_label) == 0:
            return torch.tensor(0).cuda(device_id), torch.tensor(0).cuda(device_id), torch.tensor(0).cuda(device_id)

        loss, correct, count = torch.tensor(0.).cuda(device_id), torch.tensor(0).cuda(device_id), torch.tensor(0).cuda(device_id)
        for i in range(batch_size):
            range_mask = range_mlm_label[i] > 0  # (seq_len)
            if torch.sum(range_mask) == 0:
                continue
            seq_emb = encoded_states[i]  # (seq_len, h)
            range_emb = seq_emb[range_mask]  # (num_masks, h)
            range_label = range_mlm_label[i][range_mask]  # (num_masks, )

            range_logits = torch.matmul(self.range_affine(range_emb), seq_emb.transpose(1, 0))  # (num_masks, seq_len)
            sep_mask = candi_cell_token_mask[i].unsqueeze(0).repeat(range_logits.size(0), 1)  # (num_masks, seq_len)
            range_logits = range_logits.masked_fill(sep_mask == 0, -1e9)  # (num_masks, seq_len)
            range_lprobs = F.log_softmax(range_logits, dim=-1)  # (num_masks, seq_len)
            range_loss = F.nll_loss(
                range_lprobs,
                range_label,
                reduction="mean",
                ignore_index=tokenizer.DEFAULT_RANGE_MLM_LABEL
            )
            range_pred = range_lprobs.argmax(dim=-1)  # (num_masks, )
            range_correct = torch.sum(range_pred.eq(range_label))
            loss += range_loss
            correct += range_correct
            count += torch.sum(range_mask)
        loss /= (count + 1e-6)
        return loss, correct, count


# FP Head
class FPHead(nn.Module):
    """Fine-tuning head for the task of formula prediction."""

    def __init__(self, config):
        super(FPHead, self).__init__()
        self.models = {"Transformer": TransformerLM,
                       "LSTM": LSTMLM, "LSTM_attn": LSTMLM}
        self.generation_model = self.models[config.generation_model](
            config)
        self.test = config.test

    def forward(self, encoded_states, formula_cell_states, src_sketch, tgt_sketch, candi_cell_token_mask, range_label, range_map):
        """
        Args:
            encoded_states: (b, seq_len, h)
            formula_cell_states: (b, h)
            src_sketch: (b, max_sketch_len)
            tgt_sketch: (b, max_sketch_len)
            candi_cell_token_mask: (b, seq_len), 0 or 1
            range_label: (b, max_sketch_len), 0 or ref token index, match with tgt_sketch
        """
        # class_count = sketch_class.size(0) + 1e-6
        sketch_loss, sketch_top1_crt, sketch_top5_crt, sketch_cnt = torch.tensor(0), 0, 0, 0
        range_loss, range_crt, range_cnt = torch.tensor(0), 0, 0
        fp_top1_crt, fp_top5_crt, fp_cnt = 0, 0, 0

        sketch_loss, range_loss = self.generation_model(
            encoded_states, formula_cell_states, src_sketch, tgt_sketch, candi_cell_token_mask, range_label)

        if self.test or random.random() < 0.03:
            encoded_states = encoded_states[0:1, :]  # (1, seq_len, h)
            formula_cell_states = formula_cell_states[0:1, :]  # (1, h)
            target_tokens = [t.item() for t in list(tgt_sketch[0, :])]  # List[int]
            candi_cell_token_mask = candi_cell_token_mask[0:1, :]  # (1, seq_len)
            range_label = [t.item() for t in list(range_label[0, :])]  # List[int], w. <END>, w/o. <START>
            range_map = range_map[0]  # Dict[int,str]

            generated_text, generated_tokens, \
            sketch_cnt, sketch_top1_crt, sketch_top5_crt, \
            range_cnt, range_crt, \
            fp_cnt, fp_top1_crt, fp_top5_crt \
                = self.generation_model.generate(encoded_states, formula_cell_states, target_tokens, candi_cell_token_mask, range_label, range_map)  # only the first in batch
            # if not self.test:
            # # if not sketch_top1_crt:
            #     complete_tgt_tokens = [FP_VOCAB['<START>']] + target_tokens
            #     complete_tgt_tokens = [t for t in complete_tgt_tokens if t != tokenizer.FP_PAD_TAG]
            #     complete_range_label = [tokenizer.DEFAULT_RANGE_LABEL] + range_label
            #     complete_range_label = complete_range_label[:len(complete_tgt_tokens)]
            #     gt_text = get_text(complete_tgt_tokens, generated=False, range_label=complete_range_label, explicit_range=True, range_map=range_map)
            #     # ic(generated_text)
            #     # ic(gt_text)
        return sketch_loss, range_loss, sketch_top1_crt, sketch_top5_crt, sketch_cnt, range_crt, range_cnt, fp_top1_crt, fp_top5_crt, fp_cnt


# Context Augmentation Head
class ContextAugHead(nn.Module):
    """Fine-tuning head for the task of context augmentation."""
    def __init__(self, config):
        super(ContextAugHead, self).__init__()
        self.uniform_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = act.ACT_FCN[config.hidden_act]
        self.tanh = nn.Tanh()
        self.predict_linear = nn.Linear(config.hidden_size, 3)
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor([0.15, 1.0, 1.0]))

        self.aggregator = 'avg'
        self.aggr_funcs = {"sum": self.token_sum,
                           "avg": self.token_avg}

    def token_sum(self, token_states, indicator):
        """take the sum of token encodings (not including [SEP]s) as cell encodings """
        x_mask = indicator.unsqueeze(1)  # [batch_size, 1, seq_len]
        y_mask = x_mask.transpose(-1, -2)  # [batch_size, seq_len, 1]
        mask_matrix = y_mask.eq(x_mask).float()  # [batch_size, seq_len, seq_len]
        sum_states = mask_matrix.matmul(token_states)  # [batch_size, seq_len, hidden_size]
        return sum_states

    def token_avg(self, token_states, indicator):
        """take the average of token encodings (not including [SEP]s) as cell encodings """
        x_mask = indicator.unsqueeze(1)  # [batch_size, 1, seq_len]
        y_mask = x_mask.transpose(-1, -2)  # [batch_size, seq_len, 1]
        mask_matrix = y_mask.eq(x_mask).float()  # [batch_size, seq_len, seq_len]
        sum_matrix = torch.sum(mask_matrix, dim=-1)
        mask_matrix = mask_matrix.true_divide(sum_matrix.unsqueeze(-1))
        cell_states = mask_matrix.matmul(token_states)  # [batch_size, seq_len, hidden_size]
        return cell_states

    def forward(self, encoded_states, indicator, ca_label):
        # get cell encodings from token sequence
        cell_states = self.aggr_funcs[self.aggregator](encoded_states, indicator)

        ca_label = ca_label.contiguous().view(-1)  # (b * seq_len)
        cell_states = cell_states.contiguous().view(ca_label.size()[0], -1)  # (b * seq_len, h)
        ca_logits = cell_states[ca_label > -1, :]  # (b * seq_len, h)
        ca_label = ca_label[ca_label > -1]

        # separator
        sep_logits = self.uniform_linear(ca_logits[0::2, :])
        sep_logits = self.tanh(sep_logits)
        sep_logits = self.predict_linear(sep_logits)
        sep_predict = sep_logits.argmax(dim=-1)
        sep_labels = ca_label[0:: 2]
        sep_correct = torch.sum(sep_predict.eq(sep_labels).float())
        sep_loss = self.loss(sep_logits, sep_labels)
        sep_count = torch.tensor(sep_logits.size()[0] + 1e-6)
        sep_result_dict = self.evaluate_ca_f1(sep_predict, sep_labels)

        # token-sum
        tok_logits = self.uniform_linear(ca_logits[1::2, :])
        tok_logits = self.tanh(tok_logits)
        tok_logits = self.predict_linear(tok_logits)
        tok_predict = tok_logits.argmax(dim=-1)
        tok_labels = ca_label[1:: 2]  # [batch-variant copied num]
        tok_correct = torch.sum(tok_predict.eq(tok_labels).float())   # scalar
        tok_loss = self.loss(tok_logits, tok_labels)  # scalar
        tok_count = torch.tensor(tok_logits.size()[0] + 1e-6)         # 1d tensor
        tok_result_dict = self.evaluate_ca_f1(tok_predict, tok_labels)

        return (sep_loss, sep_result_dict), (tok_loss, tok_result_dict)
        # return (sep_loss, sep_predict, sep_labels), (tok_loss, tok_predict, tok_labels)

    def evaluate_ca_f1(self, predict, labels):
        """
        Args:
            predict: (num_labels,)
            labels: (num_labels,)
        """
        result_dict = {}
        for target in range(3):
            result_dict[target] = dict()
            precision, recall, f1, tp, fp, fn = self._evaluate_ca_f1(predict, labels, target)
            result_dict[target]['precision'] = precision
            result_dict[target]['recall'] = recall
            result_dict[target]['f1'] = f1
            result_dict[target]['tp'] = tp
            result_dict[target]['fp'] = fp
            result_dict[target]['fn'] = fn
        return result_dict

    def _evaluate_ca_f1(self, predict, labels, target):
        """
        Args:
            predict: (num_labels,)
            labels: (num_labels,)
            target: target label
        """
        tp = torch.sum(labels[predict == target] == target)
        fp = torch.sum(labels[predict == target] != target)
        tn = torch.sum(labels[predict != target] != target)
        fn = torch.sum(labels[predict != target] == target)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1, tp, fp, fn


# SR Context Head
class SRContextHead(nn.Module):
    """Pretrain head for the task of SR Context"""
    def __init__(self, config):
        super(SRContextHead, self).__init__()
        self.uniform_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = act.ACT_FCN[config.hidden_act]
        self.tanh = nn.Tanh()
        self.num_sr_context_types = 6
        self.predict_linear = nn.Linear(config.hidden_size, self.num_sr_context_types)
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0, 1.0]))

        self.aggregator = 'avg'
        self.aggr_funcs = {"sum": self.token_sum,
                           "avg": self.token_avg}

    def token_sum(self, token_states, indicator):
        """take the sum of token encodings (not including [SEP]s) as cell encodings """
        x_mask = indicator.unsqueeze(1)  # [batch_size, 1, seq_len]
        y_mask = x_mask.transpose(-1, -2)  # [batch_size, seq_len, 1]
        mask_matrix = y_mask.eq(x_mask).float()  # [batch_size, seq_len, seq_len]
        sum_states = mask_matrix.matmul(token_states)  # [batch_size, seq_len, hidden_size]
        return sum_states

    def token_avg(self, token_states, indicator):
        """take the average of token encodings (not including [SEP]s) as cell encodings """
        x_mask = indicator.unsqueeze(1)  # [batch_size, 1, seq_len]
        y_mask = x_mask.transpose(-1, -2)  # [batch_size, seq_len, 1]
        mask_matrix = y_mask.eq(x_mask).float()  # [batch_size, seq_len, seq_len]
        sum_matrix = torch.sum(mask_matrix, dim=-1)
        mask_matrix = mask_matrix.true_divide(sum_matrix.unsqueeze(-1))
        cell_states = mask_matrix.matmul(token_states)  # [batch_size, seq_len, hidden_size]
        return cell_states

    def forward(self, encoded_states, indicator, sr_context_label):
        # get cell encodings from token sequence
        if torch.sum(sr_context_label > 0) == 0:
            return torch.tensor(0.), {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

        cell_states = self.aggr_funcs[self.aggregator](encoded_states, indicator)

        sr_context_label = sr_context_label.contiguous().view(-1)  # (b * seq_len)
        cell_states = cell_states.contiguous().view(sr_context_label.size()[0], -1)  # (b * seq_len, h)
        ca_logits = cell_states[sr_context_label > -1, :]  # (b * seq_len, h)
        sr_context_label = sr_context_label[sr_context_label > -1]

        # separator
        sep_logits = self.uniform_linear(ca_logits[0::2, :])
        sep_logits = self.tanh(sep_logits)
        sep_logits = self.predict_linear(sep_logits)
        sep_predict = sep_logits.argmax(dim=-1)
        sep_labels = sr_context_label[0:: 2]
        sep_correct = torch.sum(sep_predict.eq(sep_labels).float())
        sep_loss = self.loss(sep_logits, sep_labels)
        sep_count = torch.tensor(sep_logits.size()[0] + 1e-6)
        sep_result_dict = self.evaluate_ca_f1(sep_predict, sep_labels)

        # token-sum
        tok_logits = self.uniform_linear(ca_logits[1::2, :])
        tok_logits = self.tanh(tok_logits)
        tok_logits = self.predict_linear(tok_logits)
        tok_predict = tok_logits.argmax(dim=-1)
        tok_labels = sr_context_label[1:: 2]  # [batch-variant copied num]
        tok_correct = torch.sum(tok_predict.eq(tok_labels).float())   # scalar
        tok_loss = self.loss(tok_logits, tok_labels)  # scalar
        tok_count = torch.tensor(tok_logits.size()[0] + 1e-6)         # 1d tensor
        tok_result_dict = self.evaluate_ca_f1(tok_predict, tok_labels)

        return (sep_loss, sep_result_dict), (tok_loss, tok_result_dict)

    def evaluate_ca_f1(self, predict, labels):
        """
        Args:
            predict: (num_labels,)
            labels: (num_labels,)
        """
        result_dict = {}
        for target in range(self.num_sr_context_types):
            result_dict[target] = dict()
            precision, recall, f1, tp, fp, tn, fn = self._evaluate_ca_f1(predict, labels, target)
            result_dict[target]['precision'] = precision
            result_dict[target]['recall'] = recall
            result_dict[target]['f1'] = f1
            result_dict[target]['tp'] = tp
            result_dict[target]['fp'] = fp
            result_dict[target]['tn'] = tn
            result_dict[target]['fn'] = fn
        return result_dict

    def _evaluate_ca_f1(self, predict, labels, target):
        """
        Args:
            predict: (num_labels,)
            labels: (num_labels,)
            target: target label
        """
        tp = torch.sum(labels[predict == target] == target)
        fp = torch.sum(labels[predict == target] != target)
        tn = torch.sum(labels[predict != target] != target)
        fn = torch.sum(labels[predict != target] == target)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1, tp, fp, tn, fn



