#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Aggregated Model Frameworks for Pre-Training. """

import torch.nn as nn
import model.heads as hds
import model.backbones as bbs


# %% Pre-training Models
class TUTAbase(nn.Module):
    def __init__(self, config):
        super(TUTAbase, self).__init__()
        self.backbone = bbs.BbForBase(config)
        self.mlm_head = hds.MlmHead(config)
        self.clc_head = hds.ClcHead(config)
        self.tcr_head = hds.TcrHead(config)

    def forward(self,
                token_id, num_mag, num_pre, num_top, num_low,
                token_order, pos_top, pos_left, format_vec, indicator,
                mlm_label, clc_label, tcr_label
                ):
        encoded_states = self.backbone(
            token_id=token_id,
            num_mag=num_mag,
            num_pre=num_pre,
            num_top=num_top,
            num_low=num_low,
            token_order=token_order,
            pos_top=pos_top,
            pos_left=pos_left,
            format_vec=format_vec,
            indicator=indicator)
        mlm_triple = self.mlm_head(encoded_states, mlm_label)
        sep_triple, tok_triple = self.clc_head(encoded_states, indicator, clc_label)
        tcr_triple = self.tcr_head(encoded_states, indicator, tcr_label)
        return mlm_triple, sep_triple, tok_triple, tcr_triple


class TUTA(nn.Module):
    def __init__(self, config):
        super(TUTA, self).__init__()
        self.backbone = bbs.BACKBONES[config.target](config)
        self.mlm_head = hds.MlmHead(config)
        self.clc_head = hds.ClcHead(config)
        self.tcr_head = hds.TcrHead(config)

    def forward(self,
                token_id, num_mag, num_pre, num_top, num_low,
                token_order, pos_row, pos_col, pos_top, pos_left,
                format_vec, indicator, mlm_label, clc_label, tcr_label
                ):
        encoded_states = self.backbone(
            token_id=token_id,
            num_mag=num_mag,
            num_pre=num_pre,
            num_top=num_top,
            num_low=num_low,
            token_order=token_order,
            pos_row=pos_row,
            pos_col=pos_col,
            pos_top=pos_top,
            pos_left=pos_left,
            format_vec=format_vec,
            indicator=indicator)
        mlm_triple = self.mlm_head(encoded_states, mlm_label)
        sep_triple, tok_triple = self.clc_head(encoded_states, indicator, clc_label)
        tcr_triple = self.tcr_head(encoded_states, indicator, tcr_label)
        return mlm_triple, sep_triple, tok_triple, tcr_triple


# %% Dowsntream Models
class TUTAforCTC(nn.Module):
    def __init__(self, config):
        super(TUTAforCTC, self).__init__()
        self.backbone = bbs.BBS[config.target](config)
        self.ctc_head = hds.CtcHead(config)

    def forward(self,
                token_id, num_mag, num_pre, num_top, num_low,
                token_order, pos_top, pos_left, format_vec, indicator, ctc_label
                ):
        encoded_states = self.backbone(
            token_id, num_mag, num_pre, num_top, num_low,
            token_order, pos_top, pos_left, format_vec, indicator
        )
        sep_triple, tok_triple = self.ctc_head(encoded_states, indicator, ctc_label)
        return sep_triple, tok_triple


class TUTAbaseforCTC(nn.Module):
    def __init__(self, config):
        super(nn.Module, self).__init__()
        self.backbone = bbs.BBS[config.target](config)
        self.ctc_head = hds.CtcHead(config)

    def forward(self,
                token_id, num_mag, num_pre, num_top, num_low,
                token_order, pos_row, pos_col, pos_top, pos_left, format_vec,
                indicator, ctc_label
                ):
        encoded_states = self.backbone(
            token_id, num_mag, num_pre, num_top, num_low,
            token_order, pos_row, pos_col, pos_top, pos_left, format_vec, indicator
        )
        sep_triple, tok_triple = self.ctc_head(encoded_states, indicator, ctc_label)
        return sep_triple, tok_triple


# Formula TUTA
class FPTUTA(nn.Module):
    def __init__(self, config):
        super(FPTUTA, self).__init__()
        self.backbone = bbs.BACKBONES[config.target](config)
        self.fp_head = hds.FPHead(config)

    def get_formula_cell_states(self, encoded_states, formula_label):
        batch_size = encoded_states.size(0)
        hidden_size = encoded_states.size(-1)
        formula_cell_states = encoded_states[formula_label == 1].view(batch_size, -1, hidden_size)  # (b, 2, h)
        formula_cell_states = formula_cell_states[:, 0, :]  # (b, h)
        return formula_cell_states

    def forward(self,
                token_id, num_mag, num_pre, num_top, num_low,
                token_order, pos_row, pos_col, pos_top, pos_left,
                format_vec, indicator, formula_label, src_sketch, tgt_sketch,
                candi_cell_token_mask, range_label, range_map
                ):
        encoded_states = self.backbone(  # (b, seq_len, h)
            token_id=token_id,
            num_mag=num_mag,
            num_pre=num_pre,
            num_top=num_top,
            num_low=num_low,
            token_order=token_order,
            pos_row=pos_row,
            pos_col=pos_col,
            pos_top=pos_top,
            pos_left=pos_left,
            format_vec=format_vec,
            indicator=indicator)
        formula_cell_states = self.get_formula_cell_states(encoded_states, formula_label)  # (b, h)
        fp_tuple = self.fp_head(encoded_states, formula_cell_states, src_sketch, tgt_sketch, candi_cell_token_mask,
                                range_label, range_map)
        return fp_tuple


class FortapTUTA(nn.Module):
    def __init__(self, config):
        super(FortapTUTA, self).__init__()
        self.backbone = bbs.BACKBONES[config.target](config)
        self.mlm_head = hds.MlmHead(config)
        self.sr_head = hds.SRHead(config)
        self.nr_head = hds.NRHead(config)
        self.sr_context_head = hds.SRContextHead(config)
        self.op_mlm_head = hds.OpMLMHead(config)
        self.range_mlm_head = hds.RangeMLMHead(config)

    def forward(self,
                token_id, num_mag, num_pre, num_top, num_low,
                token_order, pos_row, pos_col, pos_top, pos_left,
                format_vec, indicator, mlm_label, sr_label, nr_label, op_appear_flag, sr_context_label,
                op_mlm_label, range_mlm_label, candi_cell_token_mask
                ):
        encoded_states = self.backbone(
            token_id=token_id,
            num_mag=num_mag,
            num_pre=num_pre,
            num_top=num_top,
            num_low=num_low,
            token_order=token_order,
            pos_row=pos_row,
            pos_col=pos_col,
            pos_top=pos_top,
            pos_left=pos_left,
            format_vec=format_vec,
            indicator=indicator)
        mlm_triple = self.mlm_head(encoded_states, mlm_label)
        sr_triple = self.sr_head(encoded_states, sr_label)
        nr_triple = self.nr_head(encoded_states, nr_label, op_appear_flag)
        sr_context_tuple, _ = self.sr_context_head(encoded_states, indicator, sr_context_label)
        op_mlm_triple = self.op_mlm_head(encoded_states, op_mlm_label)
        range_mlm_triple = self.range_mlm_head(encoded_states, candi_cell_token_mask, range_mlm_label)
        return mlm_triple, sr_triple, nr_triple, sr_context_tuple, op_mlm_triple, range_mlm_triple


MODELS = {
    "formula_prediction": FPTUTA,
    "fortap": FortapTUTA
}
