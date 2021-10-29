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


MODELS = {
    "tuta": TUTA, 
    "tuta_explicit": TUTA, 
    "base": TUTAbase
}



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

    
class TUTAForTTC(nn.Module):
    def __init__(self, config):
        super(TutaForTTC, self).__init__()
        self.backbone = bbs.BACKBONES[config.target](config)
        self.ttc_head = hds.TtcHead(config)

    def forward(
        self, 
        token_id, num_mag, num_pre, num_top, num_low, 
        token_order, pos_row, pos_col, pos_top, pos_left, 
        format_vec, indicator, ttc_label
    ):
        encoded_states = self.backbone(
            token_id, num_mag, num_pre, num_top, num_low, 
            token_order, pos_row, pos_col, pos_top, pos_left, 
            format_vec, indicator
        )
        loss, prediction = self.ttc_head(encoded_states, ttc_label)

        return loss, prediction, ttc_label

class TUTAbaseforTTC(nn.Module):
    def __init__(self, config):
        super(TUTAbaseforTTC, self).__init__()
        self.backbone = bbs.BACKBONES[config.target](config)
        self.ttc_head = hds.TtcHead(config)

    def forward(
        self, 
        token_id, num_mag, num_pre, num_top, num_low, 
        token_order, pos_top, pos_left, 
        format_vec, indicator, ttc_label
    ):
        encoded_states = self.backbone(
            token_id, num_mag, num_pre, num_top, num_low, 
            token_order, pos_top, pos_left, 
            format_vec, indicator
        )
        loss, prediction = self.ttc_head(encoded_states, ttc_label)

        return loss, prediction, ttc_label

    
