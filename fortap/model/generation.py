from typing import Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import random
import math
from icecream import ic
import sys
import os
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.realpath(__file__)))))

import fortap.tokenizer as tokenizer
import fortap.model.act_funcs as act
from fortap.constants import FP_VOCAB, REV_FP_VOCAB


def get_text(ids, generated=True, range_label=None, explicit_range=False, range_map=None):
    """
    Args
        ids: List[int], with <START> and <END>
        range_label: List[int], with <START> and <END>
    """
    tokens = []
    if generated:
        if explicit_range:
            assert range_map is not None
            for id in ids:
                if id >= 0:
                    token = REV_FP_VOCAB[id]
                else:
                    if -id in range_map:
                        token = range_map[-id]
                    else:
                        token = '<WR>'
                tokens.append(token)
        else:
            ids = [id if id >= 0 else FP_VOCAB['<RANGE>'] for id in ids]
            tokens = [REV_FP_VOCAB[id] for id in ids]
    else:  # ground truth:
        if explicit_range:
            assert range_label is not None and range_map is not None
            for i, id in enumerate(ids):
                if range_label[i] != tokenizer.DEFAULT_RANGE_LABEL:
                    token = range_map[range_label[i]]
                else:
                    token = REV_FP_VOCAB[id]
                tokens.append(token)
        else:
            tokens = [REV_FP_VOCAB[id] for id in ids]
    return " ".join(tokens)


def get_accuracy(generated_list, gt_tokens, range_label, range_map, topk=5):
    """Compute accuracy of the beam search results.
    Args:
        generated_list: List[List], w. both <START> and <END>
        gt_tokens: List[int], w. <END>, w/o. <START>
        range_label: List[int], w. <END>, w/o. <START>
        range_map: map from range_id to coord str
    """
    decode_log_f = open('decode_log.txt', 'a')
    gt_tokens = [FP_VOCAB['<START>']] + gt_tokens
    gt_tokens = [t for t in gt_tokens if t != tokenizer.FP_PAD_TAG]
    range_label = [tokenizer.DEFAULT_RANGE_LABEL] + range_label
    range_label = range_label[:len(gt_tokens)]

    # sketch acc
    sketch_cnt = 1
    generated_sketches = [get_text(s) for s in generated_list][:topk]
    gt_sketch = get_text(gt_tokens, generated=False)
    sketch_top1_acc = generated_sketches[0] == gt_sketch
    sketch_topk_acc = False
    for i in range(min(topk, len(generated_sketches))):
        if generated_sketches[i] == gt_sketch:
            sketch_topk_acc = True
    ic(gt_sketch)
    ic(generated_sketches)

    # range acc
    range_cnt = 0
    range_acc = False
    if generated_sketches[0] == gt_sketch:  # top1 range acc
        range_cnt = 1
        if len(generated_list[0]) == len(range_label):
            match = True
            for j in range(len(range_label)):
                if (range_label[j] != tokenizer.DEFAULT_RANGE_LABEL and range_label[j] != -generated_list[0][j]) \
                        or (range_label[j] == tokenizer.DEFAULT_RANGE_LABEL and gt_tokens[j] == FP_VOCAB['<RANGE>']):  # ref cells not in the same row/col
                    match = False
                    break
            range_acc = match

    # fp acc
    fp_cnt = 1
    generated_formulas = [get_text(s, generated=True, range_label=None, explicit_range=True, range_map=range_map) for s in generated_list][:topk]
    gt_formula = get_text(gt_tokens, generated=False, range_label=range_label, explicit_range=True, range_map=range_map)
    fp_top1_acc = generated_formulas[0] == gt_formula
    fp_topk_acc = False
    for i in range(min(topk, len(generated_formulas))):
        if generated_formulas[i] == gt_formula:
            fp_topk_acc = True
    ic(gt_formula)
    ic(generated_formulas)
    print()

    decode_log_f.write(f"{str(fp_top1_acc)}\n")
    decode_log_f.write(f"generated sketch: {generated_sketches[0]}\n")
    decode_log_f.write(f"gt sketch: {gt_sketch}\n")
    decode_log_f.write(f"generated formulas: {generated_formulas[0]}\n")
    decode_log_f.write(f"gt formulas: {gt_formula}\n")
    decode_log_f.write('\n')

    return sketch_cnt, sketch_top1_acc, sketch_topk_acc, range_cnt, range_acc, fp_cnt, fp_top1_acc, fp_topk_acc


class BeamSearch(object):
    def __init__(self, beam_size, beam_alpha, beam_gamma, ideal_length, max_length, grammar_restriction=True):
        self.beam_size, self.beam_alpha, self.beam_gamma, self.ideal_length, self.max_length = beam_size, beam_alpha, beam_gamma, ideal_length, max_length
        # self.GRAMMAR_IDS = tokenizer.GRAMMAR_IDS
        self.grammar_restriction = grammar_restriction

    def get_seq_score(self, seq, log_prob, length):
        """Calculate sequence score for ordering sequences."""
        return log_prob / ((length + 1) ** self.beam_alpha) + self.beam_gamma * self.ideal_length / length

    def grammar_check(self, seq, log_prob, grammar_info):
        """Avoid grammatically incorrect generated sequence."""
        valid = True
        if self.grammar_restriction:
            log_prob[tokenizer.FP_PAD_TAG] = float("-inf")
            last_tok = grammar_info["last_tok"]
            assert seq[-1] == last_tok
            parent_num = grammar_info["parenthesis"]
            bracket_num = grammar_info["brackets"]
            if parent_num <= 0:
                for token in self.GRAMMAR_IDS[")"]:
                    log_prob[token] = float("-inf")
            if bracket_num <= 0:
                for token in self.GRAMMAR_IDS["]"]:
                    log_prob[token] = float("-inf")
            for i in range(len(seq) - 3):  # avoid (([RANGE]))
                if valid and seq[i] in self.GRAMMAR_IDS["("] and seq[i + 1] in self.GRAMMAR_IDS["("] and seq[
                    i + 2] not in self.GRAMMAR_IDS["("]:
                    no_parenthesis = True
                    for j in range(i + 2, len(seq) - 1):
                        if no_parenthesis and seq[j] in self.GRAMMAR_IDS[")"] and seq[j + 1] in self.GRAMMAR_IDS[")"]:
                            valid = False
                            break
                        if seq[j] in self.GRAMMAR_IDS["("] or seq[j] in self.GRAMMAR_IDS[")"]:
                            no_parenthesis = False
        return log_prob, valid

    def get_grammar_info(self, grammar_info, new_token):
        if new_token in self.GRAMMAR_IDS["("]:
            grammar_info["parenthesis"] += 1
        elif new_token in self.GRAMMAR_IDS[")"]:
            grammar_info["parenthesis"] -= 1
        elif new_token in self.GRAMMAR_IDS["["]:
            grammar_info["brackets"] += 1
        elif new_token in self.GRAMMAR_IDS["]"]:
            grammar_info["brackets"] -= 1
        grammar_info["last_tok"] = new_token
        return grammar_info

    def beam_search(self, seq, sketch_logits_func, range_logits_func, encoded_states, candi_cell_token_mask,
                    formula_cell_states, last_token, sketch_only):
        # (seq, log_prob, length, normalized_log_prob)
        init_grammar_info = {"last_tok": FP_VOCAB['<START>'],
                             "parenthesis": 0, "brackets": 0}
        sequences = [(seq, 0, 1, init_grammar_info, 0)]
        finding_num = self.beam_size
        length = 1
        sketch_ids = []
        while finding_num > 0 and length < self.max_length:
            new_sequences = []
            for seq, log_prob, cur_len, grammar_info, seq_score in sequences:
                logits = sketch_logits_func(seq[None, :], encoded_states, formula_cell_states, last_token)
                if type(logits) == tuple:
                    logits = logits[0]
                lprobs = F.log_softmax(
                    logits, dim=-1).view(logits.size(-1))  # [vocab_size]
                topk = lprobs.topk(self.beam_size)
                values = topk.values
                indices = topk.indices
                for beam_i in range(self.beam_size):
                    new_seq = torch.cat((seq, indices[beam_i].unsqueeze(0)), 0)
                    grammar_info = {}
                    seq_score = self.get_seq_score(
                        new_seq, log_prob + values[beam_i], cur_len + 1)
                    new_sequences.append(
                        (new_seq, log_prob + values[beam_i], cur_len + 1, grammar_info, seq_score))
            new_sequences.sort(key=lambda x: x[-1], reverse=True)
            sequences = []
            for info in new_sequences[:self.beam_size]:
                if int(info[0][-1]) == FP_VOCAB['<END>']:
                    finding_num -= 1
                    prob = torch.exp(info[1])
                    sketch_ids.append((prob, info[0]))
                else:
                    sequences.append(info)
            length += 1

        if not sketch_ids:
            for seq, log_prob, length, grammar_info, seq_score in sequences:
                prob = torch.exp(log_prob)
                sketch_ids.append((prob, seq))
                # final_ids.append((prob, torch.tensor([0, 1]).cuda(1)))

        # tensor id to int id
        sketch_ids_int = []
        for prob, seq in sketch_ids:
            if seq.size() == torch.Size([]):
                seq = [seq.item()]
            elif seq.size(0) > 1:
                seq = seq.detach().cpu().tolist()
            sketch_ids_int.append((prob, seq))

        if sketch_only:
            return sketch_ids_int

        final_ids_int = []
        for prob, sketch in sketch_ids:
            final_ids_int.append((prob, []))
            src_sketch = sketch
            if int(sketch[-1]) == FP_VOCAB['<END>']:
                src_sketch = sketch[:-1]
            sketch_logits, sketch_hidden = sketch_logits_func(src_sketch[None, :], encoded_states,
                                                              formula_cell_states)  # (1, max_sketch_len, tgt_vocab), (1, max_sketch_len, 2*h)
            range_logits, range_hidden = range_logits_func(sketch_hidden, encoded_states, candi_cell_token_mask,
                                                           formula_cell_states)  # (1, max_sketch_len, seq_len), (1, max_sketch_len, h)
            for idx, fp_id in enumerate(src_sketch):
                if int(fp_id) == FP_VOCAB['<RANGE>']:
                    range_prob, range_id = torch.max(F.softmax(range_logits.squeeze(0)[idx - 1], dim=0),  # idx-1 since decoder output is one step earlier than input
                                                     dim=0)  # idx of tok in seq
                    final_ids_int[-1][1].append(-range_id.item())
                else:
                    final_ids_int[-1][1].append(fp_id.item())
            final_ids_int[-1][1].append(FP_VOCAB['<END>'])

        return final_ids_int


class LMbase(nn.Module):
    """Base class for language model."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.padding_idx = tokenizer.FP_PAD_TAG
        self.tokenizer = args.tokenizer
        self.embedding = nn.Embedding(
            len(FP_VOCAB) + 1, args.hidden_size, padding_idx=tokenizer.FP_PAD_TAG)
        self.embedding.padding_idx = self.padding_idx
        self.bs = BeamSearch(self.args.beam_size, self.args.beam_alpha,
                             self.args.beam_gamma, self.args.ideal_length, self.args.max_length)

    def forward(self, encoded_states, formula_cell_states, src_sketch, tgt_sketch, candi_cell_token_mask, range_label,
                reduce=True, **unused):
        """
        Args:
            encoded_states: (b, seq_len, h)
            formula_cell_states: (b, h)
            src_sketch: (b, max_sketch_len)
            tgt_sketch: (b, max_sketch_len)
            candi_cell_token_mask: (b, seq_len)
            range_label: (b, max_sketch_len)
        """
        sketch_logits, sketch_hidden = self.sketch_logits(  # (b, max_sketch_len, tgt_vocab), (b, max_sketch_len, 2*h)
            src_sketch, encoded_states, formula_cell_states)
        sketch_lprobs = F.log_softmax(sketch_logits, dim=-1).view(-1, sketch_logits.size(
            -1))  # (b*max_sketch_len, tgt_vocab)
        range_logits, range_hidden = self.range_logits(  # (b, max_sketch_len, seq_len), (b, max_sketch_len, h)
            sketch_hidden, encoded_states, candi_cell_token_mask, formula_cell_states
        )
        range_lprobs = F.log_softmax(range_logits, dim=-1).view(-1,
                                                                range_logits.size(-1))  # (b*max_sketch_len, seq_len)

        sketch_loss = F.nll_loss(
            sketch_lprobs,
            tgt_sketch.view(-1),
            reduction="mean",
            ignore_index=self.padding_idx,
        )
        range_label_flat = range_label.view(-1).clone()
        range_label_flat[range_label_flat >= self.args.max_seq_len - 1] = tokenizer.DEFAULT_RANGE_LABEL
        range_loss = F.nll_loss(
            range_lprobs,
            range_label_flat,
            reduction="mean",
            ignore_index=tokenizer.DEFAULT_RANGE_LABEL
        )
        return sketch_loss, range_loss

    @torch.no_grad()
    def generate(self, encoded_states, formula_cell_states, target, candi_cell_token_mask, range_label, range_map):
        """
        Args:
            encoded_states: (b, seq_len, h)
            formula_cell_states: (b, h)
            candi_cell_token_mask: (b, seq_len)
            target: List[int] of length 'max_sketch_len'
            range_label: List[int]
            range_map: Dict[int, str]
        """
        device = formula_cell_states.device
        seq = torch.LongTensor([self.tokenizer.fp_tok2id('<START>')]).to(
            device)
        bs_gen_tokens = self.bs.beam_search(
            seq, self.sketch_logits, self.range_logits, encoded_states, candi_cell_token_mask, formula_cell_states,
            last_token=True, sketch_only=False)
        gen_tokens = []
        gen_tokens.extend(bs_gen_tokens)

        # self.tmp_encoded_states = encoded_states
        # self.tmp_formula_cell_states = formula_cell_states
        gen_tokens.sort(key=lambda x: x[0], reverse=True)
        gen_tokens = [t[1] for t in gen_tokens]  # List[List[int]]
        # ic(gen_tokens)

        gen_text = []
        for ids in gen_tokens:
            text = get_text(ids, explicit_range=True, range_map=range_map)
            gen_text.append(text)

        sketch_cnt, sketch_top1_acc, sketch_topk_acc, range_cnt, range_acc, fp_cnt, fp_top1_acc, fp_topk_acc\
            = get_accuracy(gen_tokens, target, range_label, range_map, topk=5)
        return gen_text, gen_tokens, sketch_cnt, sketch_top1_acc, sketch_topk_acc, range_cnt, range_acc, \
               fp_cnt, fp_top1_acc, fp_topk_acc


# LSTM
class LSTMLM(LMbase):

    def __init__(self, args):
        super().__init__(args)
        self.decoder = torch.nn.LSTM(args.hidden_size, args.hidden_size, num_layers=args.LSTM_num_layers,
                                            batch_first=True, bidirectional=False)
        self.out_proj = nn.Linear(
            2 * args.hidden_size, len(FP_VOCAB))
        self.attn = nn.MultiheadAttention(
            args.gen_hidden_size, args.gen_num_attention_heads)
        self.range_encoder = torch.nn.LSTM(2 * args.hidden_size, args.hidden_size, num_layers=args.LSTM_num_layers,
                                           batch_first=True, bidirectional=False)
        self.range_affine = nn.Linear(args.hidden_size, args.hidden_size)

    def sketch_logits(self, source, encoded_states, formula_cell_states, last_token=False,
                      **unused):
        seq_emb = self.embedding(source)  # (b, max_len, h)
        h0 = formula_cell_states[None, :, :].repeat(
            self.args.LSTM_num_layers, 1, 1)
        c0 = torch.zeros_like(h0).to(seq_emb.device)
        seq_dec, (hn, cn) = self.decoder(seq_emb, (h0, c0))
        if "attn" in self.args.generation_model:
            seq_dec = seq_dec.transpose(0, 1)
            """ LM-style
            mask = ~torch.tril(torch.ones((seq_emb.size(1), seq_emb.size(
                1)), dtype=bool)).to(self.out_proj.weight.device)  # TODO: not sure
            atten, _ = self.attn(seq_dec, seq_dec,
                                 seq_dec, attn_mask=mask)
            """
            # seq2seq-style
            atten, _ = self.attn(seq_dec, encoded_states, encoded_states)
            atten = atten.transpose(0, 1)  # (b, seq_len, h)
            seq_dec = seq_dec.transpose(
                0, 1)  # [bsz, seq_len, hidden_size=768]

            # [bsz, seq_len, 2 * hidden_size=1536]
            output = torch.cat((atten, seq_dec), -1)
        else:
            output = torch.cat((seq_dec, seq_dec), -1)
        if last_token:
            output = output[:, -1:, :]
        logits = self.out_proj(output)
        return logits, output

    def range_logits(self, sketch_hidden, encoded_states, candi_cell_token_mask, formula_cell_states):
        """
        Args:
            sketch_hidden: (b, max_sketch_len, 2*h)
            encoded_states: (b, seq_len, h)
            candi_cell_token_mask: (b, seq_len)
            formula_cell_states: (b, h)
        """
        h0 = formula_cell_states[None, :, :].repeat(  # (n, b, h), note 'b' is not at the first
            self.args.LSTM_num_layers, 1, 1
        )
        c0 = torch.zeros_like(h0).to(sketch_hidden.device)  # (n, b, h)
        range_hidden, (hn, cn) = self.range_encoder(sketch_hidden,
                                                    (h0, c0))  # (b, max_sketch_len, h)
        range_logits = torch.bmm(self.range_affine(range_hidden),
                                 encoded_states.transpose(2, 1))  # (b, max_sketch_len, seq_len)
        candi_cell_token_mask = candi_cell_token_mask.unsqueeze(1).repeat(1, range_logits.size(1),
                                                                          1)  # (b, max_sketch_len, seq_len)
        range_logits = range_logits.masked_fill(candi_cell_token_mask == 0, -1e9)
        return range_logits, range_hidden


# # Transformer
# ACT2FN = {
#     "relu": F.relu,
#     "gelu": F.gelu,
#     "tanh": torch.tanh,
#     "sigmoid": torch.sigmoid,
# }


class TransformerLM(LMbase):

    def __init__(self, args):
        super().__init__(args)
        self.args.max_position_embeddings = 100
        self.args.pad_token_id = tokenizer.PAD_ID
        self.decoder = TransformerDecoder(self.args, self.embedding)
        self.out_linear = nn.Linear(
            args.hidden_size + args.hidden_size, args.hidden_size + args.hidden_size)
        self.out_proj = nn.Linear(
            args.hidden_size + args.hidden_size, len(self.tokenizer.vocab))
        self.activation_fn = act.ACT_FCN[args.hidden_act]

    def logits(self, source, all_states, formula_cell_states, last_token=False, **unused):
        # seq2seq-style

        # place [PAD] before source sequence. embedding of [PAD] will be replaced by formula_cell_states
        extra_form_seq = source.new_zeros((source.size(0), 1))

        source = torch.cat((extra_form_seq, source), 1)
        source, decoder_padding_mask, causal_mask = _prepare_decoder_inputs(
            source, causal_mask_dtype=torch.float)
        hidden = self.decoder(
            source,
            all_states,
            encoder_padding_mask=None,
            decoder_padding_mask=decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            formula_cell_states=formula_cell_states
        )
        formula_cell_states = formula_cell_states[:, None, :].repeat(
            1, hidden.size(1), 1)
        hidden = torch.cat((hidden, formula_cell_states), -1)

        if last_token:
            hidden = hidden[:, -1:, :]
        else:
            hidden = hidden[:, 1:, :]

        logits = self.out_proj(self.activation_fn(self.out_linear(hidden)))
        return logits, hidden


def make_padding_mask(input_ids, padding_idx=tokenizer.PAD_ID):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _prepare_decoder_inputs(decoder_input_ids, causal_mask_dtype=torch.float
                            ):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = tokenizer.PAD_ID

    bsz, tgt_len = decoder_input_ids.size()

    decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)

    # never mask leading token, even if it is pad
    if decoder_padding_mask is not None and decoder_padding_mask.shape[1] > 1:
        decoder_padding_mask[:, 0] = decoder_padding_mask[:, 1]

    tmp = fill_with_neg_inf(torch.zeros(tgt_len, tgt_len))
    mask = torch.arange(tmp.size(-1))
    tmp.masked_fill_(mask < (mask + 1).view(tmp.size(-1), 1), 0)
    causal_mask = tmp.to(dtype=causal_mask_dtype,
                         device=decoder_input_ids.device)
    return decoder_input_ids, decoder_padding_mask, causal_mask


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        padding_idx = tokenizer.PAD_ID
        self.shared = nn.Embedding(
            len(self.tokenizer.vocab), config.hidden_size, padding_idx)

        self.encoder = TransformerEncoder(config, self.shared)
        self.decoder = TransformerDecoder(config, self.shared)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
            self,
            input_ids,
            decoder_input_ids=None,
            attention_mask=None,
            decoder_attention_mask=None,
            encoder_outputs: Optional[Tuple] = None,
            **kwargs,
    ):

        # make masks if user doesn't supply
        decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_decoder_inputs(
            input_ids,
            decoder_input_ids=decoder_input_ids,
            causal_mask_dtype=self.shared.weight.dtype,
        )

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs,
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
        )

        return decoder_outputs


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(
            self.embed_dim, config.gen_num_attention_heads, dropout=config.attention_dropout_prob)
        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.hidden_dropout_prob
        self.activation_fn = act.ACT_FCN[config.hidden_act]

        # self.activation_fn = ACT2FN[config.hidden_act]
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.fc1 = nn.Linear(self.embed_dim, config.gen_hidden_size)
        self.fc2 = nn.Linear(config.gen_hidden_size, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x, key=x, key_padding_mask=encoder_padding_mask
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.hidden_dropout_prob, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if torch.isinf(x).any() or torch.isnan(x).any():
            clamp_value = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config:
    """

    def __init__(self, config, embed_tokens):
        super().__init__()

        self.dropout = config.hidden_dropout_prob
        self.layerdrop = config.hidden_dropout_prob

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(
            embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings,
                embed_dim,
                self.padding_idx,
                config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList([EncoderLayer(config)
                                     for _ in range(config.gen_num_layers)])
        self.layernorm_embedding = LayerNorm(
            embed_dim) if config.normalize_embedding else nn.Identity()

        self.layer_norm = LayerNorm(
            config.hidden_size) if config.add_final_layer_norm else None

    def forward(
            self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        """
        # check attention mask and invert

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.transpose(0, 1)  # B x T x C -> T x B x C

        for encoder_layer in self.layers:

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                continue
            else:
                x = encoder_layer(x, attention_mask)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = Attention(
            embed_dim=self.embed_dim,
            num_heads=config.gen_num_attention_heads,
            dropout=config.attention_dropout_prob,
        )
        self.dropout = config.hidden_dropout_prob
        self.activation_fn = act.ACT_FCN[config.hidden_act]
        # self.activation_fn = ACT2FN[config.hidden_act]
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = Attention(
            self.embed_dim,
            config.gen_num_attention_heads,
            dropout=config.attention_dropout_prob,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.gen_hidden_size)
        self.fc2 = nn.Linear(config.gen_hidden_size, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
            self,
            x,
            encoder_hidden_states,
            encoder_attn_mask=None,
            causal_mask=None,
            decoder_padding_mask=None,
    ):
        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention

        x = self.self_attn(
            query=x,
            key=x,
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.hidden_dropout_prob, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.hidden_dropout_prob
        self.layerdrop = config.hidden_dropout_prob
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(
            config.hidden_size) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.hidden_size, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings,
                config.hidden_size,
                self.padding_idx,
                config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.gen_num_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(
            config.hidden_size) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(
            config.hidden_size) if config.add_final_layer_norm else None

    def forward(
            self,
            input_ids,
            encoder_hidden_states,
            encoder_padding_mask,
            decoder_padding_mask,
            decoder_causal_mask,
            formula_cell_states=None,
            **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `[bsz, seq_len]`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_padding_mask: for ignoring pad tokens
            decoder_causal_mask: mask the future tokens
        """

        # embed positions
        positions = self.embed_positions(input_ids)

        x = self.embed_tokens(input_ids)  # [bsz, seq_len, hidden_size]
        # place formula_cell_states at the embedding of [PAD] (before the embedding of [FORM_START])
        if formula_cell_states is not None:
            x[:, 0, :] = formula_cell_states

        x = x * self.embed_scale

        x += positions
        x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        # decoder layers

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            x = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                causal_mask=decoder_causal_mask,
            )

        if self.layer_norm:
            x = self.layer_norm(x)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        return x


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            encoder_decoder_attention=False,  # otherwise self_attention
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * \
               num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    def _shape(self, tensor, seq_len, bsz):
        # change to (bsz * self.num_heads, seq_len, self.head_dim)
        return tensor.contiguous().view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
            self,
            query,
            key: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        tgt_len, bsz, _ = query.size()

        q = self.scaling * self.q_proj(query)
        if self.encoder_decoder_attention:
            k = self.k_proj(key)
            v = self.v_proj(key)
        else:  # self-attention
            k = self.k_proj(query)
            v = self.v_proj(query)

        assert q is not None
        assert k is not None
        assert v is not None
        q = self._shape(q, tgt_len, bsz)
        k = self._shape(k, -1, bsz)
        # (bsz * self.num_heads, src_len, self.head_dim)
        v = self._shape(v, -1, bsz)
        src_len = k.size(1)

        # (bsz * self.num_heads, tgt_len, src_len)
        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention.view(bsz, self.num_heads, tgt_len, src_len)

        if attn_mask is not None:  # [26, 26], -inf
            attention = attn_mask + attention

        if key_padding_mask is not None:
            if key_padding_mask.size(0) == key_padding_mask.size(1):
                reshaped = ~key_padding_mask.unsqueeze(0).unsqueeze(1)
            else:
                reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # debug(attention=attention)
            attention = attention.masked_fill(
                reshaped, float("-inf"))  # [128, 8, 26, 26]
            # debug(copy=attention)
        attention = attention.view(bsz * self.num_heads, tgt_len, src_len)
        attention = F.softmax(attention, dim=-1)

        attention_prob = F.dropout(
            attention, p=self.dropout, training=self.training)
        attention_out = torch.bmm(attention_prob, v)
        attention_out = attention_out.transpose(0, 1).contiguous().view(
            tgt_len, bsz, self.embed_dim)
        return self.out_proj(attention_out)


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, offset):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = offset
        assert padding_idx is not None
        num_embeddings += offset
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]

        positions = torch.arange(
            seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions + self.offset)


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        if embedding_dim % 2 != 0:
            raise NotImplementedError(
                f"odd embedding_dim {embedding_dim} not supported")
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """Identical to the XLM create_sinusoidal_embeddings except features are not interleaved.
        The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim)
              for j in range(dim)] for pos in range(n_pos)]
        )
        # This line breaks for odd n_pos
        out[:, 0: dim // 2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, dim // 2:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        out.requires_grad = False
        return out

    @torch.no_grad()
    def forward(self, input_ids):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]

        # starts at 0, ends at 1-seq_len
        positions = torch.arange(
            seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)
