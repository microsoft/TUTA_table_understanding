#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tabular Cell Type Classification
"""

from ast import parse
import os
import json
import random
import argparse
import json_lines
import torch
import torch.nn as nn

import utils as ut
import reader as rdr
import tokenizer as tknr
import model.backbones as bbs
import model.act_funcs as act
import model.pretrains as ptm
from optimizer import AdamW
import sys
random.seed(0)
torch.manual_seed(0)


# %% Reader and Tokenizer Classes
class TctcReader(rdr.SheetReader):
    def __init__(self, args):
        self.tree_depth = args.tree_depth
        self.node_degree = args.node_degree
        self.row_size = args.row_size
        self.column_size = args.column_size
        self.target = args.target

    def get_inputs(self, hier_table):
        cell_matrix, merged_regions = hier_table["Cells"], hier_table["MergedRegions"]
        row_number, column_number = len(cell_matrix), len(cell_matrix[0])
        if (row_number > self.row_size) or (column_number > self.column_size):
            # print("Fail for extreme sizes: {} rows, {} columns ".format(row_number, column_number))
            return None

        try:
            top_root, left_root = hier_table["TopTreeRoot"], hier_table["LeftTreeRoot"]
            top_position_list, left_position_list = None, None
            top_position_list = self.read_header(top_root, merged_regions, row_number, column_number, True)
            left_position_list = self.read_header(left_root, merged_regions, row_number, column_number, False)
            if top_position_list is None:
                top_position_list = self.read_header(None, merged_regions, row_number, column_number, True)
            if left_position_list is None:
                left_position_list = self.read_header(None, merged_regions, row_number, column_number, False)
            if top_position_list is None:
                print("Top Position List is None!")
            if left_position_list is None:
                print("Left Position List is None!")
        except:
            print("Error in read header. ")
            return None

        string_matrix, format_matrix = self.info_from_matrix(cell_matrix, merged_regions)
        header_rows, header_columns = hier_table["TopHeaderRowsNumber"], hier_table["LeftHeaderColumnsNumber"]
        return string_matrix, (top_position_list, left_position_list), (header_rows, header_columns), format_matrix


# %% Tokenizer Class
class TctcTokenizer(tknr.TableTokenizer):
    def no_sampling(self, token_matrix):
        sampling_mask = [[1 for _ in token_matrix[0]] for _ in token_matrix]
        return sampling_mask

    # def simple_sampling(self, token_matrix, label_matrix, sample_rate=[0.0, 0.0, 0.8, 0.0, 0.0, 0.1]):
    def simple_sampling(self, token_matrix, label_matrix, sample_rate=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        sampling_mask = [[1 for _ in token_matrix[0]] for _ in token_matrix]
        for irow, token_row in enumerate(token_matrix):
            for icol, tokens in enumerate(token_row):
                if (tknr.EMP_ID in tokens):    # if the cell is empty: [102, 2] 102 means CLS
                    tctc_label = -1
                tctc_label = label_matrix[irow][icol]
                
                for i, sr in enumerate(sample_rate):
                    if (tctc_label == i) and (random.random() < sr):
                        sampling_mask[irow][icol] = 0
        
        return sampling_mask

# %% Tokenizer Class
class TctcTok(TctcTokenizer):
    def init_table_seq(self, root_context=""):
        """Initialize table sequence with CLS_ID at head, add context if provided. """
        context_tokens, context_number = self.tokenize_text(cell_string=root_context, add_separate=False, max_cell_len=8)
        token_list = [ [tknr.CLS_ID] + context_tokens ]
        num_list = [ [self.wordpiece_tokenizer.default_num] + context_number ]
        pos_list = [ (self.row_size, self.column_size, [-1] * self.tree_depth, [-1] * self.tree_depth) ] 
        format_list = [ self.default_format ]
        ind_list = [ [-1] + [-2 for _ in context_tokens] ]
        label_list = [[-1] + [-1 for ct in context_tokens]]
        cell_num = 1
        seq_len = len(token_list[0])
        return token_list, num_list, pos_list, format_list, ind_list, label_list, cell_num, seq_len

    def create_table_seq(self, sampling_matrix, token_matrix, number_matrix, position_lists, format_matrix, label_matrix, 
                               range, max_seq_len, max_cell_length, add_separate=True):
        seqs = []
        label_dict = {}
        start_row = 0
        # spit tables exceeded the maximum length to smaller ones
        while start_row < len(token_matrix):
            token_list, num_list, pos_list, format_list, ind_list, label_list, cell_num, seq_len = self.init_table_seq(root_context="")
            top_pos_list, left_pos_list = position_lists
            top, bottom, left, right = range
            icell = 0
            mark_exceed_len = False
            for irow, token_row in enumerate(token_matrix):
                if mark_exceed_len: break
                if irow < start_row:continue
                for icol, token_cell in enumerate(token_row):
                    if sampling_matrix[irow][icol] == 0:
                        continue
                    token_cell = token_cell[:max_cell_length]
                    cell_len = len(token_cell)
                    if cell_len + seq_len >= max_seq_len:
                        if irow - 3 > start_row:
                            start_row = irow - 3
                        elif irow - 2 > start_row:
                            start_row = irow - 2
                        elif irow - 1 > start_row:
                            start_row = irow - 1
                        elif irow > start_row:
                            start_row = irow
                        else:
                            start_row = irow + 1
                        seqs.append([token_list, num_list, pos_list, format_list, ind_list, label_list])
                        mark_exceed_len = True
                        break
                    if (top <= irow <= bottom) and (left <= icol <= right):
                        pos_list.append( (irow, icol, top_pos_list[icell], left_pos_list[icell]) )
                        icell += 1
                        format_vector = []
                        for ivec, vec in enumerate(format_matrix[irow-top][icol-left]):
                            format_vector.append( min(vec, self.format_range[ivec]) / self.format_range[ivec] )
                        format_list.append( format_vector )
                    else:
                        if irow < top:
                            pos_list.append( (irow, icol, [-1,31,31,irow%256], [-1,31,31,icol%256]))
                        else:
                            pos_list.append( (irow, icol, [-1,31,63,irow%256], [-1,31,63,icol%256]))
                        format_list.append( self.default_format )
                    token_list.append( token_cell )
                    num_list.append( number_matrix[irow][icol][:cell_len] )
                    ind_list.append(  [cell_num*2] * cell_len )

                    tctc_label = label_matrix[irow][icol]
                    if str(irow) + "_" + str(icol) in label_dict: tctc_label = -1
                    label_list.append( [-1 for _ in token_cell] )
                    label_list[-1][0] = tctc_label
                    if add_separate == True:
                        ind_list[-1][0] -= 1
                        label_list[-1][1] = tctc_label
                    label_dict[str(irow) + "_" + str(icol)] = True
                    seq_len += cell_len
                    cell_num += 1
            if mark_exceed_len: continue
            seqs.append([token_list, num_list, pos_list, format_list, ind_list, label_list])
            start_row = len(token_matrix)
        return seqs



# Utility functions
def map_annotations_to_labels(
    annotations, 
    mapping_dict={"metadata": 0, "notes": 1, "data": 2, "attributes": 3, "header": 4, "derived": 5, None: -1}
):
    label_matrix = []
    for anno_row in annotations:
        label_matrix.append( [mapping_dict[anno] for anno in anno_row] )
    return label_matrix

def str2col(column_str, offset=ord("A"), multiplier=26):
    column_index = 0
    for char in column_str:
        column_index *= multiplier
        column_index += ord(char) - offset + 1
    # print("convert column from str {} to index {}".format(column_str, column_index))
    return column_index

def separate_str_digit(position):
    i, L = 0, len(position)
    while (i < L) and (not position[i].isdigit()):
        i += 1
    row = int(position[i: ])
    column = str2col(position[: i])
    return row, column

def parse_range(range_address):
    top_lft, btm_rgt = [item.strip() for item in range_address.strip().split(':')]   # ["A5", "CB43"] --> [(5,1)]
    top, left = separate_str_digit(top_lft)
    bottom, right = separate_str_digit(btm_rgt)
    range = [top-1, bottom-1, left-1, right-1]
    # print("Parse Address {} into {}".format(range_address, range))
    return range



def lists_to_inputs(lists, target, max_seq_len, args):
    token_list, num_list, pos_list, format_list, ind_list, label_list = lists

    token_id, num_mag, num_pre, num_top, num_low = [], [], [], [], []
    token_order, pos_row, pos_col, pos_top, pos_left = [], [], [], [], []
    format_vec, indicator, tctc_labels = [], [], []

    for tokens, num_feats, (row, col, ttop, tleft), fmt, ind, label in zip(token_list, num_list, pos_list, format_list, ind_list, label_list):
        cell_len = len(tokens)
        token_id.extend(tokens)
        num_mag.extend([f[0] for f in num_feats])
        num_pre.extend([f[1] for f in num_feats])
        num_top.extend([f[2] for f in num_feats])
        num_low.extend([f[3] for f in num_feats])

        token_order.extend([ii for ii in range(cell_len)])
        pos_row.extend([row for _ in range(cell_len)])
        pos_col.extend([col for _ in range(cell_len)])
        entire_top = ut.UNZIPS[target](zipped=ttop, node_degree=args.node_degree, total_node=args.total_node)
        pos_top.extend([entire_top for _ in range(cell_len)])
        entire_left = ut.UNZIPS[target](zipped=tleft, node_degree=args.node_degree, total_node=args.total_node)
        pos_left.extend([entire_left for _ in range(cell_len)])

        format_vec.extend( [fmt for _ in range(cell_len)] )
        indicator.extend(ind)
        tctc_labels.extend(label)
    
    if (len(token_id) > max_seq_len) or (max(tctc_labels) == -1):
        # print(":( current sequence length {} exceeds the upper bound {}".format(len(token_id), max_seq_len))
        return None
    # print(":) current sequence length {} meets the upper bound {}".format(len(token_id), max_seq_len))
    return (token_id, num_mag, num_pre, num_top, num_low, token_order, pos_row, pos_col, pos_top, pos_left, format_vec, indicator, tctc_labels)


def create_sample(hier_table, flat_table, reader, tokenizer, args):
    # get label_matrix from flat_table
    
    table_range = parse_range(hier_table["RangeAddress"])
    reader_output = reader.get_inputs(hier_table)
    if reader_output is None:
        return [None]
    string_matrix = flat_table["table_array"]
    label_matrix = map_annotations_to_labels(flat_table["annotations"])
    substr_matrix, position_lists, header_info, format_matrix = reader_output
    # add formula prefix to cell string
    formula_cell_prefix = "=#"
    for i in range(len(format_matrix)):
        for j in range(len(format_matrix[0])):
            if format_matrix[i][j][7] > 0:
                string_matrix[i][j] = formula_cell_prefix + string_matrix[i][j]
    merge_string_matrix = [["" for _ in string_matrix[0]] for _ in string_matrix]
    for i , _ in enumerate(string_matrix):
        for j, _ in enumerate(string_matrix[0]):
            merge_string_matrix[i][j] =  string_matrix[i][j]
            try:
                if len(substr_matrix[i][j]) > len(string_matrix[i][j]):
                    merge_string_matrix[i][j] = substr_matrix[i][j]   
            except:
                pass
    token_matrix, number_matrix = tokenizer.tokenize_string_matrix(
        string_matrix=merge_string_matrix, add_separate=True, max_cell_len=args.max_seq_len*1000
    )
    sampling_matrix = tokenizer.simple_sampling(token_matrix, label_matrix)
    input_tuples = []
    seqs = tokenizer.create_table_seq(
            sampling_matrix=sampling_matrix, 
            token_matrix=token_matrix, 
            number_matrix=number_matrix, 
            position_lists=position_lists, 
            format_matrix=format_matrix, 
            label_matrix=label_matrix, 
            range=table_range, max_seq_len=args.max_seq_len, max_cell_length=args.max_cell_length
        )
        
    for seq in seqs:
        token_list, num_list, pos_list, format_list, ind_list, label_list = seq
        input_tuple = lists_to_inputs((token_list, num_list, pos_list, format_list, ind_list, label_list), args.target, sys.maxsize, args)
        input_tuples.append(input_tuple)
    return input_tuples


def create_hier_id(blobname, sheetname):
    blobname = blobname.split('.')[0]
    unique_id = blobname + '-' + sheetname
    return unique_id


def build_datadict(flat_repo, hier_dir, reader, tokenizer, args):    # key table-sheet 的id， value 对应的输入数据
    datadict = {}
    files = [os.path.join(hier_dir, f) for f in os.listdir(hier_dir)]
    for file in files:
        with open(file, "r") as fr_hier:
            hier_table = json.load(fr_hier)
            blobname, sheetname = hier_table["BlobName"], hier_table["SheetName"]
            blobname = blobname.split('.')[0]
            if "/" in blobname:
                blobname = blobname.split("/")[1]
            for flat_table in flat_repo:
                filename, tableid = flat_table["file_name"], flat_table["table_id"]
                filename = filename.split('.')[0]
                if (filename == blobname) and (tableid == sheetname):
                    model_inputs = create_sample(hier_table, flat_table, reader, tokenizer, args)
                    unique_id = create_hier_id(blobname, sheetname)
                    for i, model_input in enumerate(model_inputs):
                        if model_input == None: continue
                        datadict[unique_id+str(i)] = model_input  # might be None!!
    print("create a datadict of size: ", len(datadict))
    return datadict


def build_flat_repo(flat_json_path):    # transfer deex.jl to list of sheet dict
    # Build tctc repository using json-lines
    flat_repo = []
    with open(flat_json_path, "rb") as fr_flat:
        for item in json_lines.reader(fr_flat):
            flat_repo.append( item )
    print("collected {} source samples from: {}".format(len(flat_repo), flat_json_path))
    return flat_repo

def source_content(content_list, datadict):
    dataset = []
    for content in content_list:
        fname, sname = content["fname"], content["sname"]
        for i in range(1000):
            tmpid = create_hier_id(fname, sname) + str(i)
            if tmpid not in datadict:
                # print("Not Found Error: Blobname ({}), SheetName ({}) unable to Match!".format(fname, sname))
                continue
            if datadict[tmpid] is None:
                # print("Value Error: Data in Blobname ({}), SheetName ({}) fail at Pre-processing!".format(fname, sname))
                continue
            dataset.append( datadict[tmpid] )
    return dataset


def stat_dataset(dataset, name):
    counts = [0 for _ in range(6)]
    for sample in dataset:
        tctc_labels = sample[-1]
        for lbl in tctc_labels:
            if lbl<0 or lbl>5:continue
            counts[lbl] += 1
    print("Dataset: {} counts: {}".format(name, counts))



def create_dynamic_dataset_folds(folds_path, flat_json_path, hier_dir, reader, tokenizer, args):
    print("execute create_dynamic_dataset_folds")
    flat_repo = build_flat_repo(flat_json_path)
    with open(folds_path, "r") as fr:
        folds = json.load(fr)    # folds: list of five json file
    dataset_couples = [ [] for _ in range(5) ]
    for iepoch in range(args.dataset_num):
        dyn_datadict = build_datadict(flat_repo, hier_dir, reader, tokenizer, args)  # sampling from the intersection between hier_dir and flat
    
        print("iepoch:", iepoch)
        for i, fold in enumerate(folds):
            dyn_trainset = source_content(fold["train"], dyn_datadict)
            stat_dataset(dyn_trainset, "Train #{}".format(i))
            dyn_testset = source_content(fold["test"], dyn_datadict)
            stat_dataset(dyn_testset, "Test #{}".format(i))
            print("Train Size: {}, Test Size: {}".format(len(dyn_trainset), len(dyn_testset)))
            dataset_couples[i].append( (dyn_trainset, dyn_testset) )
    print("Build Dataset Couples: {}".format([len(dc) for dc in dataset_couples]))
    return dataset_couples
            


# %% Model Classes
class TctcHead(nn.Module):
    def __init__(self, config):
        super(TctcHead, self).__init__()
        self.uniform_linear_tok = nn.Linear(config.hidden_size, config.hidden_size)
        self.uniform_linear_sep = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = act.ACT_FCN[config.hidden_act]
        self.tanh = nn.Tanh()
        self.predict_linear = nn.Linear(config.hidden_size, config.num_ctc_type)
        self.loss = nn.CrossEntropyLoss()

        self.aggregator = config.aggregator
        self.aggr_funcs = {"sum": self.token_sum, "avg": self.token_avg}
    
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

    def forward(self, encoded_states, indicator, tctc_label):
        # get cell encodings from token sequence
        cell_states = self.aggr_funcs[self.aggregator](encoded_states, indicator)

        tctc_label = tctc_label.contiguous().view(-1)
        cell_states = cell_states.contiguous().view(tctc_label.size()[0], -1)
        tctc_logits = cell_states[tctc_label > -1, :]       # [batch_total_cell_num, hidden_size]
        tctc_label = tctc_label[tctc_label > -1]

        # separator
        sep_logits = self.uniform_linear_sep(tctc_logits[0::2, :])
        sep_logits = self.tanh(sep_logits)
        sep_logits = self.predict_linear(sep_logits)
        sep_predict = sep_logits.argmax(dim=-1)
        sep_labels = tctc_label[0: : 2]
        sep_loss = self.loss(sep_logits, sep_labels)

        # token-aggregation
        tok_logits = self.uniform_linear_tok(tctc_logits[1::2, :])
        tok_logits = self.tanh(tok_logits)
        tok_logits = self.predict_linear(tok_logits)
        tok_predict = tok_logits.argmax(dim=-1)  
        tok_labels = tctc_label[1: : 2]                                # [batch-variant copied num]
        tok_loss = self.loss(tok_logits, tok_labels)                  # scalar
        return (sep_loss, sep_predict, sep_labels), (tok_loss, tok_predict, tok_labels)



# %% Model Architecture Class
class TUTAForCTC(nn.Module):
    def __init__(self, config):
        super(TUTAForCTC, self).__init__()
        self.backbone = bbs.BACKBONES[config.target](config)
        self.tctc_head = TctcHead(config)

    def forward(self, token_id, num_mag, num_pre, num_top, num_low, \
                token_order, pos_row, pos_col, pos_top, pos_left, format_vec, \
                indicator, tctc_label):
        encoded_states = self.backbone(token_id, num_mag, num_pre, num_top, num_low, \
                                       token_order, pos_row, pos_col, pos_top, pos_left, format_vec, indicator)
        sep_triple, tok_triple = self.tctc_head(encoded_states, indicator, tctc_label) 
        return sep_triple, tok_triple




# training and testing pipeline
def dynamic_pipeline(args, model, dataset_couples, no_decay=['bias', 'gamma', 'beta']):
    def evaluate(args, testset):
        # print("Start Evaluation with {} Instances: ".format(len(testset)))
        model.eval()
        sep_confusion_matrix = [[0 for _ in range(args.num_ctc_type)] for _ in range(args.num_ctc_type)]
        tok_confusion_matrix = [[0 for _ in range(args.num_ctc_type)] for _ in range(args.num_ctc_type)]
        for i, tensors in enumerate(
            ut.load_dataset_batch_withpad(
                dataset=testset, 
                batch_size=args.batch_size, 
                defaults=[0,11,11,11,11,0,256,256,args.default_tree_position,args.default_tree_position,args.default_format,0,-1], 
                device_id=args.device_id
            )
        ):
            with torch.no_grad():
                token_id, num_mag, num_pre, num_top, num_low, token_order, pos_row, pos_col, pos_top, pos_left, fmt_vec, ind, tctc = tensors
                (_, sep_pred, sep_gold), (_, tok_pred, tok_gold) = model(
                    token_id=token_id, num_mag=num_mag, num_pre=num_pre, num_top=num_top, num_low=num_low,
                    token_order=token_order, pos_row=pos_row, pos_col=pos_col, pos_top=pos_top, pos_left=pos_left, 
                    format_vec=fmt_vec.float(), indicator=ind, tctc_label=tctc
                )
                for spd, sgd in zip(sep_pred.tolist(), sep_gold.tolist()):
                    sep_confusion_matrix[spd][sgd] += 1
                for tpd, tgd in zip(tok_pred.tolist(), tok_gold.tolist()):
                    tok_confusion_matrix[tpd][tgd] += 1

        # compute confusion matrix
        sep_precision, sep_recall = [], []
        tok_precision, tok_recall = [], []
        for iclass in range(args.num_ctc_type):
            class_sep_precision = sep_confusion_matrix[iclass][iclass] / (sum(sep_confusion_matrix[iclass]) + 1e-6)
            sep_precision.append(class_sep_precision)
            class_sep_recall = sep_confusion_matrix[iclass][iclass] / (sum([line[iclass] for line in sep_confusion_matrix]) + 1e-6)
            sep_recall.append(class_sep_recall)
            class_tok_precision = tok_confusion_matrix[iclass][iclass] / (sum(tok_confusion_matrix[iclass]) + 1e-6)
            tok_precision.append(class_tok_precision)
            class_tok_recall = tok_confusion_matrix[iclass][iclass] / (sum([line[iclass] for line in tok_confusion_matrix]) + 1e-6)
            tok_recall.append(class_tok_recall)
        
        sep_f1 = [(2*p*r)/(p+r+1e-6) for p,r in zip(sep_precision, sep_recall)]
        print("[SEP] f1: ", [round(value, 3) for value in sep_f1])

        tok_f1 = [(2*p*r)/(p+r+1e-6) for p,r in zip(tok_precision, tok_recall)]
        print("[TOK] f1: ", [round(value, 3) for value in tok_f1], sum(tok_f1)/6)
        if args.sep_or_tok != 0:
            return [s for s,t in zip(sep_f1, tok_f1)]
        else: 
            return [t for s,t in zip(sep_f1, tok_f1)]
    
    # Do Training
    param_optimizer = list(model.named_parameters())
    print("tuning all of the model parameters (backbone + ctc_head)")
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)

    total_sep_loss, total_tok_loss = 0.0, 0.0
    best_result = [0.0] * args.num_ctc_type
    early_stopping_count = 0
    for iepoch in range(args.epochs_num):
        trainset, testset = dataset_couples[iepoch % args.dataset_num]
        random.shuffle(trainset)
        print("Start Training", iepoch, args.report_steps)
        model.train()
        for ii, tensors in enumerate(
            ut.load_dataset_batch_withpad(
                dataset=trainset,
                batch_size=args.batch_size, 
                defaults=[0,11,11,11,11,0,256,256,args.default_tree_position,args.default_tree_position,args.default_format,0,-1], 
                device_id=args.device_id 
            ) 
        ):
            model.zero_grad()
            token_id, num_mag, num_pre, num_top, num_low, token_order, pos_row, pos_col, pos_top, pos_left, fmt_vec, ind, tctc = tensors
            (sep_loss, _, _), (tok_loss, _, _) = model(
                token_id=token_id, num_mag=num_mag, num_pre=num_pre, num_top=num_top, num_low=num_low,
                token_order=token_order, pos_row=pos_row, pos_col=pos_col, pos_top=pos_top, pos_left=pos_left, 
                format_vec=fmt_vec.float(), indicator=ind, tctc_label=tctc
            )
            loss = sep_loss * args.sep_weight + tok_loss * (1. - args.sep_weight)
            total_sep_loss += sep_loss.item()
            total_tok_loss += tok_loss.item()
            if (ii+1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: [SEP] {:.3f}, [TOK] {:.3f}".\
                    format(iepoch, ii+1, total_sep_loss / args.report_steps, total_tok_loss / args.report_steps))
                total_sep_loss, total_tok_loss = 0.0, 0.0
            loss.backward()
            optimizer.step()

        result = evaluate(args, testset)
        if sum(result) >= sum(best_result):
            best_result = result
            ut.save_model(model, args.output_model_path)
        else:
            early_stopping_count += 1
        if early_stopping_count > args.early_stopping_bound:
            break

    return best_result



# %% Main Procedure
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # i/o paths
    parser.add_argument("--folds_path", type=str, default="./folds_deex5.json", help="Path of the splitted folds content json.")
    parser.add_argument("--flat_json_path", type=str, default="./deex.jl", help="Path of the tctc unzipped jsonlines.")
    parser.add_argument("--hier_dir", type=str, default="./deex", help="Directory of the hierarchical json files.")
    # add train/dev/test files
    parser.add_argument("--pretrained_model_path", type=str, default="./pretrain-tuta-v4-400K256.bin-600000", help="Path of the pretrained bert/ts model.")
    parser.add_argument("--output_model_path", type=str, default="ctc_out.bin", help="Fine-tuned model path.")
    # model configurations
    parser.add_argument("--vocab_path", type=str, default="./vocab/bert_vocab.txt", help="Path of the vocabulary file.")
    parser.add_argument("--context_repo_path", type=str, default="./vocab/context_repo_init.txt", help="TXT of pre-collected context pieces.")
    parser.add_argument("--cellstr_repo_path", type=str, default="./vocab/cellstr_repo_init.txt", help="TXT of pre-collected context pieces.")
    parser.add_argument("--hidden_size", type=int, default=768, help="Size of the hidden states.")
    parser.add_argument("--intermediate_size", type=int, default=3072, help="Size of the intermediate layer.")
    parser.add_argument("--magnitude_size", type=int, default=10, help="Max magnitude of numeric values.")
    parser.add_argument("--precision_size", type=int, default=10, help="Max precision of numeric values.")
    parser.add_argument("--top_digit_size", type=int, default=10, help="Most significant digit from '0' to '9'.")
    parser.add_argument("--low_digit_size", type=int, default=10, help="Least significant digit from '0' to '9'.")
    parser.add_argument("--max_cell_length", type=int, default=16, help="Maximum number of tokens in one cell string.")
    parser.add_argument("--row_size", type=int, default=2560, help="Max number of rows in table.")
    parser.add_argument("--column_size", type=int, default=2560, help="Max number of columns in table.")
    parser.add_argument("--tree_depth", type=int, default=4, help="Maximum depth of top & left header tree.")
    parser.add_argument("--node_degree", type=str, default="32,32,64,256", help="Maximum number of children of each tree node.")
    parser.add_argument("--attention_distance", type=int, default=2, help="Maximum distance for attention visibility.")
    parser.add_argument("--attention_step", type=int, default=0, help="Step size of attention distance to add for each layer.")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="Number of the attention heads.")
    parser.add_argument("--num_encoder_layers", type=int, default=12, help="Number of the encoding layers.")
    parser.add_argument("--hidden_dropout_prob", type=int, default=0.1, help="Dropout probability for hidden layers.")
    parser.add_argument("--attention_dropout_prob", type=int, default=0.1, help="Dropout probability for attention.")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-6)
    parser.add_argument("--hidden_act", type=str, default="gelu", help="Activation function for hidden layers.")
    parser.add_argument("--learning_rate", type=float, default=8e-6, help="Learning rate during fine-tuning.")

    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum length of the table sequence.")
    parser.add_argument("--max_cell_num", type=int, default=256, help="Maximum cell number.")  # useful ??
    parser.add_argument("--text_threshold", type=float, default=0.5, help="Probability threshold to sample text in data region.")
    parser.add_argument("--value_threshold", type=float, default=0.1, help="Prob to sample value in data region.")
    parser.add_argument("--clc_rate", type=float, default=0.3)
    parser.add_argument("--wcm_rate", type=float, default=0.3, help="Proportion of masked cells doing whole-cell-masking.")
    parser.add_argument("--add_separate", type=bool, default=True, help="Whether to add [SEP] as aggregate cell representation.")
    parser.add_argument("--num_ctc_type", type=int, default=6, help="Number of cell types for classification.")

    parser.add_argument("--attn_method", type=str, default="add", choices=["max", "add"])
    parser.add_argument("--hier_or_flat", type=str, default="both", choices=["hier", "flat", "both"])
    parser.add_argument("--org_or_weigh", type=str, default="original", choices=["original", "weighted"])
    parser.add_argument("--num_format_feature", type=int, default=11)
    parser.add_argument("--sep_or_tok", type=int, default=0, choices=[0, 1])
    parser.add_argument("--sep_weight", type=float, default=0.0, help="Weight to be multiplied on SEP loss.")
    parser.add_argument("--aggregator", type=str, default="sum", choices=["sum", "avg"], help="Aggregation method from token to cell.")

    # model choices
    parser.add_argument("--target", type=str, default="tuta", help="Pre-training objectives.")

    # training options
    parser.add_argument("--batch_size", type=int, default=2, help="Size of the input batch.")
    parser.add_argument("--report_steps", type=int, default=200, help="Specific steps to print prompt.")
    parser.add_argument("--epochs_num", type=int, default=40, help="Number of epochs for fine-tune.")
    parser.add_argument("--dataset_num", type=int, default=1, help="Times of distinct data sampling.")
    parser.add_argument("--early_stopping_bound", type=int, default=100)
    parser.add_argument("--device_id", type=int, default=None, help="Designated GPU id if not None.")
    args = parser.parse_args()
    args.node_degree = [int(degree) for degree in args.node_degree.split(',')]
    args.total_node = sum(args.node_degree)
    args.default_tree_position = [args.total_node for _ in args.node_degree]
    print("node degree: ", args.node_degree)

    reader = TctcReader(args)
    tokenizer = TctcTok(args)
    args.vocab_size = len(tokenizer.vocab)
    args.default_format = tokenizer.default_format
    print("Default Format: ", args.default_format)
    model = TUTAForCTC(args)

    if args.device_id is not None:
        print("Using devices {} for testing".format(args.device_id))
        model.cuda(args.device_id)

    # create datasets for five folds
    dataset_couples = create_dynamic_dataset_folds(args.folds_path, args.flat_json_path, args.hier_dir, reader, tokenizer, args)
    folds_results = []
    for iset, fold_couples in enumerate(dataset_couples):    # calculate every fold for 100 epochs
        print("\nGo On to Couple #{}".format(iset+1))
        print("fold_length:", len(dataset_couples))
        ut.init_tuta_loose(model=model, tuta_path=args.pretrained_model_path)
        f1_list = dynamic_pipeline(args, model, fold_couples)
        print("F1 List: ", [round(fl, 3) for fl in f1_list], "\n\n")
        folds_results.append(f1_list)
    
    # post-calculate f1 results
    average_f1 = []
    for ilbl in range(args.num_ctc_type):
        f1_collection = [res[ilbl] for res in folds_results]
        f1_collection_no_zero = [f1 for f1 in f1_collection if ( f1 > 1e-6)]
        f1_collection_with_zero = [f1 for f1 in f1_collection]
        print(ilbl, f1_collection_with_zero)
        average_f1.append( sum(f1_collection_no_zero) / (len(f1_collection_no_zero) + 1e-6))
    print("Average F1: ", [round(af, 3) for af in average_f1])
    print("Macro Acc.: {:.4f}".format(sum(average_f1) / (len(average_f1) + 1e-6)))
    
    

if __name__ == "__main__":
    main()
