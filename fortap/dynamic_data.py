#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loader for semi-processed table inputs.
"""

import torch
import random
import pickle
from icecream import ic

from utils import UNZIPS
from tokenizer import PAD_ID, DEFAULT_SR_LABEL, DEFAULT_NR_LABEL, FP_PAD_TAG, \
    DEFAULT_RANGE_LABEL, DEFAULT_OP_MLM_LABEL, DEFAULT_RANGE_MLM_LABEL, DEFAULT_SR_CONTEXT_LABEL
from constants import NR_AGGR_TO_INDEX


class FPDataloader(object):
    def __init__(self, args, proc_id, proc_num, do_shuffle=True):
        self.proc_id = proc_id
        self.do_shuffle = do_shuffle
        self.proc_num = proc_num
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.chunk_size = args.chunk_size  # less than the minimum file size
        # load private data sets
        self.f_reads, self.private_datasets, self.dataset_progress = [], [], []
        for ipath, dataset_path in enumerate(args.dataset_paths):
            if ipath % proc_num == proc_id:
                self.f_reads.append(open(dataset_path, "rb"))
                self.private_datasets.append(dataset_path)
                self.dataset_progress.append(0)
        self.set_count = len(self.private_datasets)
        print("DataLoader #{} assigned {} sets: {}".format(proc_id, self.set_count, self.private_datasets))

        # only need to read dataset once when buffer is big enough to load the entire dataset
        self.repeat_read_dataset = args.repeat_read_dataset
        self.start = 0
        self.end = 0
        self.buffer = []

        self.min_cell_len = 16
        self.max_cell_num = args.max_cell_num
        self.max_seq_len = args.max_seq_len
        self.max_cell_length = args.max_cell_length
        self.magnitude_size = args.magnitude_size
        self.precision_size = args.precision_size
        self.top_digit_size = args.top_digit_size
        self.low_digit_size = args.low_digit_size
        self.row_size = args.row_size
        self.column_size = args.column_size
        self.tree_depth = args.tree_depth
        self.node_degree = args.node_degree
        self.total_node = sum(self.node_degree)
        self.default_pos = [self.total_node] * self.tree_depth

        self.num_format_feature = args.num_format_feature
        self.default_format = [0.25, 0.25, 0., 0., 0., 0., 0., 0., 0., 1., 1.]

        self.tokenizer = args.tokenizer
        self.max_disturb_num = args.max_disturb_num
        self.disturb_prob = args.disturb_prob
        self.clc_rate = args.clc_rate
        self.add_separate = args.add_separate
        self.hier_or_flat = args.hier_or_flat
        self.target = args.target
        self.test = args.test

        self.num_missing_formula_label = 0

    def _fill_buf(self):
        if len(self.buffer) > 0 and not self.repeat_read_dataset:
            if self.do_shuffle:
                random.shuffle(self.buffer)
            self.start = 0
            self.end = len(self.buffer)
        else:  # load new buffer anyway
            self.buffer = []
            while len(self.buffer) < self.buffer_size:
                set_index = random.randint(0, self.set_count - 1)
                chunk = []  # a chunk from a random data set
                while len(chunk) < self.chunk_size:
                    try:
                        tables = pickle.load(self.f_reads[set_index])
                        chunk.extend(tables)
                    except EOFError:
                        if not self.repeat_read_dataset:
                            break
                        self.f_reads[set_index].seek(0)
                        tables = pickle.load(self.f_reads[set_index])
                        chunk.extend(tables)
                print("DataLoader #{}, pickle loaded chunk of size {} from {}".format(self.proc_id, len(chunk),
                                                                                      self.private_datasets[set_index]))
                semi_input = self.sift_and_prep(chunk)
                print(
                    "DataLoader #{}, tokenizer resulted {} inputs from {} tables".format(self.proc_id, len(semi_input),
                                                                                         len(chunk)))
                self.buffer.extend(semi_input)
                self.dataset_progress[set_index] += len(semi_input)

            if self.do_shuffle:
                random.shuffle(self.buffer)
            self.start = 0
            self.end = len(self.buffer)
            data_step_msg = ["{} ({})".format(dataset.split('/')[-1], steps) for dataset, steps in
                             zip(self.private_datasets, self.dataset_progress)]
            print("DataLoader #{} dataset steps: {}".format(self.proc_id, data_step_msg))

    def exist_formula(self, cell_formula_label):
        formula_label = []
        for c in cell_formula_label:
            formula_label.extend(c)
        formula_label = formula_label[: self.max_seq_len]
        if all([m == 0 for m in formula_label]):
            return False
        return True

    def sift_and_prep(self, instances):
        semi_input = []
        num_input_formulas, num_formulas = 0, 0
        num_none, num_exceed_max_cell_len, num_exceed_max_seq_len = 0, 0, 0
        for ins in instances:
            token_matrix, number_matrix, position_lists, header_info, format_or_text, \
            table_range, formula_dict, file_path, meta_data = ins
            format_matrix, context = None, None
            if isinstance(format_or_text, str):  # wiki, title
                context = (format_or_text,)
            elif isinstance(format_or_text, list):  # sheet, format_matrix
                format_matrix = format_or_text
            elif isinstance(format_or_text, tuple):  # wdc, context = (title, page_title, text_before, text_after)
                context = format_or_text
            else:
                print("Unsupported data type at 'format_or_text': ", type(format_or_text))

            if self.hier_or_flat == "hier":
                header_rows, header_columns = header_info
                if (header_rows <= 1) and (header_columns <= 1):
                    continue
            elif self.hier_or_flat == "flat":
                header_rows, header_columns = header_info
                if (header_rows > 1) or (header_columns > 1):
                    continue

            for (row, col), formula_info in formula_dict.items():
                num_input_formulas += 1
                results = self.tokenizer.fp_preprocess(
                    token_matrix=token_matrix,
                    number_matrix=number_matrix,
                    position_lists=position_lists,
                    header_info=header_info,
                    formula_row=row,
                    formula_col=col,
                    table_range=table_range,
                    formula_info=formula_info,
                    format_matrix=format_matrix,
                    context=context,
                    add_sep=self.add_separate,
                )
                if results is None:
                    num_none += 1
                if len(results[0]) > self.max_cell_num:
                    num_exceed_max_cell_len += 1
                token_seq = [tok for cell in results[0] for tok in cell]
                if len(token_seq) > self.max_seq_len:
                    num_exceed_max_seq_len += 1
                semi_input.append(results)

        return semi_input

    def _empty(self):
        return self.start + self.batch_size >= self.end

    def __del__(self):
        for fr in self.f_reads:
            fr.close()

    def __iter__(self):
        while True:
            if self._empty():
                if len(self.buffer) > 0 and self.test:  # test done
                    return None
                else:
                    self._fill_buf()
            if not self.buffer:
                print("Warning: worker {}'s data buffer is empty".format(self.proc_id))

            semi_input = self.buffer[self.start: self.start + self.batch_size]
            self.start += self.batch_size
            batch_max_seq_len, batch_max_sketch_len = 0, 0

            all_token_id, all_num_mag, all_num_pre, all_num_top, all_num_low = [], [], [], [], []
            all_token_order, all_pos_row, all_pos_col, all_pos_top, all_pos_left = [], [], [], [], []
            all_format_vec, all_indicator = [], []
            all_formula_label, all_src_sketch, all_tgt_sketch, all_candi_cell_token_mask, all_range_label, all_range_map = [], [], [], [], [], []

            for (tok_list, num_list, pos_list, fmt_list, cell_ind, cell_formula_label, complete_sketch, cell_candi_mask,
                 range_label, range_map) in semi_input:
                token_id, num_mag, num_pre, num_top, num_low = [], [], [], [], []
                token_order, pos_row, pos_col, pos_top, pos_left = [], [], [], [], []
                format_vec, indicator = [], []
                formula_label, candi_cell_token_mask = [], []

                cell_num = len(tok_list)
                for icell in range(cell_num):
                    tokens = tok_list[icell]
                    cell_len = len(tokens)
                    token_id.extend(tokens)
                    token_order.extend([ii for ii in range(cell_len)])

                    num_feats = num_list[icell]
                    num_mag.extend([f[0] for f in num_feats])
                    num_pre.extend([f[1] for f in num_feats])
                    num_top.extend([f[2] for f in num_feats])
                    num_low.extend([f[3] for f in num_feats])

                    row, col, ttop, tleft = pos_list[icell]
                    pos_row.extend([row for _ in range(cell_len)])
                    pos_col.extend([col for _ in range(cell_len)])
                    entire_top = UNZIPS[self.target](ttop, self.node_degree, self.total_node)
                    pos_top.extend([entire_top for _ in range(cell_len)])
                    entire_left = UNZIPS[self.target](tleft, self.node_degree, self.total_node)
                    pos_left.extend([entire_left for _ in range(cell_len)])

                    format_vec.extend([fmt_list[icell] for _ in range(cell_len)])
                    indicator.extend(cell_ind[icell])
                    formula_label.extend(cell_formula_label[icell])
                    candi_cell_token_mask.extend(cell_candi_mask[icell])

                seq_len = len(token_id)
                # if seq_len > self.max_seq_len:  # stop if exceed seq_len bound
                #     continue
                batch_max_seq_len = max(batch_max_seq_len, seq_len)
                batch_max_sketch_len = max(batch_max_sketch_len, len(complete_sketch) - 1)  # w/o. <start> or <end>

                # append to overall instance set
                all_token_id.append(token_id)
                all_num_mag.append(num_mag)
                all_num_pre.append(num_pre)
                all_num_top.append(num_top)
                all_num_low.append(num_low)
                all_token_order.append(token_order)
                all_pos_row.append(pos_row)
                all_pos_col.append(pos_col)
                all_pos_top.append(pos_top)
                all_pos_left.append(pos_left)
                all_format_vec.append(format_vec)
                all_indicator.append(indicator)
                all_formula_label.append(formula_label)
                all_src_sketch.append(complete_sketch[:-1])
                all_tgt_sketch.append(complete_sketch[1:])
                all_candi_cell_token_mask.append(candi_cell_token_mask)
                if len(complete_sketch[1:]) != len(range_label):
                    ic(len(complete_sketch[1:]), len(range_label))
                    ic(complete_sketch[1:])
                    ic(range_label)
                all_range_label.append(range_label.copy())
                all_range_map.append(range_map)

            # pad things to batch_max_seq_len
            batch_max_seq_len = ((batch_max_seq_len + 7) // 8) * 8  # times of 8
            batch_max_sketch_len = ((batch_max_sketch_len + 7) // 8) * 8  # times of 8
            # ic(batch_max_sketch_len)
            for isample in range(self.batch_size):
                all_token_id[isample].extend([PAD_ID] * (batch_max_seq_len - len(all_token_id[isample])))
                all_num_mag[isample].extend([self.magnitude_size + 1] * (batch_max_seq_len - len(all_num_mag[isample])))
                all_num_pre[isample].extend([self.precision_size + 1] * (batch_max_seq_len - len(all_num_pre[isample])))
                all_num_top[isample].extend([self.top_digit_size + 1] * (batch_max_seq_len - len(all_num_top[isample])))
                all_num_low[isample].extend([self.low_digit_size + 1] * (batch_max_seq_len - len(all_num_low[isample])))

                all_token_order[isample].extend([0] * (batch_max_seq_len - len(all_token_order[isample])))
                all_pos_row[isample].extend([self.row_size] * (batch_max_seq_len - len(all_pos_row[isample])))
                all_pos_col[isample].extend([self.column_size] * (batch_max_seq_len - len(all_pos_col[isample])))
                all_pos_top[isample].extend([self.default_pos] * (batch_max_seq_len - len(all_pos_top[isample])))
                all_pos_left[isample].extend([self.default_pos] * (batch_max_seq_len - len(all_pos_left[isample])))

                all_format_vec[isample].extend(
                    [self.default_format] * (batch_max_seq_len - len(all_format_vec[isample])))
                all_indicator[isample].extend([0] * (batch_max_seq_len - len(all_indicator[isample])))
                all_formula_label[isample].extend([0] * (batch_max_seq_len - len(all_formula_label[isample])))
                all_src_sketch[isample].extend([FP_PAD_TAG] * (batch_max_sketch_len - len(all_src_sketch[isample])))
                all_tgt_sketch[isample].extend([FP_PAD_TAG] * (batch_max_sketch_len - len(all_tgt_sketch[isample])))
                all_candi_cell_token_mask[isample].extend(
                    [0] * (batch_max_seq_len - len(all_candi_cell_token_mask[isample])))
                all_range_label[isample].extend([DEFAULT_RANGE_LABEL] * (
                            batch_max_sketch_len - len(all_range_label[isample])))  # !这里改变了semi_input中的range_label，很坑。

                # truncate
                all_token_id[isample] = all_token_id[isample][: self.max_seq_len]
                all_num_mag[isample] = all_num_mag[isample][: self.max_seq_len]
                all_num_pre[isample] = all_num_pre[isample][: self.max_seq_len]
                all_num_top[isample] = all_num_top[isample][: self.max_seq_len]
                all_num_low[isample] = all_num_low[isample][: self.max_seq_len]
                all_token_order[isample] = all_token_order[isample][: self.max_seq_len]
                all_pos_row[isample] = all_pos_row[isample][: self.max_seq_len]
                all_pos_col[isample] = all_pos_col[isample][: self.max_seq_len]
                all_pos_top[isample] = all_pos_top[isample][: self.max_seq_len]
                all_pos_left[isample] = all_pos_left[isample][: self.max_seq_len]
                all_format_vec[isample] = all_format_vec[isample][: self.max_seq_len]
                all_indicator[isample] = all_indicator[isample][: self.max_seq_len]
                all_formula_label[isample] = all_formula_label[isample][: self.max_seq_len]
                num_formula_labels = sum([m == 1 for m in all_formula_label[isample]])
                if num_formula_labels != 2:
                    if num_formula_labels == 1:
                        all_formula_label[isample][0] = 1
                    elif num_formula_labels == 0:
                        all_formula_label[isample][0] = 1
                        all_formula_label[isample][1] = 1  # dummy
                    self.num_missing_formula_label += 1
                    # ic(self.num_missing_formula_label)
                all_candi_cell_token_mask[isample] = all_candi_cell_token_mask[isample][: self.max_seq_len]

            yield (
                torch.LongTensor(all_token_id),
                torch.LongTensor(all_num_mag),
                torch.LongTensor(all_num_pre),
                torch.LongTensor(all_num_top),
                torch.LongTensor(all_num_low),
                torch.LongTensor(all_token_order),
                torch.LongTensor(all_pos_row),
                torch.LongTensor(all_pos_col),
                torch.LongTensor(all_pos_top),
                torch.LongTensor(all_pos_left),
                torch.FloatTensor(all_format_vec),
                torch.LongTensor(all_indicator),
                torch.LongTensor(all_formula_label),
                torch.LongTensor(all_src_sketch),
                torch.LongTensor(all_tgt_sketch),
                torch.LongTensor(all_candi_cell_token_mask),
                torch.LongTensor(all_range_label),
                all_range_map
            )


class FortapDataloader(object):
    def __init__(self, args, proc_id, proc_num, do_shuffle=True):
        self.proc_id = proc_id
        self.do_shuffle = do_shuffle
        self.proc_num = proc_num
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.chunk_size = args.chunk_size  # less than the minimum file size
        # load private data sets
        self.f_reads, self.private_datasets, self.dataset_progress = [], [], []
        for ipath, dataset_path in enumerate(args.dataset_paths):
            if ipath % proc_num == proc_id:
                self.f_reads.append(open(dataset_path, "rb"))
                self.private_datasets.append(dataset_path)
                self.dataset_progress.append(0)
        self.set_count = len(self.private_datasets)
        print("DataLoader #{} assigned {} sets: {}".format(proc_id, self.set_count, self.private_datasets))

        # only need to read dataset once when buffer is big enough to load the entire dataset
        self.repeat_read_dataset = args.repeat_read_dataset
        self.start = 0
        self.end = 0
        self.buffer = []

        self.min_cell_len = 16
        self.max_cell_num = args.max_cell_num
        self.max_seq_len = args.max_seq_len
        self.max_cell_length = args.max_cell_length
        self.magnitude_size = args.magnitude_size
        self.precision_size = args.precision_size
        self.top_digit_size = args.top_digit_size
        self.low_digit_size = args.low_digit_size
        self.row_size = args.row_size
        self.column_size = args.column_size
        self.tree_depth = args.tree_depth
        self.node_degree = args.node_degree
        self.total_node = sum(self.node_degree)
        self.default_pos = [self.total_node] * self.tree_depth

        self.num_format_feature = args.num_format_feature
        self.default_format = [0.25, 0.25, 0., 0., 0., 0., 0., 0., 0., 1., 1.]

        self.tokenizer = args.tokenizer
        self.max_disturb_num = args.max_disturb_num
        self.disturb_prob = args.disturb_prob
        self.clc_rate = args.clc_rate
        self.add_separate = args.add_separate
        self.hier_or_flat = args.hier_or_flat
        self.target = args.target
        self.formula_mlm_prop = args.formula_mlm_prop

    def _fill_buf(self):
        if len(self.buffer) > 0 and not self.repeat_read_dataset:
            if self.do_shuffle:
                random.shuffle(self.buffer)
            self.start = 0
            self.end = len(self.buffer)
        else:  # load new buffer anyway
            self.buffer = []
            while len(self.buffer) < self.buffer_size:
                set_index = random.randint(0, self.set_count - 1)
                chunk = []  # a chunk from a random data set
                while len(chunk) < self.chunk_size:
                    try:
                        tables = pickle.load(self.f_reads[set_index])
                        chunk.extend(tables)
                    except EOFError:
                        if not self.repeat_read_dataset:
                            break
                        self.f_reads[set_index].seek(0)
                        tables = pickle.load(self.f_reads[set_index])
                        chunk.extend(tables)
                print("DataLoader #{}, pickle loaded chunk of size {} from {}".format(self.proc_id, len(chunk),
                                                                                      self.private_datasets[set_index]))
                semi_input = self.sift_and_prep(chunk)
                print(
                    "DataLoader #{}, tokenizer resulted {} inputs from {} tables".format(self.proc_id, len(semi_input),
                                                                                         len(chunk)))
                self.buffer.extend(semi_input)
                self.dataset_progress[set_index] += len(semi_input)

            if self.do_shuffle:
                random.shuffle(self.buffer)
            self.start = 0
            self.end = len(self.buffer)
            data_step_msg = ["{} ({})".format(dataset.split('/')[-1], steps) for dataset, steps in
                             zip(self.private_datasets, self.dataset_progress)]
            print("DataLoader #{} dataset steps: {}".format(self.proc_id, data_step_msg))

    def sift_and_prep(self, instances):
        semi_input = []
        cnt_sample = 0
        for ins in instances:
            # token_matrix, number_matrix, position_lists, header_info, format_or_text, table_range, formula_dict = ins
            token_matrix, number_matrix, position_lists, header_info, format_or_text, table_range, formula_dict, file_path = ins
            format_matrix, context = None, None
            if isinstance(format_or_text, str):  # wiki, title
                context = (format_or_text,)
            elif isinstance(format_or_text, list):  # sheet, format_matrix
                format_matrix = format_or_text
            elif isinstance(format_or_text, tuple):  # wdc, context = (title, page_title, text_before, text_after)
                context = format_or_text
            else:
                print("Unsupported data type at 'format_or_text': ", type(format_or_text))

            if self.hier_or_flat == "hier":
                header_rows, header_columns = header_info
                if (header_rows <= 1) and (header_columns <= 1):
                    continue
            elif self.hier_or_flat == "flat":
                header_rows, header_columns = header_info
                if (header_rows > 1) or (header_columns > 1):
                    continue

            for (row, col), formula_info in formula_dict.items():
                results = self.tokenizer.pretrain_preprocess(
                    token_matrix=token_matrix,
                    number_matrix=number_matrix,
                    position_lists=position_lists,
                    header_info=header_info,
                    formula_row=row,
                    formula_col=col,
                    table_range=table_range,
                    formula_info=formula_info,
                    format_matrix=format_matrix,
                    context=context,
                    file_path=None,
                    add_sep=self.add_separate,
                    max_disturb_num=self.max_disturb_num,
                    disturb_prob=self.disturb_prob,
                    formula_mlm_prop=self.formula_mlm_prop
                )
                if (results is None) or (len(results[0]) > self.max_cell_num):
                    continue
                token_seq = [tok for cell in results[0] for tok in cell]
                if len(token_seq) > self.max_seq_len:
                    continue
                semi_input.append(results)
                cnt_sample += 1
        return semi_input

    def _empty(self):
        return self.start + self.batch_size >= self.end

    def __del__(self):
        for fr in self.f_reads:
            fr.close()

    def __iter__(self):
        while True:
            if self._empty():
                self._fill_buf()
            if not self.buffer:
                print("Warning: worker {}'s data buffer is empty".format(self.proc_id))

            semi_input = self.buffer[self.start: self.start + self.batch_size]
            self.start += self.batch_size
            batch_max_seq_len = 0

            all_token_id, all_num_mag, all_num_pre, all_num_top, all_num_low = [], [], [], [], []
            all_token_order, all_pos_row, all_pos_col, all_pos_top, all_pos_left = [], [], [], [], []
            all_format_vec, all_indicator = [], []
            all_mlm_label = []
            all_sr_label, all_nr_label, all_op_appear_flag = [], [], []
            all_sr_context_label = []
            all_op_mlm_label, all_range_mlm_label = [], []
            all_candi_cell_token_mask = []

            for (tok_list, num_list, pos_list, fmt_list, cell_ind, cell_mlm, cell_sr, cell_nr,
                 cell_sr_context, cell_op_mlm_label, cell_range_mlm_label, cell_candi_mask) in semi_input:
                token_id, num_mag, num_pre, num_top, num_low = [], [], [], [], []
                token_order, pos_row, pos_col, pos_top, pos_left = [], [], [], [], []
                format_vec, indicator = [], []
                mlm_label = []
                sr_label, nr_label = [], []
                sr_context_label = []
                op_mlm_label, range_mlm_label = [], []
                candi_cell_token_mask = []

                cell_num = len(tok_list)
                for icell in range(cell_num):
                    tokens = tok_list[icell]
                    cell_len = len(tokens)
                    token_id.extend(tokens)
                    token_order.extend([ii for ii in range(cell_len)])

                    num_feats = num_list[icell]
                    num_mag.extend([f[0] for f in num_feats])
                    num_pre.extend([f[1] for f in num_feats])
                    num_top.extend([f[2] for f in num_feats])
                    num_low.extend([f[3] for f in num_feats])

                    pos_feats = pos_list[icell]
                    for i, (row, col, ttop, tleft) in enumerate(pos_feats):
                        pos_row.append(row)
                        pos_col.append(col)
                        entire_top = UNZIPS[self.target](ttop, self.node_degree, self.total_node)
                        pos_top.append(entire_top)
                        entire_left = UNZIPS[self.target](tleft, self.node_degree, self.total_node)
                        pos_left.append(entire_left)

                    format_vec.extend(fmt_list[icell])
                    indicator.extend(cell_ind[icell])
                    mlm_label.extend(cell_mlm[icell])
                    op_mlm_label.extend(cell_op_mlm_label[icell])
                    range_mlm_label.extend(cell_range_mlm_label[icell])
                    sr_label.extend(cell_sr[icell])
                    nr_label.extend(cell_nr[icell])
                    sr_context_label.extend(cell_sr_context[icell])
                    candi_cell_token_mask.extend(cell_candi_mask[icell])

                op_appear_flag = [0] * len(NR_AGGR_TO_INDEX)
                for i in range(len(NR_AGGR_TO_INDEX)):
                    if i in nr_label:
                        op_appear_flag[i] = 1

                seq_len = len(token_id)
                if seq_len > self.max_seq_len:  # stop if exceed seq_len bound
                    continue
                batch_max_seq_len = max(batch_max_seq_len, seq_len)

                try:
                    assert len(token_id) == len(num_mag) == len(token_order) == len(pos_row) \
                           == len(pos_top) == len(format_vec) == len(indicator) == len(op_mlm_label) \
                           == len(range_mlm_label) == len(candi_cell_token_mask) \
                           == len(mlm_label) == len(sr_label) == len(nr_label) == len(sr_context_label)
                except:
                    print('Not same input length in dynamic loader.')
                    ic(len(token_id), len(num_mag), len(token_order), len(pos_row),
                       len(pos_top), len(format_vec), len(indicator), len(op_mlm_label),
                       len(range_mlm_label), len(candi_cell_token_mask),
                       len(mlm_label), len(sr_label), len(nr_label), len(sr_context_label))

                # append to overall instance set
                all_token_id.append(token_id)
                all_num_mag.append(num_mag)
                all_num_pre.append(num_pre)
                all_num_top.append(num_top)
                all_num_low.append(num_low)
                all_token_order.append(token_order)
                all_pos_row.append(pos_row)
                all_pos_col.append(pos_col)
                all_pos_top.append(pos_top)
                all_pos_left.append(pos_left)
                all_format_vec.append(format_vec)
                all_indicator.append(indicator)
                all_mlm_label.append(mlm_label)
                all_sr_label.append(sr_label)
                all_nr_label.append(nr_label)
                all_op_appear_flag.append(op_appear_flag)
                all_sr_context_label.append(sr_context_label)
                all_op_mlm_label.append(op_mlm_label)
                all_range_mlm_label.append(range_mlm_label)
                all_candi_cell_token_mask.append(candi_cell_token_mask)

            # pad things to batch_max_seq_len
            batch_max_seq_len = ((batch_max_seq_len + 7) // 8) * 8  # times of 8
            for isample in range(self.batch_size):
                all_token_id[isample].extend([PAD_ID] * (batch_max_seq_len - len(all_token_id[isample])))
                all_num_mag[isample].extend([self.magnitude_size + 1] * (batch_max_seq_len - len(all_num_mag[isample])))
                all_num_pre[isample].extend([self.precision_size + 1] * (batch_max_seq_len - len(all_num_pre[isample])))
                all_num_top[isample].extend([self.top_digit_size + 1] * (batch_max_seq_len - len(all_num_top[isample])))
                all_num_low[isample].extend([self.low_digit_size + 1] * (batch_max_seq_len - len(all_num_low[isample])))

                all_token_order[isample].extend([0] * (batch_max_seq_len - len(all_token_order[isample])))
                all_pos_row[isample].extend([self.row_size] * (batch_max_seq_len - len(all_pos_row[isample])))
                all_pos_col[isample].extend([self.column_size] * (batch_max_seq_len - len(all_pos_col[isample])))
                all_pos_top[isample].extend([self.default_pos] * (batch_max_seq_len - len(all_pos_top[isample])))
                all_pos_left[isample].extend([self.default_pos] * (batch_max_seq_len - len(all_pos_left[isample])))

                all_format_vec[isample].extend(
                    [self.default_format] * (batch_max_seq_len - len(all_format_vec[isample])))
                all_indicator[isample].extend([0] * (batch_max_seq_len - len(all_indicator[isample])))
                all_mlm_label[isample].extend([-1] * (batch_max_seq_len - len(all_mlm_label[isample])))
                all_sr_label[isample].extend([DEFAULT_SR_LABEL] * (batch_max_seq_len - len(all_sr_label[isample])))
                all_nr_label[isample].extend([DEFAULT_NR_LABEL] * (batch_max_seq_len - len(all_nr_label[isample])))
                all_sr_context_label[isample].extend([DEFAULT_SR_CONTEXT_LABEL] * (batch_max_seq_len - len(all_sr_context_label[isample])))
                all_op_mlm_label[isample].extend(
                    [DEFAULT_OP_MLM_LABEL] * (batch_max_seq_len - len(all_op_mlm_label[isample])))
                all_range_mlm_label[isample].extend(
                    [DEFAULT_RANGE_MLM_LABEL] * (batch_max_seq_len - len(all_range_mlm_label[isample])))
                all_candi_cell_token_mask[isample].extend(
                    [0] * (batch_max_seq_len - len(all_candi_cell_token_mask[isample])))

            yield (
                torch.LongTensor(all_token_id),
                torch.LongTensor(all_num_mag),
                torch.LongTensor(all_num_pre),
                torch.LongTensor(all_num_top),
                torch.LongTensor(all_num_low),
                torch.LongTensor(all_token_order),
                torch.LongTensor(all_pos_row),
                torch.LongTensor(all_pos_col),
                torch.LongTensor(all_pos_top),
                torch.LongTensor(all_pos_left),
                torch.FloatTensor(all_format_vec),
                torch.LongTensor(all_indicator),
                torch.LongTensor(all_mlm_label),
                torch.LongTensor(all_sr_label),
                torch.LongTensor(all_nr_label),
                torch.LongTensor(all_op_appear_flag),
                torch.LongTensor(all_sr_context_label),
                torch.LongTensor(all_op_mlm_label),
                torch.LongTensor(all_range_mlm_label),
                torch.LongTensor(all_candi_cell_token_mask)
            )

DataLoaders = {
    "formula_prediction": FPDataloader,
    "fortap": FortapDataloader
}
