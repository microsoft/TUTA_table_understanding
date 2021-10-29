#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loader for semi-processed table inputs, do MLM & CLC & TCR dynamically. 
"""

import torch
import random
import pickle
from utils import UNZIPS
from tokenizer import PAD_ID


class DynamicDataLoader(object):
    def __init__(self, args, proc_id, proc_num, do_shuffle=True):
        self.proc_id = proc_id
        self.do_shuffle = do_shuffle
        self.proc_num = proc_num
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.chunk_size = args.chunk_size    # less than the minimum file size
        # load private data sets
        self.f_reads, self.private_datasets, self.dataset_progress = [], [], []
        for ipath, dataset_path in enumerate(args.dataset_paths):
            if ipath % proc_num == proc_id:
                self.f_reads.append( open(dataset_path, "rb") )
                self.private_datasets.append( dataset_path )
                self.dataset_progress.append( 0 )
        self.set_count = len(self.private_datasets)
        print("DataLoader #{} assigned {} sets: {}".format(proc_id, self.set_count, self.private_datasets))

        # only need to read dataset once when buffer is big enough to load the entire dataset
        self.repeat_read_dataset = True
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

    def _fill_buf(self):
        if len(self.buffer) > 0 and not self.repeat_read_dataset:
            if self.do_shuffle:
                random.shuffle(self.buffer)
            self.start = 0
            self.end = len(self.buffer)
        else:    # load new buffer anyway
            self.buffer = []
            while len(self.buffer) < self.buffer_size:
                set_index = random.randint(0, self.set_count - 1)
                chunk = []    # a chunk from a random data set
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
                print("DataLoader #{}, pickle loaded chunk of size {} from {}".format(self.proc_id, len(chunk), self.private_datasets[set_index]))
                semi_input = self.sift_and_prep(chunk)
                print("DataLoader #{}, tokenier resulted {} inputs from {} tables".format(self.proc_id, len(semi_input), len(chunk)))
                self.buffer.extend( semi_input )
                self.dataset_progress[set_index] += len(semi_input)

            if self.do_shuffle:
                random.shuffle(self.buffer)
            self.start = 0
            self.end = len(self.buffer)
            data_step_msg = ["{} ({})".format(dataset.split('/')[-1], steps) for dataset, steps in zip(self.private_datasets, self.dataset_progress)]
            print("DataLoader #{} dataset steps: {}".format(self.proc_id, data_step_msg))

    def sift_and_prep(self, instances):
        semi_input = []
        for ins in instances:
            token_matrix, number_matrix, position_lists, header_info, format_or_text = ins
            format_matrix, context = None, None
            if isinstance(format_or_text, str):    # wiki, title
                context = (format_or_text, )
            elif isinstance(format_or_text, list): # sheet, format_matrix
                format_matrix = format_or_text
            elif isinstance(format_or_text, tuple): # wdc, context = (title, page_title, text_before, text_after)
                context = format_or_text
            else:
                print("Unsupported data type at last position: ", type(format_or_text))

            if self.hier_or_flat == "hier":
                header_rows, header_columns = header_info
                if (header_rows <= 1) and (header_columns <= 1):
                    continue
            elif self.hier_or_flat == "flat":
                header_rows, header_columns = header_info
                if (header_rows > 1) or (header_columns > 1):
                    continue
            sampling_matrix = self.tokenizer.sampling(
                token_matrix=token_matrix, 
                number_matrix=number_matrix, 
                header_info=header_info, 
                max_disturb_num=self.max_disturb_num, 
                disturb_prob=self.disturb_prob, 
                clc_rate=self.clc_rate
            )
            results = self.tokenizer.objective_preprocess(
                sampling_matrix=sampling_matrix, 
                token_matrix=token_matrix, 
                number_matrix=number_matrix, 
                position_lists=position_lists, 
                format_matrix=format_matrix, 
                context=context, 
                add_sep=self.add_separate
            )
            if (results is None) or (len(results[0]) > self.max_cell_num):
                continue
            token_seq = [tok for cell in results[0] for tok in cell]
            if len(token_seq) > self.max_seq_len:
                continue
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
                self._fill_buf()
            if not self.buffer:
                print("Warning: worker {}'s data buffer is empty".format(self.proc_id))

            semi_input = self.buffer[self.start: self.start+self.batch_size] 
            self.start += self.batch_size
            batch_max_seq_len = 0

            all_token_id, all_num_mag, all_num_pre, all_num_top, all_num_low = [], [], [], [], []
            all_token_order, all_pos_row, all_pos_col, all_pos_top, all_pos_left = [], [], [], [], []           
            all_format_vec, all_indicator = [], []
            all_mlm_label, all_clc_label, all_tcr_label = [], [], []

            for (tok_list, num_list, pos_list, fmt_list, cell_ind, cell_mlm, cell_clc, cell_tcr) in semi_input:
                token_id, num_mag, num_pre, num_top, num_low = [], [], [], [], []
                token_order, pos_row, pos_col, pos_top, pos_left = [], [], [], [], []
                format_vec, indicator = [], []
                mlm_label, clc_label, tcr_label = [], [], []

                cell_num = len(tok_list)
                for icell in range(cell_num):
                    tokens = tok_list[icell]
                    cell_len = len(tokens)
                    token_id.extend(tokens)
                    token_order.extend([ii for ii in range(cell_len)])
                    mlm_label.extend(cell_mlm[icell])

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

                    format_vec.extend( [fmt_list[icell] for _ in range(cell_len)] )
                    indicator.extend(cell_ind[icell])
                    clc_label.extend(cell_clc[icell])
                    tcr_label.extend(cell_tcr[icell])

                seq_len = len(token_id)
                if seq_len > self.max_seq_len:  # stop if exceed seq_len bound
                    continue
                batch_max_seq_len = max(batch_max_seq_len, seq_len)

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
                all_clc_label.append(clc_label)
                all_tcr_label.append(tcr_label)
                        
            # pad things to batch_max_seq_len
            batch_max_seq_len = ((batch_max_seq_len + 7) // 8) * 8
            for isample in range(self.batch_size):
                all_token_id[isample].extend( [PAD_ID] * (batch_max_seq_len - len(all_token_id[isample])) )
                all_num_mag[isample].extend( [self.magnitude_size + 1] * (batch_max_seq_len - len(all_num_mag[isample])) )
                all_num_pre[isample].extend( [self.precision_size + 1] * (batch_max_seq_len - len(all_num_pre[isample])) )
                all_num_top[isample].extend( [self.top_digit_size + 1] * (batch_max_seq_len - len(all_num_top[isample])) )
                all_num_low[isample].extend( [self.low_digit_size + 1] * (batch_max_seq_len - len(all_num_low[isample])) )

                all_token_order[isample].extend([0] * (batch_max_seq_len - len(all_token_order[isample])))
                all_pos_row[isample].extend( [self.row_size] * (batch_max_seq_len - len(all_pos_row[isample])) )
                all_pos_col[isample].extend( [self.column_size] * (batch_max_seq_len - len(all_pos_col[isample])) )
                all_pos_top[isample].extend( [self.default_pos] * (batch_max_seq_len - len(all_pos_top[isample])) )
                all_pos_left[isample].extend( [self.default_pos] * (batch_max_seq_len - len(all_pos_left[isample])) )

                all_format_vec[isample].extend( [self.default_format] * (batch_max_seq_len - len(all_format_vec[isample])) )
                all_indicator[isample].extend([0] * (batch_max_seq_len - len(all_indicator[isample])))
                all_mlm_label[isample].extend([-1] * (batch_max_seq_len - len(all_mlm_label[isample])))
                all_clc_label[isample].extend([0] * (batch_max_seq_len - len(all_clc_label[isample])))
                all_tcr_label[isample].extend([-1] * (batch_max_seq_len - len(all_tcr_label[isample])))

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
                torch.LongTensor(all_clc_label), 
                torch.LongTensor(all_tcr_label)
            )



class DynamicDataLoaderBase(object):
    def __init__(self, args, proc_id, proc_num, do_shuffle=True):
        self.proc_id = proc_id
        self.do_shuffle = do_shuffle
        self.proc_num = proc_num
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.chunk_size = args.chunk_size   
        # load private data sets
        self.f_reads, self.private_datasets, self.dataset_progress = [], [], []
        for ipath, dataset_path in enumerate(args.dataset_paths):
            if ipath % proc_num == proc_id:
                self.f_reads.append( open(dataset_path, "rb") )
                self.private_datasets.append( dataset_path )
                self.dataset_progress.append( 0 )
        self.set_count = len(self.private_datasets)
        print("DataLoader #{} assigned {} sets: {}".format(proc_id, self.set_count, self.private_datasets))

        self.repeat_read_dataset = True
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
        self.add_separate = args.add_separate
        self.hier_or_flat = args.hier_or_flat
        self.clc_rate = args.clc_rate

        self.target = args.target

    def _fill_buf(self):
        if len(self.buffer) > 0 and not self.repeat_read_dataset:
            if self.do_shuffle:
                random.shuffle(self.buffer)
            self.start = 0
            self.end = len(self.buffer)
        else:   
            self.buffer = []
            while len(self.buffer) < self.buffer_size:
                set_index = random.randint(0, self.set_count - 1)
                chunk = []  
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
                print("DataLoader #{}, pickle loaded chunk of size {} from {}".format(self.proc_id, len(chunk), self.private_datasets[set_index]))
                semi_input = self.sift_and_prep(chunk)
                print("DataLoader #{}, tokenier resulted {} inputs from {} tables".format(self.proc_id, len(semi_input), len(chunk)))
                self.buffer.extend( semi_input )
                self.dataset_progress[set_index] += len(semi_input)

            if self.do_shuffle:
                random.shuffle(self.buffer)
            self.start = 0
            self.end = len(self.buffer)
            data_step_msg = ["{} ({})".format(dataset.split('/')[-1], steps) for dataset, steps in zip(self.private_datasets, self.dataset_progress)]
            print("DataLoader #{} dataset steps: {}".format(self.proc_id, data_step_msg))

    def sift_and_prep(self, instances):
        semi_input = []
        for ins in instances:
            token_matrix, number_matrix, position_lists, header_info, format_or_text = ins
            format_matrix, context = None, None
            if isinstance(format_or_text, str):    # wiki, title
                context = (format_or_text, )
            elif isinstance(format_or_text, list): # sheet, format_matrix
                format_matrix = format_or_text
            elif isinstance(format_or_text, tuple): # wdc, context = (title, page_title, text_before, text_after)
                context = format_or_text
            else:
                print("Unsupported data type at last position: ", type(format_or_text))

            if self.hier_or_flat == "hier":
                header_rows, header_columns = header_info
                if (header_rows <= 1) and (header_columns <= 1):
                    continue
            elif self.hier_or_flat == "flat":
                header_rows, header_columns = header_info
                if (header_rows > 1) or (header_columns > 1):
                    continue
            sampling_matrix = self.tokenizer.sampling(
                token_matrix=token_matrix, 
                number_matrix=number_matrix, 
                header_info=header_info, 
                max_disturb_num=self.max_disturb_num, 
                disturb_prob=self.disturb_prob, 
                clc_rate=self.clc_rate
            )
            results = self.tokenizer.objective_preprocess(
                sampling_matrix=sampling_matrix, 
                token_matrix=token_matrix, 
                number_matrix=number_matrix, 
                position_lists=position_lists, 
                format_matrix=format_matrix, 
                context=context, 
                add_sep=self.add_separate
            )
            if (results is None) or (len(results[0]) > self.max_cell_num):
                continue
            token_seq = [tok for cell in results[0] for tok in cell]
            if len(token_seq) > self.max_seq_len:
                continue
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
                self._fill_buf()
            if not self.buffer:
                print("Warning: worker {}'s data buffer is empty".format(self.proc_id))

            semi_input = self.buffer[self.start: self.start+self.batch_size] 
            self.start += self.batch_size
            batch_max_seq_len = 0

            all_token_id, all_num_mag, all_num_pre, all_num_top, all_num_low = [], [], [], [], []
            all_token_order, all_pos_top, all_pos_left, all_format_vec, all_indicator = [], [], [], [], [] 
            all_mlm_label, all_clc_label, all_tcr_label = [], [], []

            for (tok_list, num_list, pos_list, fmt_list, cell_ind, cell_mlm, cell_clc, cell_tcr) in semi_input:
                token_id, num_mag, num_pre, num_top, num_low = [], [], [], [], []
                token_order, pos_top, pos_left, format_vec, indicator = [], [], [], [], []
                mlm_label, clc_label, tcr_label = [], [], []

                cell_num = len(tok_list)
                for icell in range(cell_num):
                    tokens = tok_list[icell]
                    cell_len = len(tokens)
                    token_id.extend(tokens)
                    token_order.extend([ii for ii in range(cell_len)])
                    mlm_label.extend(cell_mlm[icell])

                    num_feats = num_list[icell]
                    num_mag.extend([f[0] for f in num_feats])
                    num_pre.extend([f[1] for f in num_feats])
                    num_top.extend([f[2] for f in num_feats])
                    num_low.extend([f[3] for f in num_feats])

                    row, col, ttop, tleft = pos_list[icell]
                    entire_top = UNZIPS[self.target](ttop, self.node_degree, self.total_node)
                    pos_top.extend([entire_top for _ in range(cell_len)])
                    entire_left = UNZIPS[self.target](tleft, self.node_degree, self.total_node)
                    pos_left.extend([entire_left for _ in range(cell_len)])

                    format_vec.extend( [fmt_list[icell] for _ in range(cell_len)] )
                    indicator.extend(cell_ind[icell])
                    clc_label.extend(cell_clc[icell])
                    tcr_label.extend(cell_tcr[icell])

                seq_len = len(token_id)
                if seq_len > self.max_seq_len:  # stop if exceed seq_len bound
                    continue
                batch_max_seq_len = max(batch_max_seq_len, seq_len)

                # append to overall instance set
                all_token_id.append(token_id)
                all_num_mag.append(num_mag)
                all_num_pre.append(num_pre)
                all_num_top.append(num_top)
                all_num_low.append(num_low)
                all_token_order.append(token_order)
                all_pos_top.append(pos_top)
                all_pos_left.append(pos_left)
                all_format_vec.append(format_vec)
                all_indicator.append(indicator)
                all_mlm_label.append(mlm_label)
                all_clc_label.append(clc_label)
                all_tcr_label.append(tcr_label)
                        
            # pad things to batch_max_seq_len
            batch_max_seq_len = ((batch_max_seq_len + 7) // 8) * 8
            for isample in range(self.batch_size):
                all_token_id[isample].extend( [PAD_ID] * (batch_max_seq_len - len(all_token_id[isample])) )
                all_num_mag[isample].extend( [self.magnitude_size + 1] * (batch_max_seq_len - len(all_num_mag[isample])) )
                all_num_pre[isample].extend( [self.precision_size + 1] * (batch_max_seq_len - len(all_num_pre[isample])) )
                all_num_top[isample].extend( [self.top_digit_size + 1] * (batch_max_seq_len - len(all_num_top[isample])) )
                all_num_low[isample].extend( [self.low_digit_size + 1] * (batch_max_seq_len - len(all_num_low[isample])) )

                all_token_order[isample].extend([0] * (batch_max_seq_len - len(all_token_order[isample])))
                all_pos_top[isample].extend( [self.default_pos] * (batch_max_seq_len - len(all_pos_top[isample])) )
                all_pos_left[isample].extend( [self.default_pos] * (batch_max_seq_len - len(all_pos_left[isample])) )

                all_format_vec[isample].extend( [self.default_format] * (batch_max_seq_len - len(all_format_vec[isample])) )
                all_indicator[isample].extend([0] * (batch_max_seq_len - len(all_indicator[isample])))
                all_mlm_label[isample].extend([-1] * (batch_max_seq_len - len(all_mlm_label[isample])))
                all_clc_label[isample].extend([0] * (batch_max_seq_len - len(all_clc_label[isample])))
                all_tcr_label[isample].extend([-1] * (batch_max_seq_len - len(all_tcr_label[isample])))

            yield (
                torch.LongTensor(all_token_id), 
                torch.LongTensor(all_num_mag), 
                torch.LongTensor(all_num_pre), 
                torch.LongTensor(all_num_top), 
                torch.LongTensor(all_num_low), 
                torch.LongTensor(all_token_order), 
                torch.LongTensor(all_pos_top), 
                torch.LongTensor(all_pos_left), 
                torch.FloatTensor(all_format_vec), 
                torch.LongTensor(all_indicator), 
                torch.LongTensor(all_mlm_label), 
                torch.LongTensor(all_clc_label), 
                torch.LongTensor(all_tcr_label)
            )



DataLoaders = {
    "base": DynamicDataLoaderBase, 
    "tuta": DynamicDataLoader, 
    "tuta_explicit": DynamicDataLoader
}
