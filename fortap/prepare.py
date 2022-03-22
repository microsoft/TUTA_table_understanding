#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Data pre-processing. 

Build a dataset pre-processing object according to the source types. 
Distributed building and saving with multiple workers.
"""


import os
import pickle
import argparse
import reader as rdr
import tokenizer as tknr
from multiprocessing import Pool
from icecream import ic


# %% Data Set Pre-processors for tables
class SheetDataset(object):
    def __init__(self, args):
        self.reader = rdr.SheetReader(args)
        self.tokenizer = tknr.TutaTokenizer(args)
        self.input_dir = args.input_dir
        self.cache_dir = args.cache_dir
        self.output_path = args.output_path
        self.buffer_size = args.buffer_size
        self.add_separate = args.add_separate
        # self.current_fold = args.current_fold
        self.num_folds = args.num_folds
    
    def build_and_save(self, workers_num):
        # get processing file list
        file_list = sorted(os.listdir(self.input_dir))
        # file_list = file_list[: 50000]
        file_list = [os.path.join(self.input_dir, f) for f in file_list if f.endswith(".json") or f.endswith(".formula")]
        file_num = len(file_list)
        
        # assign works to parallel workers
        print("Starting {} workers for building datasets ... ".format(workers_num))
        if workers_num < 1:
            raise ValueError("workers_num should be positive numbers")
        if workers_num == 1:
            self.worker(0, file_list)
        else:
            if workers_num * 10 > file_num:
                workers_num = file_num
                print("Small Data set with {} file, recommend reduce workers_num to {}".format(file_num, workers_num))

            file_num_per_worker = file_num // workers_num
            if (file_num_per_worker * workers_num) < file_num:
                file_num_per_worker += 1
            start, end = 0, file_num_per_worker

            pool = Pool(workers_num)
            for i in range(workers_num):
                worker_file_list = file_list[start: end]
                pool.apply_async(func=self.worker, args=[i, worker_file_list])
                start += file_num_per_worker
                end += file_num_per_worker
            pool.close()
            pool.join()

        # merge intermediate datasets
        if self.num_folds == 1:
            fds = [open(self.output_path, "wb")]
        elif self.num_folds > 1:
            assert workers_num >= self.num_folds
            fds = [open(os.path.splitext(self.output_path)[0] + f'-{i}.pt', 'wb') for i in range(self.num_folds)]
        else:
            raise ValueError("num-folds must be >= 1.")
        dump_table_num, dump_formula_num = [0] * self.num_folds, [0] * self.num_folds
        for i in range(workers_num):
            ic(i)
            tmp_path = os.path.join(self.cache_dir,"dataset-tmp-"+str(i)+".pt")
            tmp_dataset_reader = open(tmp_path, "rb")
            while True:
                try:
                    instances = pickle.load(tmp_dataset_reader)
                    fold_idx = i % self.num_folds
                    dump_table_num[fold_idx] += len(instances)
                    dump_formula_num[fold_idx] += sum([len(ins[-3]) for ins in instances])  # TODO: -3 because add file_path and meta at the end
                    pickle.dump(instances, fds[fold_idx])
                except: 
                    break
            tmp_dataset_reader.close()
        ic(dump_table_num)
        ic(dump_formula_num)
        for f in fds:
            f.close()

    def worker(self, process_id, worker_file_list):
        worker_file_num = len(worker_file_list)

        print("Worker {} is building dataset from {} files... ".format(process_id, worker_file_num))
        tmp_path = os.path.join(self.cache_dir,"dataset-tmp-"+str(process_id)+".pt")
        fw = open(tmp_path, "wb")
        
        # read tables into buffer
        buffer = []
        for i,file_path in enumerate(worker_file_list):
            print(f"Worker#{process_id} preparing num#{i}: {file_path} ")
            try:
                reader_result = self.reader.result_from_file(file_path)
                if reader_result is not None:  # 'None' means failure
                    buffer.append(reader_result)
                print(f"Worker#{process_id} Num {len(buffer)}")
                if len(buffer) >= self.buffer_size:
                    instances = self.build_instances(buffer)
                    pickle.dump(instances, fw)
                    buffer = []
                    print("Worker {}, Processed {:.2f}%".format(process_id, i/worker_file_num*100))
            except Exception as e:
                print(e)
                
        if len(buffer) > 0:
            instances = self.build_instances(buffer)
            pickle.dump(instances, fw)
        # ic(dump_table_num, dump_formula_num)  # dump_len * worker_num = final dump_len
        log_f = open('./dump_log.txt', 'a')
        log_f.write(f'Dump worker#{process_id} finished!\n')
        log_f.close()
        fw.close()

    def build_instances(self, buffer):
        """Tokenize strings in buffer. """
        instances = []
        for (string_matrix, position_lists, header_info, format_matrix, table_range, formula_dict, file_path, meta_data) in buffer:
            try:
                token_matrix, number_matrix = self.tokenizer.tokenize_string_matrix(string_matrix, self.add_separate)
                instance = (token_matrix, number_matrix, position_lists, header_info, format_matrix,
                            table_range, formula_dict, file_path, meta_data)
                if instance is not None: # return 'None' if table unqualified
                    instances.append(instance)
            except Exception as e:
                print(e)
        return instances


# %% Main Pipeline

SETS = {"sheet": SheetDataset}


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # I/O options.
    parser.add_argument("--input_dir", type=str, default='./data/pretrain/spreadsheet', help="Input directory of pre-training table files.")
    parser.add_argument("--input_path", type=str, default='./data/pretrain/wiki-table-samples.json', help="Input single file containing multiple tables.")
    parser.add_argument("--source_type", type=str, default='wiki', choices=["sheet", "wiki", "wdc"])
    parser.add_argument("--cache_dir", type=str, default="./tuta/", help="Folder to save workers cache.")
    parser.add_argument("--output_path", type=str, default='./dataset.pt', help="Path to save the pre-processed dataset.")
    
    # Preprocess options.
    parser.add_argument("--vocab_path", type=str, default="./vocab/bert_vocab.txt", help="Path of the vocabulary file.")
    parser.add_argument("--context_repo_path", type=str, default="./vocab/context_repo_init.txt", help="TXT of pre-collected context pieces.")
    parser.add_argument("--cellstr_repo_path", type=str, default="./vocab/cellstr_repo_init.txt", help="TXT of pre-collected context pieces.")
    
    # model configuration options
    parser.add_argument("--magnitude_size", type=int, default=10, help="Max magnitude of numeric values.")
    parser.add_argument("--precision_size", type=int, default=10, help="Max precision of numeric values.")
    parser.add_argument("--top_digit_size", type=int, default=10, help="Most significant digit from '0' to '9'.")
    parser.add_argument("--low_digit_size", type=int, default=10, help="Least significant digit from '0' to '9'.")
    parser.add_argument("--row_size", type=int, default=256, help="Max number of rows in table.")
    parser.add_argument("--column_size", type=int, default=256, help="Max number of columns in table.")
    parser.add_argument("--tree_depth", type=int, default=4, help="Maximum depth of top & left header tree.")
    parser.add_argument("--node_degree", type=str, default="32,32,64,256", help="Maximum number of children of each tree node.")
    parser.add_argument("--num_format_feature", type=int, default=11, help="Number of features of the format vector.")
    parser.add_argument("--attention_distance", type=int, default=8, help="Maximum distance for attention visibility.")
    parser.add_argument("--attention_step", type=int, default=0, help="Step size of attention distance to add for each layer.")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="Number of the attention heads.")
    parser.add_argument("--num_encoder_layers", type=int, default=12, help="Number of the encoding layers.")
    parser.add_argument("--num_tcr_type", type=int, default=2, help="Number of table-context classes.")
    parser.add_argument("--hidden_dropout_prob", type=int, default=0.1, help="Dropout probability for hidden layers.")
    parser.add_argument("--attention_dropout_prob", type=int, default=0.1, help="Dropout probability for attention.")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-6)
    parser.add_argument("--hidden_act", type=str, default="gelu", help="Activation function for hidden layers.")    
    # data size/processing options
    parser.add_argument("--max_seq_len", type=int, default=256, help="Maximum length of the table sequence.")
    parser.add_argument("--max_cell_num", type=int, default=256, help="Maximum cell number used in data loaders.")
    parser.add_argument("--max_cell_length", type=int, default=64, help="Maximum number of tokens in one cell string.")
    parser.add_argument("--max_disturb_num", type=int, default=20, help="Maximum number of cells to be disturbed per table.")
    parser.add_argument("--disturb_prob", type=float, default=0.15, help="Probability to be disturbed per cell.")
    parser.add_argument("--add_separate", type=bool, default=True, help="Whether to add [SEP] as aggregate cell representation.")
    parser.add_argument("--text_threshold", type=float, default=0.5, help="Probability threshold to sample text in data region.")
    parser.add_argument("--value_threshold", type=float, default=0.1, help="Prob to sample value in data region.")
    parser.add_argument("--clc_rate", type=float, default=0.3)
    parser.add_argument("--hier_or_flat", type=str, default="both", choices=["hier", "flat", "both"])
    parser.add_argument("--wcm_rate", type=float, default=0.3, help="Proportion of masked cells doing whole-cell-masking.")    

    parser.add_argument("--buffer_size", type=int, default=500000, help="The buffer size of documents in memory.")
    parser.add_argument("--valid_num", type=int, default=100000, help="Number of tables to be included in the validation set.")
    parser.add_argument("--processes_num", type=int, default=1, help="Split the whole dataset and process in parallel.")

    parser.add_argument("--num_folds", type=int, default=1, help="number of all folds")

    args = parser.parse_args()
    args.node_degree = [int(degree) for degree in args.node_degree.split(',')]
                             
    # Build and save dataset.
    dataset = SETS[args.source_type](args)
    dataset.build_and_save(args.processes_num)


if __name__ == "__main__":
    main()
