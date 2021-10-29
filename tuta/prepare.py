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
    
    def build_and_save(self, workers_num):
        # get processing file list
        file_list = os.listdir(self.input_dir)
        file_list = [os.path.join(self.input_dir, f) for f in file_list if f.endswith(".json")]
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
        f_writer = open(self.output_path, "wb")
        for i in range(workers_num):
            tmp_path = os.path.join(self.cache_dir,"dataset-tmp-"+str(i)+".pt")
            tmp_dataset_reader = open(tmp_path, "rb")
            while True:
                try:
                    instances = pickle.load(tmp_dataset_reader)
                    pickle.dump(instances, f_writer)
                except: 
                    break
            tmp_dataset_reader.close()
            os.remove(tmp_path)
        f_writer.close()

    def worker(self, process_id, worker_file_list):
        worker_file_num = len(worker_file_list)

        print("Worker {} is building dataset from {} files... ".format(process_id, worker_file_num))
        tmp_path = os.path.join(self.cache_dir,"dataset-tmp-"+str(process_id)+".pt")
        fw = open(tmp_path, "wb")
        
        # read tables into buffer
        buffer = []
        for i,file_path in enumerate(worker_file_list):
            reader_result = self.reader.result_from_file(file_path)
            if reader_result is not None:
                buffer.append(reader_result)

            if len(buffer) >= self.buffer_size:
                instances = self.build_instances(buffer)
                pickle.dump(instances, fw)
                buffer = []
                print("Worker {}, Processed {:.2f}%".format(process_id, i/worker_file_num*100))
                
        if len(buffer) > 0:
            instances = self.build_instances(buffer)
            pickle.dump(instances, fw)
        fw.close()

    def build_instances(self, buffer):
        """Tokenize strings in buffer. """
        instances = []
        for (string_matrix, position_lists, header_info, format_matrix) in buffer:
            try:
                token_matrix, number_matrix = self.tokenizer.tokenize_string_matrix(string_matrix, self.add_separate)
                instance = (token_matrix, number_matrix, position_lists, header_info, format_matrix)
                if instance is not None: # return 'None' if table unqualified
                    instances.append(instance)
            except:
                continue
        return instances



class WikiDataset(object):
    def __init__(self, args):
        self.reader = rdr.WikiReader(args)
        self.tokenizer = tknr.TutaTokenizer(args)
        self.input_path = args.input_path
        self.cache_dir = args.cache_dir
        self.output_path = args.output_path
        self.buffer_size = args.buffer_size
        self.add_separate = args.add_separate
    
    def build_and_save(self, workers_num):
        # get processing table list
        table_list = self.reader.tables_from_bigfile(self.input_path)
        table_num = len(table_list)
        
        # assign tables to parallel workers
        print("Starting {} workers for building datasets ... ".format(workers_num))
        if workers_num < 1:
            raise ValueError("workers_num should be positive numbers")
        if workers_num == 1:
            self.worker(0, table_list)
        else:
            if workers_num * 10 > table_num:
                workers_num = table_num
                print("Small Data set with {} file, recommend reduce workers_num to {}".format(table_num, workers_num))

            table_num_per_worker = table_num // workers_num
            if (table_num_per_worker * workers_num) < table_num:
                table_num_per_worker += 1
            start, end = 0, table_num_per_worker

            pool = Pool(workers_num)
            for i in range(workers_num):
                worker_table_list = table_list[start: end]
                pool.apply_async(func=self.worker, args=[i, worker_table_list])
                start += table_num_per_worker
                end += table_num_per_worker
            pool.close()
            pool.join()

        # merge intermediate datasets
        f_writer = open(self.output_path, "wb")
        for i in range(workers_num):
            tmp_path = os.path.join(self.cache_dir,"dataset-tmp-"+str(i)+".pt")
            tmp_dataset_reader = open(tmp_path, "rb")
            while True:
                try:
                    instances = pickle.load(tmp_dataset_reader)
                    pickle.dump(instances, f_writer)
                except: 
                    break
            tmp_dataset_reader.close()
            os.remove(tmp_path)
        f_writer.close()

    def worker(self, process_id, worker_table_list):
        worker_table_num = len(worker_table_list)

        print("Worker {} is building dataset from {} files... ".format(process_id, worker_table_num))
        tmp_path = os.path.join(self.cache_dir,"dataset-tmp-"+str(process_id)+".pt")
        fw = open(tmp_path, "wb")
        
        # read tables into buffer
        buffer = []
        for i,table in enumerate(worker_table_list):
            result = self.reader.result_from_table(table)
            if result is not None:
                buffer.append(result)
            if len(buffer) >= self.buffer_size:
                instances = self.build_instances(buffer)
                pickle.dump(instances, fw)
                buffer = []
                print("Worker {}, Processed {:.2f}%".format(process_id, i/worker_table_num*100))
        if len(buffer) > 0:
            instances = self.build_instances(buffer)
            pickle.dump(instances, fw)
        fw.close()

    def build_instances(self, buffer):
        """Tokenize reader-results in buffer. """
        instances = []
        for (string_matrix, position_lists, header_info, title) in buffer:
            try:
                token_matrix, number_matrix = self.tokenizer.tokenize_string_matrix(string_matrix, self.add_separate)
                instance = (token_matrix, number_matrix, position_lists, header_info, title)
                if instance is not None: # return 'None' if table unqualified
                    instances.append(instance)
            except:
                continue
        return instances



class WdcDataset(object):
    def __init__(self, args):
        self.reader = rdr.WdcReader(args)
        self.tokenizer = tknr.TutaTokenizer(args)
        self.input_dir = args.input_dir
        self.cache_dir = args.cache_dir
        self.output_path = args.output_path
        self.buffer_size = args.buffer_size
        self.add_separate = args.add_separate
    
    def build_and_save(self, workers_num):
        # get directories
        dir_list = os.listdir(self.input_dir)
        dir_list = [os.path.join(self.input_dir, d) for d in dir_list if (d != ".DS_Store")]
        print("Start working on {} sub-directories...".format(len(dir_list)))
        for subdir in dir_list:
            print("Start working on directory: ", subdir)
            self.build_subdir(subdir, workers_num)
            print("Finished!")

    def build_subdir(self, subdir, workers_num):
        # get processing file list
        file_list = os.listdir(subdir)
        file_list = [os.path.join(subdir, f) for f in file_list if (f != ".DS_Store")]
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
        f_writer = open(self.output_path, "wb")
        for i in range(workers_num):
            tmp_path = os.path.join(self.cache_dir,"dataset-tmp-"+str(i)+".pt")
            tmp_dataset_reader = open(tmp_path, "rb")
            while True:
                try:
                    instances = pickle.load(tmp_dataset_reader)
                    pickle.dump(instances, f_writer)
                except:
                    break
            tmp_dataset_reader.close()
            os.remove(tmp_path)
        f_writer.close()

    def worker(self, process_id, worker_file_list):
        worker_file_num = len(worker_file_list)

        print("Worker {} is building dataset from {} files... ".format(process_id, worker_file_num))
        tmp_path = os.path.join(self.cache_dir,"dataset-tmp-"+str(process_id)+".pt")
        fw = open(tmp_path, "wb")
        
        # read tables into buffer
        buffer = []
        for i,file_path in enumerate(worker_file_list):
            reader_results = self.reader.results_from_bigfile(file_path)
            if reader_results is not None:
                buffer.extend(reader_results)

            if len(buffer) >= self.buffer_size:
                instances = self.build_instances(buffer)
                pickle.dump(instances, fw)
                buffer = []
                print("Worker {}, Processed {:.2f}%".format(process_id, i/worker_file_num*100))
        if len(buffer) > 0:
            instances = self.build_instances(buffer)
            pickle.dump(instances, fw)
        fw.close()

    def build_instances(self, buffer):
        """Tokenize strs in buffer. """
        instances = []
        for (string_matrix, position_lists, header_info, title) in buffer:
            try:
                token_matrix, number_matrix = self.tokenizer.tokenize_string_matrix(string_matrix, self.add_separate)
                instance = (token_matrix, number_matrix, position_lists, header_info, title)
                if instance is not None: # return 'None' if table unqualified
                    instances.append(instance)
            except:
                continue
        return instances



# %% Main Pipeline 

SETS = {"sheet": SheetDataset, "wiki": WikiDataset, "wdc": WdcDataset}


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # I/O options.
    parser.add_argument("--input_dir", type=str, default='../data/pretrain/spreadsheet', help="Input directory of pre-training table files.")
    parser.add_argument("--input_path", type=str, default='../data/pretrain/wiki-table-samples.json', help="Input single file containing multiple tables.")
    parser.add_argument("--source_type", type=str, default='wiki', choices=["sheet", "wiki", "wdc"])
    parser.add_argument("--cache_dir", type=str, default="./", help="Folder to save workers cache.")
    parser.add_argument("--output_path", type=str, default='../dataset.pt', help="Path to save the pre-processed dataset.")
    
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

    args = parser.parse_args()
    args.node_degree = [int(degree) for degree in args.node_degree.split(',')]
                             
    # Build and save dataset.
    dataset = SETS[args.source_type](args)
    dataset.build_and_save(args.processes_num)


if __name__ == "__main__":
    main()
