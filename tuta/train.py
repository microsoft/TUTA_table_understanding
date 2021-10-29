#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do Pre-Training of TUTA Model (variants)

"""

import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

import tokenizer as tknr
import model.pretrains as ptm
import trainers as tnr
import dynamic_data as dymdata

from utils import init_tuta_loose, init_with_bert_weight
from optimizer import AdamW, WarmupLinearSchedule
from torch.nn.parallel import DistributedDataParallel



def worker(proc_id, gpu_ranks, args, model):
    if args.dist_train:    # multiple GPU mode
        rank = gpu_ranks[proc_id] % args.world_size
        gpu_id = gpu_ranks[proc_id] % args.device_count
    elif args.single_gpu:  # single GPU mode
        rank = None
        gpu_id = proc_id
    else:  # CPU mode
        rank = None
        gpu_id = None

    if args.dist_train:
        train_loader = dymdata.DataLoaders[args.target](args, rank, args.world_size, True)
    else:
        train_loader = dymdata.DataLoaders[args.target](args, 0, 1, True)

    if gpu_id is not None: 
        torch.cuda.set_device(gpu_id)
        model.cuda(gpu_id)

    # build optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay)) ], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) ], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.total_steps*args.warmup, t_total=args.total_steps)

    if args.dist_train:
        # initialize multiprocessing distributed training environment
        dist.init_process_group(
            backend=args.backend,
            init_method=args.master_ip,
            world_size=args.world_size,
            rank=rank
        )
        model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)  # find_unused_parameters=True
        print("Worker {} is training ... ".format(rank))
    else:
        print("Worker is training ...")

    tnr.TRAINERS[args.target](args, gpu_id, rank, train_loader, model, optimizer, scheduler)


def train_and_validate(args):
    args.tokenizer = tknr.TutaTokenizer(args)
    args.vocab_size = len(args.tokenizer.vocab)

    model = ptm.MODELS[args.target](args)
    if args.load_type == "bert":
        model = init_with_bert_weight(args, model)
    elif args.load_type == "tuta":
        init_tuta_loose(model=model, tuta_path=args.pretrained_model_path)
    else:
        init_tuta_loose(model=model, tuta_path=None)

    if args.dist_train:   # multiple GPU mode
        mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args, model), daemon=False)
    elif args.single_gpu: # single GPU mode
        worker(args.gpu_id, None, args, model)
    else:                 # CPU mode
        worker(None, None, args, model)



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # vocabulary options
    parser.add_argument("--vocab_path", type=str, default="./vocab/bert_vocab.txt", help="Path of the vocabulary file.")
    parser.add_argument("--context_repo_path", type=str, default="./vocab/context_repo_init.txt", help="TXT of pre-collected context pieces.")
    parser.add_argument("--cellstr_repo_path", type=str, default="./vocab/cellstr_repo_init.txt", help="TXT of pre-collected context pieces.")
    
    # model configuration options
    parser.add_argument("--hidden_size", type=int, default=768, help="Size of the hidden states.")
    parser.add_argument("--intermediate_size", type=int, default=3072, help="Size of the intermediate layer.")
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

    # verion options
    parser.add_argument("--target", type=str, default="tuta", choices=["tuta", "tuta_explicit", "base"], help="Model variants.")
    parser.add_argument("--attn_method", type=str, default="add", choices=["max", "add"])
    
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
    parser.add_argument("--clc_weight", type=float, default=1.0, help="Weight assigned to clc loss.")

    # training options
    parser.add_argument("--batch_size", type=int, default=12, help="Size of the input batch.")
    parser.add_argument("--total_steps", type=int, default=1000000, help="Total training steps.")
    parser.add_argument("--report_steps", type=int, default=100, help="Specific steps to print prompt.")
    parser.add_argument("--save_checkpoint_steps", type=int, default=100000, help="Specific steps to save model checkpoint.")
    parser.add_argument("--buffer_size", type=int, default=500000, help="The buffer size of instances in memory.")
    parser.add_argument("--chunk_size", type=int, default=50000, help="Mininum chunk size from a random data set.")

    # io options
    parser.add_argument("--dataset_paths", type=str, default='../dataset.pt', help="Paths of the preprocessed dataset.")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Path of the pretrained bert/ts model.")
    parser.add_argument("--load_type", type=str, default="tuta", choices=["tuta", "bert", None])
    parser.add_argument("--output_model_path", type=str, default="tuta.bin", help="Path of the output model.")
    
    # optimizer options
    parser.add_argument("--warmup", type=float, default=0.1, help="Warm up value.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate.")

    # gpu options
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes (GPUs) for training.")
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int, help="List of ranks of each process."
                        " Each process has a unique integer rank whose value in the interval [0, world_size], and runs in a single GPU.")
    parser.add_argument("--master_ip", default="tcp://localhost:12345", type=str, help="IP-Port of master for training.")
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl", type=str, help="Distributed backend.")
    
    args = parser.parse_args()

    args.node_degree = [int(degree) for degree in args.node_degree.split(',')]
    if args.target == "tuta_explicit":
        args.node_degree = [32, 32, 64, 160]
    print("node degree: ", args.node_degree)

    # convert '+'-connected dataset_paths into list of strings
    args.dataset_paths = args.dataset_paths.split('+')

    ranks_num = len(args.gpu_ranks)
    if args.world_size > 1:
        assert torch.cuda.is_available(), "No available GPUs." 
        assert ranks_num <= args.world_size, "Started processes exceed `world_size` upper limit." 
        assert ranks_num <= torch.cuda.device_count(), "Started processes exceeds the available GPUs." 
        # multiple GPU mode
        args.dist_train = True
        args.ranks_num = ranks_num
        args.device_count = torch.cuda.device_count()
        print("Using distributed mode for training.")
    elif args.world_size == 1 and ranks_num == 1:
        assert torch.cuda.is_available(), "No available GPUs." 
        # single GPU mode.
        args.gpu_id = args.gpu_ranks[0]
        assert args.gpu_id <= torch.cuda.device_count(), "Invalid specified GPU device." 
        args.dist_train = False
        args.single_gpu = True
        print("Using single GPU: {} for training.".format(args.gpu_id))
    else:
        # CPU mode.
        assert ranks_num == 0, "GPUs are specified, please check the arguments."
        args.dist_train = False
        args.single_gpu = False
        print("Using CPU mode for training.")

    train_and_validate(args)


if __name__ == "__main__":
    main()
