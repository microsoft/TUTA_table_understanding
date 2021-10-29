#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainers for Pre-training Model Variants
"""

import time
import torch
from utils import save_model


def train_base(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss = 0.
    total_mlm_loss, total_mlm_crt, total_mlm_cnt = 0., 0., 0.
    total_sep_loss, total_sep_crt, total_sep_cnt = 0., 0., 0.
    total_tok_loss, total_tok_crt, total_tok_cnt = 0., 0., 0.
    total_tcr_loss, total_tcr_crt, total_tcr_cnt = 0., 0., 0.
    total_rand_num, total_rand_den = 0., 0.
    
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        token_id, num_mag, num_pre, num_top, num_low, \
        token_order, pos_top, pos_left, format_vec, indicator, \
        mlm_label, clc_label, tcr_label = next(loader_iter)

        while (clc_label.size()[0] == 0) or (torch.min(clc_label) >= 0) or (torch.max(mlm_label) == -1):
            token_id, num_mag, num_pre, num_top, num_low, \
            token_order, pos_top, pos_left, format_vec, indicator, \
            mlm_label, clc_label, tcr_label = next(loader_iter)

        model.zero_grad()
        if gpu_id is not None:
            token_id = token_id.cuda(gpu_id)
            num_mag = num_mag.cuda(gpu_id)
            num_pre = num_pre.cuda(gpu_id)
            num_top = num_top.cuda(gpu_id)
            num_low = num_low.cuda(gpu_id)

            token_order = token_order.cuda(gpu_id)
            pos_top = pos_top.cuda(gpu_id)
            pos_left = pos_left.cuda(gpu_id)
            format_vec = format_vec.cuda(gpu_id)
            indicator = indicator.cuda(gpu_id)

            mlm_label = mlm_label.cuda(gpu_id)
            clc_label = clc_label.cuda(gpu_id)
            tcr_label = tcr_label.cuda(gpu_id)
        
        # forward
        mlm_triple, sep_triple, tok_triple, tcr_triple = model(
            token_id, num_mag, num_pre, num_top, num_low, 
            token_order, pos_top, pos_left, format_vec, indicator, 
            mlm_label, clc_label, tcr_label
        )
        mlm_loss, mlm_crt, mlm_cnt = mlm_triple
        sep_loss, sep_crt, sep_cnt = sep_triple
        tok_loss, tok_crt, tok_cnt = tok_triple
        tcr_loss, tcr_crt, tcr_cnt = tcr_triple

        # backward
        loss = mlm_loss + (sep_loss + tok_loss) * args.clc_weight + tcr_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_mlm_loss += mlm_loss.item()
        total_mlm_crt += mlm_crt.item()
        total_mlm_cnt += mlm_cnt.item()

        total_sep_loss += sep_loss.item()
        total_sep_crt += sep_crt.item()
        total_sep_cnt += sep_cnt.item()

        total_tok_loss += tok_loss.item()
        total_tok_crt += tok_crt.item()
        total_tok_cnt += tok_cnt.item()

        rand_num = torch.sum((clc_label > 0).long(), dim=-1)
        rand_den = torch.sum(rand_num, dim=-1).item() // 2
        rand_num = torch.sum((rand_num > 0).long(), dim=-1).item()
        total_rand_num += rand_num
        total_rand_den += rand_den

        total_tcr_loss += tcr_loss.item()
        total_tcr_crt += tcr_crt.item()
        total_tcr_cnt += tcr_cnt.item()

        total_loss = total_mlm_loss + (total_sep_loss + total_tok_loss) * args.clc_weight + total_tcr_loss
        
        if steps % args.report_steps == 0  and (not args.dist_train or (args.dist_train and rank == 0)):
            elapsed = time.time() - start_time
            done_tokens = \
                args.batch_size * token_id.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * token_id.size(1) * args.report_steps 
            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| total_loss {:7.2f}"
                  "| mlm_loss {:7.2f}"
                  "| mlm_acc {:.3f}"
                  "| sep_loss {:7.2f}"
                  "| sep_acc {:.3f}"
                  "| tok_loss {:7.2f}"
                  "| tok_acc {:.3f}"
                  "| rand_acc {:.3f}"
                  "| tcr_loss {:7.2f}"
                  "| tcr_acc {:.3f}".format(
                    steps, total_steps, 
                    done_tokens / elapsed, 
                    total_loss / args.report_steps, 
                    total_mlm_loss / args.report_steps, 
                    total_mlm_crt / total_mlm_cnt, 
                    total_sep_loss / args.report_steps, 
                    total_sep_crt / total_sep_cnt, 
                    total_tok_loss / args.report_steps, 
                    total_tok_crt / total_tok_cnt, 
                    total_rand_num / total_rand_den, 
                    total_tcr_loss / args.report_steps, 
                    total_tcr_crt / total_tcr_cnt))   
            total_loss = 0.
            total_mlm_loss, total_mlm_crt, total_mlm_cnt = 0., 0., 0.
            total_sep_loss, total_sep_crt, total_sep_cnt = 0., 0., 0.
            total_tok_loss, total_tok_crt, total_tok_cnt = 0., 0., 0.
            total_tcr_loss, total_tcr_crt, total_tcr_cnt = 0., 0., 0.
            total_rand_num, total_rand_den = 0., 0.
            start_time = time.time()
        if steps % args.save_checkpoint_steps == 0 and (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))
        steps += 1


def train_tuta(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss = 0.
    total_mlm_loss, total_mlm_crt, total_mlm_cnt = 0., 0., 0.
    total_sep_loss, total_sep_crt, total_sep_cnt = 0., 0., 0.
    total_tok_loss, total_tok_crt, total_tok_cnt = 0., 0., 0.
    total_tcr_loss, total_tcr_crt, total_tcr_cnt = 0., 0., 0.
    total_rand_num, total_rand_den = 0., 0.
    
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        token_id, num_mag, num_pre, num_top, num_low, \
        token_order, pos_row, pos_col, pos_top, pos_left, format_vec, indicator, \
        mlm_label, clc_label, tcr_label = next(loader_iter)

        while (clc_label.size()[0] == 0) or (torch.min(clc_label) >= 0) or (torch.max(mlm_label) == -1):
            token_id, num_mag, num_pre, num_top, num_low, \
            token_order, pos_row, pos_col, pos_top, pos_left, format_vec, indicator, \
            mlm_label, clc_label, tcr_label = next(loader_iter)

        model.zero_grad()
        if gpu_id is not None:
            token_id = token_id.cuda(gpu_id)
            num_mag = num_mag.cuda(gpu_id)
            num_pre = num_pre.cuda(gpu_id)
            num_top = num_top.cuda(gpu_id)
            num_low = num_low.cuda(gpu_id)

            token_order = token_order.cuda(gpu_id)
            pos_row = pos_row.cuda(gpu_id)
            pos_col = pos_col.cuda(gpu_id)
            pos_top = pos_top.cuda(gpu_id)
            pos_left = pos_left.cuda(gpu_id)

            format_vec = format_vec.cuda(gpu_id)
            indicator = indicator.cuda(gpu_id)

            mlm_label = mlm_label.cuda(gpu_id)
            clc_label = clc_label.cuda(gpu_id)
            tcr_label = tcr_label.cuda(gpu_id)

        
        # forward
        mlm_triple, sep_triple, tok_triple, tcr_triple = model(
            token_id, num_mag, num_pre, num_top, num_low, 
            token_order, pos_row, pos_col, pos_top, pos_left, format_vec, indicator, 
            mlm_label, clc_label, tcr_label
        )
        mlm_loss, mlm_crt, mlm_cnt = mlm_triple
        sep_loss, sep_crt, sep_cnt = sep_triple
        tok_loss, tok_crt, tok_cnt = tok_triple
        tcr_loss, tcr_crt, tcr_cnt = tcr_triple

        # backward
        loss = mlm_loss + (sep_loss + tok_loss) * args.clc_weight + tcr_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_mlm_loss += mlm_loss.item()
        total_mlm_crt += mlm_crt.item()
        total_mlm_cnt += mlm_cnt.item()

        total_sep_loss += sep_loss.item()
        total_sep_crt += sep_crt.item()
        total_sep_cnt += sep_cnt.item()

        total_tok_loss += tok_loss.item()
        total_tok_crt += tok_crt.item()
        total_tok_cnt += tok_cnt.item()

        rand_num = torch.sum((clc_label > 0).long(), dim=-1)
        rand_den = torch.sum(rand_num, dim=-1).item() // 2
        rand_num = torch.sum((rand_num > 0).long(), dim=-1).item()
        total_rand_num += rand_num
        total_rand_den += rand_den

        total_tcr_loss += tcr_loss.item()
        total_tcr_crt += tcr_crt.item()
        total_tcr_cnt += tcr_cnt.item()

        total_loss = total_mlm_loss + (total_sep_loss + total_tok_loss) * args.clc_weight + total_tcr_loss
        
        if steps % args.report_steps == 0  and (not args.dist_train or (args.dist_train and rank == 0)):
            elapsed = time.time() - start_time
            done_tokens = \
                args.batch_size * token_id.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * token_id.size(1) * args.report_steps 
            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| total_loss {:7.2f}"
                  "| mlm_loss {:7.2f}"
                  "| mlm_acc {:.3f}"
                  "| sep_loss {:7.2f}"
                  "| sep_acc {:.3f}"
                  "| tok_loss {:7.2f}"
                  "| tok_acc {:.3f}"
                  "| rand_acc {:.3f}"
                  "| tcr_loss {:7.2f}"
                  "| tcr_acc {:.3f}".format(
                    steps, total_steps, 
                    done_tokens / elapsed, 
                    total_loss / args.report_steps, 
                    total_mlm_loss / args.report_steps, 
                    total_mlm_crt / total_mlm_cnt, 
                    total_sep_loss / args.report_steps, 
                    total_sep_crt / total_sep_cnt, 
                    total_tok_loss / args.report_steps, 
                    total_tok_crt / total_tok_cnt, 
                    total_rand_num / total_rand_den, 
                    total_tcr_loss / args.report_steps, 
                    total_tcr_crt / total_tcr_cnt))   
            total_loss = 0.
            total_mlm_loss, total_mlm_crt, total_mlm_cnt = 0., 0., 0.
            total_sep_loss, total_sep_crt, total_sep_cnt = 0., 0., 0.
            total_tok_loss, total_tok_crt, total_tok_cnt = 0., 0., 0.
            total_tcr_loss, total_tcr_crt, total_tcr_cnt = 0., 0., 0.
            total_rand_num, total_rand_den = 0., 0.
            start_time = time.time()
        if steps % args.save_checkpoint_steps == 0 and (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))
        steps += 1



TRAINERS = {
    "base": train_base, 
    "tuta": train_tuta, 
    "tuta_explicit": train_tuta, 
}
