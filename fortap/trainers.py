#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainers for Pre-training Model Variants
"""

import time
import torch
import torch.nn.functional as F
from icecream import ic
import json

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

        if steps % args.report_steps == 0 and (not args.dist_train or (args.dist_train and rank == 0)):
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
        if steps % args.save_checkpoint_steps == 0 and (
                not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))
        steps += 1


def train_tuta_fp(args, gpu_id, rank, loader, model, optimizer, scheduler):
    if args.test:
        model.eval()
    else:
        model.train()
    start_time = time.time()
    total_loss = 0.
    total_sketch_loss, total_sketch_top1_crt, total_sketch_top5_crt, total_sketch_cnt = 0., 0., 0., 0.
    total_range_loss, total_range_crt, total_range_cnt = 0, 0, 0
    total_fp_loss, total_fp_top1_crt, total_fp_top5_crt, total_fp_cnt = 0, 0, 0, 0

    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        try:
            next_batch = next(loader_iter)
        except:  # testin done
            print('Test done.')
            msg = "sketch_top1_acc {:7.3f}" \
                  "| sketch_top5_acc {:7.3f}" \
                  "| range_acc {:7.3f}" \
                  "| fp_top1_acc {:7.3f}" \
                  "| fp_top5_acc {:7.3f}".format(
                total_sketch_top1_crt / (total_sketch_cnt + 1e-9),  # record of all
                total_sketch_top5_crt / (total_sketch_cnt + 1e-9),
                total_range_crt / (total_range_cnt + 1e-9),
                total_fp_top1_crt / (total_fp_cnt + 1e-9),
                total_fp_top5_crt / (total_fp_cnt + 1e-9)
            )
            print(msg)
            break
        token_id, num_mag, num_pre, num_top, num_low, \
        token_order, pos_row, pos_col, pos_top, pos_left, format_vec, indicator, \
        formula_label, src_sketch, tgt_sketch, \
        candi_cell_token_mask, range_label, range_map = next_batch

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

            formula_label = formula_label.cuda(gpu_id)
            src_sketch = src_sketch.cuda(gpu_id)
            tgt_sketch = tgt_sketch.cuda(gpu_id)

            candi_cell_token_mask = candi_cell_token_mask.cuda(gpu_id)
            range_label = range_label.cuda(gpu_id)

        # forward
        fp_tuple = model(
            token_id, num_mag, num_pre, num_top, num_low,
            token_order, pos_row, pos_col, pos_top, pos_left, format_vec, indicator,
            formula_label, src_sketch, tgt_sketch,
            candi_cell_token_mask, range_label, range_map
        )
        sketch_loss, range_loss, sketch_top1_crt, sketch_top5_crt, sketch_cnt, range_crt, range_cnt, fp_top1_crt, fp_top5_crt, fp_cnt = fp_tuple

        # backward
        if steps <= args.sketch_warmup_steps:
            range_loss_weight = 0.
        else:
            range_loss_weight = args.range_loss_weight
        loss = sketch_loss + range_loss_weight * range_loss
        if not args.test:
            loss.backward()
            optimizer.step()
            scheduler.step()

        total_loss += (sketch_loss.item() + range_loss_weight * range_loss.item())

        total_sketch_loss += sketch_loss.item()
        total_sketch_top1_crt += sketch_top1_crt
        total_sketch_top5_crt += sketch_top5_crt
        # ic(sketch_top1_crt)
        # ic(sketch_top5_crt)
        # ic(total_sketch_top1_crt)
        # ic(total_sketch_top5_crt)
        # ic(total_sketch_cnt)
        total_sketch_cnt += sketch_cnt
        total_range_loss += range_loss_weight * range_loss.item()
        total_range_crt += range_crt
        total_range_cnt += range_cnt
        total_fp_loss += (sketch_loss.item() + range_loss_weight * range_loss.item())
        total_fp_top1_crt += fp_top1_crt
        total_fp_top5_crt += fp_top5_crt
        total_fp_cnt += fp_cnt

        if steps % args.report_steps == 0 and (not args.dist_train or (args.dist_train and rank == 0)):
            elapsed = time.time() - start_time
            done_tokens = \
                args.batch_size * token_id.size(1) * args.report_steps * args.world_size \
                    if args.dist_train \
                    else args.batch_size * token_id.size(1) * args.report_steps
            msg = "process#{} " \
                  "| {:8d}/{:8d} steps" \
                  "| total_loss {:7.2f}" \
                  "| sketch_loss {:7.2f}" \
                  "| sketch_top1_acc {:7.3f}" \
                  "| sketch_top5_acc {:7.3f}" \
                  "| range_loss {:7.2f}" \
                  "| range_acc {:7.3f}" \
                  "| fp_top1_acc {:7.3f}" \
                  "| fp_top5_acc {:7.3f}".format(
                rank,
                steps, total_steps,
                total_loss / args.report_steps,
                total_sketch_loss / args.report_steps,  # record of report steps
                total_sketch_top1_crt / (total_sketch_cnt + 1e-9),  # record of all
                total_sketch_top5_crt / (total_sketch_cnt + 1e-9),
                total_range_loss / args.report_steps,
                total_range_crt / (total_range_cnt + 1e-9),
                total_fp_top1_crt / (total_fp_cnt + 1e-9),
                total_fp_top5_crt / (total_fp_cnt + 1e-9)
            )
            print(msg)
            total_loss = 0.
            total_sketch_loss, total_range_loss, total_fp_loss = 0., 0., 0.
            start_time = time.time()
        if steps % args.save_checkpoint_steps == 0 and (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))
        steps += 1


def train_tuta_fortap(args, gpu_id, rank, loader, model, optimizer, scheduler):
    if args.test:
        model.eval()
    else:
        model.train()
    start_time = time.time()
    total_loss = 0.
    total_mlm_loss, total_mlm_crt, total_mlm_cnt = 0., 0., 0.
    total_sr_loss, total_sr_samples, total_sr_correct = 0., 0., 0.
    total_nr_loss, total_nr_samples, total_nr_correct = 0., 0., 0.
    total_op_mlm_loss, total_op_mlm_crt, total_op_mlm_cnt = 0., 0., 0.
    total_range_mlm_loss, total_range_mlm_crt, total_range_mlm_cnt = 0., 0., 0.
    total_sr_context_loss = 0
    total_sr_context_stats = {}
    num_sr_context_classes = 6
    for target in range(num_sr_context_classes):
        total_sr_context_stats[target] = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        total_sr_context_stats[target] = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    cnt_skip_batch = 0
    while True:
        if steps == total_steps + 1:
            break
        token_id, num_mag, num_pre, num_top, num_low, \
        token_order, pos_row, pos_col, pos_top, pos_left, format_vec, indicator, \
        mlm_label, sr_label, nr_label, op_appear_flag, sr_context_label, \
        op_mlm_label, range_mlm_label, candi_cell_token_mask \
            = next(loader_iter)

        # skip batch
        while (sr_label.size()[0] == 0) or (torch.sum(sr_label == 1) == 0):
            token_id, num_mag, num_pre, num_top, num_low, \
            token_order, pos_row, pos_col, pos_top, pos_left, format_vec, indicator, \
            mlm_label, sr_label, nr_label, op_appear_flag, sr_context_label, \
            op_mlm_label, range_mlm_label, candi_cell_token_mask \
                = next(loader_iter)
            cnt_skip_batch += 1

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

            sr_label = sr_label.cuda(gpu_id)
            nr_label = nr_label.cuda(gpu_id)
            op_appear_flag = op_appear_flag.cuda(gpu_id)
            sr_context_label = sr_context_label.cuda(gpu_id)

            op_mlm_label = op_mlm_label.cuda(gpu_id)
            range_mlm_label = range_mlm_label.cuda(gpu_id)
            candi_cell_token_mask = candi_cell_token_mask.cuda(gpu_id)

        # forward
        mlm_triple, sr_triple, nr_triple, sr_context_tuple, op_mlm_triple, range_mlm_triple = model(
            token_id, num_mag, num_pre, num_top, num_low,
            token_order, pos_row, pos_col, pos_top, pos_left, format_vec, indicator,
            mlm_label, sr_label, nr_label, op_appear_flag, sr_context_label,
            op_mlm_label, range_mlm_label, candi_cell_token_mask
        )
        mlm_loss, mlm_crt, mlm_cnt = mlm_triple
        sr_loss, sr_logits, sr_labels = sr_triple
        nr_loss, nr_logits, nr_labels = nr_triple
        op_mlm_loss, op_mlm_crt, op_mlm_cnt = op_mlm_triple
        range_mlm_loss, range_mlm_crt, range_mlm_cnt = range_mlm_triple
        sr_context_loss, sr_context_result_dict = sr_context_tuple

        # backward
        loss = mlm_loss * args.mlm_weight + sr_loss * args.sr_weight + nr_loss * args.nr_weight \
               + op_mlm_loss * args.op_mlm_weight + range_mlm_loss * args.range_mlm_weight \
               + sr_context_loss * args.sr_context_weight
        if not args.test:
            loss.backward()
            optimizer.step()
            scheduler.step()

        total_mlm_loss += mlm_loss.item()
        total_mlm_crt += mlm_crt.item()
        total_mlm_cnt += mlm_cnt.item()

        total_sr_loss += sr_loss.item()
        for i in range(len(sr_logits)):
            sr_logits_one = sr_logits[i]  # (num_sr, 2)
            sr_labels_one = sr_labels[i]  # (num_sr, )
            sr_preds_one = F.softmax(sr_logits_one, dim=-1).argmax(dim=-1)
            num_samples = sr_logits_one.size(0)
            num_correct = torch.sum(sr_preds_one == sr_labels_one).item()
            total_sr_samples += num_samples
            total_sr_correct += num_correct

        total_nr_loss += nr_loss.item() * args.nr_weight
        for i in range(len(nr_logits)):
            nr_logits_one = nr_logits[i]  # (num_sr, len(NR_AGGR))
            nr_labels_one = nr_labels[i]  # (num_sr, )
            nr_preds_one = F.softmax(nr_logits_one, dim=-1).argmax(dim=-1)
            num_samples = nr_logits_one.size(0)
            num_correct = torch.sum(nr_preds_one == nr_labels_one).item()
            total_nr_samples += num_samples
            total_nr_correct += num_correct

        total_op_mlm_loss += op_mlm_loss.item()
        total_op_mlm_crt += op_mlm_crt.item()
        total_op_mlm_cnt += op_mlm_cnt.item()

        total_range_mlm_loss += range_mlm_loss.item()
        total_range_mlm_crt += range_mlm_crt.item()
        total_range_mlm_cnt += range_mlm_cnt.item()

        total_sr_context_loss += sr_context_loss.item()
        for target in range(num_sr_context_classes):
            total_sr_context_stats[target]['tp'] += sr_context_result_dict[target]['tp'].item()
            total_sr_context_stats[target]['fp'] += sr_context_result_dict[target]['fp'].item()
            total_sr_context_stats[target]['fn'] += sr_context_result_dict[target]['fn'].item()
            total_sr_context_stats[target]['tn'] += sr_context_result_dict[target]['tn'].item()

        total_loss = total_mlm_loss * args.mlm_weight + total_sr_loss * args.sr_weight + total_nr_loss * args.nr_weight \
                     + total_op_mlm_loss * args.op_mlm_weight + total_range_mlm_loss * args.range_mlm_weight \
                     + total_sr_context_loss * args.sr_context_weight

        if steps % args.report_steps == 0 and (not args.dist_train or (args.dist_train and rank == 0)):
            elapsed = time.time() - start_time
            done_tokens = \
                args.batch_size * token_id.size(1) * args.report_steps * args.world_size \
                    if args.dist_train \
                    else args.batch_size * token_id.size(1) * args.report_steps
            for target in range(num_sr_context_classes):
                tp, fp, tn, fn = \
                    total_sr_context_stats[target]['tp'], \
                    total_sr_context_stats[target]['fp'], \
                    total_sr_context_stats[target]['tn'], \
                    total_sr_context_stats[target]['fn']
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6)
                total_sr_context_stats[target]['precision'] = round(precision, 3)
                total_sr_context_stats[target]['recall'] = round(recall, 3)
                total_sr_context_stats[target]['f1'] = round(f1, 3)

            msg = "process#{} " \
                  "| {:8.2f} tokens/s" \
                  "| {:8d}/{:8d} steps" \
                  "| total_loss {:7.2f}" \
                  "| mlm_loss {:7.2f}" \
                  "| mlm_acc {:.3f}" \
                  "| sr_loss {:7.2f}" \
                  "| sr_acc {:.3f}" \
                  "| nr_loss {:7.2f}" \
                  "| nr_acc {:.3f}" \
                  "| op_mlm_loss {:7.2f}" \
                  "| op_mlm_acc {:.3f}" \
                  "| range_mlm_loss {:7.2f}" \
                  "| range_mlm_acc {:.3f}".format(
                rank,
                done_tokens / elapsed,
                steps, total_steps,
                total_loss / args.report_steps,
                total_mlm_loss / args.report_steps,
                total_mlm_crt / (total_mlm_cnt + 1e-6),
                total_sr_loss / args.report_steps,
                total_sr_correct / total_sr_samples,
                total_nr_loss / args.report_steps,
                total_nr_correct / total_nr_samples,
                total_op_mlm_loss / args.report_steps,
                total_op_mlm_crt / total_op_mlm_cnt,
                total_range_mlm_loss / args.report_steps,
                total_range_mlm_crt / total_range_mlm_cnt,
            )
            print(msg)
            print(json.dumps(total_sr_context_stats))
            with open('pretrain_v5_log.txt', 'a') as f:
                f.write(f"{msg}\n")
            total_loss = 0.
            total_mlm_loss, total_mlm_crt, total_mlm_cnt = 0., 0., 0.
            total_sr_loss, total_sr_samples, total_sr_correct = 0., 0., 0.
            total_nr_loss, total_nr_samples, total_nr_correct = 0., 0., 0.
            total_op_mlm_loss, total_op_mlm_crt, total_op_mlm_cnt = 0., 0., 0.
            total_range_mlm_loss, total_range_mlm_crt, total_range_mlm_cnt = 0., 0., 0.
            total_sr_context_loss = 0.
            total_sr_context_stats = {}
            for target in range(num_sr_context_classes):
                total_sr_context_stats[target] = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
            start_time = time.time()
        if steps % args.save_checkpoint_steps == 0 and (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))
        steps += 1


TRAINERS = {
    "formula_prediction": train_tuta_fp,
    'fortap': train_tuta_fortap
}
