#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utility functions

"""

import torch


# %% Index Converters (layout, sequential)
def layout2seq(row_id, col_id, column_number):
    return row_id * column_number + col_id

def seq2layout(cell_id, column_number):
    row_id = cell_id // column_number
    col_id = cell_id - (row_id * column_number)
    return row_id, col_id

# %% unzipping functions for tree positions
def zip_to_index(zipped, node_degree=[32,32,64,256], total_node=384):
    index = [-1 for _ in zipped]
    offset = 0
    for ilayer, zp in enumerate(zipped):
        if -1 < zp < node_degree[ilayer]:
            index[ilayer] = offset + zp
        else:
            index[ilayer] = total_node
        offset += node_degree[ilayer]
    return index


def zip_to_orgindex(zipped, node_degree=[32,32,64,256]):
    index = [-1 for _ in zipped]
    for ilayer, zp in enumerate(zipped):
        if -1 < zp < node_degree[ilayer]:
            index[ilayer] = zp
        else:
            index[ilayer] = node_degree[ilayer]
    return index


UNZIPS = {
    "tuta": zip_to_index, 
    "base": zip_to_index, 
    "tuta_explicit": zip_to_index
}



# %% model initialization choices
def init_ts32(model, ts32_path, strict=False, initializer_range=0.01):
    if ts32_path:
        if strict:  # strict load
            model.load_state_dict(torch.load(ts32_path, map_location=torch.device("cpu")), strict=strict)
            print("Parameters initiated from {}".format(ts32_path))
        else:       # partial load
            for n,p in list(model.named_parameters()):
                    if 'gamma' not in n and 'beta' not in n:
                        p.data.normal_(0, initializer_range)
            print("Parameters first initiated randomly within range ", initializer_range)   
                
            pretrained_dict = torch.load(ts32_path, map_location=torch.device("cpu"))
            # print("Pretrained Dict: ", list(pretrained_dict.keys()), "\n")
            current_dict = model.state_dict()
            # print("Current Dict: ", list(current_dict.keys()), "\n")
            updated_dict = {k:v for (k,v) in pretrained_dict.items() if k in current_dict}
            model.load_state_dict(updated_dict, strict=strict)
            print("{} parameters (pretrained: {}, current: {}) further initiated from {}".
                  format(len(updated_dict), len(pretrained_dict), len(current_dict), ts32_path))                
    else:           # random init
        for n,p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, initializer_range)
        print("Parameters initiated randomly within range {}".format(initializer_range))


def init_tuta_loose(model, tuta_path, initializer_range=0.02):
    # random initialize within the sepcified range
    for n,p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, initializer_range)
    print("Parameters initiated randomly within range {}".format(initializer_range))
    if tuta_path is None:
        return

    # load model parameters from ts_path
    pretrained_dict = torch.load(tuta_path, map_location=torch.device("cpu"))

    num_target_fit, num_target_expand = 0, 0
    target_dict = model.state_dict()
    for name, params in pretrained_dict.items():
        if name not in target_dict:
            continue
        if params.size() == target_dict[name].size():
            target_dict[name] = params
            num_target_fit += 1
        else:
            old_size, _ = params.size()
            new_size, _ = target_dict[name].size()
            target_dict[name][: min(old_size, new_size), :] = params[: min(old_size, new_size), :]
            print("model's state_dict expand parameter {} from size {} to {}".format(name, old_size, new_size))
            num_target_expand += 1
    print("{} parameters (fit: {}, expand: {}) initiated from {} in {}". \
        format(num_target_fit+num_target_expand, num_target_fit, num_target_expand, len(pretrained_dict), tuta_path))   
    model.load_state_dict(target_dict, strict=True)


def init_with_bert_weight(args, ts_model, initializer_range=1e-3):
    for n,p in list(ts_model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, initializer_range)
    print("Parameters initiated randomly within range {}".format(initializer_range))

    bert_dict = torch.load(args.pretrained_model_path)
    selected_dict = {"backbone.embeddings.token_weight.weight": bert_dict["bert.embeddings.word_embeddings.weight"], 
                     "backbone.embeddings.LayerNorm.weight": bert_dict["bert.embeddings.LayerNorm.weight"], 
                     "backbone.embeddings.LayerNorm.bias": bert_dict["bert.embeddings.LayerNorm.bias"]}
    layer_num = args.num_encoder_layers
    suffixes = [".attention.self.query.weight", ".attention.self.query.bias", 
                ".attention.self.key.weight", ".attention.self.key.bias", 
                ".attention.self.value.weight", ".attention.self.value.bias", 
                ".attention.output.dense.weight", ".attention.output.dense.bias", 
                ".attention.output.LayerNorm.weight", ".attention.output.LayerNorm.bias", 
                ".intermediate.dense.weight", ".intermediate.dense.bias", 
                ".output.dense.weight", ".output.dense.bias", 
                ".output.LayerNorm.weight", ".output.LayerNorm.bias"]
    for ilayer in range(layer_num):
        for suffix in suffixes:
            bert_key = "bert.encoder.layer." + str(ilayer) + suffix
            select_key = "backbone.encoder.layer." + str(ilayer) + suffix
            selected_dict[select_key] = bert_dict[bert_key]

    ts_model.load_state_dict(selected_dict, strict=False)
    print("Selected Keys: ", sorted(list(selected_dict.keys())))
    return ts_model


def save_model(model, save_path):
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)


# %% batch loader
def dataset_to_tensors(dataset):
    """return a list of collected tensor from a pre-processed data set. """
    tensor_num = len(dataset[0])
    tensors = [torch.LongTensor([sample[i] for sample in dataset]) for i in range(tensor_num)]
    return tensors


def load_tensor_batch(tensors, batch_size):
    total_num = tensors[0].size()[0]
    print("collect {} valid samples in total, starting loading...".format(total_num))
    load_num = (total_num + batch_size - 1) // batch_size
    tensor_num = len(tensors)
    for j in range(load_num):
        yield [tensors[i][j * batch_size: (j+1) * batch_size] for i in range(tensor_num)]

def load_dataset_batch(dataset, batch_size):
    tensors = dataset_to_tensors(dataset)
    for batch_list in load_tensor_batch(tensors, batch_size):
        yield batch_list


def load_tensor_batch_withpad(dataset, batch_size, defaults, device_id=None):
    total_num = len(dataset)
    print("collect {} valid samples in total, starting loading...".format(total_num))
    load_iters = (total_num + batch_size - 1) // batch_size
    tensor_num = len(dataset[0])
    for j in range(load_iters):
        start, end = j * batch_size, min((j+1) * batch_size, total_num)
        # compute max sequence length of current batch
        batch_max_seqlen = 0
        for idx in range(start, end):
            batch_max_seqlen = max(batch_max_seqlen, len(dataset[idx][0]))
        # print("batch max sequence length: ", batch_max_seqlen)
        # pad each input
        assert len(defaults) == tensor_num, "Number of Tensors: {} not matching Given Defaults: {}".\
            format(len(defaults), tensor_num)
        batch_data = []
        for i in range(tensor_num):
            batch_data.append( [] )
            pad = defaults[i]
            for idx in range(start, end):
                short_tensor = dataset[idx][i]
                align_tensor = short_tensor + [pad for _ in range(batch_max_seqlen - len(short_tensor))]
                batch_data[-1].append( align_tensor )
        if device_id is not None:
            yield [torch.LongTensor(chunk).to(device_id) for chunk in batch_data]
        else:
            yield [torch.LongTensor(chunk) for chunk in batch_data]


def load_dataset_batch_withpad(dataset, batch_size, defaults, device_id):
    # tensors = dataset_to_tensors(dataset)
    for batch_list in load_tensor_batch_withpad(dataset, batch_size, defaults, device_id):
        yield batch_list
