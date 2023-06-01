# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com

# %%
import random
import os
import pdb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch_geometric.nn import MessagePassing, max_pool
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import Sampler, DataLoader
from torch_geometric.nn import max_pool_x
from model_utils import gpu, to_long

import sys
sys.path.append(r"./utils")
sys.path.append(r"./common")
from dataset import ArgoDataset, collate_fn
from config import get_params

def worker_init_fn(pid):
    np_seed = int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)

# %%
# %%

def gather(data):
    actors = gpu(data["feats"])
    batch_size = len(actors)
    num_actors = [len(x) for x in actors]

    actors = [x.transpose(1, 2) for x in actors]
    actors = torch.cat(actors, 0)

    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i]).to(actors.device)
        actor_idcs.append(idcs)
        count += num_actors[i]

    graphs = to_long(gpu(data["graph"]))
    node_idcs = []

    for i in range(batch_size):
        actor_idcs.append(count)
        idcs = torch.arange(count, count + graphs[i]["num_nodes"]).to(
            graphs[i]["feats"].device
        )
        node_idcs.append(idcs)
        count = count + graphs[i]["num_nodes"]

    print(actors.shape)
    print(graphs[0]["feats"].shape)
    print(graphs[1]["feats"].shape)
    print(graphs[2]["feats"].shape)

    exit(0)
    return actors, actor_idcs

class SubGraph(nn.Module):
    """
    Subgraph that computes all vectors in a polyline, and get a polyline-level feature
    """

    def __init__(self, in_channels, num_subgraph_layres=3, hidden_unit=64):
        super(SubGraph, self).__init__()
        self.num_subgraph_layres = num_subgraph_layres
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layres):
            self.layer_seq.add_module(
                f'glp_{i}', GraphLayerProp(in_channels, hidden_unit))
            in_channels *= 2

    def forward(self, data):
        features, idxs = gather(data)
        

        return None#out_data

# %%
class GraphLayerProp(MessagePassing):
    """
    Message Passing mechanism for infomation aggregation
    """

    def __init__(self, in_channels, hidden_unit=64, verbose=False):
        super(GraphLayerProp, self).__init__(
            aggr='max')  # MaxPooling aggragation
        self.verbose = verbose
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, in_channels)
        )

    def forward(self, x, edge_index):
        if self.verbose:
            print(f'x before mlp: {x}')
        x = self.mlp(x)
        if self.verbose:
            print(f"x after mlp: {x}")
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        if self.verbose:
            print(f"x after mlp: {x}")
            print(f"aggr_out: {aggr_out}")
        return torch.cat([x, aggr_out], dim=1)
    


def masked_softmax(X, valid_len):
    """
    masked softmax for attention scores
    args:
        X: 3-D tensor, valid_len: 1-D or 2-D tensor
    """
    if valid_len is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_len.dim() == 1:
            valid_len = torch.repeat_interleave(
                valid_len, repeats=shape[1], dim=0)
        else:
            valid_len = valid_len.reshape(-1)
        # Fill masked elements with a large negative, whose exp is 0
        X = X.reshape(-1, shape[-1])
        for count, row in enumerate(X):
            try:
                X[count][int(valid_len[count]):] = -1e6
            except:
                break
            #row[int(valid_len[count]):] = -1e6
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class SelfAttentionLayer(nn.Module):
    """
    Self-attention layer. no scale_factor d_k
    """

    def __init__(self, in_channels, global_graph_width, need_scale=False):
        super(SelfAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.q_lin = nn.Linear(in_channels, global_graph_width)
        self.k_lin = nn.Linear(in_channels, global_graph_width)
        self.v_lin = nn.Linear(in_channels, global_graph_width)
        self.scale_factor_d = 1 + \
            int(np.sqrt(self.in_channels)) if need_scale else 1

    def forward(self, x, valid_len):
        # print(x.shape)
        # print(self.q_lin)
        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)
        scores = torch.bmm(query, key.transpose(1, 2))
        attention_weights = masked_softmax(scores, valid_len)
        return torch.bmm(attention_weights, value)


class TrajPredMLP(nn.Module):
    """Predict one feature trajectory, in offset format"""

    def __init__(self, in_channels, out_channels, hidden_unit):
        super(TrajPredMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, out_channels)
        )

    def forward(self, x):
        return self.mlp(x)



class HGNN(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self, in_channels, out_channels, num_subgraph_layers=3, num_global_graph_layer=1, subgraph_width=64, global_graph_width=64, traj_pred_mlp_width=64):
        super(HGNN, self).__init__()
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers)
        self.subgraph = SubGraph(
            in_channels, num_subgraph_layers, subgraph_width)
        self.self_atten_layer = SelfAttentionLayer(
            self.polyline_vec_shape, global_graph_width, need_scale=False)
        self.traj_pred_mlp = TrajPredMLP(
            global_graph_width, out_channels, traj_pred_mlp_width)

    def forward(self, data):
        """
        args: 
            data (Data): [x, y, cluster, edge_index, valid_len]

        """
        # data = (data.x, 
        # data.edge_index,
        # data.y,
        # data.cluster,
        # data.valid_len,
        # data.time_step_len,
        # data.batch,
        # data.ptr)
        # time_step_len = int(time_step_len[0])
        valid_lens = None#valid_len

        sub_graph_out = self.subgraph(data)

        # x = sub_graph_out.view(-1, time_step_len, self.polyline_vec_shape)
        # #print(x.shape)
        # out = self.self_atten_layer(x, valid_lens)
        # #print(out.shape)
        # # from pdb import set_trace
        # # set_trace()
        # pred = self.traj_pred_mlp(out[:, [0]].squeeze(1))
        return torch.ones(1,1)


# %%
if __name__ == "__main__":
    epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    decay_lr_factor = 0.9
    decay_lr_every = 10
    lr = 0.005
    in_channels, out_channels = 8, 60
    show_every = 10
    os.chdir('..')
    # get model
    model = HGNN(in_channels, out_channels).to(device)
    config = get_params()

    dataset = ArgoDataset(config["train_split"], config, train=True)
    # exit(0)
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    for i, data in enumerate(train_loader):
        # print(1)
        out = model(data)
        exit()

        
# %%
    '''
    def get_data_path_ls(dir_):
        return [os.path.join(dir_, data_path) for data_path in os.listdir(dir_)]

    # data preparation
    DIR = 'input_data'
    data_path_ls = get_data_path_ls(DIR)

    # hyper parameters
    epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    decay_lr_factor = 0.9
    decay_lr_every = 10
    lr = 0.005
    in_channels, out_channels = 7, 60
    show_every = 10

    # get model
    model = HGNN(in_channels, out_channels).to(device)
    # print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)

    # overfit the small dataset
    # [FIXME]: unparallel batch processing
    model.train()
    for epoch in range(epochs):
        accum_loss = .0
        loss = None
        np.random.shuffle(data_path_ls)
        for sample_id, data_p in enumerate(data_path_ls):
            data = pd.read_pickle(data_p)
            y = data['GT'].values[0].reshape(-1).astype(np.float32)
            y = torch.from_numpy(y)

            # No Batch stuffs
            # optimizer.zero_grad()
            # out = model(data)
            # loss = F.mse_loss(out, y)
            # accum_loss += loss.item()
            # loss.backward()
            # optimizer.step()

            if sample_id % batch_size == 0:
                if loss:
                    accum_loss += loss.item()

                    loss /= batch_size
                    loss.backward()
                    optimizer.step()
                    loss = None
                optimizer.zero_grad()

            out = model(data)
            sample_loss = F.mse_loss(out, y)
            loss = sample_loss if loss is None else sample_loss + loss
        if sample_id % batch_size == 0:
            if loss:
                accum_loss += loss.item()

                loss /= batch_size
                loss.backward()
                optimizer.step()
                loss = None
            optimizer.zero_grad()

        scheduler.step()
        print(
            f"loss at epoch {epoch}: {accum_loss / len(data_path_ls):.3f}, lr{optimizer.state_dict()['param_groups'][0]['lr']: .3f}")

    # eval result on the identity dataset
    model.eval()
    data_path_ls = get_data_path_ls(DIR)

    with torch.no_grad():
        accum_loss = .0
        for sample_id, data_p in enumerate(data_path_ls):
            print(f"sample id: {sample_id}")
            data = pd.read_pickle(data_p)
            y = data['GT'].values[0].reshape(-1).astype(np.float32)
            y = torch.from_numpy(y)

            out = model(data)
            loss = F.mse_loss(out, y)

            accum_loss += loss.item()
            print(f"loss for sample {sample_id}: {loss.item():.3f}")
            show_predict_result(data, out, y, data['TARJ_LEN'].values[0])
            plt.show()

    print(f"eval overall loss: {accum_loss / len(data_path_ls):.3f}")
    # print(y)

    # edge_index = torch.tensor(
    #     [[1, 2, 0, 2, 0, 1],
    #         [0, 0, 1, 1, 2, 2]], dtype=torch.long)
    # x = torch.tensor([[3, 1, 2], [2, 3, 1], [1, 2, 3]], dtype=torch.float)
    # y = torch.tensor([1, 2, 3, 4], dtype=torch.float)
    # data = Data(x=x, edge_index=edge_index, y=y)

    # data = pd.read_pickle('./input_data/features_4791.pkl')
    # all_in_features_, y = data['POLYLINE_FEATURES'].values[0], data['GT'].values[0].reshape(-1).astype(np.float32)
    # traj_mask_, lane_mask_ = data["TRAJ_ID_TO_MASK"].values[0], data['LANE_ID_TO_MASK'].values[0]
    # y = torch.from_numpy(y)
    # in_channels, out_channels = all_in_features_.shape[1], y.shape[0]
    # print(f"all feature shape: {all_in_features_.shape}, gt shape: {y.shape}")
    # print(f"len of trajs: {traj_mask_}, len of lanes: {lane_mask_}")'''

# %%


# %%
