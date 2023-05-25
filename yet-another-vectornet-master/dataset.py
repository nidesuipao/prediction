# %%

import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data, Dataset
from torch_geometric.loader import DataLoader
import torch
from utils.feature_utils import compute_feature_for_one_seq, encoding_features, save_features
from utils.config import DATA_DIR, LANE_RADIUS, OBJ_RADIUS, OBS_LEN, INTERMEDIATE_DATA_DIR
from tqdm import tqdm


# %%
def get_fc_edge_index(num_nodes, start=0):
    """
    return a tensor(2, edges), indicing edge_index
    """
    to_ = np.arange(num_nodes, dtype=np.int64)
    edge_index = np.empty((2, 0))
    for i in range(num_nodes):
        from_ = np.ones(num_nodes, dtype=np.int64) * i
        # FIX BUG: no self loop in ful connected nodes graphs
        edge_index = np.hstack((edge_index, np.vstack((np.hstack([from_[:i], from_[i+1:]]), np.hstack([to_[:i], to_[i+1:]])))))
    edge_index = edge_index + start

    return edge_index.astype(np.int64), num_nodes + start
# %%


class GraphData(Data):
    """
    override key `cluster` indicating which polyline_id is for the vector
    """

    def __inc__(self, key, value, *args):
        if key == 'edge_index':
            return self.x.size(0)
        elif key == 'cluster':
            return int(self.cluster.max().item()) + 1
        else:
            return 0

# %%


class GraphDataset(Dataset):
    """
    dataset object similar to `torchvision` 
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if not hasattr (self, 'file_names') or len(self.file_names) == 0:
            self.file_names = []
            for data_path in sorted(os.listdir(self.root)):
                if not data_path.endswith('pkl'):
                    continue
                self.file_names.append(data_path+ ".pt")
        return self.file_names
   
    def download(self):
        pass

    def process(self):

        def get_data_path_ls(dir_):
            return [os.path.join(dir_, data_path) for data_path in os.listdir(dir_)]
        
        # make sure deterministic results
        data_path_ls = sorted(get_data_path_ls(self.root))
        
        #计算valid_len_ls
        valid_len_ls = []
        for data_p in tqdm(data_path_ls):
            if not data_p.endswith('pkl'):
                continue
            data = pd.read_pickle(data_p)
            all_in_features = data['POLYLINE_FEATURES'].values[0]
            cluster = all_in_features[:, -1].reshape(-1).astype(np.int32)
            valid_len_ls.append(cluster.max())

        data_ls = []
        padd_to_index = np.max(valid_len_ls)
        feature_len = -1
        idx = 0
        for data_p in tqdm(data_path_ls):
            if not data_p.endswith('pkl'):
                continue
            x_ls = []
            y = None
            cluster = None
            edge_index_ls = []
            # 读取数据
            data = pd.read_pickle(data_p)
            # 每个图有点和线组成,每个点有很多属性特征
            all_in_features = data['POLYLINE_FEATURES'].values[0]
            add_len = data['TARJ_LEN'].values[0]
            # 构建子图时指定哪些点属于同一子图
            cluster = all_in_features[:, -1].reshape(-1).astype(np.int32)
            valid_len = cluster.max()
            # 预测的y值,预测30个坐标值
            y = data['GT'].values[0].reshape(-1).astype(np.float32)

            traj_mask, lane_mask = data["TRAJ_ID_TO_MASK"].values[0], data['LANE_ID_TO_MASK'].values[0]
            agent_id = 0
            edge_index_start = 0
            assert all_in_features[agent_id][
                -1] == 0, f"agent id is wrong. id {agent_id}: type {all_in_features[agent_id][4]}"

            # 构建图模型,将点用线连接起来
            for id_, mask_ in traj_mask.items():
                data_ = all_in_features[mask_[0]:mask_[1]]
                edge_index_, edge_index_start = get_fc_edge_index(
                    data_.shape[0], start=edge_index_start)
                x_ls.append(data_)
                edge_index_ls.append(edge_index_)

            for id_, mask_ in lane_mask.items():
                data_ = all_in_features[mask_[0]+add_len: mask_[1]+add_len]
                edge_index_, edge_index_start = get_fc_edge_index(
                    data_.shape[0], edge_index_start)
                x_ls.append(data_)
                edge_index_ls.append(edge_index_)
            edge_index = np.hstack(edge_index_ls)
            x = np.vstack(x_ls)
            tup = [x, y, cluster, edge_index]

            if feature_len == -1:
                feature_len = x.shape[1]

            # [x, y, cluster, edge_index, valid_len]
            tup[0] = np.vstack(
                [tup[0], np.zeros((padd_to_index - tup[-2].max(), feature_len), dtype=tup[0].dtype)])
            tup[-2] = np.hstack(
                [tup[2], np.arange(tup[-2].max()+1, padd_to_index+1)])
            g_data = GraphData(
                x=torch.from_numpy(tup[0]),
                y=torch.from_numpy(tup[1]),
                cluster=torch.from_numpy(tup[2]),
                edge_index=torch.from_numpy(tup[3]),
                valid_len=torch.tensor([valid_len]),
                time_step_len=torch.tensor([padd_to_index + 1])
            )
            torch.save(g_data, os.path.join(self.root, "processed/", self.file_names[idx]))
            idx += 1

        #     g_ls.append(g_data)
        #     print(g_data)
        # data, slices = self.collate(g_ls)
        # torch.save((data, slices), self.processed_paths[0])
            
    
    def get(self, idx):
        data = torch.load(os.path.join(self.root, "processed/", self.file_names[idx]))
        return data

    def len(self):
        return len(self.processed_file_names)


# %%
if __name__ == "__main__":
    for folder in os.listdir(DATA_DIR):
        dataset_input_path = os.path.join(
            INTERMEDIATE_DATA_DIR, f"{folder}_intermediate")

        print(dataset_input_path)
        dataset = GraphDataset(dataset_input_path)
        # for data in dataset:
        #     print(data)


        batch_iter = DataLoader(dataset, batch_size=10)
        
        batch = next(iter(batch_iter))

# %%
