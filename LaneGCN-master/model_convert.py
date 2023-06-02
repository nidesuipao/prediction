from importlib import import_module

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import ArgoTestDataset
from utils import Logger, load_pretrain
from viz_utils import show_predict_result
import argparse
import os

parser = argparse.ArgumentParser(description="Argoverse Motion Forecasting in Pytorch")
parser.add_argument(
    "-m", "--model", default="lanegcn", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true", default=True)
parser.add_argument(
    "--split", type=str, default="val", help='data split, "val" or "test"'
)
parser.add_argument(
    "--weight", default="/home/nio/Documents/prediction/LaneGCN-master/results/lanegcn/30.pt", type=str, metavar="WEIGHT", help="checkpoint path"
)

args = parser.parse_args()

model = import_module("lanegcn")
config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model()

ckpt_path = args.weight
if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(config["save_dir"], ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        load_pretrain(net, ckpt["state_dict"])
        net.eval()

# Data loader for evaluation
dataset = ArgoTestDataset(args.split, config, train=False)

data_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
        )   


x = next(iter(data_loader))
data = dict(x)
print(net(data))


torch.onnx.export(net, data, 'model.onnx')
#print(data)

# traced_script_module = torch.jit.trace(model, (data['x'], data['edge_index'], data['y'], data['cluster'],data['valid_len'],data['time_step_len'],data['batch'],data['ptr']))



#print(traced_script_module(data['x'], data['edge_index'], data['y'], data['cluster'],data['valid_len'],data['time_step_len'],data['batch'],data['ptr']).size())

# data = next(iter(test_loader))
# #y = torch.cat([data.y], 0).view(-1, 60)

# traced_script_module = torch.jit.trace(model, (data.x, 
#         data.edge_index,
#         data.y,
#         data.cluster,
#         data.valid_len,
#         data.time_step_len,
#         data.batch,
#         data.ptr))


# traced_script_module.save("model.pt")




