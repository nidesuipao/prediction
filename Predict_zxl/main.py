import os
import argparse
import numpy as np
import random
import sys
import time
import shutil
from importlib import import_module
from numbers import Number

import torch
from torch.utils.data import Sampler, DataLoader

from common.config import get_params
from utils.dataset import ArgoDataset, collate_fn

from model.lanegcn import Net, Loss, PostProcess
from model.model_utils import Optimizer

config = get_params()
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

def worker_init_fn(pid):
    np_seed = int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


def main():
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net(config)
    loss = Loss(config).to(device)
    post_process = PostProcess(config).to(device)
    opt = Optimizer(net.parameters(), config)
    

    save_dir = config["save_dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    src_dirs = [root_path]
    dst_dirs = [os.path.join(save_dir, "files")]
    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for f in files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    # Data loader for training
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

    # Data loader for evaluation
    dataset = ArgoDataset(config["val_split"], config, train=False)
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    epoch = config["epoch"]
    remaining_epochs = int(np.ceil(config["num_epochs"] - epoch))
    for i in range(remaining_epochs):
        train(epoch + i, config, train_loader, net, loss, post_process, opt, val_loader)



def train(epoch, config, train_loader, net, loss, post_process, opt, val_loader):
    net.train()

    num_batches = len(train_loader)
    epoch_per_batch = 1.0 / num_batches
    save_iters = int(np.ceil(config["save_freq"] * num_batches))
    display_iters = int(
        config["display_iters"] / (config["batch_size"])
    )
    val_iters = int(config["val_iters"] / (config["batch_size"]))
    # print(val_iters)

    start_time = time.time()
    metrics = dict()

    for i, data in enumerate(train_loader):
        epoch += epoch_per_batch
        data = dict(data)
        output = net(data)
        
        loss_out = loss(output, data)
        post_out = post_process(output, data)
        post_process.append(metrics, loss_out, post_out)

        opt.zero_grad()
        loss_out["loss"].backward()
        lr = opt.step(epoch)

        num_iters = int(np.round(epoch * num_batches))

        if num_iters % save_iters == 0 or epoch >= config["num_epochs"]:
            save_ckpt(net, opt, config["save_dir"], epoch)

        if num_iters % display_iters == 0:
            dt = time.time() - start_time
            # metrics = sync(metrics)
            post_process.display(metrics, dt, epoch, lr)
            start_time = time.time()
            metrics = dict()

        if num_iters % val_iters == 0:
            val(config, val_loader, net, loss, post_process, epoch)

        if epoch >= config["num_epochs"]:
            val(config, val_loader, net, loss, post_process, epoch)
            return


def val(config, data_loader, net, loss, post_process, epoch):
    net.eval()
    start_time = time.time()
    preds= {}
    cities = {}
    gts = {}
    
    metrics = dict()
    total_i = 0
    for i, data in enumerate(data_loader):
        data = dict(data)
        with torch.no_grad():
            
            output = net(data)
            results = [x[0:1].detach().cpu().numpy() for x in output["reg"]]
            loss_out = loss(output, data)
            post_out = post_process(output, data)
            metrics = post_process.append(metrics, loss_out, post_out)

        for i, pred_traj in enumerate(results):
            preds[total_i] = pred_traj.squeeze()
            cities[total_i] = data["city"][i]
            gts[total_i] = data["gt_preds"][i][0] if "gt_preds" in data else None
            total_i += 1
	    
    from argoverse.evaluation.eval_forecasting import (
            compute_forecasting_metrics, get_displacement_errors_and_miss_rate
        )
    # Max #guesses (K): 6
    metrics6 = get_displacement_errors_and_miss_rate(preds, gts, 6, 30, 2)
    print(f"minADE6:{metrics6['minADE']:3f}, minFDE6:{metrics6['minFDE']:3f}, MissRate6:{metrics6['MR']:3f}")
    # Max #guesses (K): 1
    metrics1 = get_displacement_errors_and_miss_rate(preds, gts, 1, 30, 2)
    print(f"minADE:{metrics1['minADE']:3f}, minFDE:{metrics1['minFDE']:3f}, MissRate:{metrics1['MR']:3f}")
    dt = time.time() - start_time
    # metrics = sync(metrics)
    # if hvd.rank() == 0:
    post_process.display(metrics, dt, epoch)
    net.train()




def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.pt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict},
        os.path.join(save_dir, save_name),
    )









if __name__ == '__main__':
    main()





