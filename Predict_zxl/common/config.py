import argparse
import os
import sys
sys.path.append(r"./model")
from model_utils import StepLR


def get_params():
    parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
    parser.add_argument("-m", "--model", default="lanegcn", type=str, metavar="MODEL", help="model name"
    )
    parser.add_argument("--seed", default=123, type=int, help="random seed"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", default="", type=str, metavar="RESUME", help="checkpoint path")
    parser.add_argument("--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path")

    parser.add_argument("--display_iters", default=2, type=int, help="")
    parser.add_argument("--val_iters", default=2, type=int, help="iters to val")
    parser.add_argument("--save_freq", default=2, type=int, help="how frequent to save model")
    parser.add_argument("--epoch", default=0, type=int, help="start training epoch")
    parser.add_argument("--num_epochs", default=36, type=int, help="total epoch")
    parser.add_argument("--lr", default=[1e-3, 1e-4], type=list, help="learning rate")
    parser.add_argument("--lr_epochs", default=[32], type=list, help="when to step learning rate")
    parser.add_argument("--opt", default="adam", type=str, help="optim")
    parser.add_argument("--save_dir", default="results", type=str, help="directory to save")
    parser.add_argument("--batch_size", default=2, type=int, help="training batch_size")
    parser.add_argument("--val_batch_size", default=2, type=int, help="val batch size")
    parser.add_argument("--workers", default=2, type=int, help="")
    parser.add_argument("--rootdata", default="/home/nio/Documents/mydata", type=str, help="")

    parser.add_argument("--rot_aug", default=False, type=bool, help="")
    parser.add_argument("--pred_range", default=[-100.0, 100.0, -100.0, 100.0], type=list, help="")
    parser.add_argument("--num_scales", default=6, type=int, help="")
    parser.add_argument("--n_actor", default=128, type=int, help="")
    parser.add_argument("--n_map", default=128, type=int, help="")
    parser.add_argument("--actor2map_dist", default=7.0, type=float, help="")
    parser.add_argument("--map2actor_dist", default=6.0, type=float, help="")
    parser.add_argument("--actor2actor_dist", default=100.0, type=float, help="")
    parser.add_argument("--pred_size", default=30, type=int, help="")
    parser.add_argument("--pred_step", default=1, type=int, help="")
    parser.add_argument("--num_mods", default=6, type=int, help="")
    parser.add_argument("--cls_coef", default=1.0, type=float, help="")
    parser.add_argument("--reg_coef", default=1.0, type=float, help="")
    parser.add_argument("--mgn", default=0.2, type=float, help="")
    parser.add_argument("--cls_th", default=2.0, type=float, help="")
    parser.add_argument("--cls_ignore", default=0.2, type=float, help="")

    config = vars(parser.parse_args())


    config["val_workers"] = config["workers"]

    """Dataset"""
    # Raw Dataset
    rootdata = config['rootdata']
    config["train_split"] = os.path.join(
        rootdata, "dataset/train/data"
    )
    config["val_split"] = os.path.join(rootdata, "dataset/val/data")
    config["test_split"] = os.path.join(rootdata, "dataset/test_obs/data")

    # Preprocessed Dataset
    config["preprocess"] = False#True # whether use preprocess or not
    config["preprocess_train"] = os.path.join(
        rootdata, "dataset","preprocess", "train_crs_dist6_angle90.p")
    config["preprocess_val"] = os.path.join(
        rootdata,"dataset", "preprocess", "val_crs_dist6_angle90.p")
    config['preprocess_test'] = os.path.join(rootdata, "dataset",'preprocess', 'test_test.p')

    config["num_preds"] = config["pred_size"] // config["pred_step"]

    config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])

    return config
