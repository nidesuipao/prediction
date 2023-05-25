#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
import torch
import random

color_dict = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"}


def show_doubled_lane(polygon, rot, orig):
    """
    args: ndarray in shape of (n, 2)
    returns:
    """
    polygon = np.matmul(polygon.reshape(-1, 2), rot) #+ orig.reshape(-1, 2)
    xs, ys = polygon[:, 0] + orig[0], polygon[:, 1] + orig[1]
    plt.plot(xs, ys, '--', color='grey')


def show_traj(ctr, traj, type_, rot, orig):
    """
    args: ndarray in shape of (n, 2)
    returns:
    """
    if type_ == "AGENT":
        color = color_dict[type_]
    else:
        colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
        color = "#"
        for i in range(6):
            color += colorArr[random.randint(0,14)]

    
    valid_traj = traj[[not np.all(traj[i] == 0) for i in range(traj.shape[0])], :2] 
    valid_traj = valid_traj.cumsum(axis=0)
    valid_traj = ctr - valid_traj
    valid_traj = np.matmul(valid_traj.reshape(-1, 2), rot) #+ orig.reshape(-1, 2)
    xs, ys = valid_traj[:, 0] + orig[0], valid_traj[:, 1] + orig[1]
    plt.plot(xs, ys, color=color)


def reconstract_polyline(features, traj_mask, lane_mask, add_len):
    traj_ls, lane_ls = [], []
    for id_, mask in traj_mask.items():
        data = features[mask[0]: mask[1]]
        traj = np.vstack((data[:, 0:2], data[-1, 2:4]))
        traj_ls.append(traj)
    for id_, mask in lane_mask.items():
        data = features[mask[0]+add_len: mask[1]+add_len]
        # lane = np.vstack((data[:, 0:2], data[-1, 3:5]))
        # change lanes feature to (xs, ys, zs, xe, ye, ze, polyline_id)
        lane = np.vstack((data[:, 0:2], data[-1, 2:4]))
        lane_ls.append(lane)
    return traj_ls, lane_ls


def show_pred_and_gt(pred_y, y):

    valid_y = y > 0
    y  = y[valid_y].reshape(-1, 2)
    plt.plot(y[:, 0], y[:, 1], color='yellow', linewidth = 2)
    for i in pred_y:
        i = i[valid_y].reshape(-1, 2)
        plt.scatter(i[:, 0], i[:, 1], marker='*', s = 1)


def show_predict_result(data, pred_ys, ys, save_path, show_lane=True, loss = None):

    lanes = data['graph'][0]['ctrs'].detach().numpy().reshape((-1, 2))
    rot = data['rot'][0].detach().numpy()
    orig = data['orig'][0].detach().numpy()
    node_idcs = data['graph'][0]['node_idcs']

    plt.figure(figsize=(40,40))
    plt.xlim([orig[0] - 75, orig[0] + 75])
    plt.ylim([orig[1] - 75, orig[1] + 75])
    


    trajs = data['feats'][0]
    ctrs = data['ctrs'][0]
    type_ = 'AGENT'
    for i in range(len(trajs)):
        traj = trajs[i].detach().numpy()
        ctr = ctrs[i].detach().numpy()
        show_traj(ctr, traj, type_, rot, orig)
        type_ = 'OTHERS'

    pred_ys = np.array(list(pred_ys.values()))
    for pre_idx in range(len(pred_ys)):

        pred_y = pred_ys[pre_idx].reshape((-1, 30, 2))
        y = ys[pre_idx].detach().numpy().reshape((-1, 2))
        show_pred_and_gt(pred_y, y)
    if show_lane:
        for num in node_idcs:
            show_doubled_lane(lanes[num,:], rot, orig)

    
    # plt.text(0.01, 0.95, "loss: " + loss, transform = plt.gca().transAxes, fontsize = 15, c = 'black')
    plt.savefig(save_path)
    plt.close()
    plt.clf()
