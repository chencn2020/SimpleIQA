import sys
import argparse
import builtins
import os
import random
import shutil
import time
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import ConcatDataset

from SimpleIQA.utils import log_writer
from SimpleIQA.utils.dataset import data_loader
from SimpleIQA.utils.toolkit import *

from SimpleIQA.AwesomeIQA.MANIQA import maniqa
from SimpleIQA.AwesomeIQA.MobileIQA import MobileViT_IQA
from SimpleIQA.AwesomeIQA.CLIPIQA import ClipIQA
from SimpleIQA.AwesomeIQA.HyperIQA import HyperIQA
from SimpleIQA.AwesomeIQA.TeacherIQA import studentNetwork as SN
from SimpleIQA.AwesomeIQA.TOPIQ_NR import topiq_arch as TOPIQ_NR

from tqdm import tqdm

import argparse
import yaml

import warnings
warnings.filterwarnings('ignore')
loger_path = None

import shutup
shutup.please()
def init(config):
    global loger_path
    
    train_cfg = config.train  # 方便引用
    
    if train_cfg.dist_url == "env://" and train_cfg.world_size == -1:
        train_cfg.world_size = int(os.environ["WORLD_SIZE"])

    train_cfg.distributed = train_cfg.world_size > 1 or train_cfg.multiprocessing_distributed

    print("train_cfg.distributed", train_cfg.distributed)

    loger_path = os.path.join(train_cfg.save_path, "log")
    os.makedirs(loger_path, exist_ok=True)
    printArgs(config, loger_path)
    
    sys.stdout = log_writer.Logger(os.path.join(loger_path, "training_logs.log"))
    print("All train and test data will be saved in: ", train_cfg.save_path)
    print("----------------------------------")
    print(
        "Begin Time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    )
    setup_seed(train_cfg.seed)

    # Save the traning files.
    file_backup = os.path.join(train_cfg.save_path, "training_files")
    os.makedirs(file_backup, exist_ok=True)
    shutil.copy(
        os.path.basename(__file__),
        os.path.join(file_backup, os.path.basename(__file__)),
    )
    
    shutil.copy(
        os.path.basename('train.sh'),
        os.path.join(file_backup, 'train.sh'),
    )
    
    shutil.copy(
        train_cfg.config,
        os.path.join(file_backup, os.path.basename(train_cfg.config)),
    )

    save_folder_list = ["SimpleIQA"]
    for save_folder in save_folder_list:
        save_folder_path = os.path.join(file_backup, save_folder)
        if os.path.exists(save_folder_path):
            shutil.rmtree(save_folder_path)
        shutil.copytree(save_folder, save_folder_path)

def main(config):
    init(config)
    ngpus_per_node = torch.cuda.device_count()

    if config.train.multiprocessing_distributed:
        config.train.world_size = ngpus_per_node * config.train.world_size
        print(config.train.world_size, ngpus_per_node, ngpus_per_node)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        main_worker(config.train.gpu, ngpus_per_node, config)

    print("End Time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

def _align_for_resample(x, times):
    n = (len(x) // times) * times
    return x[:n]

def main_worker(gpu, ngpus_per_node, config):
    train_cfg = config.train
    opt_cfg = config.optimizer
    dataset_cfg = config.dataset

    # GPU 日志输出
    if gpu == 0:
        loger_path = os.path.join(train_cfg.save_path, "log")
        sys.stdout = log_writer.Logger(os.path.join(loger_path, "training_logs_GPU0.log"))
    train_cfg.gpu = gpu

    # 非主进程屏蔽打印
    if train_cfg.multiprocessing_distributed and train_cfg.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if train_cfg.gpu is not None:
        print(f"Use GPU: {train_cfg.gpu} for training")

    # 初始化分布式
    if train_cfg.distributed:
        if train_cfg.dist_url == "env://" and train_cfg.rank == -1:
            train_cfg.rank = int(os.environ["RANK"])
        if train_cfg.multiprocessing_distributed:
            train_cfg.rank = train_cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=train_cfg.dist_backend,
            init_method=train_cfg.dist_url,
            world_size=train_cfg.world_size,
            rank=train_cfg.rank,
        )

    # 设置随机种子
    seed = train_cfg.seed + train_cfg.rank * 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print("seed", seed, "rank", train_cfg.rank)

    # 创建模型
    print("Take Model:", train_cfg.model)
    if train_cfg.model == "maniqa":
        model = maniqa.MANIQA()
    elif train_cfg.model == "MobileViT_IQA":
        model = MobileViT_IQA.MobileViT_IQA()
    elif train_cfg.model == "clipiqa":
        model = ClipIQA.CLIPIQA()
    elif train_cfg.model == "hyperiqa":
        model = HyperIQA.HyperNet(16, 112, 224, 112, 56, 28, 14, 7)
    elif train_cfg.model == "teacheriqa":
        model = SN.StudentNetwork()
    elif train_cfg.model == "topiq_nr":
        model = TOPIQ_NR.CFANet(semantic_model_name='resnet50', model_name='cfanet_nr_res50', backbone_pretrain=True)
    else:
        print("config.model", train_cfg.model)
        raise NotImplementedError("Only PromptIQA")

    model = model.cuda()
    if train_cfg.distributed:
        if train_cfg.gpu is not None:
            torch.cuda.set_device(train_cfg.gpu)
            model.cuda(train_cfg.gpu)
            train_cfg.batch_size //= ngpus_per_node
            train_cfg.workers = int((train_cfg.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[train_cfg.gpu], find_unused_parameters=True
            )
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

    # 损失函数
    if config.loss.criterion == "l1":
        print("Use L1 Loss")
        criterion = nn.L1Loss().cuda()
    elif config.loss.criterion == "mse":
        print("Use MSE Loss")
        criterion = nn.MSELoss().cuda()
    elif config.loss.criterion == "smooth_l1":
        print("Use Smooth L1 Loss")
        criterion = nn.SmoothL1Loss().cuda()
    else:
        print("config.train.criterion", config.loss.criterion)
        raise NotImplementedError("Only L1, MSE, Smooth L1 Loss")

    if config.optimizer.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay
        )
    else:
        raise NotImplementedError("Only Adam Optimizer")

    prompt_num = train_cfg.batch_size - 1
    print("prompt_num", prompt_num)

    # 加载数据
    train_data_list, train_prompt_list, test_data_list = [], {}, []
    train_ori_data = []
    for dataset in dataset_cfg.train_dataset + dataset_cfg.zero_shot_dataset:
        path, train_index, test_index = get_data(dataset=dataset, split_seed=train_cfg.seed)
        if dataset not in dataset_cfg.zero_shot_dataset:
            print(f"---Load {dataset}")
            train_dataset = data_loader.Data_Loader(
                train_cfg.batch_size, dataset, path, train_index,
                istrain=True, dataset_cfg=dataset_cfg
            )
            train_ori_data.append(train_dataset)
            train_data_list.append(train_dataset.get_samples())
            train_prompt_list[dataset] = train_dataset.get_prompt(prompt_num, "fix")
        else:
            print(f"---Loading Zero Shot Dataset {dataset}")

        test_dataset = data_loader.Data_Loader(
            train_cfg.batch_size, dataset, path, test_index,
            istrain=False, dataset_cfg=dataset_cfg
        )
        test_data_list.append(test_dataset.get_samples())

    print("train_prompt_list", train_prompt_list.keys())
    combined_train_samples = ConcatDataset(train_data_list)
    combined_test_samples = ConcatDataset(test_data_list)

    print("train_dataset", len(combined_train_samples))
    print("test_dataset", len(combined_test_samples))

    train_sampler = torch.utils.data.distributed.DistributedSampler(combined_train_samples)
    test_sampler = torch.utils.data.distributed.DistributedSampler(combined_test_samples, shuffle=False, drop_last=False)

    train_loader = torch.utils.data.DataLoader(
        combined_train_samples,
        batch_size=1,
        shuffle=(train_sampler is None),
        num_workers=train_cfg.workers,
        drop_last=True,
        sampler=train_sampler,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        combined_test_samples,
        batch_size=1,
        shuffle=False,
        num_workers=train_cfg.workers,
        sampler=test_sampler,
        drop_last=False,
        pin_memory=True,
    )

    # 训练循环
    best_srocc, best_plcc, best_krcc, best_mae = 0.0, 0.0, 0.0, float("inf")
    weight = {data: 1 for data in train_prompt_list.keys()}

    for epoch in range(train_cfg.epochs):
        if train_cfg.distributed:
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
            
        adjust_learning_rate(optimizer, epoch, config)

        print("Weight:", weight)
        pred_scores, gt_scores = train(train_loader, model, criterion, optimizer, train_cfg, epoch, weight)

        gt_scores = [item for sublist in gather_together(gt_scores) for item in sublist]
        pred_scores = [item for sublist in gather_together(pred_scores) for item in sublist]
        train_srocc, train_plcc, train_krcc, train_mae = cal_srocc_plcc(pred_scores, gt_scores)

        print(
            f"Train SROCC: {round(train_srocc, 4)}, "
            f"Train PLCC: {round(train_plcc, 4)}, "
            f"Train KRCC: {round(train_krcc, 4)}, "
            f"Train MAE: {round(train_mae, 4)}"
        )

        print("reshuffle data.")
        for d_re in train_data_list:
            d_re.reshuffle()

        pred_scores, gt_scores, path = test(test_loader, model)
        print("Summary---")

        gt_scores = gather_together(gt_scores)
        pred_scores = gather_together(pred_scores)

        gt_score_dict, pred_score_dict = {}, {}
        for sublist in gt_scores:
            for k, v in sublist.items():
                gt_score_dict.setdefault(k, []).extend(v)
        for sublist in pred_scores:
            for k, v in sublist.items():
                pred_score_dict.setdefault(k, []).extend(v)

        gt_score_dict = dict(sorted(gt_score_dict.items()))
        test_srocc, test_plcc, test_krcc, test_mae = 0, 0, 0, 0
        for k, v in gt_score_dict.items():
            if dataset_cfg.re_sample:
                gt = _align_for_resample(gt_score_dict[k], dataset_cfg.re_sample_times)
                pred = _align_for_resample(pred_score_dict[k], dataset_cfg.re_sample_times)
                gt_score_dict[k] = np.mean(
                    np.reshape(np.array(gt), (-1, dataset_cfg.re_sample_times)), axis=1
                )
                pred_score_dict[k] = np.mean(
                    np.reshape(np.array(pred), (-1, dataset_cfg.re_sample_times)), axis=1
                )
            
            test_srocc_, test_plcc_, test_krcc_, test_mae_ = cal_srocc_plcc(gt_score_dict[k], pred_score_dict[k])
            print(f'\t{"[Zero Short]" if k not in weight else ""} {k} Dataset [{len(gt_score_dict[k])} samples]: '
                  f'{round(test_srocc_, 4)}, {round(test_plcc_, 4)}, '
                  f'{round(test_krcc_, 4)}, {round(test_mae_, 4)}')
            if k in weight:
                test_srocc += test_srocc_
                test_plcc += test_plcc_
                test_krcc += test_krcc_
                test_mae += test_mae_

        total_dataset = len(weight.keys())
        print(f"AVG Performance [WO Zero Shot]:\n\tSROCC, PLCC, KLCC, MAE")
        print(f"\t{test_srocc / total_dataset}, {test_plcc / total_dataset}, "
              f"{test_krcc / total_dataset}, {test_mae / total_dataset}")

        if not train_cfg.multiprocessing_distributed or (
            train_cfg.multiprocessing_distributed and train_cfg.rank % ngpus_per_node == 0
        ):
            if test_srocc + test_plcc + test_krcc > best_srocc + best_plcc + best_krcc:
                best_srocc, best_plcc, best_krcc, best_mae = test_srocc, test_plcc, test_krcc, best_mae
                save_checkpoint(
                    {"state_dict": model.module.state_dict()},
                    is_best=True,
                    filename=os.path.join(train_cfg.save_path, "best_model.pth.tar"),
                )
                print("Best Model Saved.")

    print(f"Best SROCC: {best_srocc}, PLCC: {best_plcc}")

def test(test_loader, model):
    """Training"""
    pred_scores = {}
    gt_scores = {}
    path = []

    batch_time = AverageMeter("Time", ":6.3f")
    srocc = AverageMeter("SROCC", ":6.2f")
    plcc = AverageMeter("PLCC", ":6.2f")
    krcc = AverageMeter("KRCC", ":6.2f")
    mae = AverageMeter("MAE", ":6.2f")
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, srocc, plcc],
        prefix="Testing ",
    )

    model.train(False)
    with torch.no_grad():
        for index, (img_or, label_or, paths, dataset_type) in enumerate((test_loader)):
            dataset_type = dataset_type[0]
            # prompt_dataset = promt_data_loader[dataset_type]
            t = time.time()

            img = img_or.squeeze(0).cuda()
            label = label_or.squeeze(0)[:, -1].view(-1).cuda()

            pred = model(img).view(-1)

            if dataset_type not in pred_scores:
                pred_scores[dataset_type] = []

            if dataset_type not in gt_scores:
                gt_scores[dataset_type] = []

            pred_scores[dataset_type] = pred_scores[dataset_type] + pred.cpu().tolist()
            gt_scores[dataset_type] = gt_scores[dataset_type] + label.cpu().tolist()
            path = path + list(paths)

            batch_time.update(time.time() - t)

            if index % 100 == 0:
                for k, v in pred_scores.items():
                    test_srocc, test_plcc, test_krcc, test_mae = cal_srocc_plcc(pred_scores[k], gt_scores[k])
                srocc.update(test_srocc)
                plcc.update(test_plcc)
                krcc.update(test_krcc)
                mae.update(test_mae)

                progress.display(index)

    model.train(True)
    return pred_scores, gt_scores, path


def train(train_loader, model, loss_fun, optimizer, config, epoch, weight):
    """Training"""
    print("------------Training Logs---------------")
    epoch_loss = []
    pred_scores = []
    gt_scores = []

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()
    
    for index, (img, label, _, data_name) in enumerate((train_loader)):
        img = img.squeeze(0)
        label = label.squeeze(0)[:, -1].view(-1).cuda()

        data_time.update(time.time() - end)

        optimizer.zero_grad()

        pred = model(img).view(-1)

        loss = loss_fun(pred.squeeze(), label.float().detach())
        loss = loss * weight[data_name[0]]
        epoch_loss.append(loss.item())

        losses.update(loss.item(), img.size(0))

        loss.backward()
        optimizer.step()
        
        pred_scores = pred_scores + pred.cpu().tolist()
        gt_scores = gt_scores + label.cpu().tolist()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if index % 100 == 0:
            progress.display(index)
        
    progress.display(index)

    return pred_scores, gt_scores
import argparse
import yaml

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and k in d:
            recursive_update(d[k], v)
        else:
            d[k] = v
    return d

from types import SimpleNamespace
def dict_to_namespace(d):
    """递归把 dict 转成 SimpleNamespace，方便用点号访问"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config file path")
    args_cli = parser.parse_args()

    cfg_dict = load_yaml(args_cli.config)
    cfg = dict_to_namespace(cfg_dict)
    cfg.train.config = args_cli.config

    # 确保类型正确
    cfg.optimizer.lr = float(cfg.optimizer.lr)
    cfg.optimizer.weight_decay = float(cfg.optimizer.weight_decay)

    main(cfg)