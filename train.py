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

import argparse
import yaml

import warnings
warnings.filterwarnings('ignore')
loger_path = None

import shutup
shutup.please()
def init(config):
    global loger_path
    if config.dist_url == "env://" and config.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    print("config.distributed", config.distributed)

    loger_path = os.path.join(config.save_path, "log")
    if not os.path.isdir(loger_path):
        os.makedirs(loger_path)
    sys.stdout = log_writer.Logger(os.path.join(loger_path, "training_logs.log"))
    print("All train and test data will be saved in: ", config.save_path)
    print("----------------------------------")
    print(
        "Begin Time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    )
    printArgs(config, loger_path)
    setup_seed(config.seed)

    # Save the traning files.
    file_backup = os.path.join(config.save_path, "training_files")
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
        config.config,
        os.path.join(file_backup, os.path.basename(config.config)),
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
    if config.multiprocessing_distributed:
        config.world_size = ngpus_per_node * config.world_size

        print(config.world_size, ngpus_per_node, ngpus_per_node)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config)

    print("End Time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

def main_worker(gpu, ngpus_per_node, config):
    
    if gpu == 0:
        loger_path = os.path.join(config.save_path, "log")
        sys.stdout = log_writer.Logger(os.path.join(loger_path, "training_logs_GPU0.log")) # The print info will be saved here
    config.gpu = gpu

    # suppress printing if not master
    if config.multiprocessing_distributed and config.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if config.gpu is not None:
        print("Use GPU: {} for training".format(config.gpu))

    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            config.rank = config.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=config.dist_backend,
            init_method=config.dist_url,
            world_size=config.world_size,
            rank=config.rank,
        )

    seed = config.seed + config.rank * 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('seed', seed, 'rank', config.rank)

    print('Take Model: ', config.model)
    if config.model == 'promptiqa':
        # model = SimpleIQA.PromptIQA()
        pass
    elif config.model == 'maniqa':
        model = maniqa.MANIQA().cuda()
    elif config.model == 'MobileViT_IQA':
        model = MobileViT_IQA.MobileViT_IQA().cuda()
    elif config.model == 'clipiqa':
        model = ClipIQA.CLIPIQA().cuda()
    else:
        raise NotImplementedError('Only PromptIQA')

    if config.distributed:
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.workers = int((config.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[config.gpu], find_unused_parameters=True
            )
            print("Model Distribute.")
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    
    criterion = nn.L1Loss().cuda(config.gpu)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )

    prompt_num = config.batch_size - 1 # the number of prompts is batch_size / num_gpus - 1
    print('prompt_num', prompt_num)

    train_data_list, train_prompt_list, test_data_list = [], {}, [] # train_prompt_list save the ISPP from different datasets
    train_ori_data = []
    for dataset in config.dataset + config.zero_shot_dataset: # loading the datasets
        path, train_index, test_index = get_data(dataset=dataset, split_seed=config.seed)
            
        if dataset not in config.zero_shot_dataset:
            print('---Load ', dataset)
            
            train_dataset = data_loader.Data_Loader(config.batch_size, dataset, path, train_index, istrain=True, resize_size=config.resize_size)
            train_ori_data.append(train_dataset)
            train_data_list.append(train_dataset.get_samples()) # get the data_folder
            train_prompt_list[dataset] = train_dataset.get_prompt(prompt_num, 'fix') # The ISPP for testing is sampled from training data and is fixed.
        else:
            print('---Loading Zero Shot Dataset ', dataset)

        test_dataset = data_loader.Data_Loader(config.batch_size, dataset, path, test_index, istrain=False, resize_size=config.resize_size)
        test_data_list.append(test_dataset.get_samples())
        
    print('train_prompt_list', train_prompt_list.keys())
    combined_train_samples = ConcatDataset(train_data_list) # combine the training and testing dataset
    combined_test_samples = ConcatDataset(test_data_list)

    print("train_dataset", len(combined_train_samples))
    print("test_dataset", len(combined_test_samples))

    train_sampler = torch.utils.data.distributed.DistributedSampler(combined_train_samples)
    test_sampler = torch.utils.data.distributed.DistributedSampler(combined_test_samples)

    train_loader = torch.utils.data.DataLoader(
        combined_train_samples,
        batch_size=1, # please keep the bs to 1. More details about the bs can be found in ```folders.py```
        shuffle=(train_sampler is None),
        num_workers=config.workers,
        drop_last=True,
        sampler=train_sampler,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        combined_test_samples,
        batch_size=1,
        shuffle=(test_sampler is None),
        num_workers=config.workers,
        sampler=test_sampler,
        drop_last=False,
        pin_memory=True,
    )

    best_srocc, best_plcc, best_krcc, best_mae = 0.0, 0.0, 0.0, float('inf')
    weight = {}
    for data in train_prompt_list.keys(): # the loss weight for different datasets
        weight[data] = 1
        
    for epoch in range(config.epochs):
        if config.distributed:
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, config)

        # train for one epoch
        print('Weight: ', weight)
        pred_scores, gt_scores = train(train_loader, model, criterion, optimizer, config, epoch, weight)
        
        # gather all the results from all gpus
        gt_scores = gather_together(gt_scores)  
        gt_scores = [item for sublist in gt_scores for item in sublist]
        pred_scores = gather_together(pred_scores) 
        pred_scores = [item for sublist in pred_scores for item in sublist]
        train_srocc, train_plcc, train_krcc, train_mae = cal_srocc_plcc(pred_scores, gt_scores)

        print(
            "Train SROCC: {}, Train PLCC: {}, Train KRCC: {}, Train MAE: {}".format(
                round(train_srocc, 4), round(train_plcc, 4), round(train_krcc, 4), round(train_mae, 4)
            )
        )

        print('reshuffle data.') # reorganize the training batch
        for d_re in train_data_list:
            d_re.reshuffle()

        pred_scores, gt_scores, path = test(
            test_loader, model
        )
        print('Summary---')

        # gather all the testing results from all gpus
        gt_scores = gather_together(gt_scores)
        pred_scores = gather_together(pred_scores) 

        gt_score_dict, pred_score_dict = {}, {}
        for sublist in gt_scores:
            for k, v in sublist.items():
                if k not in gt_score_dict:
                    gt_score_dict[k] = v
                else:
                    gt_score_dict[k] = gt_score_dict[k] + v
        
        for sublist in pred_scores:
            for k, v in sublist.items():
                if k not in pred_score_dict:
                    pred_score_dict[k] = v
                else:
                    pred_score_dict[k] = pred_score_dict[k] + v

        gt_score_dict = dict(sorted(gt_score_dict.items()))
        test_srocc, test_plcc, test_krcc, test_mae = 0, 0, 0, 0
        for k, v in gt_score_dict.items():
            test_srocc_, test_plcc_, test_krcc_, test_mae_ = cal_srocc_plcc(gt_score_dict[k], pred_score_dict[k])
            print('\t{} Dataset: {} Test SROCC: {}, PLCC: {}, KRCC: {}, MAE: {}'.format("[Zero Short]" if k not in weight else "" , k, round(test_srocc_, 4), round(test_plcc_, 4), round(test_krcc_, 4), round(test_mae_, 4)))
            if k in weight:
                test_srocc += test_srocc_
                test_plcc += test_plcc_
                test_krcc += test_krcc_
                test_mae += test_mae_
            
        print(f'AVG Performance [WO Zero Shot]: \n\tSROCC, PLCC, KLCC, MAE')
        total_dataset = len(gt_score_dict.keys())
        print(f'\t{test_srocc / total_dataset}, {test_plcc / total_dataset}, {test_krcc / total_dataset}, {test_mae / total_dataset}')
        
        if not config.multiprocessing_distributed or (
                config.multiprocessing_distributed and config.rank % ngpus_per_node == 0
        ):
            if test_srocc + test_plcc + test_krcc > best_srocc + best_plcc + best_krcc:
                best_srocc, best_plcc, best_krcc, best_mae = test_srocc, test_plcc, test_krcc, best_mae
                save_checkpoint(
                    {
                        "state_dict": model.module.state_dict(),
                    },
                    is_best=True,
                    filename=os.path.join(config.save_path, f'best_model.pth.tar'),
                )
                print("Best Model Saved.")

    print('Best SROCC: {}, PLCC: {}'.format(best_srocc, best_plcc))

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
        for index, (img_or, label_or, paths, dataset_type) in enumerate(test_loader):
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

    random_flipping_rate = config.random_flipping_rate
    random_scale_rate = config.random_scale_rate
    
    for index, (img, label, _, data_name) in enumerate(train_loader):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ===== 原 argparse 参数保留 =====
    parser.add_argument("--seed", type=int, default=570908)
    parser.add_argument("--dataset", nargs='+', default=None, type=str)
    parser.add_argument("--zero_shot_dataset", nargs='+', default=[])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=44)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--T_max", type=int, default=50)
    parser.add_argument("--eta_min", type=int, default=0)
    parser.add_argument("-j", "--workers", default=32, type=int)
    parser.add_argument("--world-size", default=-1, type=int)
    parser.add_argument("--rank", default=-1, type=int)
    parser.add_argument("--dist-url", default="tcp://224.66.41.62:23456", type=str)
    parser.add_argument("--dist-backend", default="nccl", type=str)
    parser.add_argument("--multiprocessing-distributed", action="store_true")
    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument("--resize_size", default=[224, 224], type=int, nargs=2)
    parser.add_argument("--random_flipping_rate", default=0.1, type=float)
    parser.add_argument("--random_scale_rate", default=0.5, type=float)
    parser.add_argument("--model", default='promptiqa', type=str)
    parser.add_argument("--save_path", type=str, default="./save_logs/Matrix_Comparation_Koniq_bs_25")
    parser.add_argument("--config", type=str, help="YAML config file path")
    args = parser.parse_args()

    if args.config:
        yaml_cfg = load_yaml(args.config)
        args = recursive_update(args, yaml_cfg)

    args.lr = float(args.lr) # 确保学习率为浮点型，防止被 YAML 字符串覆盖导致优化器报错
    args.weight_decay = float(args.weight_decay)  # 确保权重衰减是浮点数

    main(args)
