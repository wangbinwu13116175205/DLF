import sys, json, argparse, random, re, os, shutil
sys.path.append("src/")
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import math
import os.path as osp
import networkx as nx
import pdb
from Bio.Cluster import kcluster
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import optim
import torch.multiprocessing as mp
import ct
from src.model import DSTG
from utils.metrics import masked_mae
from utils.args import get_public_config
from utils.engine import BaseEngine
from utils.dataloader import load_dataset


pin_memory = True
n_work = 16
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def get_config():
    parser = get_public_config()
    parser.add_argument('--adj_type', type=str, default='normlap')
    parser.add_argument('--adp_adj', type=int, default=1)
    parser.add_argument('--init_dim', type=int, default=64)
    parser.add_argument('--dilation_channels', type=int, default=128)
    parser.add_argument('--skip_dim', type=int, default=256)
    parser.add_argument('--end_dim', type=int, default=256)

    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--result', type=str, default='1')
    parser.add_argument('--clip_grad_value', type=float, default=5)
    parser.add_argument('--conf', type=str, default='DLF.json')
    args = parser.parse_args()
    return args
def update(src, tmp):
    for key in tmp:
        if key!= "gpuid":
            src[key] = tmp[key]



def init(args):    
    conf_path = osp.join(args.conf)
    info = ct.load_json_file(conf_path)
    info["time"] = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    update(vars(args), info)
    vars(args)["path"] = osp.join(args.model_path, args.logname+args.time)
    ct.mkdirs(args.path)
    del info
def get_model(args):
    model = DSTG(input_dim=args.input_dim,
                    output_dim=args.output_dim,
                    dropout=args.dropout,
                    hidden_channels=args.hidden_channels,
                    dilation_channels=args.dilation_channels,
                    skip_channels=args.skip_dim,
                    end_channels=args.end_dim,device=args.device
                    )
    return model
def get_engine(args,optimizer,model,loss_fn,lrate,month):
    engine = BaseEngine(device=device,
                            model=model,
                            sampler=None,
                            loss_fn=loss_fn,
                            lrate=lrate,
                            optimizer=optimizer,
                            scheduler=None,
                            clip_grad_value=args.clip_grad_value,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            log_dir=args.save_model_path,
                            logger=args.logger,
                            seed=args.seed,month=month,learning_method=args.learning_method,args=args)
def init_log(args):
    log_dir, log_filename = args.path, args.logname
    logger = logging.getLogger(__name__)
    ct.mkdirs(log_dir)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(osp.join(log_dir, log_filename+".log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("logger name:%s", osp.join(log_dir, log_filename+".log"))
    path=osp.join(log_dir, log_filename)
    vars(args)["logger"] = logger
    vars(args)["save_model_path"] = path
    return logger

def main(args):
    logger = init_log(args)
    logger.info("params : %s", vars(args))
    ct.mkdirs(args.save_data_path)
    loss_fn = masked_mae
    #scheduler = None
    vars(args)["train_time"]=0 
    for month in np.arange(args.begin_month, args.end_month+1,0.5):
        vars(args)["month"]=month
        if (month-1)%3==0:
                    model = get_model(args)
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
                    scheduler=None
                   
                    engine = BaseEngine(device=device,
                            model=model,
                            sampler=None,
                            loss_fn=loss_fn,
                            lrate=args.lrate,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            clip_grad_value=args.clip_grad_value,
                            max_epochs=1,#args.max_epochs,
                            patience=args.patience,
                            log_dir=args.save_model_path,
                            logger=args.logger,
                            seed=args.seed,month=month,learning_method=args.learning_method,args=args)
                    dataloader, scaler= load_dataset(args,args.logger)
                    engine.update_adj(dataloader['train_adj'],dataloader['val_adj'],dataloader['test_adj'])
                    model= get_model(args) 
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
                    if args.self_surlearning:
                        engine.load_self_model(args.sur_save_model_path,model,optimizer)
                    engine.update_dataloader(dataloader, scaler)
                    engine.down_prediction()
        else:
                engine.load_savedmodel(args.save_model_path)
                dataloader, scaler= load_dataset(args,args.logger,subgraph_detect=True) 
                engine.update_dataloader(dataloader, scaler)
                engine.update_adj(dataloader['train_adj'],dataloader['val_adj'],dataloader['test_adj']) 
                engine.updatalr(args.fintue_lr)
                engine.free_par()
                engine.update_month(args.month)
                engine.online_train()
    engine.get_results(args)
    engine.get_results_final(args)
if __name__ == "__main__":
    args= get_config()
    set_seed(args.seed)
    init(args)


    device = torch.device("{}".format(args.device)) if torch.cuda.is_available() and args.device != -1 else "cpu"
    vars(args)["device"] = device
    main(args)
