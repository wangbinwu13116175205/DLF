import os
import pickle
import torch
import numpy as np
import threading
import multiprocessing as mp
import calendar
import argparse
import math
import random
import scipy.sparse as sp
from scipy.sparse import linalg
import scipy.stats
import os.path as osp
import networkx as nx
import os.path as osp
import argparse
import ct
import re
import json
from torch_geometric.utils import to_dense_batch,k_hop_subgraph
from scipy.optimize import linprog


def load_adj_from_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)


def get_dataset_info(dataset):
    base_dir = os.getcwd() + '/data/'
    d = {
         'CA': [base_dir+'ca', base_dir+'ca/ca_rn_adj.npy', 8600],
         'GLA': [base_dir+'gla', base_dir+'gla/gla_rn_adj.npy', 3834],
         'GBA': [base_dir+'gba', base_dir+'gba/gba_rn_adj.npy', 2352],
         'SD': [base_dir+'sd', base_dir+'sd/sd_rn_adj.npy', 716],
        }
    assert dataset in d.keys()
    return d[dataset]

def calculate_normalized_laplacian(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))

    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = sp.eye(adj_mx.shape[0]) - d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt).tocoo()
    return res
def update(src, tmp):
    for key in tmp:
        if key!= "gpuid":
            src[key] = tmp[key]

def load_json_file(file_path):
    with open(file_path, "r") as f:
        s = f.read()
        s = re.sub('\s',"", s)
    return json.loads(s)

def init(args):    
    conf_path = osp.join(args.conf)
    info = ct.load_json_file(conf_path)
    update(vars(args), info)
    #vars(args)["path"] = osp.join(args.model_path, args.logname+args.time)
    #ct.mkdirs(args.path)
    del info

def calculate_scaled_laplacian(adj_mx, lambda_max=None, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    res = (2 / lambda_max * L) - I
    return res


def calculate_sym_adj(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    rowsum = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt)
    return res


def calculate_asym_adj(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    rowsum = np.array(adj_mx.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    res = d_mat_inv.dot(adj_mx)
    return res


def calculate_cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L.copy()]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[i - 1]) - LL[i - 2])
    return np.asarray(LL)

def normalize_adj_mx(adj_mx, adj_type, return_type='dense'):
    if adj_type == 'normlap':
        adj = [calculate_normalized_laplacian(adj_mx)]
    elif adj_type == 'scalap':
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adj_type == 'symadj':
        adj = [calculate_sym_adj(adj_mx)]
    elif adj_type == 'transition':
        adj = [calculate_asym_adj(adj_mx)]
    elif adj_type == 'doubletransition':
        adj = [calculate_asym_adj(adj_mx), calculate_asym_adj(np.transpose(adj_mx))]
    elif adj_type == 'identity':
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        return []
    
    if return_type == 'dense':
        adj = [a.astype(np.float32).todense() for a in adj]
    elif return_type == 'coo':
        adj = [a.tocoo() for a in adj]
    return adj
class DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, pad_last_sample=False):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        #logger.info('Sample num: ' + str(self.idx.shape[0]) + ', Batch num: ' + str(self.num_batch))
        
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon


    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx


    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]


    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1])
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = len(idx_ind) // 2
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, args=(x, y, idx_ind, start_index, end_index))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y)
                self.current_ind += 1

        return _wrapper()
    
class DataLoader13(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, pad_last_sample=False):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        #logger.info('Sample num: ' + str(self.idx.shape[0]) + ', Batch num: ' + str(self.num_batch))
        
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon

        
    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx

    def retrain_shuffle(self):
        len_month1=288*30
        perm = np.random.permutation(len_month1)
        idx12= self.idx[:len_month1]
        idx1 = idx12[perm]
        perm = np.random.permutation(len_month1)

        idx23 = self.idx[len_month1:len_month1*2]
        idx2  =  idx23[perm]
        perm = np.random.permutation(self.size-len_month1*2)
        idx34 = self.idx[len_month1*2:]
        idx3= idx34[perm]
        idx=np.concatenate([idx1,idx2,idx3])
        self.idx = idx

    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]

    def get_idx(self):
        return self.idx

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1])
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = len(idx_ind) // 2
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, args=(x, y, idx_ind, start_index, end_index))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y)
                self.current_ind += 1

        return _wrapper()


class DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, pad_last_sample=False):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        #logger.info('Sample num: ' + str(self.idx.shape[0]) + ', Batch num: ' + str(self.num_batch))
        
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon


    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx


    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]


    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1])
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = len(idx_ind) // 2
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, args=(x, y, idx_ind, start_index, end_index))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)


    def transform(self, data):
        return (data - self.mean) / self.std


    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class DataLoader_self(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, pad_last_sample=False):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        #logger.info('Sample num: ' + str(self.idx.shape[0]) + ', Batch num: ' + str(self.num_batch))
        
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon


    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx

    def retrain_shuffle(self):
        len_month1=288*30
        perm = np.random.permutation(len_month1)
        idx12= self.idx[:len_month1]
        idx1 = idx12[perm]
        perm = np.random.permutation(len_month1)

        idx23 = self.idx[len_month1:len_month1*2]
        idx2  =  idx23[perm]
        perm = np.random.permutation(self.size-len_month1*2)
        idx34 = self.idx[len_month1*2:]
        idx3= idx34[perm]
        idx=np.concatenate([idx1,idx2,idx3])
        self.idx = idx
        
    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):

        K=int(idx_ind*0.5)
        random_integers = np.random.choice(np.arange(start_idx,  end_idx + 1), size=K, replace=False)
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]

    def get_idx(self):
        return self.idx

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1])
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = 2
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, args=(x, y, idx_ind, start_index, end_index))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y)
                self.current_ind += 1

        return _wrapper()
class selflearning_DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, pad_last_sample=False):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        #logger.info('Sample num: ' + str(self.idx.shape[0]) + ', Batch num: ' + str(self.num_batch))
        
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon


    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx

    def retrain_shuffle(self):
        len_month1=288*30
        perm = np.random.permutation(len_month1)
        idx12= self.idx[:len_month1]
        idx1 = idx12[perm]
        perm = np.random.permutation(len_month1)

        idx23 = self.idx[len_month1:len_month1*2]
        idx2  =  idx23[perm]
        perm = np.random.permutation(self.size-len_month1*2)
        idx34 = self.idx[len_month1*2:]
        idx3= idx34[perm]
        idx=np.concatenate([idx1,idx2,idx3])
        self.idx = idx

    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        K=int(len(idx_ind)*0.75)
        random_integers = np.random.choice(np.arange(start_idx,end_idx + 1), size=K, replace=False)
        for i in range(start_idx, end_idx):
                x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
                y[i] = self.data[idx_ind[i] + self.x_offsets, :, :1]
                if i in random_integers:
                    x[i,:,:,0]=0
    def get_idx(self):
        return self.idx

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1])
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = 1
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, args=(x, y, idx_ind, start_index, end_index))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y,idx_ind)
                self.current_ind += 1

        return _wrapper()
def process_data_idx(data_path, args, logger=None):
    ptr = np.load(os.path.join(data_path, args.data_year, 'his.npz'))
    #logger.info('Data shape: ' + str(ptr['data'].shape))
    vars(args)["year_length"]=ptr['data'].shape[0]
    dataloader = {}
    if args.mode=='retrin':
        idx_list=get_idx_retrain(args)
    if args.mode=='online':
        idx_list=get_idx_online(args)

def get_index(month):
    if (month-1)%3==0:
        day_gap=90
    else:
        day_gap=15
    day_high=(month-1)*30
    day_low=day_high-day_gap

    data_len=day_gap*288
    val_high=day_high*288
    val_len=int(data_len*0.3)
    val_low=val_high-val_len

    train_high=val_low
    train_low=train_high-(data_len-val_len)

    test_low=val_high
    test_high=test_low+15*288
    return np.array([i for i in range(int(train_low),int(train_high))]),np.array([i for i in range(int(val_low),int(val_high))]),np.array([i for i in range(int(test_low),int(test_high))]) 

def get_adj_idx(month):
    adj=np.load('/home/wbw/ICLR/DLF/data/original/2023_12_3new_adj.npy')
    res_adj_len=adj.shape[0]
    adj_list={}
    adj_idx={}
    #data_final=data_our
    for month1 in range(12,2,-1):
        res_adj_len=int(res_adj_len*0.95)
        res_node_list=[i for i in range(res_adj_len)]
        adj_lin=adj[res_node_list,:]
        adj_lin=adj_lin[:,res_node_list]
        adj_list[month1]=adj_lin
        adj_idx[month1]=res_node_list
    for i in range(3,12):
        adj1= adj_list[i]
        adj2= adj_list[i+1]
        for a in range(adj1.shape[0]):
            for b in range(adj1.shape[1],adj2.shape[1]):
                 adj_list[i+1][a][b]=0
    return adj_list[month],adj_idx[month]


class DataLoader13(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, pad_last_sample=False):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        #logger.info('Sample num: ' + str(self.idx.shape[0]) + ', Batch num: ' + str(self.num_batch))
        
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon


    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx

    def retrain_shuffle(self):
        len_month1=288*30
        perm = np.random.permutation(len_month1)
        idx12= self.idx[:len_month1]
        idx1 = idx12[perm]
        perm = np.random.permutation(len_month1)

        idx23 = self.idx[len_month1:len_month1*2]
        idx2  =  idx23[perm]
        perm = np.random.permutation(self.size-len_month1*2)
        idx34 = self.idx[len_month1*2:]
        idx3= idx34[perm]
        idx=np.concatenate([idx1,idx2,idx3])
        self.idx = idx

    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]

    def get_idx(self):
        return self.idx

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1])
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = len(idx_ind) // 2
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, args=(x, y, idx_ind, start_index, end_index))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y,idx_ind)
                self.current_ind += 1

        return _wrapper()


def self_load_dataset(args, logger=None,subgraph_detect=False):
    ptr = np.load("/home/wbw/ICLR/DLF/data/original/1402_our_his.npz")
    #logger.info('Data shape: ' + str(ptr['data'].shape))
    vars(args)["year_length"]=ptr['data'].shape[0]
    dataloader = {}
    idx_list={}
    idx_list['train'],idx_list['val'],idx_list['test']=get_index(args.month)
    month_per=(args.month-1)//3
    month_old=int(month_per*3+1)
    month_new=int(args.month-0.5)
    if subgraph_detect:
            subgraph,node_list=online_select_data(args,month_old,month_new)
            adj_idx=np.load('/home/wbw/ICLR/DLF/data/adj/'+str(month_new)+'_ouradjidx.npy')
            adj_idx=adj_idx[node_list]
            adj=subgraph
    else:
            adj=np.load(osp.join(args.graph_path, str(month_new)+"_ouradj.npy"))
            adj_idx=np.load('/home/wbw/ICLR/DLF/data/adj/'+str(month_new)+'_ouradjidx.npy')
    adj_mx = normalize_adj_mx(adj, args.adj_type)
    supports = [torch.tensor(i) for i in adj_mx]
    for cat in ['train']:
        dataloader[cat + '_loader'] = selflearning_DataLoader(ptr['data'][:,adj_idx,:3], idx_list[cat], \
                                                        args.seq_len, args.horizon, args.bs, logger)

        logger.info(cat+'Data length: ' + str(len(idx_list[cat])))
        dataloader[cat+'_adj']=supports
    for cat in ['val']:
        dataloader[cat + '_loader'] = selflearning_DataLoader(ptr['data'][:,adj_idx,:3], idx_list[cat], \
                                                        args.seq_len, args.horizon, args.bs, logger)

        logger.info(cat+'Data length: ' + str(len(idx_list[cat])))
        dataloader[cat+'_adj']=supports
    for cat in ['test']:
        adj=np.load(osp.join(args.graph_path, str(int(args.month))+"_ouradj.npy"))
        adj_idx=np.load('/home/wbw/ICLR/DLF/data/adj/'+str(int(args.month))+'_ouradjidx.npy')  
        adj_mx = normalize_adj_mx(adj, args.adj_type)
        supports = [torch.tensor(i) for i in adj_mx]
        dataloader[cat + '_loader'] = DataLoader13(ptr['data'][:,adj_idx,:3], idx_list[cat], \
                                                        args.seq_len, args.horizon, args.bs, logger)
        dataloader[cat+'_adj']=supports

    scaler = StandardScaler(mean=ptr['mean'], std=ptr['std'])
    return dataloader, scaler

def load_dataset(args, logger=None,subgraph_detect=False):
    ptr = np.load("/home/wbw/ICLR/DLF/data/original/1402_our_his.npz")
    #logger.info('Data shape: ' + str(ptr['data'].shape))
    vars(args)["year_length"]=ptr['data'].shape[0]
    dataloader = {}
    idx_list={}
    idx_list['train'],idx_list['val'],idx_list['test']=get_index(args.month)
    month_new=int(args.month-0.5)
    if args.month==int(args.month):
        month_per=(args.month-1)//3
        month_old=int(month_per*3+1)
    else:
        month_per=(args.month-1)//3
        month_old=int(month_per*3+1)
    if subgraph_detect:
            subgraph,node_list=online_select_data(args,month_old,month_new)
            adj_idx=np.load('data/adj/'+str(month_new)+'_ouradjidx.npy')
            adj_idx=adj_idx[node_list]
            adj=subgraph
    else:
            adj=np.load(osp.join(args.graph_path, str(month_new)+"_ouradj.npy"))
            adj_idx=np.load('data/adj/'+str(month_new)+'_ouradjidx.npy')
    adj_mx = normalize_adj_mx(adj, args.adj_type)
    supports = [torch.tensor(i) for i in adj_mx]
    for cat in ['train','val']:
        dataloader[cat + '_loader'] = DataLoader13(ptr['data'][:,adj_idx,:3], idx_list[cat], \
                                                        args.seq_len, args.horizon, args.bs, logger)

        logger.info(cat+'Data length: ' + str(len(idx_list[cat])))
        dataloader[cat+'_adj']=supports
    for cat in ['test']:
        adj=np.load(osp.join(args.graph_path, str(int(args.month))+"_ouradj.npy"))
        adj_idx=np.load('data/adj/'+str(int(args.month))+'_ouradjidx.npy')  
        adj_mx = normalize_adj_mx(adj, args.adj_type)
        supports = [torch.tensor(i) for i in adj_mx]
        dataloader[cat + '_loader'] = DataLoader13(ptr['data'][:,adj_idx,:3], idx_list[cat], \
                                                        args.seq_len, args.horizon, args.bs, logger)
        dataloader[cat+'_adj']=supports
    scaler = StandardScaler(mean=ptr['mean'], std=ptr['std'])
    return dataloader, scaler

def get_data_detect_weak(month,args):
    ptr = np.load(os.path.join(args.data_path, '2018/his.npz'))
    path='data/'+args.mode+'_idx/'
    #month=int(month)
    path=os.path.join(path,str(month)+'.npz')
    idx_list=np.load(path)
    idx_list1=list(idx_list['train'])+list(idx_list['val'])
    idx=np.array(idx_list1)
    month_int=int(month-0.5)
    path_adj='data/adj/'+str(month_int)+'.npz'
    adj_data=np.load(path_adj)
    adj_idx=adj_data['idx']
    data=ptr['data'][idx,:,0]
    data=data[:,adj_idx]
    N=data.shape[-1]
    data=data.reshape(-1,96,N)
    data=data[:14,:,:]
    data=data.reshape(-1,7,96,N)#1eee
    return data.sum(axis=0)  #7,288,N
def get_idx_retrain(args):
    idx_list={}
    sum_day=0
    sum_last_day=0
    month=math.trunc(args.month)
    if month==3:
            for i in range(1,4):
                sum_day=sum_day+calendar.monthrange(int(args.data_year),i)[1]
    else:
            for i in range(month-2,month+1):
                sum_day=sum_day+calendar.monthrange(int(args.data_year),i)[1]
            for i in range(1,month-2):
                sum_last_day=sum_last_day+calendar.monthrange(int(args.data_year),i)[1]
    train_low=sum_last_day*96
    train_length=int(sum_day*96*0.9)
    train_high=train_low+train_length
    idx_list['train']=np.array([i for i in range(train_low,train_high)])
    val_low=train_high
    val_length=sum_day*96-train_length
    val_high=train_high+val_length
    idx_list['val'] =np.array([i for i in range(val_low,val_high)])
    test_low=val_high
    sum_day_next=calendar.monthrange(int(args.data_year),month+1)[1]
    test_length=sum_day_next*96
    test_high=test_low+test_length
    idx_list['test'] =np.array([i for i in range(test_low,test_high)])
    
    path='data/'+args.mode+'_idx/'
    path=os.path.join(path,str(args.month)+'.npz')
    np.savez(path,train=idx_list['train'],test=idx_list['test'],val=idx_list['val'])


def get_idx_online(args):
    idx_list={}

    if args.learning=='online':
        month_int=math.trunc(args.month)
        month_all_length=calendar.monthrange(int(args.data_year),month_int)[1]*96
        if args.month-month_int==0:
            sum_length=15*96
        else:
            sum_length=month_all_length-15*96
        train_low=args.last_data_point
        train_length=int(sum_length*0.8)
        train_high=train_low+train_length
        idx_list['train']=np.array([i for i in range(train_low,train_high)])
       
        val_length=sum_length-train_length
        val_low=train_high
        val_high=val_low+val_length
        idx_list['val'] =np.array([i for i in range(val_low,val_high)])

        if args.month-month_int==0:
            test_length=month_all_length-sum_length
        else:
            test_length=15*96

        test_length=month_all_length-sum_length
        test_low=val_high
        test_high=test_low+test_length
        idx_list['test'] =np.array([i for i in range(test_low,test_high)])

    vars(args)["last_data_point"]=val_high

    path='data/'+args.mode+'_idx/'
    path=os.path.join(path,str(args.month)+'.npz')
    np.savez(path,train=idx_list['train'],test=idx_list['test'],val=idx_list['val'])

    return idx_list
def dfs(adj_matrix, node, visited, hop_count):
    visited[node] = True
    if hop_count == 0:
        return
    for i in range(len(adj_matrix)):
        if adj_matrix[node][i] != 0 and not visited[i]:
            dfs(adj_matrix, i, visited, hop_count - 1)

def get_nhop_subgraph(adj_matrix, nodes,n_hop):
    num_nodes = len(adj_matrix)
    visited = np.zeros(num_nodes, dtype=bool)
    for node in nodes:
        dfs(adj_matrix, node, visited, n_hop)
    subgraph_adj_matrix = adj_matrix[visited][:, visited]
    return subgraph_adj_matrix
def process_adj(args):
    adj=np.load(args.graph_path)
    size=adj.shape[0]
    elements_to_remove=[]
    lst = list(range(1, size)) 
    res_list=lst
    for i in range(13,1,-1):
        size=len(res_list)
        lst=list(range(0, size)) 
        rand_size=int(size*0.05)
        rand=random.sample(lst,rand_size)
        rand.sort(reverse=True)
        for index in rand:
            res_list.pop(index)
        adj1=adj[res_list]
        adj2=adj1[:,res_list]
        path='data/adj/'+str(i)+'.npz'
        np.savez(path,adj=adj2,idx=res_list)

def get_n_hop_subgraph2(adj_matrix, nodes, n):
    graph = nx.from_numpy_array(adj_matrix)
    subgraph_nodes = set(nodes)
    for _ in range(n):
        neighbors = set()
        for node in subgraph_nodes:
            neighbors.update(graph.neighbors(node))
        subgraph_nodes.update(neighbors)
    subgraph = graph.subgraph(subgraph_nodes)
    return subgraph,subgraph_nodes

def online_select_data(args,month_old,month_new):

    node_list = list()
    old_adj=np.load(osp.join(args.graph_path, str(month_old)+"_ouradj.npy"))
    new_adj=np.load(osp.join(args.graph_path, str(month_new)+"_ouradj.npy"))
    
    old_node_size = old_adj.shape[0]
    new_node_size = new_adj.shape[0]
    vars(args)["graph_size"]= new_node_size
    if new_node_size>old_node_size:
        node_list.extend(list(range(old_node_size,new_node_size)))       #new nodes 
    if  args.detect:  
        vars(args)["new_topk"] = int(0.01*new_node_size)
        vars(args)["replay_topk"] = int(0.04*new_node_size)                
        evolution_node_list,replay_node_list=select_evol_nodes(args)
        node_list.extend(list(evolution_node_list))
        node_list.extend(list(replay_node_list))                         
        node_list = list(set(node_list))
    if len(node_list) < int(0.2*args.graph_size):
        new_graph_list=list(range(new_node_size))
        res_list=[]
        for x in new_graph_list:
            if x not in node_list:
                res_list.append(x)
        node_list_new = random.sample(res_list, int(0.2*new_node_size)-len(node_list))
    node_list.extend(node_list_new)  
    node_list = list(set(node_list))
    if len(node_list) != 0 :
        num_hops=2
        subgraph2,node_idx=get_n_hop_subgraph2(new_adj,node_list,num_hops) 
    return nx.to_numpy_array(subgraph2), list(node_idx)
def get_feature(data, graph, args, model, adj):
    node_size = data.shape[1]
    
    dataloader = DataLoader()
    for X, label in dataloader.get_iterator():
        data = data.to(args.device, non_blocking=True)
        feature, _ = to_dense_batch(model.feature(X, adj), batch=args.batch_size)#B*N,T,D
        node_size = feature.size()[1]

        feature = feature.permute(1,0,2)

        return feature.cpu().detach().numpy()

def get_adj(month, args):
    month=int(month)
    adj = np.load(osp.join(args.graph_path, str(month)+"_adj.npz"))["adj"]
    adj_mx = normalize_adj_mx(adj, args.adj_type)
    supports = [torch.tensor(i) for i in adj_mx]
    return supports

def wasserstein_distance(p, q, D):
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    D = np.array(D)
    D = D.reshape(-1)

    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    myresult = result.fun

    return myresult

def evolution_detect_mode1(x, y):
    x, y = np.array(x), np.array(y)
    x_norm = (x**2).sum(axis=1, keepdims=True)**0.5
    y_norm = (y**2).sum(axis=1, keepdims=True)**0.5
    p = x_norm[:, 0] / x_norm.sum()
    q = y_norm[:, 0] / y_norm.sum()
    D = 1 - np.dot(x / x_norm, (y / y_norm).T)
    return wasserstein_distance(p, q, D)

def evolution_detect_mode2(q,p):
    evolution=0
    node_list=[]
    #q:T,N,D
    node_size=q.shape[1]
    for node in range(node_size):
        for i in range(q.shape[-1]):
            evolution=evolution+scipy.stats.wasserstein_distance(q[:,i,node],p[:,i,node])
        node_list.append(evolution)
    return node_list

def select_evol_nodes(args):
    old_data = get_data_detect_weak(args.month-0.5,args).sum(axis=0)
    new_data = get_data_detect_weak(args.month,args).sum(axis=0)
    node_size = old_data.shape[-1]
    score = []
    for node in range(node_size):
        score.append(scipy.stats.wasserstein_distance(old_data[:,node], new_data[:,node]))
    return np.argpartition(np.asarray(score), -args.new_topk)[-args.new_topk:], np.argpartition(np.asarray(score), args.replay_topk)[:args.replay_topk]







