import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd

import numpy as np
import logging
import pdb

import scipy.sparse as sp
import copy
from metrics import masked_mape
from metrics import masked_rmse
from metrics import compute_all_metrics

global result
global train_time 
result= {3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}},'time':{}}
new_result= {3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}},'time':{}}
train_time =0
def calculate_normalized_laplacian(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))

    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = sp.eye(adj_mx.shape[0]) - d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt).tocoo()
    return res

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

class BaseEngine():
    def __init__(self, device, model, sampler, loss_fn, lrate, optimizer, \
                 scheduler, clip_grad_value, max_epochs, patience, log_dir, logger, seed,month,learning_method,args):
        super().__init__()
        self._device = device
        self.model = model
        self.model.to(self._device)

        self._dataloader = None
        self._scaler = None

        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value

        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed
        self._month = month
        self._learning_method=learning_method
        self._logger.info('The number of parameters: {}'.format(self.model.param_num())) 
        #self._result={3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}},'time':{}}
        self._time=0
        self._args=args
        self.mode1=model
                
    def updata_adj_list_retrain(self):
        self.adj_list={}
        for month in range(3):
            adj=np.load('data/adj/'+str(int(self._args.month-1-month))+'_ouradj.npy')
            adj_mx = normalize_adj_mx(adj, self._args.adj_type)
            supports = [torch.tensor(i) for i in adj_mx]
            self.adj_list[self._args.month-1-month]=supports 


    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)


    def _to_numpy(self, tensors):
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        else:
            return tensors.detach().cpu().numpy()


    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [torch.tensor(array, dtype=torch.float32) for array in nparray]
        else:
            return torch.tensor(nparray, dtype=torch.float32)


    def _inverse_transform(self, tensors):
        def inv(tensor):
            return self._scaler.inverse_transform(tensor)

        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        else:
            return inv(tensors)

    def updatalr(self,lr):
        self._lrate=lr
    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #torch.save({'model_state_dict': gnn_model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
        filename = 'final_model_learning_{}_month_{}.pt'.format(self._learning_method,self._month)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))


    def load_model(self, save_path):
        filename = 'final_model_learning_{}_month_{}.pt'.format(self._learning_method,self._month)
        #self._logger.info('Val loss decrease from {:.4f} to {:.4f}'.format(min_loss, mvalid_loss))
        self._logger.info('Start test {}, load model from {}'.format(self._month+0.5,os.path.join(save_path, filename)))
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))   
    def update_o(self,op):
        self._optimizer=op
    def load_model2(self, save_path):
        filename = 'final_model_learning_{}_month_{}.pt'.format(self._learning_method,self._month)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))
    def load_self_model(self, save_path,model=None,op=None):
        filename = 'final_model_learning_{}_month_{}.pt'.format(self._learning_method,self._args.month)
        self._logger.info('Start down_finetune {}, load model from {}'.format(self._month,os.path.join(save_path, filename)))
        para_state_dict=torch.load(os.path.join(save_path, filename))
        res_state_dice={name:key for name,key in para_state_dict.items() if 'transformer' in name}
        model.load_state_dict(res_state_dice,strict=False)
        self.model=model
        self.model.to(self._device)
        self._optimizer=op
        del res_state_dice,para_state_dict
    def load_model4(self,path):
        self._logger.info('Start train, load model from {}'.format(path))
        self.model.load_state_dict(torch.load(path))
        self.evaluate2('test')
    def update_dataloader(self,dataloader,scaler):
        self._dataloader=dataloader 
        self._scaler=scaler

    def down_prediction_batch(self):
        #self.updata_adj_list()
        self.model.train()
        
        train_loss = []
        train_mape = []
        train_rmse = []


        self._dataloader['train_loader'].retrain_shuffle()
        loss_ewc=0
        for X, label,idx in self._dataloader['train_loader'].get_iterator():
            
            self._optimizer.zero_grad()
            max_adj=min(idx)//(30*288)
            adj=self.adj_list[max_adj+1]
            adj=self._to_device(adj)
            adj_idx=adj[0].shape[0]
            X=X[:,:,:adj_idx,:]
            label=label[:,:,:adj_idx,:]
            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))


            pred = self.model(X, adj,label)
            #print(type(pred),type(label))
            pred, label = self._inverse_transform([pred, label])
   
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('Check mask value', mask_value)
            #print(loss_ewc,self._loss_fn(pred, label, mask_value))
            loss = self._loss_fn(pred, label, mask_value)+loss_ewc
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)

    def load_savedmodel(self, save_path):
        if self._month==int(self._month):
            saved_month=(self._args.month-1)//3*3.0+1.0
        else:
            saved_month=self._month
        filename = 'final_model_learning_{}_month_{}.pt'.format(self._learning_method,saved_month)
        self._logger.info('Start train {}, load model from {}'.format(self._month+0.5,os.path.join(save_path, filename)))
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))


    def self_train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        #self._dataloader['train_loader'].shuffle()
        loss_ewc=0
        for X, label,_ in self._dataloader['train_loader'].get_iterator():
            
            self._optimizer.zero_grad()
            
            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))

            
            pred = self.model(X, self.train_adj,label)
            #print(type(pred),type(label))
            pred, label = self._inverse_transform([pred, label])
   
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('Check mask value', mask_value)
            loss = self._loss_fn(pred, label, mask_value)+loss_ewc
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)
    
        
    def down_prediction(self):
        self._logger.info('month: {}, Start training!'.format(self._month))
        wait = 0
        min_loss = np.inf
        #for epoch in range(self._args.epoch_max):
        self.updata_adj_list_retrain()
        for epoch in range(1):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse = self.down_prediction_batch()
            t2 = time.time()
            self._args.train_time =self._args.train_time+t2-t1


            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = self.evaluate('val')
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}'
            self._logger.info(message.format(epoch + 1, mtrain_loss, mtrain_rmse, mtrain_mape, \
                                             mvalid_loss, mvalid_rmse, mvalid_mape))

            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info('Val loss decrease from {:.4f} to {:.4f}'.format(min_loss, mvalid_loss))
                min_loss = mvalid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch + 1, min_loss))
                    break

        self.evaluate('test')

    def online_train(self):
        self._logger.info('month: {}, Start training!'.format(self._month))
        wait = 0
        min_loss = np.inf
        #for epoch in range(self._args.epoch_max):
        for epoch in range(1):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse = self.train_batch()
            t2 = time.time()
            self._args.train_time =self._args.train_time+t2-t1


            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = self.evaluate('val')
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}'
            self._logger.info(message.format(epoch + 1, mtrain_loss, mtrain_rmse, mtrain_mape, \
                                             mvalid_loss, mvalid_rmse, mvalid_mape))

            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info('Val loss decrease from {:.4f} to {:.4f}'.format(min_loss, mvalid_loss))
                min_loss = mvalid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch + 1, min_loss))
                    break

        self.evaluate('test')

    def updata_lr(self,lr_rate):
        self._lrate=lr_rate
    def self_learning_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        #self._dataloader['train_loader'].shuffle()
        loss_ewc=0
        for X, label,_ in self._dataloader['train_loader'].get_iterator():
            
            self._optimizer.zero_grad()
            
            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))

            
            pred = self.model(X, self.train_adj,label)
            #print(type(pred),type(label))
            pred, label = self._inverse_transform([pred, label])
   
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('Check mask value', mask_value)
            loss = self._loss_fn(pred, label, mask_value)+loss_ewc
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)

    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()
        loss_ewc=0
        for X, label,_ in self._dataloader['train_loader'].get_iterator():
            
            self._optimizer.zero_grad()
            
            X, label = self._to_device(self._to_tensor([X, label]))

            
            pred = self.model(X, self.train_adj,label)
            #print(type(pred),type(label))
            pred, label = self._inverse_transform([pred, label])
   
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('Check mask value', mask_value)
            loss = self._loss_fn(pred, label, mask_value)+loss_ewc
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)
    def train(self):
        self._logger.info('month: {}, Start training!'.format(self._month))
        wait = 0
        min_loss = np.inf
        #for epoch in range(self._args.epoch_max):
        for epoch in range(self._args.epoch_max):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse = self.train_batch()
            t2 = time.time()
            self._args.train_time =self._args.train_time+t2-t1


            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = self.evaluate('val')
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}'
            self._logger.info(message.format(epoch + 1, mtrain_loss, mtrain_rmse, mtrain_mape, \
                                             mvalid_loss, mvalid_rmse, mvalid_mape, ))

            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info('Val loss decrease from {:.4f} to {:.4f}'.format(min_loss, mvalid_loss))
                min_loss = mvalid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch + 1, min_loss))
                    break

        self.evaluate('test')
    def update_adj(self,train_adj,val_adj,test_adj):
        self.train_adj=self._to_device(train_adj)
        self.val_adj=self._to_device(val_adj)
        self.test_adj=self._to_device(test_adj)
    def evaluate2(self, mode):
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for X, label,_ in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                if mode == 'test':
                    pred = self.model(X, self.test_adj,label)
                elif mode == 'val':
                    pred = self.model(X, self.val_adj,label)
                elif mode == 'train':
                    pred = self.model(X, self.train_adj,label)
                pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == 'test' or mode == 'train':
                pre_time=[3,6,12]
                for i in pre_time:
                    res = compute_all_metrics(preds[:,:i,:], labels[:,:i,:], mask_value)
                    log = 'T {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                    self._logger.info(log.format(i, np.mean(res[0]), np.mean(res[2]), np.mean(res[1])))
                    result[i]["mae"][self._month] = res[0]
                    result[i]["mape"][self._month] = res[1]
                    result[i]["rmse"][self._month] = res[2]
    def evaluate4(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for X, label,_ in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                if mode == 'test':
                    pred = self.model(X, self.test_adj,label)
                elif mode == 'val':
                    pred = self.model(X, self.val_adj,label)
                elif mode == 'train':
                    pred = self.model(X, self.train_adj,label)
                pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == 'test' or mode == 'train':
                pre_time=[3,6,12]
                test_mae = []
                test_mape = []
                test_rmse = []
                for i in pre_time:
                    res = compute_all_metrics(preds[:,:i,:], labels[:,:i,:], mask_value)
                    log = 'T {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                    self._logger.info(log.format(i, np.mean(res[0]), np.mean(res[2]), np.mean(res[1])))
                    result[i]["mae"][self._month] = res[0]
                    result[i]["mape"][self._month] = res[1]
                    result[i]["rmse"][self._month] = res[2]
                result['time'][self._month]=self._time
                
                for i in pre_time:
                    res = compute_all_metrics(preds[:,:i,:], labels[:,:i,:], mask_value)
                    log = 'T {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                    self._logger.info(log.format(i, np.mean(res[0]), np.mean(res[2]), np.mean(res[1])))
                    new_result[i]["mae"][self._month] = res[0]
                    new_result[i]["mape"][self._month] = res[1]
                    new_result[i]["rmse"][self._month] = res[2]
                new_result['time'][self._month]=self._time

    def evaluate(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for X, label,_ in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                if mode == 'test':
                    pred = self.model(X, self.test_adj,label)
                elif mode == 'val':
                    pred = self.model(X, self.val_adj,label)
                elif mode == 'train':
                    pred = self.model(X, self.train_adj,label)
                pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == 'test' or mode == 'train':
                pre_time=[3,6,12]
                test_mae = []
                test_mape = []
                test_rmse = []
                for i in pre_time:
                    res = compute_all_metrics(preds[:,:i,:], labels[:,:i,:], mask_value)
                    log = 'T {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                    self._logger.info(log.format(i, np.mean(res[0]), np.mean(res[2]), np.mean(res[1])))
                    result[i]["mae"][self._month] = res[0]
                    result[i]["mape"][self._month] = res[1]
                    result[i]["rmse"][self._month] = res[2]
                result['time'][self._month]=self._time

    def update_model(self,model):
        self.model = model
        self.model.to(self._device)
    def update_month(self,month):
        self._month=month

    def free_par(self):
        for name,parameter in self.model.named_parameters():
            if 'season' in name:
                parameter.requires_grad = False
    def unfree_par(self):
        for name,parameter in self.model.named_parameters():
            if 'season' in name:
                parameter.requires_grad = True  
    def get_results(self,args):
        for i in [3, 6, 12]:
            for j in ['mae', 'rmse', 'mape']:
                info = ""
                if j == 'mape':
                    for month in np.arange(args.begin_month, args.end_month+1,0.5):
                        metric=0
                        if i in result:
                            if j in result[i]:
                                if month in result[i][j]:
                                    metric=result[i][j][month]*100
                                    info+="{:.2f}\t".format(metric)
                    args.logger.info("{}\t{}\t".format(i,j) + info)
                else:
                    for month in np.arange(args.begin_month, args.end_month+1,0.5):
                        metric=0
                        if i in result:
                            if j in result[i]:
                                if month in result[i][j]:
                                    metric=result[i][j][month]
                                    info+="{:.2f}\t".format(metric)
                    args.logger.info("{}\t{}\t".format(i,j) + info)

    def get_results_final(self,args):
        for i in [3, 6, 12]:
            for j in ['mae', 'rmse', 'mape']:
                info = ""
                if j == 'mape':
                    for month in np.arange(args.begin_month, args.end_month+1,1.0):
                        metric=0
                        if i in result:
                            if j in result[i]:
                                if month in result[i][j]:
                                    metric=result[i][j][month]*100+result[i][j][month+0.5]*100
                        info+="{:.2f}\t".format(metric/2)
                else:
                    for month in np.arange(args.begin_month, args.end_month+1,1.0):
                        metric=0
                        if i in result:
                            if j in result[i]:
                                if month in result[i][j]:
                                    metric=result[i][j][month]+result[i][j][month+0.5]
                        info+="{:.2f}\t".format(metric/2)
                args.logger.info("{}\t{}\t".format(i,j) + info)
    
        info = "total_time\t{}\t".format(self._args.train_time)
        args.logger.info(info)        

    def get_results_retrain(self,args):
        for i in [3, 6, 12]:
            for j in ['mae', 'rmse', 'mape']:
                info = ""
                if j == 'mape':
                    for month in np.arange(args.begin_month, args.end_month+1,1):
                        metric=0
                        if i in result:
                            if j in result[i]:
                                if month in result[i][j]:
                                    metric=result[i][j][month]*100
                        info+="{:.2f}\t".format(metric)
                else:
                    for month in np.arange(args.begin_month, args.end_month+1,1):
                        metric=0
                        if i in result:
                            if j in result[i]:
                                if month in result[i][j]:
                                    metric=result[i][j][month]
                        info+="{:.2f}\t".format(metric)
                args.logger.info("{}\t{}\t".format(i,j) + info)

    
        info = "total_time\t{}\t".format(self._time)
        args.logger.info(info)    