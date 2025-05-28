
import os
import torch
import numpy as np
import random
import argparse
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import tqdm
from tensorflow import keras
import pandas as pd
import warnings


class MFBasedModel(torch.nn.Module):

    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim)

    def forward(self, x):
        uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))
        iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))
        emb = torch.cat([uid_emb, iid_emb], dim=1)
        x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
        return x


class Run():
    def __init__(self,
                 config
                 ):
        self.use_cuda = config['use_cuda']
        self.version = config['version']
        self.base_model = config['base_model']
        self.root = config['root']
        self.ratio = config['ratio']
        self.task = config['task']
        self.src = config['src_tgt_pairs'][self.task]['src']
        self.tgt = config['src_tgt_pairs'][self.task]['tgt']
        self.uid_all = config['src_tgt_pairs'][self.task]['uid']
        self.iid_all = config['src_tgt_pairs'][self.task]['iid']
        self.batchsize_src = config['src_tgt_pairs'][self.task]['batchsize_src']
        self.batchsize_tgt = config['src_tgt_pairs'][self.task]['batchsize_tgt']
        self.batchsize_meta = config['src_tgt_pairs'][self.task]['batchsize_meta']
        self.batchsize_map = config['src_tgt_pairs'][self.task]['batchsize_map']
        self.batchsize_test = config['src_tgt_pairs'][self.task]['batchsize_test']
        self.batchsize_aug = self.batchsize_src
        self.algo = config['algo']

        self.epoch = config['epoch']
        self.emb_dim = config['emb_dim']
        self.meta_dim = config['meta_dim']
        self.num_fields = config['num_fields']
        self.lr = config['lr']
        self.wd = config['wd']

        self.reward_name = config['reward_name']
        self.rl_lr = config['rl_lr']
        self.env_stage = config['env_stage']
        self.action_space = config['action_space']
        self.as_learn_method = config['as_learn_method']


        self.input_root = self.root
        self.src_path = self.input_root + '/pretrain.csv'
        self.tgt_path = self.input_root + 'train_tgt.csv'
        self.meta_path = self.input_root + '/train_meta.csv'
        self.test_path = self.input_root + '/test.csv'


        self.results = {'tgt_mae': 10, 'tgt_rmse': 10,
                        'aug_mae': 10, 'aug_rmse': 10,
                        'emcdr_mae': 10, 'emcdr_rmse': 10,
                        'ptupcdr_mae': 10, 'ptupcdr_rmse': 10,
                        'rlcdr_mae': 10, 'rlcdr_rmse': 10
                        }
        self.spath=config['modeldir']
        self.latent_dim = config['emb_dim']

    def seq_extractor(self, x):
        x = x.rstrip(']').lstrip('[').split(', ')
        for i in range(len(x)):
            try:
                x[i] = int(x[i])
            except:
                x[i] = self.iid_all
        return np.array(x)

    def record_num(self, path):
        data = pd.read_csv(path, header=None)
        return data.shape[0]

    def read_log_data(self, path, batchsize, history=False):
        if not history:
            cols = ['uid', 'iid', 'y']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data = pd.read_csv(path, header=None)
            data.columns = cols
            X = torch.tensor(data[x_col].values, dtype=torch.long)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=True)
            return data_iter
        else:
            data = pd.read_csv(path, header=None)
            cols = ['uid', 'iid', 'y', 'pos_seq']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data.columns = cols
            pos_seq = keras.preprocessing.sequence.pad_sequences(data.pos_seq.map(self.seq_extractor), maxlen=160,
                                                                 padding='post')
            pos_seq = torch.tensor(pos_seq, dtype=torch.long)
            id_fea = torch.tensor(data[x_col].values, dtype=torch.long)
            X = torch.cat([id_fea, pos_seq], dim=1)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=True)
            return data_iter

    def read_map_data(self):
        cols = ['uid', 'iid', 'y', 'pos_seq']
        data = pd.read_csv(self.meta_path, header=None)
        data.columns = cols
        X = torch.tensor(data['uid'].unique(), dtype=torch.long)
        y = torch.tensor(np.array(range(X.shape[0])), dtype=torch.long)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_map, shuffle=True)
        return data_iter

    def read_aug_data(self):
        cols_train = ['uid', 'iid', 'y']
        x_col = ['uid', 'iid']
        y_col = ['y']
        src = pd.read_csv(self.src_path, header=None)
        src.columns = cols_train
        tgt = pd.read_csv(self.tgt_path, header=None)
        tgt.columns = cols_train

        X_src = torch.tensor(src[x_col].values, dtype=torch.long)
        y_src = torch.tensor(src[y_col].values, dtype=torch.long)
        X_tgt = torch.tensor(tgt[x_col].values, dtype=torch.long)
        y_tgt = torch.tensor(tgt[y_col].values, dtype=torch.long)
        X = torch.cat([X_src, X_tgt])
        y = torch.cat([y_src, y_tgt])
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_aug, shuffle=True)

        return data_iter

    def get_data(self):
        print('========Reading data========')


        data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
        print('tgt {} iter / batchsize = {} '.format(len(data_tgt), self.batchsize_tgt))


        return data_tgt

    def get_model(self):
        if self.base_model == 'MF':
            model = MFBasedModel(self.uid_all, self.iid_all,self.emb_dim)
        return model.cuda() if self.use_cuda else model

    def get_optimizer(self, model):
        optimizer_tgt = torch.optim.Adam(params=model.parameters(), lr=self.lr, weight_decay=self.wd)

        return optimizer_tgt

    def eval_mae(self, model, data_loader):
        print('Evaluating MAE:')
        model.eval()
        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            for X, y in data_loader:

                pred = model(X)
                targets.extend(y.squeeze(1).tolist())
                predicts.extend(pred.tolist())
        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts)
        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()

    def train(self, data_loader, model, criterion, optimizer, epoch, mapping=False):
        print('Training Epoch {}:'.format(epoch + 1))
        model.train()

        for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            if mapping:
                src_emb, tgt_emb = model(X)
                loss = criterion(src_emb, tgt_emb)
            else:
                pred = model(X)
                loss = criterion(pred, y.squeeze().float())
            model.zero_grad()
            loss.backward()
            optimizer.step()


    def update_results(self, mae, rmse, phase):
        if mae < self.results[phase + '_mae']:
            self.results[phase + '_mae'] = mae
        if rmse < self.results[phase + '_rmse']:
            self.results[phase + '_rmse'] = rmse


    def TgtOnly(self, model, data_tgt,criterion, optimizer,savepath):
        print('=========TgtOnly========')

        for i in range(50):
            self.train(data_tgt, model, criterion, optimizer, i)


            torch.save(model.state_dict(), savepath)


    def main(self):
        model = self.get_model()
        data_tgt= self.get_data()
        optimizer_tgt = self.get_optimizer(model)
        criterion = torch.nn.MSELoss()
        self.TgtOnly(model, data_tgt, criterion, optimizer_tgt,self.spath)


def prepare(config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_data_mid', default=0)
    parser.add_argument('--process_data_ready', default=0)
    parser.add_argument('--task', default='1')
    parser.add_argument('--base_model', default='MF')
    parser.add_argument('--seed', type=int, default= 10)
    parser.add_argument('--ratio', nargs="+", default=[0.8, 0.2])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_cuda', type=int, default=1)


    parser.add_argument('--rl_lr', type=float, default=0.01)


    parser.add_argument('--algo', type=str, default='BASE')
    parser.add_argument('--root', type=str, default="/<path/to/your/directory>/multidomain/_8_2/targetcd")
    parser.add_argument('--modeldir', type=str, default='/<path/to/your/directory>/pretrainmodel/targetcd/8_2/domain1/single.pt')

    args = parser.parse_args()

    with open(config_path, 'r') as f:
        config = json.load(f)

        config['base_model'] = args.base_model
        config['task'] = args.task
        config['ratio'] = args.ratio
        config['epoch'] = args.epoch
        config['lr'] = args.lr
        config['use_cuda'] = args.use_cuda
        config['gpu'] = args.gpu


        config['rl_lr'] = args.rl_lr


        config['algo'] = args.algo
        config['seed'] = args.seed
        config['root'] = args.root
        config['version'] = 'v6'
        config['modeldir'] = args.modeldir

    return args, config


if __name__ == '__main__':
    config_path = '/<path/to/your/directory>/root/singleMF/config.json'
    args, config = prepare(config_path)
    print(f"cudnn enabled: {torch.backends.cudnn.enabled}")
    print(f"number of used gpu: {torch.cuda.device_count()}")
    print(f"cuda is available: {torch.cuda.is_available()}")
    print(config['root'])
    print(config['modeldir'])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(config['seed'])
    Run(config).main()