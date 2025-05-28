import os
import torch
import numpy as np
import random
import argparse
import json
from run import Run
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from torch.utils.tensorboard import SummaryWriter

def prepare(config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="/<path/to/your/directory>/multidomain_data")
    parser.add_argument('--process_data_mid', default=0)
    parser.add_argument('--process_data_ready', default=0)
    parser.add_argument('--task', default='1')
    parser.add_argument('--task_run', default='targetcd')
    parser.add_argument('--base_model', default='MF')
    parser.add_argument('--seed', type=int, default= None)
    parser.add_argument('--ratio', nargs="+", default=[0.8, 0.2])
    parser.add_argument('--gpu', type=int,default=0)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_cuda', type=int, default=1)


    parser.add_argument('--rl_lr', type=float, default=0.01)


    parser.add_argument('--algo', type=str, default='marco')
    parser.add_argument('--domain', type=int, default=0)

    parser.add_argument('--target_domain', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--logdir', default='/<path/to/your/directory>')
    parser.add_argument('--performance', default=2)

    args = parser.parse_args()

    with open(config_path, 'r') as f:
        config = json.load(f)
        config['root'] = args.root
        config['base_model'] = args.base_model
        config['task'] = args.task
        config['task_run'] = args.task_run
        config['ratio'] = args.ratio
        config['epoch'] = args.epoch
        config['lr'] = args.lr
        config['use_cuda'] = args.use_cuda
        config['gpu'] = args.gpu


        config['rl_lr'] = args.rl_lr


        config['algo'] = args.algo
        config['seed'] = args.seed
        config['version'] = 'v6' 
        config['domain'] = args.domain

        config['target_domain'] = args.target_domain
        config['device'] = args.device
        config['logdir'] = args.logdir
        config['performance']= args.performance
    return args, config


if __name__ == '__main__':
    config_path = '/<path/to/your/directory>/config.json'
    args, config = prepare(config_path)
    print(f"cudnn enabled: {torch.backends.cudnn.enabled}")
    print(f"number of used gpu: {torch.cuda.device_count()}")
    print(f"cuda is available: {torch.cuda.is_available()}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(args.gpu)


    parent_dir = '/<path/to/your/directory>'
    os.makedirs(parent_dir, exist_ok=True)
    logdir=config['logdir']
    os.makedirs(logdir, exist_ok=True)
    for ratio in [[0.8, 0.2]]:
        config['ratio']=ratio
        config['modeldir']='/<path/to/your/directory>/pretrainmodel'+'/' + str(args.task_run)+'/' +str(int(ratio[0] * 10)) + '_' + str(int(ratio[1] * 10))+'/domain0'
        weight_dir=os.path.join(logdir, f'task_{args.task_run}_{ratio[1]}')
        os.makedirs(weight_dir, exist_ok=True)
        single_dirs=[]
        for i in range(4):
            model_dir=config['modeldir'].replace(f"domain{config['domain']}", f"domain{i}")
            single_dirs.append(f"{model_dir}/single.pt")
        config['single_dirs'] = single_dirs
        print(config['single_dirs'])

        for seed in [100,900,1000,10,10000]:
            for lambda_en1 in [0.0001,0.001,0.15,1.5,10]:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                seed_dir1 = os.path.join(parent_dir, f'task_{args.task_run}_{ratio[0]}_{ratio[1]}_{seed}_{lambda_en1}')
                os.makedirs(seed_dir1, exist_ok=True)
                config['lambda_en1']=lambda_en1
                print(config['lambda_en1'])
                writer1 = SummaryWriter(log_dir=seed_dir1)
                Run(config,writer1,weight_dir).main()
