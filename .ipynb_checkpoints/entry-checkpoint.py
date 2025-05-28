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
    parser.add_argument('--root', type=str, default="/root/autodl-tmp/root/autodl-tmp/multidomain")
    parser.add_argument('--process_data_mid', default=0)
    parser.add_argument('--process_data_ready', default=0)
    parser.add_argument('--task', default='1')
    parser.add_argument('--task_run', default='targetcd')
    parser.add_argument('--base_model', default='MF')
    parser.add_argument('--seed', type=int, default= None)
    parser.add_argument('--ratio', nargs="+", default=[0.2, 0.8])
    parser.add_argument('--gpu', type=int,default=0)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_cuda', type=int, default=1)
    # parser.add_argument('--version', default='v1')
    # parser.add_argument('--reward_name', default='r1')
    parser.add_argument('--rl_lr', type=float, default=0.01)
    # parser.add_argument('--env_stage', default='env_v6')
    # parser.add_argument('--action_space', type=int, default=3)
    parser.add_argument('--algo', type=str, default='remit')
    parser.add_argument('--domain', type=int, default=0)
    # parser.add_argument('--modeldir', type=str, default='/root/MAREC/single-pretrain/ceshi/0.5/single_model/domain0')#批量训练的时候这里要注释掉
    parser.add_argument('--target_domain', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--logdir', default='/root/autodl-tmp/new_entropy_experiment')
    parser.add_argument('--performance', default=1.52)

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
        # config['version'] = args.version
        # config['reward_name'] = args.reward_name
        config['rl_lr'] = args.rl_lr
        # config['env_stage'] = args.env_stage
        # config['action_space'] = args.action_space
        # config['as_learn_method'] = args.as_learn_method
        config['algo'] = args.algo
        config['seed'] = args.seed
        config['version'] = 'v6' if config['algo'] == 'remit' else 'v1'
        config['domain'] = args.domain
        # config['modeldir'] = args.modeldir  #这个在批量训练的时候记得注释掉
        config['target_domain'] = args.target_domain
        config['device'] = args.device
        config['logdir'] = args.logdir
        config['performance']= args.performance
    return args, config


if __name__ == '__main__':
    config_path = '/root/MAREC/coding_beifen/entropy研究/config.json'
    args, config = prepare(config_path)
    print(f"cudnn enabled: {torch.backends.cudnn.enabled}")
    print(f"number of used gpu: {torch.cuda.device_count()}")
    print(f"cuda is available: {torch.cuda.is_available()}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(args.gpu)

#=====================这里是批量批量常规的训练========================
    parent_dir = '/root/tf-logs/新实验/MAPPO_entropy'
    os.makedirs(parent_dir, exist_ok=True)
    logdir=config['logdir']
    os.makedirs(logdir, exist_ok=True)
    for ratio in [[0.8, 0.2]]:
        config['ratio']=ratio
        config['modeldir']='/root/autodl-tmp/pretrainmodel'+'/' + str(args.task_run)+'/' +str(int(ratio[0] * 10)) + '_' + str(int(ratio[1] * 10))+'/domain0'
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
        for seed in range(30):
            for lambda_en1 in [0.0001,0.001,0.15,1.5,10]:
            # random.seed(seed)
            # np.random.seed(seed)
            # torch.manual_seed(seed)
            # torch.cuda.manual_seed(seed)
                seed_dir1 = os.path.join(parent_dir, f'task_{args.task_run}_{ratio[0]}_{ratio[1]}_{seed}_{lambda_en1}')
                os.makedirs(seed_dir1, exist_ok=True)
                config['lambda_en1']=lambda_en1
                print(config['lambda_en1'])
                writer1 = SummaryWriter(log_dir=seed_dir1)
                Run(config,writer1,weight_dir).main()
                
#=====================================================

#===================这里是测试调参用的======================
# if __name__ == '__main__':
#     config_path = '/root/MAREC/coding_beifen/泛化研究/config.json'
#     args, config = prepare(config_path)
#     print(f"cudnn enabled: {torch.backends.cudnn.enabled}")
#     print(f"number of used gpu: {torch.cuda.device_count()}")
#     print(f"cuda is available: {torch.cuda.is_available()}")
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#     torch.cuda.set_device(args.gpu)
#     # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#     logdir=config['logdir']
#     os.makedirs(logdir, exist_ok=True)
#     config['modeldir']='/root/autodl-tmp/pretrainmodel'+'/' + str(args.task_run)+'/' +str(int(config['ratio'][0] * 10)) + '_' + str(int(config['ratio'][1] * 10))+'/domain0'
#     single_dirs=[]
#     for i in range(4):
#         model_dir=config['modeldir'].replace(f"domain{config['domain']}", f"domain{i}")
#         single_dirs.append(f"{model_dir}/single.pt")
#     config['single_dirs'] = single_dirs
#     print(config['single_dirs'])
#     writer = SummaryWriter(log_dir=logdir)
#     seed=10
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     # for i in range(4):
#     Run(config,writer).main()



    # single_dirs=[]
    # logdir=config['logdir']
    # os.makedirs(logdir, exist_ok=True)
    # for i in range(4):
    #     model_dir=config['modeldir'].replace(f"domain{config['domain']}", f"domain{i}")
    #     single_dirs.append(f"{model_dir}/single.pt")
    # config['single_dirs'] = single_dirs
    # print(config['single_dirs'])
    # ratio=config['ratio']
    # for i in range(5):
    #     seed_dir = os.path.join(parent_dir, f'task_{args.target_domain}_{ratio[1]}_{i}')
    #     os.makedirs(seed_dir, exist_ok=True)
    #     writer = SummaryWriter(log_dir=seed_dir)
    #     Run(config,writer).main()       
    #     writer.close()


    # for seed in [10,500,900,1000,2020]:
    #     seed_dir = os.path.join(parent_dir, f'seed_{seed}')
    #     os.makedirs(seed_dir, exist_ok=True)
    #     writer = SummaryWriter(log_dir=seed_dir)
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    # writer = SummaryWriter(log_dir=config['logdir'])
    # Run(config,writer).main()
    # writer.close()


    # for i in range(5):
    #     Run(config).main()


    # if not args.process_data_mid and not args.process_data_ready:
    #     if config['seed'] is None:
    #         print("No seed is provided. Run algorithms on five random seeds(2020, 10, 1000, 900, 500).... ")
    #         res = []
    #         for seed in [2020, 10, 1000, 900, 500]:
    #             random.seed(seed)
    #             np.random.seed(seed)
    #             torch.manual_seed(seed)
    #             torch.cuda.manual_seed(seed)
    #             res.append(Run(config).main())
    #         mae = [d[0] for d in res]
    #         rmse = [d[1] for d in res]
    #         print(f'Avg results over five runs: \n Avg/mae: {sum(mae)/len(mae)}, Avg/rmse:{sum(rmse)/len(rmse)}')
    #     else:
    #         torch.manual_seed(config['seed'])
    #         print(f"task:{config['task']}; version:{config['version']}; model:{config['base_model']}; ratio:{config['ratio']}; epoch:{config['epoch']}; lr:{config['lr']}; gpu:{config['gpu']}; seed:{config['seed']}; algo:{config['algo']}")
    #         Run(config).main()


            
