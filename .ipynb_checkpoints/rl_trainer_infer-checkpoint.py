import numpy as np
import torch
import tqdm
import torch.nn as nn
from collections import namedtuple, deque
import random
from torch.autograd import Variable
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import pandas as pd

Trajectory = namedtuple('Trajectory',
                        ['gs','s','s_v' ,'a', 'a_log_p','r'])





# class ReplayBuffer():
#     """https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory"""
#     def __init__(self):
#         self.memory = []

#     def push(self, *args):
#         """Save a transition"""
#         self.memory.append(Transition(*args))

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)

class Trainer():
    """Runs games for given agents. Optionally will visualise and save the results"""
    def __init__(self, env, model, agent, data_loader, capacity,ppo_epoch,device):
        self.env = env
        self.cdr_model = model
        self.agent = agent
        self.replay_buffer = []
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.data_loader = data_loader
        self.ppo_epoch=ppo_epoch
        self.device=device

    def initial_agent_policy(self, max_episode):
        """
        get the initial params for three Aspect Selector agent policy models.
        only one epoch of the data_meta.
        """
        self.agent.actor.train()
        self.agent.critic.train()
        # self.cdr_model.train()
        all_reward = 0
        i_episode = 0
        # data_iter = iter(self.data_loader)
        # for i in range(len(self.data_loader) - 2):
        for X, y in self.data_loader:
        # for X, y in tqdm.tqdm(self.data_loader, smoothing=0, mininterval=1.0):  
        #X[128,162],y[128,1]
            s_v=[]
            ap=[]
            a_log=[]
            i_episode = i_episode + 1
            
            global_state,state = self.env.reset(X, y)  #torch.Size([64, 11]),# state torch.Size([64, 4, 11])
            for i,j in zip(state,global_state):#i的形状 torch.Size([4, 11])
                action_prob = self.event_softmax(torch.randn(i.shape[0],1)).to(self.device)  
                action_log_prob = self.agent.ini_log(action_prob).to(self.device) #actionprob torch.Size([4, 1]),action_log_prob是([4])
                action_prob=action_prob.reshape(1,4).to(self.device) #torch.Size([1, 4])
                action_log_prob=action_log_prob.reshape(1,4).to(self.device) 
                state_value=self.agent.get_value(j, i)#(4,1)
                state_value=state_value.reshape(1,4).to(self.device) 
                s_v.append(state_value)
                ap.append(action_prob)
                a_log.append(action_log_prob)
            action1=torch.cat([x for x in ap], dim=0)#(64,4)     
            reward, _, _ = self.env.step(X, y, action1)  # reward torch.Size([128])
            all_reward = all_reward+reward.mean()
            reward=reward.unsqueeze(1).repeat(1,4)#torch.Size([64, 4])   
            action_prob1=torch.cat([x for x in a_log], dim=0) #torch.Size([64, 4])
            state_value1=torch.cat([x for x in s_v], dim=0) #torch.Size([64, 4])
            # next_state = self.env.reset(batch_next[0], batch_next[1])
            transition=Trajectory(global_state.clone().detach(),state.clone().detach(), state_value1.clone().detach(),action1.clone().detach(),
                                                 action_prob1.clone().detach(), reward.clone().detach())
            self.agent.train(transition)
            if i_episode >= max_episode:
                break

    def cdr_with_fixed_as(self, optimizer):
        """
        in the CDR process, we fix the ppo policy param and learn the CDR param
        """
        self.cdr_model.train()
        all_reward = 0
        # data_iter = iter(self.data_loader)
        # for i in range(len(self.data_loader) - 2):
        for X, y in self.data_loader:
        # for X, y in tqdm.tqdm(self.data_loader, smoothing=0, mininterval=1.0):  
        #X[128,162],y[128,1]
            s_v=[]
            ap=[]
            a_log=[]
            
            global_state,state = self.env.reset(X, y) #torch.Size([64, 11]),# state torch.Size([64, 4, 11])
            for i,j in zip(state,global_state):#i的形状 torch.Size([4, 11])
                action_prob, action_log_prob,_ = self.agent.choose_action(i)#actionprob torch.Size([4, 1]),action_log_prob是([4])
                action_prob=action_prob.reshape(1,4)#torch.Size([1, 4])
                action_log_prob=action_log_prob.reshape(1,4)
                state_value=self.agent.get_value(j, i)#(4,1)
                state_value=state_value.reshape(1,4)
                s_v.append(state_value)
                ap.append(action_prob)
                a_log.append(action_log_prob)
            action1=torch.cat([x for x in ap], dim=0)#(64,4)     
            reward, loss, _ = self.env.step(X, y, action1)  # reward torch.Size([128])
            all_reward = all_reward+reward.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            reward=reward.unsqueeze(1).repeat(1,4)#torch.Size([64, 4])   
            action_prob1=torch.cat([x for x in a_log], dim=0) #torch.Size([64, 4])
            state_value1=torch.cat([x for x in s_v], dim=0) #torch.Size([64, 4])
            # next_state = self.env.reset(batch_next[0], batch_next[1])
            self.replay_buffer.append(Trajectory(global_state.clone().detach(),state.clone().detach(), state_value1.clone().detach(),action1.clone().detach(),
                                                 action_prob1.clone().detach(), reward.clone().detach()))
                                                 


        print("all Episode: {}, all_reward: {}".format(len(self.replay_buffer), all_reward))

    def as_from_replay_buffer(self):
        """
        in the AS process, we learn from the Replay Buffer
        """
        self.replay_buffer.pop()
        for transition in tqdm.tqdm(self.replay_buffer):
            self.agent.actor.train()
            self.agent.critic.train()
            torch.autograd.set_detect_anomaly(True)
            self.agent.train(transition)   
        del  self.replay_buffer[:]

class Infer():
    def __init__(self, env, model, agent, data_loader):
        self.env = env
        self.cdr_model = model
        self.agent = agent
        self.data_loader = data_loader

    def infer_weight(self):
        self.cdr_model.eval()
        self.agent.actor.train()
        self.agent.critic.train()
        weight = []
        ids = []
        with torch.no_grad():
            for X, y in self.data_loader:
                state = self.env.reset(X, y)    # state [128, 58]
                userids, itemids = X[:, 0].cpu().tolist(), X[:, 1].cpu().tolist()
                action,_,_ = self.agent.select_action(state)     # [128, 3]
                weight.extend(action.cpu().tolist())
                ids.extend(list(zip(userids, itemids)))
        weight = [list(ids[i]) + w for i,w in enumerate(weight)]
        return weight

    def infer_emb(self):
        self.cdr_model.eval()
        self.agent.net.eval()
        # self.agent.cnet.eval()
        uiu_df = []
        uiciu_df = []
        uibiu_df = []
        with torch.no_grad():
            for X, y in self.data_loader:
            # for X, y in tqdm.tqdm(self.data_loader, smoothing=0, mininterval=1.0):
                state = self.env.reset(X, y)    # state [128, 58]
                action,_,_ = self.agent.select_action(state)     # [128, 3]
                uiu_emb, uiciu_emb, uibiu_emb = self.env.get_emb(X, y, action)   # [128]
                uiu_df.extend(uiu_emb.cpu().tolist())
                uiciu_df.extend(uiciu_emb.cpu().tolist())
                uibiu_df.extend(uibiu_emb.cpu().tolist())
        return uiu_df, uiciu_df, uibiu_df

    def eval_mae(self):
        print('Evaluating MAE:')
        self.cdr_model.eval()
        self.agent.actor.train()
        self.agent.critic.train()
        targets, predicts,weight = list(), list(),list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            for X, y in self.data_loader:
            # for X, y in tqdm.tqdm(self.data_loader, smoothing=0, mininterval=1.0):
                _,state = self.env.reset(X, y)    # state [128, 58]
                action,_,_,_ = self.agent.choose_train_action(state)     # [128, 3]
                _, _, pred = self.env.eval_step(X, y, action)   # [128]
                targets.extend(y.squeeze(1).tolist())
                predicts.extend(pred.tolist())
                weight.append(action)
        all_weight = torch.cat(weight,dim=0)
        all_weight = all_weight.squeeze()
        all_weight_numpy = all_weight.cpu().numpy()
        all_weight_df = pd.DataFrame(all_weight_numpy)
        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts)
        abs_differences = torch.abs(targets_tensor - predicts_tensor).cpu().numpy()
        all_weight_df['performance'] = abs_differences
        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item(),all_weight_df
