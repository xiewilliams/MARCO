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


class Trainer():
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

        self.agent.actor.train()
        self.agent.critic.train()

        all_reward = 0
        i_episode = 0


        for X, y in self.data_loader:


            s_v=[]
            ap=[]
            a_log=[]
            i_episode = i_episode + 1

            global_state,state = self.env.reset(X, y)
            for i,j in zip(state,global_state):
                action_prob = self.event_softmax(torch.randn(i.shape[0],1)).to(self.device)
                action_log_prob = self.agent.ini_log(action_prob).to(self.device)
                action_prob=action_prob.reshape(1,4).to(self.device)
                action_log_prob=action_log_prob.reshape(1,4).to(self.device)
                state_value=self.agent.get_value(j, i)
                state_value=state_value.reshape(1,4).to(self.device)
                s_v.append(state_value)
                ap.append(action_prob)
                a_log.append(action_log_prob)
            action1=torch.cat([x for x in ap], dim=0)
            reward, _, _ = self.env.step(X, y, action1)
            all_reward = all_reward+reward.mean()
            reward=reward.unsqueeze(1).repeat(1,4)
            action_prob1=torch.cat([x for x in a_log], dim=0)
            state_value1=torch.cat([x for x in s_v], dim=0)

            transition=Trajectory(global_state.clone().detach(),state.clone().detach(), state_value1.clone().detach(),action1.clone().detach(),
                                                 action_prob1.clone().detach(), reward.clone().detach())
            self.agent.train(transition)
            if i_episode >= max_episode:
                break

    def cdr_with_fixed_as(self, optimizer):

        self.cdr_model.train()
        all_reward = 0


        for X, y in self.data_loader:


            s_v=[]
            ap=[]
            a_log=[]

            global_state,state = self.env.reset(X, y)
            for i,j in zip(state,global_state):
                action_prob, action_log_prob,_ = self.agent.choose_action(i)
                action_prob=action_prob.reshape(1,4)
                action_log_prob=action_log_prob.reshape(1,4)
                state_value=self.agent.get_value(j, i)
                state_value=state_value.reshape(1,4)
                s_v.append(state_value)
                ap.append(action_prob)
                a_log.append(action_log_prob)
            action1=torch.cat([x for x in ap], dim=0)
            reward, loss, _ = self.env.step(X, y, action1)
            all_reward = all_reward+reward.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            reward=reward.unsqueeze(1).repeat(1,4)
            action_prob1=torch.cat([x for x in a_log], dim=0)
            state_value1=torch.cat([x for x in s_v], dim=0)

            self.replay_buffer.append(Trajectory(global_state.clone().detach(),state.clone().detach(), state_value1.clone().detach(),action1.clone().detach(),
                                                 action_prob1.clone().detach(), reward.clone().detach()))


        print("all Episode: {}, all_reward: {}".format(len(self.replay_buffer), all_reward))

    def as_from_replay_buffer(self):

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
                state = self.env.reset(X, y)
                userids, itemids = X[:, 0].cpu().tolist(), X[:, 1].cpu().tolist()
                action,_,_ = self.agent.select_action(state)
                weight.extend(action.cpu().tolist())
                ids.extend(list(zip(userids, itemids)))
        weight = [list(ids[i]) + w for i,w in enumerate(weight)]
        return weight

    def infer_emb(self):
        self.cdr_model.eval()
        self.agent.net.eval()

        uiu_df = []
        uiciu_df = []
        uibiu_df = []
        with torch.no_grad():
            for X, y in self.data_loader:

                state = self.env.reset(X, y)
                action,_,_ = self.agent.select_action(state)
                uiu_emb, uiciu_emb, uibiu_emb = self.env.get_emb(X, y, action)
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

                _,state = self.env.reset(X, y)
                action,_,_,_ = self.agent.choose_train_action(state)
                _, _, pred = self.env.eval_step(X, y, action)
                targets.extend(y.squeeze(1).tolist())
                predicts.extend(pred.tolist())
                weight.append(action)
        all_weight = torch.cat(weight,dim=0)
        all_weight = all_weight.squeeze()
        all_weight_numpy = all_weight.cpu().numpy()
        all_weight_df = pd.DataFrame(all_weight_numpy)
        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts)
        abs_differences = torch.abs(targets - predicts).cpu().numpy()
        all_weight_df['performance'] = abs_differences
        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item(),all_weight_df
