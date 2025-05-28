import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SequentialSampler
from torch.distributions import Beta


class ActorNet(nn.Module):
    def __init__(self, num_inputs, hidden_size, action_size):
        super(ActorNet, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actionscore = nn.Linear(hidden_size, action_size)

        # 初始化网络层
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        ac =self.actionscore(x)
        return ac



# class Actor_Critic_Net(nn.Module):
#     def __init__(self, num_inputs, hidden_size, action_size):
#         super(Actor_Critic_Net, self).__init__()
#         # 定义网络层
#         self.fc1 = nn.Linear(num_inputs, hidden_size)
#         self.b1 = nn.BatchNorm1d(hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.b2 = nn.BatchNorm1d(hidden_size)
#         self.actionscore = nn.Linear(hidden_size, action_size)
#         self.b3 = nn.BatchNorm1d(action_size)
#         self.critic = nn.Linear(hidden_size, 1)
#         self.b4 = nn.BatchNorm1d(1)

#         # 初始化网络层
#         self.init_weights()

#     def init_weights(self):
#         for m in self.modules():
#             if type(m) == nn.Linear:
#                 nn.init.orthogonal_(m.weight.data)
#                 nn.init.zeros_(m.bias)

#     def forward(self, x):
#         x = F.tanh(self.b1(self.fc1(x)))
#         x = F.tanh(self.b2(self.fc2(x)))
#         ac = F.softmax(self.b3(self.actionscore(x)))  # 加1确保alpha > 1
#         value = self.b4(self.critic(x)) # 加1确保beta > 1
#         return ac, value


class CriticNet(nn.Module):

    def __init__(self,hidden_size, num_inputs):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.v_head = nn.Linear(hidden_size, 1)
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)    

    def forward(self, x):
        x = F.tanh(self.fc(x))
        x = F.tanh(self.fc2(x))
        state_values = self.v_head(x)
        return state_values


class MAPPO():

    # clip_param = 0.2
    # max_grad_norm = 0.5
    # ppo_epoch = 10
    # buffer_capacity, batch_size = 10000, 10

    def __init__(self,obs_dim,state_dim,critic_input,hidden_size, action_dim, lr,gamma,lamda,epsilon,lambda_en1,device): 
        self.N=4
        self.obs_dim=obs_dim
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.gamma = gamma
        self.lr = lr
        self.lamda=lamda
        self.epsilon=epsilon
        self.use_lr_decay = False
        self.use_rnn = False
        self.add_agent_id = False
        self.use_agent_specific = True
        self.use_value_clip = True
        self.use_grad_clip = False
        self.set_adam_eps = True
        self.actor_input_dim = obs_dim
        self.critic_input_dim = state_dim
        self.device=device
        self.lambda_en1 = lambda_en1
        
        if self.use_agent_specific:
            print("------use agent specific global state------")
            self.critic_input_dim+=obs_dim
        self.actor = ActorNet(state_dim, hidden_size, 1).to(self.device)
        self.critic = CriticNet(hidden_size, critic_input).to(self.device)
        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.set_adam_eps:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)
        
        

    def choose_action(self, obs_n):
        #state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            actor_inputs=[]
            actor_inputs.append(obs_n)
            actor_inputs = torch.cat([x for x in actor_inputs], dim=-1) 
            prob = self.actor(actor_inputs) 
            action_prob=F.softmax(prob,dim=0)
            dist = Categorical(probs=action_prob)
            a_n = dist.sample()
            a_logprob_n = dist.log_prob(a_n)
            entropy = dist.entropy()
            return action_prob,a_logprob_n,entropy
    
    def ini_log(self,action_prob):
        with torch.no_grad():
            dist = Categorical(probs=action_prob)
            a_n = dist.sample()
            a_logprob_n = dist.log_prob(a_n)
        return a_logprob_n


    def get_value(self, s,obs_n):
        with torch.no_grad():
            critic_inputs = []
            s_r = torch.tensor(s, dtype=torch.float32).repeat(4, 1)
            critic_inputs.append(s_r)
            critic_inputs.append(obs_n)
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)
            v_n = self.critic(critic_inputs)
            
            return v_n
        
    def get_train_value(self, s, obs_n):   
        critic_inputs=[]
        s_r = torch.tensor(s, dtype=torch.float32).unsqueeze(1).repeat(1,4, 1)
        critic_inputs.append(s_r)
        critic_inputs.append(obs_n)   
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)
        v_n = self.critic(critic_inputs)
        
        return v_n
    
    def choose_train_action(self, obs_n):    
        actor_inputs=[]
        actor_inputs.append(obs_n)
        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)
        prob = self.actor(actor_inputs)
        action_prob=F.softmax(prob,dim=1)
        dist = Categorical(probs=action_prob)
        a_n = dist.sample()
        a_logprob_n = dist.log_prob(a_n)
        entropy = dist.entropy()
        entropy_weight = -torch.sum(action_prob * torch.log(action_prob + 1e-9), dim=1)
        return action_prob,a_logprob_n,entropy_weight,entropy
        
        
    def train(self,transition):

        # 构建张量
        gs = transition.gs[:-1, :]#torch.Size([63, 11])
        s = transition.s[:-1, :]#torch.Size([63, 4, 11])
        a = transition.a[:-1, :]#torch.Size([63, 4])
        r = transition.r[:-1, :]#torch.Size([63, 4])
        old_action_log_probs = transition.a_log_p[:-1, :] #torch.Size([63, 4])
        v_n=transition.s_v#torch.Size([64, 4])
        max_episode_len = len(r)#63

        adv = []
        gae = 0
        with torch.no_grad():
                deltas = r + self.gamma* v_n[1:,:]  - v_n[:-1, :]#shape(63,4)
                for t in reversed(range(max_episode_len)):
                    gae = deltas[t,:]+self.gamma*self.lamda*gae#torch.Size([4])
                    adv.insert(0,gae)
                adv=torch.stack(adv,dim=0).to(self.device)#adv是一个list，长度为63
        v_target = adv+v_n[:-1,:]# v_target.shape torch.Size([63, 4])
        adv = (adv - adv.mean()) / (adv.std() + 0.00001)
                
        _, a_logprob_n_now, dist_entropy,en = self.choose_train_action(s)#torch.Size([63, 4])            
        values_now = self.get_train_value(gs, s)#torch.Size([63, 4, 1])            
        values_now = values_now.squeeze()#torch.Size([63, 4])            
        ratios = torch.exp(a_logprob_n_now - old_action_log_probs.detach())  # ratios.shape=#torch.S
        surr1 = ratios * adv#这是未剪切的目标函数，即概率比率乘以优势函数。#torch.Size([63, 4])        
        surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv#torch.Size([63, 4])        
        actor_loss = -torch.min(surr1, surr2) + self.lambda_en1 * dist_entropy        
        actor_loss = actor_loss.mean()        
        if self.use_value_clip:#值裁剪        
            values_old = v_n[:-1, :].detach()#获取旧的价值估计（从训练数据中），并通过调用 detach 确保不计算这些旧价值的梯度。        
            values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old-v_target
            values_error_original = values_now - v_target        
            critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)        
            critic_loss = critic_loss.mean()
        else:
            critic_loss = (values_now - v_target[:-1, :]) ** 2
            critic_loss = critic_loss.mean()
        self.ac_optimizer.zero_grad() 
        ac_loss = actor_loss+critic_loss
        ac_loss.backward()
        if self.use_grad_clip:  # Trick 7: Gradient clip
            torch.nn.utils.clip_grad_norm_(self.ac_parameters, 0.5)
        self.ac_optimizer.step()            
        if self.use_lr_decay:
            self.lr_decay(total_steps)
    def lr_decay(self,total_steps):
        lr_now = self.lr * (1-total_steps/self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr']=lr_now
        
        
        
        
        
        
        
        # for i in range(L-1, -1, -1):
        #     return_value = r[i] + self.gamma * return_value 
        #     G.append(return_value)
        # G = G[::-1]
        # G = torch.tensor(G, dtype=torch.float).view(-1, 1).to(self.device)
        # #torch.Size([128, 1])

        # with torch.no_grad():
        #     _,s_v=self.net(s)
        #     adv = G - s_v # advantage
        #     for i in range(L-2, -1, -1):
        #         adv[i] += adv[i+1]*self.gamma # cumulated advantage
        #     adv = (G - G.mean()) / (G.std() + 0.00001)#torch.Size([128, 1])

        # # PPO 更新
        # # total_size = len(buffer) * 128  # 总状态数
        # # for _ in range(ppo_epoch):
        # #     for index in BatchSampler(
        # #             SequentialSampler(range(128)), batch_size, True):
        #     # 重新计算动作对数概率和熵
        # _, action_log_probs, dist_entropy = self.select_action(s)
        # _,state_value = self.net(s)
        # state_value = state_value.squeeze()

        # KL_Loss = old_action_log_probs.exp()*(old_action_log_probs-action_log_probs)
        # KL_Loss = KL_Loss.sum(dim=1,keepdim=True)
        # # 计算比率和损失
        # ratio_1 = torch.exp(action_log_probs - old_action_log_probs)
        # # ratio = torch.clamp(ratio_1, max=100)
        # ratio = ratio_1
        # surr1 = ratio * adv
        # surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
        # action_loss1 = -torch.min(surr1, surr2) +0.3 * KL_Loss
        # action_loss = action_loss1.mean()
        # value_loss = 0.5 * F.mse_loss(state_value, G.squeeze())
        # # 优化Actor网络
        # Loss = action_loss + value_loss
        # self.optimizer.zero_grad()
        # Loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
        # self.optimizer.step()
        # action_loss.backward(retain_graph=True)