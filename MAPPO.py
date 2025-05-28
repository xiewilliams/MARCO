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

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actionscore = nn.Linear(hidden_size, action_size)


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


        gs = transition.gs[:-1, :]
        s = transition.s[:-1, :]
        a = transition.a[:-1, :]
        r = transition.r[:-1, :]
        old_action_log_probs = transition.a_log_p[:-1, :]
        v_n=transition.s_v
        max_episode_len = len(r)

        adv = []
        gae = 0
        with torch.no_grad():
                deltas = r + self.gamma* v_n[1:,:]  - v_n[:-1, :]
                for t in reversed(range(max_episode_len)):
                    gae = deltas[t,:]+self.gamma*self.lamda*gae
                    adv.insert(0,gae)
                adv=torch.stack(adv,dim=0).to(self.device)
        v_target = adv+v_n[:-1,:]
        adv = (adv - adv.mean()) / (adv.std() + 0.00001)

        _, a_logprob_n_now, dist_entropy,en = self.choose_train_action(s)
        values_now = self.get_train_value(gs, s)
        values_now = values_now.squeeze()
        ratios = torch.exp(a_logprob_n_now - old_action_log_probs.detach())
        surr1 = ratios * adv
        surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv
        actor_loss = -torch.min(surr1, surr2) + self.lambda_en1 * dist_entropy
        actor_loss = actor_loss.mean()
        if self.use_value_clip:
            values_old = v_n[:-1, :].detach()
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
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.ac_parameters, 0.5)
        self.ac_optimizer.step()
        if self.use_lr_decay:
            self.lr_decay(total_steps)
    def lr_decay(self,total_steps):
        lr_now = self.lr * (1-total_steps/self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr']=lr_now
