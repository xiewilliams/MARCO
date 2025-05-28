import torch


class CDREnvironment():

    def __init__(self, reward_name, action_space_n, model, emb_dim, stage,device):
        self.reward_name = reward_name
        self.global_state=None
        self.state = None
        self.action_space_n = action_space_n
        self.model = model
        self.state_dim = 1+2*emb_dim
        self.critic_inputs=2+4*emb_dim
        self.stage = stage
        self.device=device
        self.criterion=torch.nn.MSELoss(reduction='none')

    def reset(self, X, y):
        iid_emb, uid_emb, uiu_emb, uiciu_emb, uibiu_emb,att_emb = self.model(X, stage=self.stage)
        tgt_pred = torch.sum(uid_emb * iid_emb, dim=1).unsqueeze(1)

        uiu_pred = torch.sum(uiu_emb * iid_emb, dim=1).unsqueeze(1)

        uiciu_pred = torch.sum(uiciu_emb * iid_emb, dim=1).unsqueeze(1)

        uibiu_pred = torch.sum(uibiu_emb * iid_emb, dim=1).unsqueeze(1)
        att_pred = torch.sum(att_emb * iid_emb, dim=1).unsqueeze(1)


        a1_state=torch.cat([iid_emb,uid_emb,tgt_pred],dim=1).unsqueeze(1)
        a2_state=torch.cat([iid_emb,uiu_emb,uiu_pred],dim=1).unsqueeze(1)
        a3_state=torch.cat([iid_emb,uiciu_emb,uiciu_pred],dim=1).unsqueeze(1)
        a4_state=torch.cat([iid_emb,uibiu_emb,uibiu_pred],dim=1).unsqueeze(1)
        self.global_state=torch.cat([iid_emb,att_emb,att_pred],dim=1)
        self.state = torch.cat([a1_state,a2_state,a3_state,a4_state], 1)
        return self.global_state, self.state

    def get_emb(self, X, y, action):
        _, _, uiu_emb, uiciu_emb, uibiu_emb = self.model(X, stage=self.stage)
        return uiu_emb, uiciu_emb, uibiu_emb

    def step(self, X, y, action):
        reward, loss, output = self.reward(X, y, action)
        return reward, loss, output
    def eval_step(self, X, y, action):
        reward, loss, output = self.eva_reward(X, y, action)

        return reward, loss, output

    def reward(self, X, y, action):
        iid_emb, uid_emb, uiu_emb, uiciu_emb, uibiu_emb,att_emb= self.model(X, stage=self.stage)
        src_map_seq_emb = torch.cat([uid_emb.unsqueeze(1),uiu_emb.unsqueeze(1), uiciu_emb.unsqueeze(1), uibiu_emb.unsqueeze(1)], 1)
        agg_uid_emb = torch.bmm(action.unsqueeze(1), src_map_seq_emb)
        emb = torch.cat([agg_uid_emb, iid_emb.unsqueeze(1)], 1)
        output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
        criterion1 = torch.nn.MSELoss(reduction='none')
        criterion2 = torch.nn.MSELoss(reduction='mean')
        loss1 = criterion1(output, y.squeeze().float())
        loss = criterion2(output, y.squeeze().float())
        if self.reward_name == 'r1':
            return -1.0 * loss1, loss, output
        elif self.reward_name == 'r2':

            return 0, loss, output
        else:
            raise ValueError("Unknown reward name")

    def eva_reward(self, X, y, action):
        iid_emb, uid_emb, uiu_emb, uiciu_emb, uibiu_emb,att_emb= self.model(X, stage=self.stage)
        src_map_seq_emb = torch.cat([uid_emb.unsqueeze(1),uiu_emb.unsqueeze(1), uiciu_emb.unsqueeze(1), uibiu_emb.unsqueeze(1)],1)
        action=action.squeeze()
        agg_uid_emb = torch.bmm(action.unsqueeze(1), src_map_seq_emb)
        emb = torch.cat([agg_uid_emb, iid_emb.unsqueeze(1)], 1)
        output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
        criterion1 = torch.nn.MSELoss(reduction='none')
        criterion2 = torch.nn.MSELoss(reduction='mean')
        loss1 = criterion1(output, y.squeeze().float())
        loss = criterion2(output, y.squeeze().float())
        if self.reward_name == 'r1':
            return -1.0 * loss1, loss, output
        elif self.reward_name == 'r2':

            return 0, loss, output
        else:
            raise ValueError("Unknown reward name")

    def criterion(self, x, y):
        return torch.abs(x - y)
