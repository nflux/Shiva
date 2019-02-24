import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROLLOUT_HIDDEN = 256
class EnvironmentModel(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=int(1024), nonlin=F.relu, norm_in=True, agent=object,maddpg=object):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (intT): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(EnvironmentModel, self).__init__()
        self.agent=agent
        self.action_size = 3
        self.param_size = 5
        self.count = 0
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, 1024)
        
        self.fc1.weight.data.normal_(0, 0.01) 
        self.fc2 = nn.Linear(1024, 512)
        self.fc2.weight.data.normal_(0, 0.01) 
        self.fc3 = nn.Linear(512, 256)
        self.fc3.weight.data.normal_(0, 0.01) 
        self.fc4 = nn.Linear(256, 128)
        self.fc4.weight.data.normal_(0, 0.01) 
 
              
        self.out_obs = nn.Linear(128, out_dim)
        self.out_obs.weight.data.normal_(0, 0.01) 
        self.out_ws = nn.Linear(128, self.agent.world_status_dim)
        self.out_ws.weight.data.normal_(0, 0.01) 
        self.out_rew = nn.Linear(128, 1)
        self.out_rew.weight.data.normal_(0, 0.01)
        self.out_ws_fn = nn.Softmax(dim=1)
        self.out_action_fn = lambda x: x

        self.nonlin = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.out_fn = lambda x: x
    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations, actions
        Outputs:
            out (PyTorch Matrix): Output of network (states)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        h4 = self.nonlin(self.fc4(h3))     
        out_obs = self.out_fn(self.out_obs(h4))
        out_ws = self.out_ws_fn(self.out_ws(h4))
        out_rew = self.out_fn(self.out_rew(h4))
        return out_obs, out_rew, out_ws
class RolloutEncoder(nn.Module):
    def __init__(self, input_shape, hidden_size=ROLLOUT_HIDDEN):
        super(RolloutEncoder, self).__init__()
        self.rnn = nn.LSTM(input_size=input_shape, hidden_size=hidden_size, batch_first=False)
        
    def forward(self, obs_v, reward_v,ws_v):
        """
        Input is in (time, batch, *) order
        """
        
        n_time = obs_v.size()[0]
        n_batch = obs_v.size()[1]
        n_items = n_time * n_batch
        obs_flat_v = obs_v.view(n_items, *obs_v.size()[2:])
        obs_flat_v = obs_flat_v.view(n_time, n_batch, -1)
        rnn_in = torch.cat((obs_flat_v, reward_v,ws_v), dim=2)
        _, (rnn_hid, _) = self.rnn(rnn_in)
        return rnn_hid.view(-1)
class I2A(nn.Module):
    def __init__(self, input_shape, n_actions, net_em, net_policy, rollout_steps):
        super(I2A, self).__init__()
        self.n_actions = n_actions
        self.rollout_steps = rollout_steps
        fc_input = input_shape + ROLLOUT_HIDDEN * n_actions
        self.fc = nn.Sequential(
            nn.Linear(fc_input, 512),
            nn.ReLU()
        )
        self.policy = nn.Linear(512, n_actions)
        # used for rollouts
        self.encoder = RolloutEncoder(EM_OUT_SHAPE)
        # save refs without registering
        object.__setattr__(self, "net_em", net_em)
        object.__setattr__(self, "net_policy", net_policy)

    def forward(self, x):
        fx = x.float()
        enc_rollouts = self.rollouts_batch(fx)
        fc_in = torch.cat((fx, enc_rollouts), dim=1)
        fc_out = self.fc(fc_in)
        return self.policy(fc_out)
    def rollouts_batch(self, batch):
        batch_size = batch.size()[0]
        batch_rest = batch.size()[1:]
        if batch_size == 1:
            obs_batch_v = batch.expand(batch_size * self.n_actions, *batch_rest)
        else:
            obs_batch_v = batch.unsqueeze(1)
            obs_batch_v = obs_batch_v.expand(batch_size, self.n_actions, *batch_rest)
            obs_batch_v = obs_batch_v.contiguous().view(-1, *batch_rest)
        actions = np.tile(np.arange(0, self.n_actions, dtype=np.int64), batch_size)
        step_obs, step_rewards = [], []
        for step_idx in range(self.rollout_steps):
            actions_t = torch.tensor(actions).to(batch.device)
            obs_next_v, reward_v = self.net_em(obs_batch_v, actions_t)
            step_obs.append(obs_next_v.detach())
            step_rewards.append(reward_v.detach())
            # don't need actions for the last step
            if step_idx == self.rollout_steps-1:
                break
            # combine the delta from EM into new observation
            cur_plane_v = obs_batch_v[:, 1:2]
            new_plane_v = cur_plane_v + obs_next_v
            obs_batch_v = torch.cat((cur_plane_v, new_plane_v), dim=1)
            # select actions
            logits_v, _ = self.net_policy(obs_batch_v)
            probs_v = F.softmax(logits_v, dim=1)
            probs = probs_v.data.cpu().numpy()
            actions = self.action_selector(probs)
        step_obs_v = torch.stack(step_obs)
        step_rewards_v = torch.stack(step_rewards)
        flat_enc_v = self.encoder(step_obs_v, step_rewards_v)
        return flat_enc_v.view(batch_size, -1)