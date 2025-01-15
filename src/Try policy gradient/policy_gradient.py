import numpy as np
import gymnasium as gym
from tqdm import tqdm
import torch
import torch.nn as nn
from copy import deepcopy
import random 
import math
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV 

# Usefull ressources 
# https://towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d
# https://dac.lip6.fr/wp-content/uploads/2019/10/coursRLD4.pdf
# https://huggingface.co/learn/deep-rl-course/unit4/introduction

# Usefull functions 
def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity 
        self.data = []
        self.index = 0 
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

## Policy Gradient
class DiscretePolicy(nn.Module):
    def __init__(self, hidden_size = (128, 128), activation='tanh'):
        super().__init__()
        super().__init__()
        self.env = TimeLimit(
                            env = HIVPatient(domain_randomization = True),
                            max_episode_steps = 200
                            )  
        self.state_dim = self.env.observation_space.shape[0]
        self.n_action = self.env.action_space.n 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nb_actions = int(self.env.action_space.n)
        self.is_disc_action = True
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = self.state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_head = nn.Linear(last_dim, self.nb_actions)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_prob = torch.softmax(self.action_head(x), dim=1)
        return action_prob

    def select_action(self, x):
        action_prob = self.forward(x)
        action = action_prob.multinomial(1)
        return action

    def get_kl(self, x):
        action_prob1 = self.forward(x)
        action_prob0 = action_prob1.detach()
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_prob = self.forward(x)
        return torch.log(action_prob.gather(1, actions.long().unsqueeze(1)))

    def get_fim(self, x):
        action_prob = self.forward(x)
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob, {}
    
class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value