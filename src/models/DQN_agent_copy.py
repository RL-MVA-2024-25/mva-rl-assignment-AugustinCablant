import numpy as np
import gymnasium as gym
from tqdm import tqdm
import torch
from copy import deepcopy
import random 
from DQNNetwork import DQNNetwork, DQN, MLP

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV   

# Have a look at : https://arxiv.org/abs/1312.5602 
# We want the environment class  

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
    
class DQN_AGENT():
    def __init__(self, config):
        self.env = TimeLimit(
                            env = HIVPatient(domain_randomization = False),
                            max_episode_steps = 200
                            )  
        self.env_dr = TimeLimit(
                            env = HIVPatient(domain_randomization = True),
                            max_episode_steps = 200
                            )  
        self.state_dim = self.env.observation_space.shape[0]
        self.n_action = self.env.action_space.n 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nb_actions = int(self.env.action_space.n)
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size, self.device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        #self.model = DQNNetwork(self.state_dim, config['nb_neurons'], self.nb_actions).to(self.device) 
        self.model = MLP(self.state_dim, 512, self.nb_actions, depth = 5, activation = torch.nn.SiLU(), normalization = 'None').to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)    #  X = states; A = actions  ; R = rewards; Y = next states; D = dones
            QYmax = self.target_model(Y).max(1)[0].detach()      # predict the Q-value of the next state for every a' (actions) (wit the target network)
            update = torch.addcmul(R, 1-D, QYmax, value = self.gamma)    # update = R + gamma * QYmax (if not done)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))   # predict the Q-value of the current state for the action taken
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, path, max_episode = 1000):
        episode_return = []
        episode = 0
        test_episode = 100
        best_score = 0
        episode_cum_reward = 0
        state, _ = self.env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            if episode % 2==0:
                dr = False
            else:
                dr = True
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = self.env.action_space.sample()
            else:
                action = self.greedy_action(state)
            # step
            next_state, reward, done, trunc, _ = self.env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps):   ### Number of gradient steps after each iteration : try to tune this parameter ...
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                if episode > test_episode:
                    test_score = evaluate_HIV(agent = self, nb_episode = 1)
                else:
                    test_score = 0
                if test_score > best_score:
                    best_score = test_score 
                    self.best_model = deepcopy(self.model).to(self.device)

                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                if dr:
                    state, _ = self.env_dr.reset()
                else:
                    state, _ = self.env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
        self.model.load_state_dict(self.best_model.state_dict())
        self.save(path)
        return episode_return
    
    def greedy_action(self, state):
        with torch.no_grad():
            Q = self.model(torch.Tensor(state).unsqueeze(0).to(self.device))
        return torch.argmax(Q).item()
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        device = torch.device('cpu')
        self.model = DQNNetwork(self.state_dim, self.config['nb_neurons'], self.nb_actions).to(device)
        self.model.load_state_dict(torch.load(path, map_location = device))
        self.model.eval()

    def act(self, observation, use_random = False):
        if use_random:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                Q = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()
    

"""

config1 = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'buffer_size': 1000000,
        'epsilon_min': 0.01,
        'epsilon_max': 1.,
        'epsilon_decay_period': 1000,
        'epsilon_delay_decay': 20,
        'batch_size': 200,
        'gradient_steps': 3,
        'update_target_strategy': 'replace', # or 'ema'
        'update_target_freq': 50,
        'update_target_tau': 0.005,
        'criterion': torch.nn.SmoothL1Loss(),
        'monitoring_nb_trials': 50,
        'nb_neurons': 256
        }


config2 = {
        'learning_rate': 0.001,
        'gamma': .9,
        'buffer_size': 1000000,
        'epsilon_min': 0.01,
        'epsilon_max': 0.01,
        'epsilon_decay_period': 1000,
        'epsilon_delay_decay': 20,
        'batch_size': 1000,
        'gradient_steps': 3,
        'update_target_strategy': 'replace', # or 'ema'
        'update_target_freq': 50,
        'update_target_tau': 0.005,
        'criterion': torch.nn.SmoothL1Loss(),
        'monitoring_nb_trials': 0,
        'nb_neurons': 256
        }

config4 = {
        'learning_rate': 0.001,
        'gamma': .8,
        'buffer_size': 1000000,
        'epsilon_min': 0.001,
        'epsilon_max': 1.,
        'epsilon_decay_period': 40000,
        'epsilon_delay_decay': 20,
        'batch_size': 1000,
        'gradient_steps': 10,
        'update_target_strategy': 'ema', # or 'ema' replace
        'update_target_freq': 50,
        'update_target_tau': 0.005,
        'criterion': torch.nn.SmoothL1Loss(),
        'monitoring_nb_trials': 0,
        'nb_neurons': 256
        }



config3 = {
        'learning_rate': 0.001,
        'gamma': .8,
        'buffer_size': 1000000,
        'epsilon_min': 0.01,
        'epsilon_max': 1.,
        'epsilon_decay_period': 40000,
        'epsilon_delay_decay': 20,
        'batch_size': 1000,
        'gradient_steps': 10,
        'update_target_strategy': 'replace', # or 'ema'
        'update_target_freq': 50,
        'update_target_tau': 0.005,
        'criterion': torch.nn.SmoothL1Loss(),
        'monitoring_nb_trials': 0,
        'nb_neurons': 256
        }



agent = DQN_AGENT(config3)
path = "src/models/DQN/config3_testDQN_copy.pt"
episode_return = agent.train(path)
"""
# To do :
# Look for different optimizers : Adam, RMSprop
# Clip the value of the loss gradient (-1, 1) i.e. Huber Loss -> torch.nn.SmoothL1Loss()
# Number of gradient steps after each iteration 
# Try double Q-learning
