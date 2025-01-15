from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import os
from numba import jit
from joblib import dump, load
#from RF_FQI_agent import RandomForestFQI
from models.DQN_agent_copy import ReplayBuffer, DQN_AGENT
from DQNNetwork import DQNNetwork, DQN, MLP
from ppo import ActorCritic, TorchWrapper
from fast_env import FastHIVPatient
import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import time

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import xgboost as xgb
import joblib
# pip install -r requirements.txt


env = TimeLimit(
    env=FastHIVPatient(domain_randomization=True), max_episode_steps=200
) 

""" 
env = TimeLimit(
                env = HIVPatient(domain_randomization = False),
                max_episode_steps = 200
                )  
"""

# ENJOY!
class ProjectAgent:
    """
    FQI agent using XGBoost for Q-value approximation.
    """
    def __init__(self):
        self.gamma = 0.99          
        self.n_actions = 4           
        self.model = None
        self.replay_buffer = []
        self.state_dim = 6
        self.epsilon = 0.2
        self.epsilon_decay = 0.991
        self.epsilon_min = 0.01
        self.n_iterations = 5
        self.all_actions_one_hot = np.eye(self.n_actions)
        self.X = np.array([])
        self.rewards_all = np.array([])
        self.dones_all = np.array([])
        self.states_next_all = np.array([])


    def act(self, observation, use_random=False):
        if (self.model is None) or (use_random and (random.random() < self.epsilon)):
            return np.random.randint(0, self.n_actions)
        else:
            duplicated_observation = np.tile(observation, (self.n_actions, 1))
            x_inputs = np.hstack((duplicated_observation, self.all_actions_one_hot))
            q_values = self.model.predict(x_inputs)
            return int(np.argmax(q_values))


    def save(self, path):
        if self.model is not None:
            joblib.dump(self.model, path)
        else:
            print("No model found to save.")


    def load(self):
        try:
            self.model = joblib.load("src/models/fqi_model.xgb")
        except:
            print("No saved model found at 'fqi_model.xgb'.")
            self.model = None


    def add_transition(self, s, a, r, s_next, done):
        self.replay_buffer.append((s, a, r, s_next, done))


    def do_fqi_training(self):
        if len(self.replay_buffer) == 0:
            return  
        states, actions, rewards, states_next, dones = zip(*self.replay_buffer)
        self.replay_buffer = []
        states = np.array(states)
        actions = np.array(actions)
        actions_one_hot = np.array([self.all_actions_one_hot[a] for a in actions])
        rewards = np.array(rewards)
        states_next = np.array(states_next)
        dones = np.array(dones)
        if self.X.size == 0:
            self.X = np.hstack((states, actions_one_hot))
            self.rewards_all = rewards
            self.dones_all = dones
            self.states_next_all = states_next
        else:
            self.X = np.vstack([self.X, np.hstack((states, actions_one_hot))])
            self.rewards_all = np.hstack([self.rewards_all, rewards])
            self.dones_all = np.hstack([self.dones_all, dones])
            self.states_next_all = np.vstack([self.states_next_all, states_next])

        if self.model is None:
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=0
            )
            self.model.fit(self.X, rewards, verbose=False)
        for _ in range(self.n_iterations):
            repeated_states = np.repeat(self.states_next_all, self.n_actions, axis=0)
            repeated_actions = np.tile(self.all_actions_one_hot, (self.states_next_all.shape[0], 1))
            SA = np.hstack((repeated_states, repeated_actions))
            q_values = self.model.predict(SA)
            q_values_reshaped = q_values.reshape(-1, 4)  
            best_q_values = np.max(q_values_reshaped, axis=1)
            targets = self.rewards_all + self.gamma * best_q_values * (1 - self.dones_all)
            self.model.fit(self.X, targets, verbose=False)


def train_fqi_agent(num_episodes, max_steps):
    agent = ProjectAgent()
    all_rewards = []

    for ep in range(num_episodes):
        s, _ = env.reset()
        ep_reward = 0

        for _ in range(max_steps):
            a = agent.act(s, use_random=True)
            s_next, r, done, truncated, _ = env.step(a)
            ep_reward += r
            agent.add_transition(s, a, r, s_next, done)
            s = s_next
            if done or truncated:
                break

        all_rewards.append(ep_reward)
        agent.do_fqi_training()

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        print(f"Episode {ep+1}/{num_episodes} | Reward: {ep_reward} | Epsilon: {agent.epsilon}")

    agent.save("src/models/fqi_model.xgb")

    # Plot training curve
    plt.plot(all_rewards)
    plt.xlabel('Number of episode')
    plt.ylabel('Return')
    plt.title('FQI Training Performance')
    plt.show()


if __name__ == "__main__":
    train_fqi_agent(num_episodes = 500, max_steps = 200)

""" 
less performant ... 
class ProjectAgent:
    def __init__(self, env, name = 'DQN'):
        self.env = env  
        self.name = name 
        self.original_env = self.env.env
        self.nb_actions = int(self.original_env.action_space.n)
        self.nb_neurons = 256
        self.state_dim = self.env.observation_space.shape[0]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def act(self, observation, use_random = False, noise = True):
        if use_random:
            return self.env.action_space.sample()
        elif self.name == 'RF_FQI':
            agent = self.load()
            return agent.greedy_action(observation)
        elif self.name == 'DQN':
            device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
            with torch.no_grad():
                Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device)) 
                if noise:
                    noise_std = 1
                    Q = Q + torch.tensor(np.random.normal(0, noise_std, size = 4), device = device)
                probs = F.softmax(Q, dim = 1)
                probs = Categorical(probs)
                action = probs.sample()
            return action
        elif self.name == 'PPO':
            with torch.no_grad():
                action, _ = self.model.get_action(torch.Tensor(observation).unsqueeze(0))
            return action
        else:
            raise ValueError("Unknown model")

    def save(self):
        if self.name == 'RF_FQI':
            filename = 'src/models/RF_FQI/Qfct'
            model = {'Qfunction': self.agent.rf_model}
            dump(model, filename, compress=9)
        elif self.name == 'DQN':
            filename = "src/models/DQN/config3.pt"
            torch.save(self.model.state_dict(), filename)
        elif self.name == 'PPO':
            filename = "src/models/DQN/config3.pt"
            torch.save(self.model.state_dict(), filename)


    def load(self):
        if self.name == 'RF_FQI':
            loaded_data = load("src/models/RF_FQI/Qfct")
            self.Qfunctions = loaded_data['Qfunctions']
        elif self.name == 'DQN':
            device = torch.device('cpu')
            state_dim, nb_neurons, n_action = self.state_dim, self.nb_neurons, self.nb_actions
            self.model = DQNNetwork(state_dim, nb_neurons, n_action).to(device)
            #self.model = MLP(self.state_dim, 512, self.nb_actions, depth = 5, activation = torch.nn.SiLU(), normalization = 'None').to(device)
            state_dict = torch.load("src/models/DQN/config3.pt", 
                                    weights_only = True, 
                                    map_location = device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            try : 
                x,_ = self.env.reset()
                self.act(x)
                print("Model loaded \n Actor compatible with environnement")
            except : 
                raise Exception("Actor incompatible with environnement")
        
        elif self.name == 'PPO':
            device = torch.device('cpu')
            state_dim, nb_neurons, n_action = self.state_dim, self.nb_neurons, self.nb_actions
            self.model = ActorCritic(TorchWrapper(env))
            state_dict = torch.load("src/models/ppo_model.pth", 
                                    weights_only = True, 
                                    map_location = device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            x,_ = self.env.reset()
            self.act(x)
            try : 
                x,_ = self.env.reset()
                self.act(x)
                print("Model loaded \n Actor compatible with environnement")
            except : 
                raise Exception("Actor incompatible with environnement")
"""