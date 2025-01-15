import numpy as np
import gymnasium as gym
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, nn, optim
from copy import deepcopy
import random 
from DQNNetwork import DQNNetwork, DQN, MLP
from torch.distributions import Categorical, Distribution
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV   
from typing import Any, Dict, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class TorchWrapper(gym.Wrapper):
    """
    Torch wrapper. Actions and observations are Tensors instead of arrays.
    """

    def step(self, action: Tensor) -> Tuple[Tensor, float, bool, Dict[str, Any]]:
        action = action.cpu().numpy()
        observation, reward, done, info, _ = self.env.step(action)
        return torch.tensor(observation), reward, done, info

    def reset(self) -> Tensor:
        observation, _ = self.env.reset()
        return torch.tensor(observation)

def layer_init(layer, std = np.sqrt(2), bias_const = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    A neural network implementing the Actor-Critic architecture for reinforcement learning.
    Designed for environments with discrete action spaces.
    """

    def __init__(self, env: gym.Env):
        """
        Initializes the actor and critic networks based on the environment's observation and action spaces.

        Parameters:
        - env (gym.Env): The environment providing observation and action space information.
        """
        super().__init__()
        
        # Actor network: maps observations to logits over action space
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),  # Input layer
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),  # Hidden layer
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),  # Output layer
        )
        
        # Critic network: maps observations to a scalar value (state value)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),  # Input layer
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),  # Hidden layer
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),  # Output layer
        )

    def get_value(self, observation: Tensor) -> Tensor:
        """
        Computes the value of the given observation using the critic network.

        Parameters:
        - observation (Tensor): The input observation(s) (shape: [batch_size, observation_dim]).

        Returns:
        - Tensor: A scalar value (or a batch of values) representing the state value(s) (shape: [batch_size]).
        """
        observation = torch.tensor(observation, dtype=torch.float32)
        return self.critic(observation).squeeze(-1)

    def get_action_distribution(self, observation: Tensor) -> Categorical:
        """
        Produces a categorical probability distribution over the action space.

        Parameters:
        - observation (Tensor): The input observation(s) (shape: [batch_size, observation_dim]).

        Returns:
        - Categorical: A distribution object based on the logits produced by the actor network.
        """
        logits = self.actor(observation)
        return Categorical(logits=logits)

    def get_action(self, observation: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Samples an action from the actor's policy and retrieves the log-probability of the sampled action.

        Parameters:
        - observation (Tensor): The input observation(s) (shape: [batch_size, observation_dim]).

        Returns:
        - action (Tensor): The sampled action(s) (shape: [batch_size]).
        - log_prob (Tensor): The log-probability of the sampled action(s) (shape: [batch_size]).
        """
        distribution = self.get_action_distribution(observation)
        action = distribution.sample()
        return action, distribution.log_prob(action)


import torch
import numpy as np
from torch import nn, optim
from tqdm import tqdm

class PPOAgent:
    def __init__(self, env, actor_critic, learning_rate=2.5e-4, gamma=0.8, gae_lambda=0.95, clip_coef=0.2,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, num_steps=128, minibatch_size=32, update_epochs=4):
        self.env = env
        self.agent = actor_critic(env)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.learning_rate = learning_rate
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        self.num_steps = num_steps
        self.minibatch_size = minibatch_size
        self.update_epochs = update_epochs
        
        self.observations = torch.zeros((num_steps + 1, *env.observation_space.shape))
        self.values = torch.zeros((num_steps + 1))
        self.actions = torch.zeros((num_steps + 1, *env.action_space.shape), dtype=torch.long)
        self.log_probs = torch.zeros((num_steps + 1))
        self.rewards = torch.zeros((num_steps + 1))
        self.dones = torch.zeros((num_steps + 1))
        
        self.global_step = 0

    def gradient_steps(self, num_updates):
        for update in tqdm(range(num_updates)):
            new_lr = (1.0 - update / num_updates) * self.learning_rate
            self.optimizer.param_groups[0]["lr"] = new_lr
            self.train()
            
    def train(self):
        step = 0
        observation = self.env.reset()
        self.observations[step] = observation
        
        with torch.no_grad():
            self.values[step] = self.agent.get_value(observation)
        
        while step < self.num_steps:
            # Compute action
            with torch.no_grad():
                action, log_prob = self.agent.get_action(self.observations[step])
            
            self.actions[step] = action
            self.log_probs[step] = log_prob
            
            observation, reward, done, info = self.env.step(action)
            if done:
                observation = self.env.reset()
                print(f"global_step={self.global_step}, episodic_return={info['episode']['r']:.2f}")
            
            step += 1
            self.global_step += 1
            
            self.observations[step] = observation
            with torch.no_grad():
                self.values[step] = self.agent.get_value(observation)
            self.rewards[step] = reward
            self.dones[step] = done
        
        advantages = torch.zeros_like(self.rewards)
        last_gae_lamda = 0
        for t in reversed(range(self.num_steps)):
            advantages[t] = (
                self.rewards[t + 1] + self.gamma * (1.0 - self.dones[t + 1]) * 
                (self.values[t + 1] + self.gae_lambda * last_gae_lamda) - self.values[t]
            )
            last_gae_lamda = advantages[t]
        returns = advantages + self.values
        
        for epoch in range(self.update_epochs):
            b_inds = np.random.permutation(self.num_steps)
            for start in range(0, self.num_steps, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                self._optimize(mb_inds, returns, advantages)
    
    def _optimize(self, mb_inds, returns, advantages):
        b_observations = self.observations[mb_inds]
        b_values = self.values[mb_inds]
        b_actions = self.actions[mb_inds]
        b_log_probs = self.log_probs[mb_inds]
        b_returns = returns[mb_inds]
        b_advantages = advantages[mb_inds]
        
        action_distribution = self.agent.get_action_distribution(b_observations)
        b_advantages = (b_advantages - torch.mean(b_advantages)) / (torch.std(b_advantages) + 1e-8)
        
        new_log_probs = action_distribution.log_prob(b_actions)
        ratio = torch.exp(new_log_probs - b_log_probs)
        pg_loss1 = -b_advantages * ratio
        pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2))
        
        entropy_loss = torch.mean(action_distribution.entropy())
        new_values = self.agent.get_value(b_observations)
        v_loss_unclipped = (new_values - b_returns) ** 2
        v_clipped = b_values + torch.clamp(new_values - b_values, -self.clip_coef, self.clip_coef)
        v_loss_clipped = (v_clipped - b_returns) ** 2
        v_loss = 0.5 * torch.mean(torch.max(v_loss_unclipped, v_loss_clipped))
        
        loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def act(self, observation):
        with torch.no_grad():
            action, _ = self.agent.get_action(observation)
        return action

    def save(self, filepath):
        torch.save(self.agent.state_dict(), filepath)

    def load(self, filepath):
        self.agent.load_state_dict(torch.load(filepath))


""" 
env = TimeLimit(
        env=HIVPatient(domain_randomization=False),  
        max_episode_steps=200,
    )
env = TorchWrapper(env)
agent = PPOAgent(env, ActorCritic)

# Entraîner l'agent
agent.gradient_steps(num_updates = 1000)
agent.save("src/models/ppo_model.pth")
"""