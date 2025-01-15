import numpy as np
import gymnasium as gym
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt 
from joblib import dump, load
from env_hiv import HIVPatient
from evaluate import evaluate_HIV  
from gymnasium.wrappers import TimeLimit

class RandomForestFQI():
    def __init__(self, env, gamma = .7, horizon = 1000):
        self.env = env
        self.states = self.env.observation_space.shape[0]
        self.n_action = self.env.action_space.n 
        self.nb_actions = int(self.env.action_space.n)
        self.gamma = gamma
        self.linear_model = None
        self.horizon = horizon

    def collect_samples(self, disable_tqdm = False, print_done_states = False):
        s, _ = self.env.reset()
        dataset = []
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in tqdm(range(self.horizon), disable = disable_tqdm):
            a = self.env.action_space.sample()
            s2, r, done, trunc, _ = self.env.step(a)
            dataset.append((s, a, r, s2, done, trunc))
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = self.env.reset()
                if done and print_done_states:
                    print("done!")
            else:
                s = s2
        self.S = np.array(S)
        self.A = np.array(A).reshape((-1, 1))
        self.R = np.array(R)
        self.S2 = np.array(S2)
        self.D = np.array(D)

    
    def train(self, iterations = 10, disable_tqdm = False):
        self.collect_samples()
        nb_samples = self.S.shape[0]
        Qfunctions = []
        SA = np.append(self.S,
                            self.A,
                            axis = 1)
        for iter in tqdm(range(iterations), disable = disable_tqdm):
            if iter==0:
                value = self.R.copy()
            else:
                Q2 = np.zeros((nb_samples, self.nb_actions))
                for a2 in range(self.nb_actions):
                    A2 = a2 * np.ones((nb_samples, 1))
                    S2A2 = np.append(self.S2, A2,axis=1)
                    Q2[:,a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = self.R + self.gamma * (1 - self.D) * max_Q2
            Q = RandomForestRegressor()
            Q.fit(SA, value)
            Qfunctions.append(Q)
        self.rf_model = Qfunctions[-1]

    def greedy_action(self, state):
        Qsa = []
        for a in range(self.nb_actions):
            sa = np.append(state, a).reshape(1, -1)
            Qsa.append(self.rf_model.predict(sa))
        return np.argmax(Qsa)
    
    def evaluate(self):
        score = evaluate_HIV(agent = self, nb_episode = 1)
        print("score:", score)
    
    def act(self, observation, use_random = False):
        if use_random:
            return self.env.action_space.sample()
        else:
            self.greedy_action(observation)

    def save(self, path):
        dump(self.Qfunctions, path)
        data_to_save = {
            'Qfunctions': self.Qfunctions[-1]}
        dump(data_to_save, path, compress = 9)

    def load(self, path):
        self.Qfunctions = load(path)
        loaded_data = load(path)
        self.Qfunctions = loaded_data['Qfunctions']


env = TimeLimit(
                env = HIVPatient(domain_randomization=False),
                max_episode_steps = 200
                )  

### Train the agent ###
agent = RandomForestFQI(env)
agent.train()
agent.evaluate()
path = ""
agent.save(path) 
