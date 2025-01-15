import numpy as np
import gymnasium as gym
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt 
from env_hiv import HIVPatient
from evaluate import evaluate_HIV  
from joblib import dump, load
from sklearn.cluster import KMeans

class RandomForestFQI():
    # Tune gamma
    def __init__(self, env, gamma = .7, horizon = 1000):
        self.states = gym.spaces.Discrete(4)
        self.actions = [np.array(pair) for pair in [[0.0, 0.0], [0.0, 0.3], [0.7, 0.0], [0.7, 0.3]]]
        self.nb_actions = len(self.actions)
        self.env = env
        self.gamma = gamma
        self.rf_model = None
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

    
    def train(self, iterations = 10, disable_tqdm = False, n_cluster = 4):
        self.collect_samples()
        nb_samples = self.S.shape[0]
        Qfunctions = []
        SA = np.append(self.S,
                            self.A,
                            axis = 1)
        Q = KMeans(n_clusters = n_cluster, random_state = 42)
        Q.fit(SA)
        Qfunctions.append(Q)
        self.clustering = Qfunctions[-1]

    def greedy_action(self, state):
        Qsa = []
        for a in range(self.nb_actions):
            sa = np.append(state, a).reshape(1, -1)
            Qsa.append(self.rf_model.predict(sa))
        return np.argmax(Qsa)
    
    def evaluate(self):
        score = evaluate_HIV(agent = self, nb_episode = 1)
        print("score:", score)

    def save(self, path):
        dump(self.Qfunctions, path)
        data_to_save = {
            'Qfunctions': self.Qfunctions[-1]}
        dump(data_to_save, path, compress = 9)

    def load(self, path):
        self.Qfunctions = load(path)
        loaded_data = load(path)
        self.Qfunctions = loaded_data['Qfunctions']