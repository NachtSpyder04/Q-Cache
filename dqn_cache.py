
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple

class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class CacheEnv:

    def __init__(self, cache_size: int = 16, num_items: int = 100):
        self.cache_size = cache_size
        self.num_items = num_items
        self.cache = []  
        self.access_counts = np.zeros(num_items)
        self.time = 0

    def reset(self):
        self.cache = []
        self.access_counts = np.zeros(self.num_items)
        self.time = 0

    def step(self, request: int) -> Tuple[np.ndarray, float, bool]:
      
        self.time += 1
        self.access_counts[request] += 1

        # Cache check
        if request in self.cache:
            self.cache.remove(request)
            self.cache.append(request)
            reward = 1.0
            hit = True
        else:
            if len(self.cache) < self.cache_size:         
                self.cache.append(request)
            reward = -1.0
            hit = False

        return self.get_state(request), reward, hit

    def evict(self, evict_idx: int, request: int):
        if evict_idx < len(self.cache):
            self.cache.pop(evict_idx)
        self.cache.append(request)

    def get_state(self, request: int) -> np.ndarray:
        # inspired from DRLCache repo
        # todo : improve state representation

        state = np.zeros(self.num_items * 3 + self.num_items)

        for item in self.cache:
            state[item] = 1


        max_freq = max(self.access_counts.max(), 1)
        state[self.num_items:self.num_items*2] = self.access_counts / max_freq

        for pos, item in enumerate(self.cache):
            state[self.num_items*2 + item] = (pos + 1) / self.cache_size

        state[self.num_items*3 + request] = 1

        return state
    

class DQNAgent:

    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)

        # todo: tune it better
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999 #todo: decay faster?
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.target_update = 1000


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

        print(f"Agent initialized on device: {self.device}")

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_eviction(self, env: CacheEnv, state: np.ndarray) -> int:
  
        if len(env.cache) == 0:
            return 0

 
        if np.random.rand() <= self.epsilon:
            return random.randrange(len(env.cache))

        # q-values 
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]

        
        # todo: optimize this, q-values for all items is not needed
        cache_q_values = [(i, q_values[item]) for i, item in enumerate(env.cache)]

        
        evict_idx = min(cache_q_values, key=lambda x: x[1])[0]
        return evict_idx

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor([t[1] for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


class LRUPolicy:

    def choose_eviction(self, cache):
        return 0  
    
class LFUPolicy:

    def __init__(self):
        self.access_counts = {}

    def choose_eviction(self, cache, access_counts):
        lfu_idx = 0
        min_count = float('inf')
        for i, item in enumerate(cache):

            if access_counts[item] < min_count:
                min_count = access_counts[item]
                lfu_idx = i

        return lfu_idx  
    
class RandomPolicy:

    def choose_eviction(self, cache):
        return random.randint(0, len(cache) - 1) if cache else 0