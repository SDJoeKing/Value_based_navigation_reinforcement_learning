import numpy as np
import random
from collections import deque

from model import Model

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, states = 37, actions = 4, gamma = 0.995, lr = 1e-4, update_every = 5, tau = 0.01):
        self.target_qnet= Model(states, actions).to(device)
        self.local_qnet = Model(states, actions).to(device)
        
        self.replay = EReplay(int(1e4))
        self.optimizer = optim.Adam(self.local_qnet.parameters(), lr = lr)
        self.action_size = actions
        self.t_step = 0
        self.gamma = gamma
        self.update_every = update_every
        self.tau = tau
      
        
    def act(self, state, e = 0.):
        self.local_qnet.eval()
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            actions = self.local_qnet(state).detach().cpu().numpy()
        
        if np.random.rand() > e:
            ## greedy action selection
            act = np.argmax(actions)
        else:
            ## random selection
            act = np.random.choice(range(self.action_size))
        self.local_qnet.train()
        return act
        
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        self.replay.add(states, actions, rewards, next_states, dones)
        
                
        # If enough samples are available in memory, get random subset and learn
        if self.replay.ready():
            experiences = self.replay.sample()
            self.learn(experiences)
            
            # to update target network
            if (self.t_step +1)%self.update_every == 0:
                self.soft_update()
               
            self.t_step += 1
    
    
    def learn(self, experiences):
        
        states, actions, rewards, next_states, dones = experiences
        
        ## implement double DQN
        target_action_max = self.local_qnet(next_states).argmax(1).detach().unsqueeze(1)
#         print(target_action_max.size())
        target = rewards + self.gamma * (1-dones) * self.target_qnet(next_states).gather(1, target_action_max)
#         print(self.local_qnet(states).size(), actions.unsqueeze(1).size())
        predict= self.local_qnet(states).gather(1, actions )
        try:
            assert(predict.size() == target.size())
        except:
            print(predict.size(), target.size(), states.size(), actions.size(), rewards.size(), next_states.size(), dones.size())
        loss = F.mse_loss(predict, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        ## Clip the gradient to -1,1
        for param in self.local_qnet.parameters():
            param.grad.data.clamp_(-1,1)
            
        self.optimizer.step()
 
    
    def soft_update(self):
        for target_param, local_param in zip(self.target_qnet.parameters(), self.local_qnet.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1-self.tau) * target_param.data)
    



## Normal experience replay, to be upgraded to be priority experience replay
class EReplay:
    def __init__(self, size, action_size = 4, batch_size = 64):
        self.memory = deque(maxlen = size)
        self.batch_size = batch_size
        self.action_size = action_size 
    
    
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self):
        experience = random.sample(self.memory, k = self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experience)
        
        states = torch.from_numpy(np.array(states)).float().to(device) 
        try:
            actions = torch.from_numpy(np.array(actions)).long().to(device).unsqueeze(1)
            rewards = torch.from_numpy(np.array(rewards)).float().to(device).unsqueeze(1)
            next_states = torch.from_numpy(np.array(next_states)).float().to(device) 
            dones = torch.from_numpy(np.array(dones)).float().to(device).unsqueeze(1)
        except:
            print(actions)        
        return (states, actions, rewards, next_states, dones)
        
    def ready(self):
        return len(self.memory) >= self.batch_size
    
    
    def __len__(self):
        return len(self.memory)


        
        