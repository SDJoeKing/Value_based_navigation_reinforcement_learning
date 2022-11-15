import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, states = 37, actions = 4, layer1 = 64, layer2=64):
        super().__init__()
        self.L1 = nn.Linear(states, layer1)
        self.L2 = nn.Linear(layer1, layer2)
        self.svalue = nn.Linear(layer2, 1) 
        self.advan = nn.Linear(layer2, actions)
        self.dropout = nn.Dropout(p = 0.1)
        self.activation = nn.ReLU()
    
    ## implementation of dueling DQN
    def forward(self, states):
        out = self.dropout(self.activation(self.L1(states)))
        out = self.dropout(self.activation(self.L2(out)))
        out_advan = self.advan(out)
        out_state = self.svalue(out)
        
        ## to cancel the advantage at maximum action value
#         print(out_state.size(), out_advan.size(),out_advan.max(1)[0].size() )
        return out_state + (out_advan - out_advan.max(1)[0].unsqueeze(1))
        
        