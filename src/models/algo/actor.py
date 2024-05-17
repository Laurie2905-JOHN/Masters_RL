import torch
import torch.nn as nn
import torch.optim as optim

# Actor
class Actor(nn.Module):
    def __init__(self, state_dim, selection_action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, selection_action_dim)
        self.fc3 = nn.Linear(128, selection_action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        selection_probs = torch.sigmoid(self.fc2(x))
        quantity = torch.relu(self.fc3(x))
        return selection_probs, quantity