#%% Libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#%% Class of voters votes

class cumulative_voting_net(nn.Module):
    def __init__(self, num_voters, num_alternatives):
        super().__init__()
        self.num_voters = num_voters
        self.num_alternatives = num_alternatives
        self.raw_votes = nn.ParameterList([
            nn.Parameter(torch.randn(num_alternatives)) for _ in range(num_voters)])

    def forward(self):
        # Apply softmax over the alternatives dimension (columns)
        raw_votes_tensor = torch.stack([param for param in self.raw_votes], dim=0)  # Shape: (num_voters, num_alternatives)
        votes = F.softmax(raw_votes_tensor, dim=1)
        return votes
    def set_voter_grad(self, voter_index, requires_grad=True):
        for i, param in enumerate(self.raw_votes):
            param.requires_grad = requires_grad if i == voter_index else False


#%% Loss
def count_votes(cumulative_voting_net):
    profile = cumulative_voting_net()
    tally = torch.sum(profile, dim=0)
    tally.requires_grad = False
    return tally

def calculate_loss(cumulative_voting_net, true_prefs):
    tally = count_votes(cumulative_voting_net)


#%% Initializing a model
num_voters = 10
num_alternatives = 3
voters = cumulative_voting_net(num_voters=num_voters, num_alternatives=num_alternatives)
true_preferences = torch.randn(num_voters, num_alternatives, requires_grad=False)
count_votes(voters)

#%%
loss = calculate_loss(voters, true_preferences)



#%% Calculating
