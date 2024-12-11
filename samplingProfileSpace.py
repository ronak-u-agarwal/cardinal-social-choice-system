#%% Standard imports
import torch
from votingClasses import Voter, VoterGroup, n, v


#%% Functions to sample preference space

# Unbiased preference space
def unbiased_preference(alts=n, out_size=v):
    preference_list = []
    for _ in range(out_size):
        one_pref = torch.randint(0, 2, (alts,))
        one_pref = one_pref * 2 - 1
        preference_list.append(one_pref)
    preference_tensor = torch.stack(preference_list)
    return preference_tensor


#%% Functions to sample interest space

# Unbiased interest space
def unbiased_interest(alts=n, out_size=v):  # generate v n-dimensional interest vectors
    interest_list = []
    for _ in range(out_size):
        rand_p = torch.rand(alts - 1)
        sorted_p, _ = torch.sort(rand_p)
        extend_p = torch.cat((torch.tensor([0.0]), sorted_p, torch.tensor([1.0])))
        segments = torch.diff(extend_p)
        interest_list.append(segments)
    interest_tensor = torch.stack(interest_list)
    return interest_tensor

# Softmax interest space

# Common interest space (e.g. election is president vs VP vs secretary, everyone thinks pres is most important)
