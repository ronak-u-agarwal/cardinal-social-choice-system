#%% Imports
import os

os.chdir('/interest-CSCS')
import torch
from votingClasses import Voter, VoterGroup, n, v
import samplingProfileSpace as sps



#%%
def take_steps(voter_dist, alpha, steps):
    # choose 2 indices randomly, shift them IF POSSIBLE, do a steps number of times
    new = voter_dist.clone()
    for _ in range(steps):
        i, j = torch.randperm(len(new))[:2]
        if new[i] > alpha:
            new[i] -= alpha
            new[j] += alpha
        else:
            epsilon = new[i].item()
            new[i] = 0.0
            new[j] += epsilon
    return new


def generate_variations(dist_tensor, voter_ids, directions=10, steps=5, alpha=0.02):
    output_tensor = []
    for _ in range(directions):
        layer_tensor = dist_tensor.clone()
        for voter_id in voter_ids:
            voter_dist = layer_tensor[voter_id]
            modified_dist = take_steps(voter_dist, alpha, steps)
            layer_tensor[voter_id] = modified_dist
        output_tensor.append(layer_tensor)
    output_tensor = torch.stack(output_tensor)
    return output_tensor


# def bad_directed_step(whole_group, group_to_optimize, dist_tensor, directions=10, steps=5, alpha=0.02):
#     group_to_optimize_ids = group_to_optimize.voter_ids
#     to_test = generate_variations(dist_tensor, group_to_optimize_ids, directions, steps, alpha)
#     avg_happiness = []
#     for i in range(len(to_test)):
#         temp_dist_tensor = to_test[i]
#         outcome = whole_group.calculate_winner(temp_dist_tensor)
#         sub_group_happiness = group_to_optimize.avg_happiness(outcome)
#         avg_happiness.append(sub_group_happiness)
#     argmax = torch.argmax(torch.tensor(avg_happiness))
#     better_dist_tensor = to_test[argmax]
#     return better_dist_tensor

def directed_step(whole_group, group_to_optimize, dist_tensor):
    ## Want to get above inputs, and output a dist_tensor that represents some sort of collective action by a subgroup
    ## Thus, all the slightly different
    ## Assuming all same preferences: everyone has same distribution
    ## Assuming mostly same preferences: hmm, try to shift interest onto things agreed upon
    ## ok so calculate winner of subgroup; that maximizes their happiness, no reason to toggle any of those
    ## make a big voter with m votes instead of 1,
    ## uniform simplex, or random walk simplex

    representative_preference = group_to_optimize.calculate_true_winner()
    representative_interest = group_to_optimize.calculate_interest()
    representative = Voter(representative_preference)
