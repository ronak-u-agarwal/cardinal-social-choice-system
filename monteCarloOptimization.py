#%% Imports
import os

import torch
from votingClasses import Voter, VoterGroup, n, v


#%%
def take_steps(point_on_simplex, alpha, steps):
    # choose 2 indices randomly, shift them IF POSSIBLE, do a steps number of times
    new = point_on_simplex.clone()
    for _ in range(steps):
        i, j = torch.randperm(len(new))[:2]
        epsilon = torch.rand(1)*alpha
        if new[i] > epsilon:
            new[i] -= epsilon
            new[j] += epsilon
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

def create_rep_voter(whole_group, group_to_represent):
    ## Use rep voter thm:
    ## create a big voter, new voting group with whole_group - group_to_optimize + rep voter
    ## initialize rep voter with preferences, interest, etc. as calculated in obsidian
    ## return a new group with normal voters, and then a big guy for group to optimize; also new dist_tensor
    ## IMPORTANT: ONLY WORKS IF NO VOTERS HAVE -1 HAPPINESS WITH RV PREFS
    ## IMPORTANT: ONLY INITIAL SETTING OF INTEREST FOR SURE WORKS AS VOTE DIST
    rep_preference = group_to_represent.group_preferences #
    rep_interest = group_to_represent.group_interest # remember group interest doesn't add up to 1 or to size, just use for happiness
    rep_size = group_to_represent.group_size  # so if the group to optimize has
    rep_id = group_to_represent.group_id
    rep = Voter(preferences=rep_preference, interest=rep_interest, voter_id=rep_id, size=rep_size)

    # Getting all the voters in whole group and not in group_to_represent
    new_voter_set = whole_group.voter_set - group_to_represent.voter_set
    new_voter_set.add(rep)
    new_voter_list = list(new_voter_set)
    new_voter_group = VoterGroup(voter_list=new_voter_list)

    return new_voter_group





