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





#%% Computationally expensive algorithm,  takes a strategy + group as input, outputs vote by
    # Setting everyone's dist, setting all disagreements to 0, then normalizing

# would be faster if these were tensor operations; pretend that we have a tensor, at beginning convert dist_dict into
# a tensor, then at the end convert it back to a dict

def dict_to_tensor(dist_dict, collab_group):
    # output dist_tensor and a list keys, where list[i] is the key for the voting dist at dist_tensor[i]
    # also, want to create a mask of 0's here
    voter_lst = list(collab_group.voter_set)
    dist_dict_copy = dist_dict.copy()

    # tensor and list of keys
    subdist_tensor = torch.tensor([dist_dict_copy[voter.id] for voter in voter_lst])
    key_list = [voter.id for voter in voter_lst] # maybe more efficient to do both inside 1 for-loop?

    # zero-mask:
    # approach: create tensor of collab-group individual prefs, then multiply by group prefs as a whole,
    # add 1, divide by 2
    ind_prefs = torch.tensor([voter.preferences for voter in voter_lst])
    group_pref = collab_group.group_preferences
    zero_mask = ((ind_prefs*group_pref)+1)/2

    return subdist_tensor, key_list, zero_mask

def collab_vote(dist_to_test, zero_mask):
    size = zero_mask.shape[0]

    # create stack of "size" dist_to_tests
    # multiply by the zero_mask
    # normalize rows


# calculate tally from non-group, add to tally of group and set equal to 1/-1 to get outcomes that can look at happiness with

#%% Representative voter to increase computational efficiency; dangerous if samples RV strats outside possibility
def create_rep_voter(whole_group, group_to_represent):
    ## Use rep voter thm:
    ## create a big voter, new voting group with whole_group - group_to_optimize + rep voter
    ## initialize rep voter with preferences, interest, etc. as calculated in obsidian
    ## return a new group with normal voters, and then a big guy for group to optimize; also new dist_tensor?
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










