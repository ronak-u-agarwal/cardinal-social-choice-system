#%%

import os
os.chdir('/Users/ronakagarwal/Coding projects/pythonProjects/modellingSocialChoiceSystems/cardinal-social-choice-system')
import torch
from votingClasses import Voter, VoterGroup, n, v
import samplingProfileSpace as sps
#from monteCarloOptimization import directed_step


#%% Unbiased voters_list
voters_list = []
preferences_tensor = sps.unbiased_preference(n, v)
interests_tensor = sps.unbiased_interest(n, v)
in_dist_tens = interests_tensor.clone()
in_dist_dict = {voter_id: in_dist_tens[voter_id].clone() for voter_id in range(in_dist_tens.size(0))}

for i in range(v):
    pref = preferences_tensor[i]
    inter = interests_tensor[i]
    voter = Voter(pref, inter, voter_id=i)
    voters_list.append(voter)

all_voters = VoterGroup(voters_list)
initial_winner = all_voters.calculate_winner(in_dist_dict)

#%% Example
# #new_tensor = directed_step(all_voters, all_voters, initial_distribution_tensor, steps=5, alpha=0.02)
# print("old winner", initial_winner)
# print("average happiness with old", all_voters.avg_happiness(initial_winner))
#
# # new_winner = all_voters.calculate_winner(new_tensor)
# print("new winner", new_winner)
# print("average happiness with new", all_voters.avg_happiness(new_winner))

#%% Proof that maximizing avg happiness for whole group is achieved?
# take whole_group, go through and compare winners being [. . . -1 . . .] to [ . . . 1 . . .], take avg happiness, compare, take winner
ideal_win = []
for i in range(n):
    sample_winners = torch.ones(n)
    happiness_with_pos = all_voters.avg_happiness(sample_winners)
    sample_winners[i] = -1
    happiness_with_neg = all_voters.avg_happiness(sample_winners)
    if happiness_with_pos > happiness_with_neg:
        ideal_win.append(1)
    else:
        ideal_win.append(-1)


#%% More example
test = voters_list[21]
print("Preferences, interest, id", test.preferences, test.interest, test.voter_id)
print("Distribution", in_dist_dict[test.voter_id])
print("Vote cast", test.cast_vote(in_dist_dict))
print("____")
print("sum of all prefs", torch.sum(preferences_tensor, dim=0))
print("actual winners", initial_winner)
print("test's happiness", test.calculate_happiness(initial_winner))
print("average happiness", all_voters.avg_happiness(initial_winner))

#%%
