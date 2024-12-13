#%%

import os
os.chdir('/Users/ronakagarwal/Coding projects/pythonProjects/modellingSocialChoiceSystems/cardinal-social-choice-system')
import torch
from votingClasses import Voter, VoterGroup, n, v
import samplingProfileSpace as sps
from monteCarloOptimization import create_rep_voter

#%% Unbiased voters_list cool new sps way
whole_unbiased_pop, initial_dist_dict = sps.create_pop()




#%% Assigning positive and negative party
pos_position = torch.tensor([1,1,1])
neg_position = torch.tensor([-1,-1,-1])
pos_party_voter_list = []
neg_party_voter_list = []

for x in whole_unbiased_pop.voter_set:
    pos_party_happiness = x.calculate_happiness(pos_position)
    if pos_party_happiness > 0:
        pos_party_voter_list.append(x)
    else:
        neg_party_voter_list.append(x)

neg_coalition = VoterGroup(neg_party_voter_list)
pos_coalition = VoterGroup(pos_party_voter_list)

pos_party_squished = create_rep_voter(whole_unbiased_pop, pos_coalition)
print(pos_party_squished.group_size, len(list(pos_party_squished.voter_set)))
neg_party_squished = create_rep_voter(whole_unbiased_pop, neg_coalition)
print(neg_party_squished.group_size, len(list(neg_party_squished.voter_set)))

pos_neg_squished = create_rep_voter(pos_party_squished, neg_coalition)
print(pos_neg_squished.group_size, len(list(pos_neg_squished.voter_set)))


#%%

squish_dict = sps.recreate_dist_dict(pos_neg_squished)
print(squish_dict)
print("squish_dict results", pos_neg_squished.calculate_tally(squish_dict))
print("original results", whole_unbiased_pop.calculate_tally(initial_dist_dict))
# The reason these are different is because the RV thm + new dict doesn't just re-represent voters;
# instead, it tallies total interest, then rescales the vote distribution to be proportional to interest, then
# casts the votes of EVERYONE in the group with the set of preferences and distribution. In other words,
party_list = list(pos_neg_squished.voter_set)
print("one group cast vote", party_list[0].cast_vote(squish_dict))
print("one group interest", party_list[0].interest)
print("second group cast vote", party_list[1].cast_vote(squish_dict))
print("second group interest", party_list[1].interest)
# we can see that the cast vote is just a scaled version of interest

#%% Making the positive party try to be strategic
## Now we shall finally try stragetic voting with pos_party in the pos_party_squished pop
pps = pos_party_squished
distribution_dict = sps.recreate_dist_dict(pps)
print("og pop true tally", whole_unbiased_pop.true_tally)
print("pps true tally", pps.true_tally)

print("pps pos party reduces waste", pps.calculate_tally(distribution_dict))


## Ah i see how i can maybe get a group to collaborate:
## if disagree with group preferences in an election, mask cast vote to 0. Then, monte
## carlo random walk on the distribution that the group follows, apply 0 mask, and renormalize

## So now, I need to make sure people would want to be in party if being in party makes them do that

## ahh so opposing party means that party forms
## another way of says it is this: if you know already that your preference will win, strategic voting = casting 0 vote
## so if the positive party forms, they gain efficiency, super positive results. Then, the negative party naturally
## forms, as negative party memebers who have a positive preference in a certain election will just not put their vote there
## So the more powerful the party, the more individuals in the party are incentivized to vote honestly


# make general function to sort people into parties ? or maybe just make parties, combine to have population vote
#%%
# #%% Testing things
# initial_winner = whole_unbiased_pop.calculate_winner(initial_dist_dict)
# print("total happiness", whole_unbiased_pop.calculate_total_happiness(initial_winner))
#
# example_sub_group = whole_unbiased_pop
# new_group = create_rep_voter(whole_unbiased_pop, example_sub_group)
# new_dict = sps.recreate_dist_dict(new_group)
#
# new_winner = new_group.calculate_winner(new_dict)
# print("winner of new subgroup", new_winner)
# print("total happiness", new_group.calculate_total_happiness(new_winner))
#




#%% Unbiased voters_list og way
# voters_list = []
# preferences_tensor = sps.unbiased_preference(n, v)
# interests_tensor = sps.unbiased_interest(n, v)
# in_dist_tens = interests_tensor.clone()
# in_dist_dict = {voter_id: in_dist_tens[voter_id].clone() for voter_id in range(in_dist_tens.size(0))}
#
# for i in range(v):
#     pref = preferences_tensor[i]
#     inter = interests_tensor[i]
#     voter = Voter(pref, inter, voter_id=i)
#     voters_list.append(voter)
#
# all_voters = VoterGroup(voters_list)
# initial_winner = all_voters.calculate_winner(in_dist_dict)


#%% Example
# #new_tensor = directed_step(all_voters, all_voters, initial_distribution_tensor, steps=5, alpha=0.02)
# print("old winner", initial_winner)
# print("average happiness with old", all_voters.avg_happiness(initial_winner))
#
# # new_winner = all_voters.calculate_winner(new_tensor)
# print("new winner", new_winner)
# print("average happiness with new", all_voters.avg_happiness(new_winner))

# #%% Proof that maximizing avg happiness for whole group is achieved?
# # take whole_group, go through and compare winners being [. . . -1 . . .] to [ . . . 1 . . .], take avg happiness, compare, take winner
# ideal_win = []
# for i in range(n):
#     sample_winners = torch.ones(n)
#     happiness_with_pos = all_voters.avg_happiness(sample_winners)
#     sample_winners[i] = -1
#     happiness_with_neg = all_voters.avg_happiness(sample_winners)
#     if happiness_with_pos > happiness_with_neg:
#         ideal_win.append(1)
#     else:
#         ideal_win.append(-1)
#

# #%% More example
# test = voters_list[21]
# print("Preferences, interest, id", test.preferences, test.interest, test.id)
# print("Distribution", in_dist_dict[test.id])
# print("Vote cast", test.cast_vote(in_dist_dict))
# print("____")
# print("sum of all prefs", torch.sum(preferences_tensor, dim=0))
# print("actual winners", initial_winner)
# print("test's happiness", test.calculate_happiness(initial_winner))
# print("average happiness", all_voters.avg_happiness(initial_winner))

#%%
