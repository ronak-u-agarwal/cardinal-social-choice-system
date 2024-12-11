
#%% Imports and variables

import os
os.chdir('/interest-CSCS')
import torch
import samplingProfileSpace as sps
from monteCarloOptimization import directed_step




#%% Variables
number_of_pairwise_elections = 3
number_of_voters = 100
n = number_of_pairwise_elections
v = number_of_voters

#%%

class Voter:
    def __init__(self, preferences, interest, voter_id=-1):
        self.preferences = preferences  # the way this voter will vote in all n elections
        self.interest = interest  # true priority vector
        self.voter_id = voter_id

    def calculate_happiness(self, outcome):
        win_loss_dist = self.preferences * outcome  # prefs is n-d vec of -1's and 1's, so is outcome. Wins are 1, loss are -1.
        scaled_win_loss_dist = win_loss_dist * self.interest
        return sum(scaled_win_loss_dist)

    def cast_vote(self, dist_tensor):
        dist = dist_tensor[self.voter_id]
        cast = dist * self.preferences
        return cast


class VoterGroup:
    def __init__(self, voter_list):
        self.voter_list = voter_list
        self.voter_ids = [n.voter_id for n in voter_list]

    def calculate_winner(self, dist_tensor):  # If just this group voted, what would the election results look like?
        cast_vote_list = []
        # For this part, take ID of all the voters, use that to get index for all the vote-prefs that matter
        for single_voter in self.voter_list:
            cast_vote_list.append(single_voter.cast_vote(dist_tensor))
        tally = torch.sum(torch.stack(cast_vote_list), dim=0)
        winners = torch.where(tally <= 0, -1, 1)  # with this scheme, -1 wins in the case of a tie
        return winners

    def avg_happiness(self, outcome):
        total_happiness = 0
        for single_voter in self.voter_list:
            total_happiness += single_voter.calculate_happiness(outcome)
        return total_happiness / (len(self.voter_list))