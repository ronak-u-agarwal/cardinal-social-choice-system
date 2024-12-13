#%% Imports and variables

import torch

#%% Variables
number_of_pairwise_elections = 3
number_of_voters = 100
n = number_of_pairwise_elections
v = number_of_voters


#%%

class Voter:
    def __init__(self, preferences, interest, voter_id=-1, size=1):
        self.preferences = preferences  # the way this voter will vote in all n elections
        self.interest = interest  # true priority vector
        self.id = voter_id
        self.size = size

    def calculate_happiness(self, outcome):
        win_loss_dist = self.preferences * outcome  # prefs is n-d vec of -1's and 1's, so is outcome. Wins are 1, loss are -1.
        return torch.dot(win_loss_dist.float(), self.interest.float())

    def cast_vote(self, dist_dict):
        dist = dist_dict[self.id]
        cast = dist * self.preferences
        return cast * self.size   ## Unlike interest, dist stays on the standard simplex, so we need to give the voter a boost

    def cast_true_vote(self):
        cast = self.preferences * self.interest
        return cast


class VoterGroup:
    def __init__(self, voter_list):
        self.voter_set = set(voter_list)
        self.ids = set([x.id for x in self.voter_set])
        self.group_id = min(self.ids)

        sizes = [x.size for x in self.voter_set]
        self.group_size = sum(sizes)
        # Calculate winner if just this voting coalition voted honestly
        cast_vote_list = []
        for single_voter in self.voter_set:
            cast_vote_list.append(single_voter.cast_true_vote())
        self.true_tally = torch.sum(torch.stack(cast_vote_list), dim=0)
        self.group_preferences = torch.where(self.true_tally <= 0, -1, 1)

        # Calculate interest this group has
        # It should stay unscaled, and then it can be used to calculate group happiness; nice
        self.group_interest = torch.abs(self.true_tally)

    def calculate_tally(self, dist_dict):  # If just this group voted, what would the election results look like?
        cast_vote_list = []
        # For this part, take ID of all the voters, use that to get index for all the vote-prefs that matter
        for single_voter in self.voter_set:
            cast_vote_list.append(single_voter.cast_vote(dist_dict))
        tally = torch.sum(torch.stack(cast_vote_list), dim=0)
        return tally

    def calculate_winner(self, dist_dict):
        tally = self.calculate_tally(dist_dict)
        winners = torch.where(tally <= 0, -1, 1)  # with this scheme, -1 wins in the case of a tie
        return winners

    def calculate_total_happiness(self, outcome):
        win_loss_dist = self.group_preferences * outcome
        return torch.dot(win_loss_dist.float(), self.group_interest.float())

    # def calculate_true_tally(self):
    #     cast_vote_list = []
    #     for single_voter in self.voter_set:
    #         cast_vote_list.append(single_voter.cast_true_vote())
    #     tally = torch.sum(torch.stack(cast_vote_list), dim=0)
    #     return tally
    #
    # def calculate_true_winner(self):
    #     tally = self.calculate_true_tally()
    #     winners = torch.where(tally <= 0, -1, 1)
    #     return winners
