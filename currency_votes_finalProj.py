#%% Libraries

import torch

#%% Base variables and parameters
number_of_pairwise_elections = 3
number_of_voters = 100
n = number_of_pairwise_elections
v = number_of_voters


#%% Classes

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


#   def (calculate loss for group),
#   def (adjust vote distribution for group for monte carlo),
#       (talk to chat about this, want to stay uniform simplex, but also only adjust each person a little)


#%% Functions to use on the distribution tensor
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

#%% Visualizing different profile spaces


#%% Monte carlo optimization of any group of voters

def directed_step(whole_group, group_to_optimize, dist_tensor, directions=10, steps=5, alpha=0.02):
    group_to_optimize_ids = group_to_optimize.voter_ids
    to_test = generate_variations(dist_tensor, group_to_optimize_ids, directions, steps, alpha)
    avg_happiness = []
    for i in range(len(to_test)):
        temp_dist_tensor = to_test[i]
        outcome = whole_group.calculate_winner(temp_dist_tensor)
        sub_group_happiness = group_to_optimize.avg_happiness(outcome)
        avg_happiness.append(sub_group_happiness)
    argmax = torch.argmax(torch.tensor(avg_happiness))
    better_dist_tensor = to_test[argmax]
    return better_dist_tensor


#%% Creating voters_list
voters_list = []
preferences_tensor = unbiased_preference(n, v)
interests_tensor = unbiased_interest(n, v)
initial_distribution_tensor = interests_tensor.clone()

for i in range(v):
    pref = preferences_tensor[i]
    inter = interests_tensor[i]
    voter = Voter(pref, inter, voter_id=i)
    voters_list.append(voter)

all_voters = VoterGroup(voters_list)
initial_winner = all_voters.calculate_winner(initial_distribution_tensor)

#%% Example
new_tensor = directed_step(all_voters, all_voters, initial_distribution_tensor, steps=5, alpha=0.02)
print("old winner", initial_winner)
print("average happiness with old", all_voters.avg_happiness(initial_winner))

new_winner = all_voters.calculate_winner(new_tensor)
print("new winner", new_winner)
print("average happiness with new", all_voters.avg_happiness(new_winner))

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
print("Distribution", initial_distribution_tensor[test.voter_id])
print("Vote cast", test.cast_vote(initial_distribution_tensor))
print("____")
print("sum of all prefs", torch.sum(preferences_tensor, dim=0))
print("actual winners", initial_winner)
print("test's happiness", test.calculate_happiness(initial_winner))
print("average happiness", all_voters.avg_happiness(initial_winner))

#%%
