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


#%% Functions to make voting populations
def create_pop(alt=n, size=v, pref_func=unbiased_preference, interest_func=unbiased_interest):
    # Return a voter group with all voters in it, as well as an initial vote_distribution_dictionary
    voters_list = []
    preferences_tensor = pref_func(alt, size)
    interests_tensor = interest_func(alt, size)
    in_dist_tens = interests_tensor.clone()
    in_dist_dict = {voter_id: in_dist_tens[voter_id].clone() for voter_id in range(in_dist_tens.size(0))}
    for i in range(size):
        pref = preferences_tensor[i]
        inter = interests_tensor[i]
        voter = Voter(pref, inter, voter_id=i)
        voters_list.append(voter)
    all_voters = VoterGroup(voters_list)
    return all_voters, in_dist_dict


def recreate_dist_dict(voter_group):
    # given a coalition of voters, create a dictionary that assigns to each its theoretic/honest voting distribution
    # be careful: can't just do interest, because rep voters have interest vectors adding up to wacky numbers
    new_dist_dict = {}
    for voter in voter_group.voter_set:
        voter_dist = voter.interest / torch.sum(voter.interest)  # normalizing in case of RV's
        new_dist_dict[voter.id] = voter_dist
    return new_dist_dict
