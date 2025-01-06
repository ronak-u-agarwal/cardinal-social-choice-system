# def generate_variations(dist_tensor, voter_ids, directions=10, steps=5, alpha=0.02):
#     output_tensor = []
#     for _ in range(directions):
#         layer_tensor = dist_tensor.clone()
#         for voter_id in voter_ids:
#             voter_dist = layer_tensor[voter_id]
#             modified_dist = take_steps(voter_dist, alpha, steps)
#             layer_tensor[voter_id] = modified_dist
#         output_tensor.append(layer_tensor)
#     output_tensor = torch.stack(output_tensor)
#     return output_tensor

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
