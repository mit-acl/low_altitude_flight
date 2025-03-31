import torch
import torch.nn as nn

'''
Relaxed Winner-Takes-All (R-WTA) loss used in the paper:
Loquercio, Antonio, et al. "Learning high-speed flight in the wild." Science Robotics 6.59 (2021): eabg5810.
'''
class RelaxedWTA(nn.Module):
    def __init__(self, epsilon=0.05):
        """
        Initialize the Relaxed Winner-Takes-All (R-WTA) loss.
        :param epsilon: Small value to relax the hard assignment.
        """
        super(RelaxedWTA, self).__init__()
        self.epsilon = epsilon

    def forward(self, expert_trajectories, predicted_trajectories, mode=3):
        """
        Compute the R-WTA loss between expert and predicted trajectories.
        :param expert_trajectories: Tensor of shape (N, E, T, D) where
                                    N = batch size,
                                    E = number of expert trajectories,
                                    T = number of timesteps,
                                    D = trajectory dimension.
        :param predicted_trajectories: Tensor of shape (N, P, T, D) where
                                       P = number of predicted trajectories.
        :return: Scalar loss value.
        """
        N, E, T, D = expert_trajectories.shape
        _, P, _, _ = predicted_trajectories.shape
        # print("Expert student shape: ", expert_trajectories.shape, predicted_trajectories.shape)
        # Compute pairwise L2 distances between expert and predicted trajectories
        distances = torch.cdist(
            expert_trajectories.reshape(N, E, -1),  # Reshape to (N, E, T*D)
            predicted_trajectories.reshape(N, P, -1),  # Reshape to (N, P, T*D)
            p=2
        )  # Shape: (N, E, P)

        # Find the closest expert trajectory for each predicted trajectory
        min_distances, min_indices = torch.min(distances, dim=1)  # Shape: (N, P)

        # Compute α weights based on closest assignment
        alpha = torch.full_like(distances, self.epsilon / (mode - 1))  # Shape: (N, E, P)
        alpha.scatter_(1, min_indices.unsqueeze(1), 1 - self.epsilon)  # Assign 1-ε to closest

        # Compute the weighted sum of squared distances
        loss = torch.sum(alpha * distances ** 2) / N

        return loss
