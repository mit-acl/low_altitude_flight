import torch
import torch.nn as nn
import open3d as o3d
import numpy as np

'''
Collision cost used in the paper:
Loquercio, Antonio, et al. "Learning high-speed flight in the wild." Science Robotics 6.59 (2021): eabg5810.
'''
class CollisionCost(nn.Module):
    def __init__(self, terrain_map, rq):
        super(CollisionCost, self).__init__()
        self.rq = rq
        self.threshold = 4.0
        pcd_copy = o3d.geometry.PointCloud(terrain_map.pcd_tsdf)
        pcd_copy = pcd_copy.voxel_down_sample(voxel_size=5.0)
        self.terrain_pcd = np.asarray(pcd_copy.points)
        self.pcd_tree = o3d.geometry.KDTreeFlann(pcd_copy)
        self.num_nb = 100

    def forward(self, predicted_trajectories_global):
        """
        :param predicted_trajectories: Tensor of shape (N, P, T, D) where
                                       P = number of predicted trajectories.
        :return: Scalar loss value.
        """
        batch, num_modes, traj_length, _ = predicted_trajectories_global.shape
        student_trajectory = predicted_trajectories_global.detach().cpu().numpy()
        cost_gt = torch.zeros(batch, num_modes)
        for i in range(batch):
            for j in range(num_modes):
                cost = 0.0
                for k in range(traj_length):
                    loc = np.copy(student_trajectory[i, j, k, :])
                    [k, idx, _] = self.pcd_tree.search_radius_vector_3d(loc, 2*self.rq)
                    neighbors = self.terrain_pcd[idx, :]

                    if len(neighbors) > 0:
                        dist_sum = 0
                        for nb in neighbors:
                            dist = np.linalg.norm(nb - loc)
                            dist_sum += dist
                        d = dist_sum / len(neighbors)
                        cost += - d**2 / self.rq**2 + self.threshold

                cost_gt[i, j] = cost

        return cost_gt
