'''
Modified from:
https://github.com/leggedrobotics/iPlanner/blob/master/iplanner/esdf_mapping.py
'''
import open3d as o3d
import numpy as np
import torch
import os

import cv2
import math
from scipy import ndimage
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial.transform import Rotation as R

class ElevationMap:
    def __init__(self, device='cpu'):
        self.map_init = False

        self.device = device
        self.pcd_tsdf = o3d.geometry.PointCloud()
        self.pcd_viz  = o3d.geometry.PointCloud()

    # def DirectLoadMap(self, tsdf_array, viz_points, start_xy, voxel_size, clear_dist):
    def DirectLoadMap(self, data, coord, params):
        self.voxel_size = params[0]
        self.clear_dist = params[1]
        self.start_x, self.start_y = coord
        self.tsdf_array = torch.tensor(data[0], device=self.device)
        self.num_x, self.num_y = self.tsdf_array.shape
        # visualization points
        self.viz_points = data[1]
        self.pcd_viz.points = o3d.utility.Vector3dVector(self.viz_points)
        self.ground_array = torch.tensor(data[2], device=self.device)
        # set cost map
        self.SetUpCostArray()
        # update pcd instance
        xv, yv = np.meshgrid(np.linspace(0, self.num_x * self.voxel_size, self.num_x), np.linspace(0, self.num_y * self.voxel_size, self.num_y), indexing="ij")
        T = np.concatenate((np.expand_dims(xv, axis=0), np.expand_dims(yv, axis=0)), axis=0)
        T = np.concatenate((T, np.expand_dims(self.cost_array.cpu(), axis=0)), axis=0)
        self.pcd_tsdf.points = o3d.utility.Vector3dVector(T.reshape(3, -1).T)
        
        self.map_init = True;

    def ShowTSDFMap(self, cost_map=True): # not run with cuda
        if not self.map_init:
            print("Error: cannot show map, map has not been init yet!")
            return;
        if cost_map:
            o3d.visualization.draw_geometries([self.pcd_tsdf])
        else:
            o3d.visualization.draw_geometries([self.pcd_viz])
        return

    def Pos2Ind(self, points):
        '''
        Compute indices w.r.t. start x and start y (lower left corner).
        '''
        # points: batch x number of points x 3
        start_xy = torch.tensor([self.start_x, self.start_y], dtype=torch.float64, device=points.device).expand(1, 1, -1)
        H = (points[:, :, 0:2] - start_xy) / self.voxel_size
        mask = torch.logical_and((H > 0).all(axis=2), (H < torch.tensor([self.num_x, self.num_y], device=points.device)[None,None,:]).all(axis=2))
        return self.NormInds(H), H[mask, :]

    def NormInds(self, H):
        '''
        Center and normalize the indices
        '''
        norm_matrix = torch.tensor([self.num_x/2.0, self.num_y/2.0], dtype=torch.float64, device=H.device)
        H = (H - norm_matrix) / norm_matrix
        return H

    def DeNormInds(self, NH):
        norm_matrix = torch.tensor([self.num_x/2.0, self.num_y/2.0], dtype=torch.float64, device=NH.device)
        NH = NH * norm_matrix + norm_matrix
        return NH

    def SaveTSDFMap(self, root_path, map_name):
        if not self.map_init:
            print("Error: map has not been init yet!")
            return;
        map_path    = os.path.join(*[root_path, "maps", "data",   map_name   + "_map.txt"])
        ground_path = os.path.join(*[root_path, "maps", "data",   map_name   + "_ground.txt"])
        params_path = os.path.join(*[root_path, "maps", "params", map_name + "_param.txt"])
        cloud_path  = os.path.join(*[root_path, "maps", "cloud",  map_name   + "_cloud.txt"])
        # save datas
        np.savetxt(map_path, self.tsdf_array.cpu())
        np.savetxt(ground_path, self.ground_array.cpu())
        np.savetxt(cloud_path, self.viz_points)
        params = [str(self.voxel_size), str(self.start_x), str(self.start_y), str(self.clear_dist)]
        with open(params_path, 'w') as f:
            for param in params:
                f.write(param)
                f.write('\n')
        print("TSDF Map saved.")
    
    def SetUpCostArray(self):
        self.cost_array = self.tsdf_array

    def ReadTSDFMap(self, root_path, map_name):
        map_path    = os.path.join(*[root_path, "maps", "data",   map_name   + "_map.txt"])
        ground_path = os.path.join(*[root_path, "maps", "data",   map_name   + "_ground.txt"])
        params_path = os.path.join(*[root_path, "maps", "params", map_name   + "_param.txt"])
        cloud_path  = os.path.join(*[root_path, "maps", "cloud",  map_name   + "_cloud.txt"])
        # open params file
        with open(params_path) as f:
            content = f.readlines()
        self.voxel_size = float(content[0])
        self.start_x    = float(content[1])
        self.start_y    = float(content[2])
        self.clear_dist = float(content[3])
        self.tsdf_array = torch.tensor(np.loadtxt(map_path), device=self.device)
        self.viz_points = np.loadtxt(cloud_path)
        self.ground_array = torch.tensor(np.loadtxt(ground_path), device=self.device)

        self.num_x, self.num_y = self.tsdf_array.shape
        # visualization points
        self.pcd_viz.points = o3d.utility.Vector3dVector(self.viz_points)
        # opne map array
        self.SetUpCostArray()
        # update pcd instance
        xv, yv = np.meshgrid(np.linspace(0, self.num_x * self.voxel_size, self.num_x), np.linspace(0, self.num_y * self.voxel_size, self.num_y), indexing="ij")
        T = np.concatenate((np.expand_dims(xv, axis=0), np.expand_dims(yv, axis=0)), axis=0)
        T = np.concatenate((T, np.expand_dims(self.cost_array.cpu().detach().numpy(), axis=0)), axis=0)
        wps = T.reshape(3, -1).T + np.array([self.start_x, self.start_y, 0.0])
        self.pcd_tsdf.points = o3d.utility.Vector3dVector(wps)

        self.map_init = True
        return


class ElevationMapCreator:
    def __init__(self, input_path, voxel_size, robot_height, robot_size, clear_dist=1.0):
        self.initialize_path_and_properties(input_path, voxel_size, robot_height, robot_size, clear_dist)
        self.terrain_pcd = o3d.geometry.PointCloud()

    def initialize_path_and_properties(self, input_path, voxel_size, robot_height, robot_size, clear_dist):
        self.input_path = input_path
        self.is_map_ready = False
        self.clear_dist = clear_dist
        self.voxel_size = voxel_size
        self.robot_height = robot_height
        self.robot_size = robot_size

        
    def update_point_cloud(self, terrain_points, is_downsample=False):
        self.terrain_pcd.points  = o3d.utility.Vector3dVector(terrain_points)
        if is_downsample:
            self.terrain_pcd  = self.terrain_pcd.voxel_down_sample(self.voxel_size)
        self.terrain_points   = np.asarray(self.terrain_pcd.points)

    def update_map_params(self):
        if (self.terrain_points.shape[0] == 0):
            print("No points received.")
            return
        self._set_map_limits_and_start_coordinates()
        self.is_map_ready = True
        print("tsdf map initialized, with size: %d, %d" %(self.num_x, self.num_y))
        
    def read_point_from_file(self, file_name, is_filter=False):
        file_path = os.path.join(self.input_path, file_name)
        pcd_load = o3d.io.read_point_cloud(file_path)
        
        self.update_point_cloud(np.asarray(pcd_load.points), is_downsample=False)
        
        if is_filter:
            terrain_p = self.filter_cloud(self.terrain_points, num_nbs=50, std_ratio=2.0)
            self.update_point_cloud(terrain_p)
        
        self.update_map_params()
    
    def create_elevation_map(self):
        if not self.is_map_ready:
            print("create tsdf map fails, no points received.")
            return
        
        pcd_tree = o3d.geometry.KDTreeFlann(self.terrain_pcd)
        elevation_map = np.zeros([self.num_x, self.num_y])

        ### Add boundary
        elevation_map[0:5, :] = 100.0
        elevation_map[:, 0:5] = 100.0
        elevation_map[-5:, :] = 100.0
        elevation_map[:, -5:] = 100.0

        grid_indices = self._index_array_of_points(self.terrain_points)
        for i, idx in enumerate(grid_indices):
            [k, candidate_indices, _] = pcd_tree.search_knn_vector_3d(self.terrain_points[i], 10)
            # [k, candidate_indices, _] = pcd_tree.search_radius_vector_3d(self.terrain_points[i], 10.0)
            candidate_points = np.asarray(self.terrain_pcd.points)[candidate_indices, :]
            if candidate_points.shape[0] > 0:
                max_idx = np.argmax(candidate_points, axis=0)
                _, _, highest_p = candidate_points[max_idx]
                elevation_map[idx[0], idx[1]] = highest_p[2]
            else:
                elevation_map[idx[0], idx[1]] = 1000.0

        elevation_map = maximum_filter(elevation_map, size=(5, 5))


        viz_points = self.terrain_points

        ground_array = elevation_map

        return [elevation_map, viz_points, ground_array], [self.start_x, self.start_y], [self.voxel_size, self.clear_dist]
    
    def filter_cloud(self, points, num_nbs=100, std_ratio=1.0):
        pcd = self._convert_to_point_cloud(points)
        filtered_pcd = self._remove_statistical_outliers(pcd, num_nbs, std_ratio)
        return np.asarray(filtered_pcd.points)
    
    def visualize_cloud(self, pcd):
        o3d.visualization.draw_geometries([pcd])
    
    def _convert_to_point_cloud(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def _remove_statistical_outliers(self, pcd, num_nbs, std_ratio):
        filtered_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=num_nbs, std_ratio=std_ratio)
        return filtered_pcd 
    
    def _index_array_of_points(self, points):
        I = np.round((points[:, :2] - np.array([self.start_x, self.start_y])) / self.voxel_size).astype(int)
        return I

    def _index_to_point(self, index):
        point = index * self.voxel_size + np.array([self.start_x, self.start_y]) - self.voxel_size / 2.0
        return point
    
    def _initialize_point_arrays(self, input_points):
        return np.zeros(input_points.shape), np.zeros(input_points.shape)
        
    def _set_map_limits_and_start_coordinates(self):
        '''
        min_x, max_x, min_y, max_y: minimum and maximum x, y coordinates
        num_x, num_y: number of grids along x, y
        start_x, start_y: the starting location of the grid map.
        '''
        max_x, max_y, _ = np.amax(self.terrain_points, axis=0) + self.clear_dist
        min_x, min_y, _ = np.amin(self.terrain_points, axis=0) - self.clear_dist
        self.num_x = np.ceil((max_x - min_x) / self.voxel_size / 10).astype(int) * 10
        self.num_y = np.ceil((max_y - min_y) / self.voxel_size / 10).astype(int) * 10
        self.start_x = (max_x + min_x) / 2.0 - self.num_x / 2.0 * self.voxel_size
        self.start_y = (max_y + min_y) / 2.0 - self.num_y / 2.0 * self.voxel_size
