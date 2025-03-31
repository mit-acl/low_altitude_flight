import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from numpy import genfromtxt

class TerrainNavDataset(Dataset):
    def __init__(self, trajectory_dir, training_data_dir, num_trajectories, transform=None, preload=False, device='cuda'):
        self.trajectory_dir = trajectory_dir
        self.training_data_dir = training_data_dir
        self.transform = transform
        self.preload = preload
        self.device = device
        self.input_limits = {"heading": 100.0}         # insert keys of values that should be normalized in features
        self.output_limits = {"desired_trajectory": 1.0, "depth_img": 10.0}     
        self.traj_steps = 10
        self.lookahead = 750
        self.num_trajectories = num_trajectories
        self.x_offset = -409.9997863769531
        self.y_offset = -410.0000305175781
        self.attitude_idx_offset = 6

        self.dataframe = self._create_dataframe()
        print('Finished creating dataset.')

    def __len__(self):
        return self.num_data
    
    def _filter_trajectorty(self, states):
        states_filtered = []
        last_state = None
        for i in range(states.shape[0]):
            if last_state is None:
                states_filtered.append(states[i, :3])
                last_state = states[i, :3]
            else:
                cur_state = states[i, :3]
                dist = np.linalg.norm(cur_state - last_state)
                if dist < 1e-3:
                    continue
                states_filtered.append(cur_state)
                last_state = cur_state
        states_filtered = np.asarray(states_filtered)
        return states_filtered
    
    def _sample_spline(self, states_filtered, fixed_distance=10.0):
        x,y,z = states_filtered[:, 0], states_filtered[:, 1], states_filtered[:, 2]
        xd = np.diff(x)
        yd = np.diff(y)
        zd = np.diff(z)
        dist = np.sqrt(xd**2+yd**2+zd**2)
        u = np.cumsum(dist)
        u = np.hstack([[0],u])

        num_points = int(np.rint(u[-1] / fixed_distance))

        t = np.linspace(0,u.max(), num_points)
        xn = np.interp(t, u, x)
        yn = np.interp(t, u, y)
        zn = np.interp(t, u, z)

        sampled_trajectory = np.asarray([np.asarray([xn[i], yn[i], zn[i]]) for i in range(len(xn))])

        return sampled_trajectory

    def _create_dataframe(self):
        data = []
        for data_idx in tqdm(range(self.num_trajectories), desc="Loading demonstrations"):
            # print("Loading: ", os.path.join(self.trajectory_dir, f"{data_idx}_states.csv"))
            trajectory = genfromtxt(os.path.join(self.trajectory_dir, f"{data_idx}_states.csv"), delimiter=',')
            trajectory = trajectory[1:, :-1]
            trajectory[:, 0] += self.x_offset
            trajectory[:, 1] += self.y_offset

            for timestep in range(0, len(trajectory), 10):
                state = trajectory[timestep]
                loc = state[:3]
                att = state[self.attitude_idx_offset:self.attitude_idx_offset+4]

                if timestep + self.lookahead >= len(trajectory):
                    goal = trajectory[-1, :3]
                else:
                    goal = trajectory[timestep+self.lookahead, :3]
                    if np.linalg.norm(goal - loc) < 50.0:
                        goal = trajectory[-1, :3]

                if np.linalg.norm(goal - loc) < 20.0:
                    break

                trajectory_filtered = self._filter_trajectorty(trajectory[timestep:, :3])
                trajectory_sampled = self._sample_spline(trajectory_filtered)
                desired_trajectory = trajectory_sampled[1:self.traj_steps+1] - loc
                assert len(desired_trajectory) <= self.traj_steps, "Trajectory wrong shape"
                if len(desired_trajectory) < self.traj_steps:
                    # print("Trajectory wrong shape.")
                    pad_size = self.traj_steps - len(desired_trajectory)
                    pad = np.tile(desired_trajectory[-1, :3], pad_size).reshape(pad_size, 3)
                    desired_trajectory = np.concatenate([desired_trajectory, pad])
                    assert len(desired_trajectory) == self.traj_steps, "Padding wrong."

                depth_img_path = os.path.join(self.training_data_dir, f"trajectory_{data_idx}/distance_to_image_plane_{timestep}.npy")
                rgb_img_path = os.path.join(self.training_data_dir, f"trajectory_{data_idx}/rgb_{timestep}.npy")
                rgb_tilted_img_path = os.path.join(self.training_data_dir, f"trajectory_{data_idx}/rgb_tilted_{timestep}.npy")
                

                data.append({
                    'demonstration_id': int(data_idx),
                    'timestep': int(timestep),
                    'rgb': rgb_img_path,
                    'rgb_tilted': rgb_tilted_img_path,
                    'depth': depth_img_path,
                    'p_x': float(loc[0]),
                    'p_y': float(loc[1]),
                    'p_z': float(loc[2]),
                    'q_w': float(att[0]),
                    'q_x': float(att[1]),
                    'q_y': float(att[2]),
                    'q_z': float(att[3]),
                    'g_x': float(goal[0]),
                    'g_y': float(goal[1]),
                    'g_z': float(goal[2]),
                    'desired_trajectory': desired_trajectory
                })
            
                        
        df = pd.DataFrame(data)
        
        # Sort by demonstration_id and then by timestep
        df = df.sort_values(['demonstration_id', 'timestep'], ignore_index=True)
        
        # Compute heading vectors and desired trajectories
        df[['hx', 'hy', 'hz']] = 0.0

        for demonstration_id in df['demonstration_id'].unique():
            demo_df = df[df['demonstration_id'] == demonstration_id]
            positions = demo_df[['p_x', 'p_y', 'p_z']].values
            goals = demo_df[['g_x', 'g_y', 'g_z']].values

            headings = self._compute_heading_vectors(positions, goals)

            df.loc[df['demonstration_id'] == demonstration_id, ['hx', 'hy', 'hz']] = headings

        
        self.num_data = len(data)
        print("Number of data: ", self.num_data)
        return df

    def _preprocess_data(self, depth_img_path, rgb_img_path, rgb_tilted_img_path, record):
        depth_img = np.load(depth_img_path)
        ### Take the log so that far away stuff is less important
        depth_img = np.log(depth_img)
        rgb_img = Image.open(rgb_img_path)
        rgb_tilted_img = Image.open(rgb_tilted_img_path)
        
        if self.transform:
            depth_img = self.transform(depth_img)
            rgb_img = self.transform(rgb_img)
            rgb_tilted_img = self.transform(rgb_tilted_img)
        
        position = torch.tensor([record['p_x'], record['p_y'], record['p_z']], dtype=torch.float32).to(self.device)
        heading = torch.tensor([record['hx'], record['hy'], record['hz']], dtype=torch.float32).to(self.device)
        attitude = torch.tensor([record['q_w'], record['q_x'], record['q_y'], record['q_z']], dtype=torch.float32).to(self.device)

        features = {
            'rgb_img': rgb_img.to(self.device),
            'rgb_tilted_img': rgb_tilted_img.to(self.device),
            'heading': heading,
            'attitude': attitude, 
            'position': position
            }

        desired_trajectory = torch.tensor(record['desired_trajectory'], dtype=torch.float32).to(self.device)
        # print("Desired trajectory shape: ", desired_trajectory.shape)
        labels = {
            'depth_img': depth_img.to(self.device),
            'desired_trajectory': desired_trajectory
            }

        return features, labels

    def __getitem__(self, idx):
        record = self.dataframe.iloc[idx]
        depth_img_path = record['depth']
        rgb_img_path = record['rgb']
        rgb_tilted_img_path = record['rgb_tilted']

        features, labels = self._preprocess_data(depth_img_path, rgb_img_path, rgb_tilted_img_path, record)

        return self._normalize_features(features), self._normalize_labels(labels)
    
    
    def _normalize_features(self, features):
        return {
            key: (value / self.input_limits[key] if key in self.input_limits else value)
            for key, value in features.items()
        }

    def _normalize_labels(self, labels):
        return {
            key: (value / self.output_limits[key] if key in self.output_limits else value)
            for key, value in labels.items()
        }

    def _compute_heading_vectors(self, positions, goals):
        headings = np.zeros_like(positions)
        for i in range(len(positions)):
            dir = goals[i] - positions[i]
            headings[i] = dir
        return headings

