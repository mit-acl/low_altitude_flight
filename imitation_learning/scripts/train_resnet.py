import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from low_altitude_nav.datasets.dataset import TerrainNavDataset
from low_altitude_nav.utils.esdf import ElevationMap

from torchvision import transforms
from tqdm import tqdm

from low_altitude_nav.models.resnet_policy import ResNetPolicy
from low_altitude_nav.loss.relaxed_wta import RelaxedWTA
from low_altitude_nav.loss.collision_cost import CollisionCost

import os
import uuid
import wandb
import argparse
import yaml
import pickle

class NetParams:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        name=config["run_name"]
    )
    wandb.run.save()

def reduction_batch_based(image_loss, M):
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor
    
def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


class DepthMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.__reduction = reduction_batch_based

    def forward(self, prediction, target):
        #preprocessing
        mask = target < 1.0

        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class AltitudeMSELoss(nn.Module):
    def __init__(self, tsdf_map: ElevationMap):
        super().__init__()

        self.tsdf_map = tsdf_map
        self.relu = nn.ReLU()
        self.mse_loss = torch.nn.MSELoss()


    def forward(self, plan_world):
        batch_size, _, num_p, point_dim = plan_world.shape
        plan_world = plan_world.view(-1, num_p, point_dim)
        batch_size = plan_world.shape[0]

        norm_inds, _ = self.tsdf_map.Pos2Ind(plan_world)
        height_grid = self.tsdf_map.ground_array.T.expand(batch_size, 1, -1, -1)
        terrain_height = F.grid_sample(height_grid, norm_inds[:, None, :, :], mode='bicubic', padding_mode='border', align_corners=False).squeeze(1).squeeze(1).to(torch.float32)

        oloss = self.relu(terrain_height - plan_world[:, :, 2])
        oloss = torch.mean(oloss) 
        zloss = self.mse_loss(plan_world[:, :, 2] - 5.0, terrain_height)

        return zloss, oloss, terrain_height


def train(config, terrain_map):
    torch.cuda.empty_cache() 

    data_config = config['data']
    train_config = config['train']

    # Define the base directory and transformations for the dataset
    expert_demo_dir = data_config['expert_demo_dir']
    training_data_dir = data_config['training_data_dir']
    img_size = data_config['img_size']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size)
    ])

    device = config['device']
    num_trajectories = data_config['num_traj']
    # Create dataset instance
    dataset = TerrainNavDataset(expert_demo_dir, training_data_dir, num_trajectories=num_trajectories, transform=transform, preload=False, device='cuda')
    features, labels = dataset[0]

    # Split dataset into training and validation sets
    train_size = int(data_config['training_data_ratio'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    batch_size = train_config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    # Instantiate the model, loss function, and optimizer
    model_config = config['model']
    model = ResNetPolicy(NetParams(**model_config)).to(device)
    model.device = device

    num_epochs = train_config['num_epochs']
    lr = train_config['lr']
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=1e-5
    )
    # Set up the learning rate scheduler with linear decay
    def lambda_lr(epoch):
        return 1 - epoch / num_epochs

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    # Set up W&B writer
    experiment_id = str(uuid.uuid4())[:8]
    run_name = config['name'] + '_resnet_rgb_'  + experiment_id
    print("Run Name: ", run_name)

    # Directory to save checkpoints. PLEASE CHANGE THIS TO YOUR OWN DIRECTORY.
    checkpoint_dir = '/home/yixuany/Workspace/LowAltitudeFlight/deliverables/low_altitude_flight/checkpoints/' + run_name
    os.makedirs(checkpoint_dir, exist_ok=False)

    # Save the config for testing
    config['run_name'] = run_name
    with open(os.path.join(checkpoint_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    wandb_init(config)


    student_loss = RelaxedWTA()
    collision_computation = CollisionCost(terrain_map, rq=5.0)
    collision_prediction_loss = nn.MSELoss()
    altitude_prediction_loss = nn.MSELoss()
    depth_prediction_loss = DepthMSELoss()
    terrain_altitude_loss = AltitudeMSELoss(terrain_map)


    # Lists to store loss values
    train_losses = []
    val_losses = []


    # Initialize variables to track the best model
    best_val_loss = float('inf')
    best_epoch = 0

    print(f"Training started on {device}")
    global_steps = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (features, labels) in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch}/{num_epochs}')):
            rgb_img = features['rgb_img']
            rgb_tilted_img = features['rgb_tilted_img']
            heading = features['heading']
            attitude = features['attitude']
            position = features['position']

            desired_position = labels['desired_trajectory']
            depth_image = labels['depth_img']

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            states = torch.cat((heading, attitude), dim=-1)
            student_path_local, collision_prediction, altitude_prediction, depth_img_prediction = model(states, rgb_img, rgb_tilted_img)
            
            student_path_global = student_path_local + position[:, None, None, :]
            collision_prediction = collision_prediction.squeeze(dim=-1)
            altitude_prediction = altitude_prediction.squeeze(dim=-1)

            expert_path = desired_position.unsqueeze(dim=1)
            collision_cost = collision_computation(student_path_global)
            collision_cost = collision_cost.to(device)

            terrain_loss, terrain_collision_loss, terrain_altitude = terrain_altitude_loss(student_path_global)
            
            alt_loss = altitude_prediction_loss(altitude_prediction.reshape(-1, 10), terrain_altitude)
            bc_loss = 0.5*student_loss(expert_path[:, :, :, :2], student_path_local[:, :, :, :2])
            c_loss = 2.0*collision_prediction_loss(collision_prediction, collision_cost)
            d_loss = 1e6*depth_prediction_loss(depth_img_prediction, depth_image)
            terrain_collision_loss = 1e3*terrain_collision_loss

            loss = bc_loss + c_loss + alt_loss + d_loss + terrain_loss + terrain_collision_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


            # # print("BC Loss", bc_loss.item(), "Depth loss: ", d_loss.item())
            if global_steps % 500 == 0:
                
                wandb.log({'Train Loss': loss.item(), 
                           "BC Loss": bc_loss.item(),
                           "Terrain Loss": terrain_loss.item(),
                            "Terrain Collision Loss": terrain_collision_loss.item(),
                           "Collision Prediction Loss": c_loss.item(),
                           "Altitude Prediction Loss": alt_loss.item(),
                            "Depth Prediction Loss": d_loss.item()}, step=global_steps)
            
            global_steps += 1
            
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)


        # Step the learning rate scheduler
        scheduler.step()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f'Validation Epoch {epoch}/{num_epochs}'):
                rgb_img = features['rgb_img']
                rgb_tilted_img = features['rgb_tilted_img']
                heading = features['heading']
                attitude = features['attitude']
                position = features['position']

                desired_position = labels['desired_trajectory']
                depth_image = labels['depth_img']

                # Forward pass
                states = torch.cat((heading, attitude), dim=-1)
                student_path_local, collision_prediction, altitude_prediction, depth_img_prediction = model(states, rgb_img, rgb_tilted_img)
                student_path_global = student_path_local + position[:, None, None, :]
                collision_prediction = collision_prediction.squeeze(dim=-1)

                expert_path = desired_position.unsqueeze(dim=1)
                collision_cost = collision_computation(student_path_global)
                collision_cost = collision_cost.to(device)

                terrain_loss, terrain_collision_loss, terrain_altitude = terrain_altitude_loss(student_path_global)
                

                alt_loss = altitude_prediction_loss(altitude_prediction.reshape(-1, 10), terrain_altitude)
                bc_loss = 0.5*student_loss(expert_path[:, :, :, :2], student_path_local[:, :, :, :2])
                c_loss = 2.0*collision_prediction_loss(collision_prediction, collision_cost)
                d_loss = 1e6*depth_prediction_loss(depth_img_prediction, depth_image)
                terrain_collision_loss = 1e3*terrain_collision_loss

                loss = bc_loss + c_loss + alt_loss + d_loss + terrain_loss + terrain_collision_loss
                
                val_loss += loss.item()
                
                wandb.log({'Validation Loss': loss.item(),
                           "Validation BC Loss": bc_loss.item(),
                           "Validation Terrain Loss": terrain_loss.item(),
                            "Validation Terrain Collision Loss": terrain_collision_loss.item(),
                           "Validation Collision Prediction Loss": c_loss.item(),
                           "Validation Altitude Prediction Loss": alt_loss.item(),
                           "Validation Depth Prediction Loss": d_loss.item()}, step=global_steps)


        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.5f}, Val Loss: {avg_val_loss:.5f}')

        # Save the model checkpoint
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, best_model_path)



    print("Run Name: ", run_name)
    print(f"Training completed. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")


def get_args():
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a policy with BC.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    with open(args.config,"r") as file_object:
        config = yaml.load(file_object,Loader=yaml.SafeLoader)

    ### Load TSDF Map
    root_path = "/home/yixuany/Workspace/LowAltitudeFlight/deliverables/low_altitude_flight/terrain/tsdf_map"
    map_name = "terrain_tsdf"
    terrain_map = ElevationMap(device='cuda')
    terrain_map.ReadTSDFMap(root_path, map_name)

    train(config, terrain_map)