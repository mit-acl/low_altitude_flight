import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from low_altitude_nav.models.multiview_depth import MultiViewDepthEstimator

class EarlyFusionResNet(nn.Module):
    """ResNet that accepts 4-channel input (RGB + Depth)."""
    def __init__(self, pretrained=True):
        super(EarlyFusionResNet, self).__init__()
        # Start with a standard ResNet
        resnet = models.resnet50(pretrained=pretrained)
        old_weight = resnet.conv1.weight
        new_channels = old_weight.shape[1] + 3 + 1  
        resnet.conv1 = nn.Conv2d(
            new_channels, 
            old_weight.shape[0],
            kernel_size=resnet.conv1.kernel_size,
            stride=resnet.conv1.stride,
            padding=resnet.conv1.padding,
            bias=False
        )
        # Re-initialize the new conv's weights. Copy existing RGB weights and init the 4th channel
        with torch.no_grad():
            resnet.conv1.weight[:, :3] = old_weight  # copy RGB
            resnet.conv1.weight[:, 3:6] = old_weight  # copy RGB
            nn.init.xavier_normal_(resnet.conv1.weight[:, 6:7])

        layers = list(resnet.children())[:-1]  # remove the last FC
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        """ x: (B, 4, H, W) --> 4 channels for RGBD """
        x = self.feature_extractor(x)
        return x 

    
class ResNetPolicy(nn.Module):
    def __init__(self, config):
        super(ResNetPolicy, self).__init__()
        self.config = config
        self._create()

    def _create(self):
        """Initialize the model components."""

        self.depth_estimation_model = MultiViewDepthEstimator()
        self.rgbd_encoder = EarlyFusionResNet(pretrained=True)

        self.state_processor = nn.Sequential(
            nn.Linear(7, 64), 
            nn.LeakyReLU(0.1),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 64)
        )
        
        self.plan_module = nn.Sequential(
            nn.Linear(2048 + 64, 2048),
            nn.LeakyReLU(0.5),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(0.5),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.5),
            nn.Linear(1024, self.config.modes*(3*10 + 10 + 1))
        )


        self.path_size = self.config.modes*self.config.state_dim*self.config.out_seq_len
        self.collison_prediction_size = self.config.modes
        self.altitude_predictions_size = self.config.modes*self.config.out_seq_len
    
    
    def forward(self, states, image, image_tilted):
        """Forward pass through the model."""        

        depth_prediction = self.depth_estimation_model(image, image_tilted)
        
        rgbd_feature = self.rgbd_encoder(torch.cat((image, image_tilted, depth_prediction), dim=1)).squeeze(dim=-1).squeeze(dim=-1)
        # print("RGBD Feature size: ", rgbd_feature.shape)

        state_feature = self.state_processor(states)
        total_embeddings = torch.cat((rgbd_feature, state_feature), dim=-1)

        plans = self.plan_module(total_embeddings)

        # print("Path size: ", plans[:, :self.path_size].shape)
        student_path_local = plans[:, :self.path_size].view(plans.shape[0], self.config.modes, -1, self.config.state_dim)
        # student_path_regularized = seperation_radius*F.normalize(student_path, p=2, dim=-1)
        # student_path_local = torch.cumsum(student_path_regularized, dim=2)

        # print("Collision prediction size: ", plans[:, self.path_size:self.path_size+self.collison_prediction_size].shape)
        collison_prediction = plans[:, self.path_size:self.path_size+self.collison_prediction_size].view(plans.shape[0], self.config.modes, -1)

        # print("Altitude prediction size: ", plans[:, self.path_size+self.collison_prediction_size:].shape)
        altitude_prediction = plans[:, self.path_size+self.collison_prediction_size:].view(plans.shape[0], self.config.modes, -1, 1)

        return student_path_local, collison_prediction, altitude_prediction, depth_prediction
