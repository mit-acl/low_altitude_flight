import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0p = self.maxpool(x0)
        
        x1 = self.layer1(x0p)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        return x0, x1, x2, x3, x4

class DepthDecoder(nn.Module):
    def __init__(self, num_features=(64, 256, 512, 1024, 2048)):
        super(DepthDecoder, self).__init__()
        
        # Blocks for upsampling, same as single-image example.
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(num_features[4], 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512 + num_features[3], 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256 + num_features[2], 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128 + num_features[1], 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64 + num_features[0], 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x0, x1, x2, x3, x4):
        d1 = self.up1(x4)
        d1 = torch.cat([d1, x3], dim=1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, x2], dim=1)
        d3 = self.up3(d2)
        d3 = torch.cat([d3, x1], dim=1)
        d4 = self.up4(d3)
        d4 = torch.cat([d4, x0], dim=1)
        depth = self.up5(d4)
        return depth

class MultiViewDepthEstimator(nn.Module):
    """
    Takes two images: forward-facing (I_f) and downward-facing (I_d).
    Produces a single depth map for the forward view (or any reference).
    """
    def __init__(self, pretrained=True):
        super(MultiViewDepthEstimator, self).__init__()
        # Two separate encoders
        self.encoderF = ResNetEncoder(pretrained=pretrained)
        self.encoderD = ResNetEncoder(pretrained=pretrained)
        
        # Single decoder
        self.decoder = DepthDecoder(num_features=(128, 256, 512, 1024, 2048))
        
        # Optional: a small conv to reduce or fuse the combined bottleneck
        # so that the channel dimension doesn't explode when we concatenate.
        self.bottleneck_fuse_0 = nn.Conv2d(64*2, 128, kernel_size=1)
        self.bottleneck_fuse_1 = nn.Conv2d(256*2, 256, kernel_size=1)
        self.bottleneck_fuse_2 = nn.Conv2d(512*2, 512, kernel_size=1)
        self.bottleneck_fuse_3 = nn.Conv2d(1024*2, 1024, kernel_size=1)
        self.bottleneck_fuse_4 = nn.Conv2d(2048*2, 2048, kernel_size=1)
        
    def forward(self, I_f, I_d):
        """
        I_f, I_d: Each is a (B,3,H,W) tensor. 
                  We assume they have the same resolution for simplicity.
        """
        # Encode forward image
        f0, f1, f2, f3, f4 = self.encoderF(I_f)
        # Encode downward image
        d0, d1, d2, d3, d4 = self.encoderD(I_d)

        # print("Encoder shapes: ", f0.shape, f1.shape, f2.shape, f3.shape, f4.shape)
        fused_x0 = torch.cat([f0, d0], dim=1)  # B x (2048+2048) x H/32 x W/32
        fused_x0 = self.bottleneck_fuse_0(fused_x0)  # B x 2048 x H/32 x W/32

        fused_x1 = torch.cat([f1, d1], dim=1)  # B x (2048+2048) x H/32 x W/32
        fused_x1 = self.bottleneck_fuse_1(fused_x1)  # B x 2048 x H/32 x W/32

        fused_x2 = torch.cat([f2, d2], dim=1)  # B x (2048+2048) x H/32 x W/32
        fused_x2 = self.bottleneck_fuse_2(fused_x2)  # B x 2048 x H/32 x W/32

        fused_x3 = torch.cat([f3, d3], dim=1)  # B x (2048+2048) x H/32 x W/32
        fused_x3 = self.bottleneck_fuse_3(fused_x3)  # B x 2048 x H/32 x W/32

        fused_x4 = torch.cat([f4, d4], dim=1)  # B x (2048+2048) x H/32 x W/32
        fused_x4 = self.bottleneck_fuse_4(fused_x4)  # B x 2048 x H/32 x W/32

        depth_pred = self.decoder(
            fused_x0, fused_x1, fused_x2, fused_x3, fused_x4
        )
        
        return depth_pred
