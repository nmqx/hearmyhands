"""
model.py — ResNet-50 Heatmap Backbone + Spatial Softmax

Architecture:
    1. ResNet-50 (pretrained or from scratch) -> [B, 2048, 8, 8] pour une image 256x256
    2. Deconvolution Head (SimpleBaseline) -> [B, NUM_KEYPOINTS, 64, 64]
    3. Spatial Softmax (Integral Pose Regression) -> [B, NUM_KEYPOINTS, 2] (coordonnées [0, 1])
    4. Visibilité via GAP -> Linear -> [B, NUM_KEYPOINTS, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

NUM_KEYPOINTS = 9


class SpatialSoftmax(nn.Module):
    """
    Transforme une heatmap 2D en coordonnées (x, y) différentiables via une
    espérance mathématique (produit pondéré par les probabilités softmax).
    """
    def __init__(self, height: int, width: int, temperature: float = 10.0):
        super().__init__()
        self.height = height
        self.width = width
        # Temperature aide à rendre la distribution softmax plus 'pointue'
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

        # Grille de coordonnées normalisées entre 0 et 1
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, height),
            torch.linspace(0.0, 1.0, width),
            indexing='ij'
        )
        self.register_buffer('pos_x', pos_x.reshape(-1))
        self.register_buffer('pos_y', pos_y.reshape(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, H, W = x.shape
        # x: [B, K, H*W]
        x_flat = x.view(B, K, H * W)
        
        # Attention Softmax
        attn = F.softmax(x_flat / self.temperature, dim=-1)
        
        # Somme pondérée
        expected_x = torch.sum(self.pos_x * attn, dim=-1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attn, dim=-1, keepdim=True)
        
        return torch.cat([expected_x, expected_y], dim=-1)  # [B, K, 2]


class DeconvHead(nn.Module):
    """ SimpleBaseline: 3 couches de déconvolution pour upsampler les features """
    def __init__(self, in_channels=2048, num_layers=3, num_filters=256, kernel_size=4):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels if i == 0 else num_filters,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.ReLU(inplace=True))
        
        self.deconv = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.deconv(x)


class HeatnoksModel(nn.Module):
    """
    Modèle Heatmap Keypoint pour 9 points clés.
    """
    def __init__(self, num_keypoints: int = NUM_KEYPOINTS, pretrained: bool = True):
        super().__init__()
        self.num_keypoints = num_keypoints

        # ── Backbone ResNet-50 ──────────────────────────────────────────────
        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = tvm.resnet50(weights=weights)

        # On garde jusqu'à layer4 (inclus) -> reduction x32 (256 -> 8)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, # -> 64
            resnet.layer1, # -> 64
            resnet.layer2, # -> 32
            resnet.layer3, # -> 16
            resnet.layer4  # -> 8
        )

        # ── Heatmap Head ───────────────────────────────────────────────────
        # Upsampling: 8x8 -> 16x16 -> 32x32 -> 64x64
        self.deconv_head = DeconvHead(in_channels=2048, num_layers=3, num_filters=256, kernel_size=4)
        
        # Projection vers le nombre de keypoints
        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels=num_keypoints,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # ── Spatial Softmax (Coord extraction) ─────────────────────────────
        # Une entrée 256x256 donne une heatmap 64x64
        self.spatial_softmax = SpatialSoftmax(height=64, width=64)

        # ── Visibility Head ────────────────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.vis_head = nn.Linear(2048, num_keypoints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            out : [B, K, 3]
                  out[..., 0:2] = (x, y) dans (0, 1) calculés via spatial softmax
                  out[..., 2]   = visibilité brutes (logits pour le BCELoss)
        """
        # Features
        feat = self.backbone(x)  # [B, 2048, H/32, W/32]
        
        # Heatmaps
        up = self.deconv_head(feat)       # [B, 256, H/4, W/4]
        heatmaps = self.final_layer(up)   # [B, K, H/4, W/4]
        
        # Coordonnées Differentiables [0, 1]
        xy = self.spatial_softmax(heatmaps)  # [B, K, 2]
        
        # Visibilité Logits
        feat_pooled = self.global_pool(feat).view(feat.size(0), -1)  # [B, 2048]
        vis = self.vis_head(feat_pooled).unsqueeze(-1)               # [B, K, 1]

        # Concat pour avoir la même signature qu'avant [B, K, 3]
        return torch.cat([xy, vis], dim=-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """ Inference: applique la sigmoïde sur la visibilité pour l'avoir sur [0, 1] """
        out = self.forward(x)
        vis = torch.sigmoid(out[..., 2:3])
        return torch.cat([out[..., :2], vis], dim=-1)

if __name__ == "__main__":
    model = HeatnoksModel(pretrained=False)
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    print(f"Input  : {tuple(dummy.shape)}")
    print(f"Output : {tuple(out.shape)}")         # L'output doit être [2, 9, 3]
    assert out.shape == (2, NUM_KEYPOINTS, 3)
    print("Test local réussi.")
