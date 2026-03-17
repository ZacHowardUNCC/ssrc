import torch
import torch.nn as nn


class SimpleNavDecoder(nn.Module):
    """
    Navigation decoder.

    Inputs
      visual_embed:  Tensor [B, 256, 4096]
      latent_action: Tensor [B, 4, 4096]

    Output
      logits: Tensor [B, 4]
    """

    def __init__(self, dim: int = 4096, num_classes: int = 4):
        super().__init__()

        self.vis_proj = nn.Linear(dim, 512)

        self.lat_proj = nn.Linear(dim, 512)

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, visual_embed: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        visual_embed:  [B, 256, 4096]
        latent_action: [B, 4, 4096]
        """

        # Mean pool over token dimension
        vis = visual_embed.mean(dim=1)   
        lat = latent_action.mean(dim=1)  

        vis = self.vis_proj(vis)         
        lat = self.lat_proj(lat)    

        x = torch.cat([vis, lat], dim=-1)
        return self.mlp(x)
