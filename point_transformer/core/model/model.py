# source: https://github.com/Pointcept

import torch
import torch.nn as nn
import torch_scatter
from .point_transformer_v3 import PointTransformerV3

class RegressionHead(nn.Module):
    def __init__(self, input_dim=512, output_channels=48):
        super().__init__()
        self.reg = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_channels)
        )
        
    def forward(self, x):
        return self.reg(x)


class RegressionPointTransformer(nn.Module):
    # Spinal regression model using Point Transformer V3 as backbone

    def __init__(
        self,
        in_channels=3,  # xyz coordinates
        num_points=16,  # 16 spinal points
        output_channels=48,  # 16 points × 3 coordinates = 48
        grid_size: float = None,
        **ptv3_kwargs
    ):
        super().__init__()
        self.provided_grid_size = grid_size

        # Point Transformer V3 encoder
        self.encoder = PointTransformerV3(
            in_channels=in_channels,
            cls_mode=True,  # Use classification mode (encoder only)
            enable_flash=False,
            **ptv3_kwargs
        )

        # Get the output dimension from the encoder
        # This will be enc_channels[-1]
        encoder_out_dim = self.encoder.enc_channels[-1]

        # Regression head 
        self.reg_head = RegressionHead(
            input_dim=encoder_out_dim,
            output_channels=output_channels
        )

        self.num_points = num_points


    def forward(self, data_dict):
        # Extract features using Point Transformer V3
        point = self.encoder(data_dict)

        # Global pooling over points in each batch
        # Method 1: Use scatter_mean with batch indices
        if hasattr(point, 'batch'):
            global_feat = torch_scatter.scatter_mean(
                src=point.feat,
                index=point.batch,
                dim=0,
                dim_size=len(point.offset) - 1  # Actual batch size
            )
        else:
            # Method 2: Fix segment_csr usage
            # segment_csr expects indptr without padding
            global_feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=point.offset,  
                reduce="mean",
            )
            # Remove the last element if it's creating an extra output
            batch_size = len(data_dict['offset']) - 1
            global_feat = global_feat[:batch_size]

        # Regression to predict spinal points - output shape (B, 48)
        predictions = self.reg_head(global_feat)

        return predictions
