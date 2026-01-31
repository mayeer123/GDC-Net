import torch
import torch.nn as nn
import torch.nn.functional as F


class GLCA_ResBlock(nn.Module):
    """
    Global-Local Channel Attention Residual Block
    - Parallel global and local feature extraction
    - No channel reduction to preserve information
    - Shared attention mechanism for adaptive feature fusion
    - Residual connection to maintain feature integrity
    """

    def __init__(self, in_channels, dilation=2, act=nn.ReLU(inplace=True)):
        super(GLCA_ResBlock, self).__init__()
        self.in_channels = in_channels
        self.dilation = dilation
        self.act = act

        # Input projection (1×1 convolution)
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn_proj = nn.BatchNorm2d(in_channels)

        # Global branch: 2 cascaded dilated convolutions
        self.global_branch = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                groups=1,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            act,
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                groups=1,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            act
        )

        # Local branch: 2 standard 3×3 convolutions
        self.local_branch = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            act,
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            act
        )

        # Shared attention subnetwork
        self.attention_subnet = nn.Sequential(
            act,
            nn.Conv2d(
                in_channels=2 * in_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
        )

        # Final feature fusion (1×1 convolution)
        self.fusion = nn.Conv2d(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize layer weights using Kaiming normalization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Residual connection
        residual = x
        B, C, H, W = x.shape

        # Step 1: Input projection and normalization
        x_proj = self.proj(x)
        x_proj = self.bn_proj(x_proj)
        x_proj = self.act(x_proj)

        # Step 2: Parallel feature extraction
        x_g = self.global_branch(x_proj)  # Global branch output
        x_l = self.local_branch(x_proj)  # Local branch output

        # Step 3: Intermediate feature fusion for attention
        x_g1 = self.global_branch[0:3](x_proj)  # Intermediate feature (global)
        x_l1 = self.local_branch[0:3](x_proj)  # Intermediate feature (local)
        x_concat_mid = torch.cat([x_g1, x_l1], dim=1)

        # Step 4: Global-local pooling
        v_g = F.adaptive_max_pool2d(x_concat_mid, output_size=1)
        v_l = F.adaptive_avg_pool2d(x_concat_mid, output_size=1)

        # Step 5: Generate attention weights
        a_g = self.attention_subnet(v_g)
        a_l = self.attention_subnet(v_l)
        a = torch.sigmoid(a_g + a_l)  # Shape: B×C×1×1

        # Expand attention weights to match concatenated feature channels (2C)
        a = torch.repeat_interleave(a, repeats=2, dim=1)  # Shape: B×(2C)×1×1

        # Step 6: Final feature fusion with attention weighting
        x_concat_final = torch.cat([x_g, x_l], dim=1)  # Shape: B×(2C)×H×W
        x_calibrated = a * x_concat_final  # Channel-wise weighting (now matching)
        x_fused = self.fusion(x_calibrated)

        # Step 7: Residual connection
        output = residual + x_fused

        return output


# Test the module (CPU-compatible)
if __name__ == "__main__":
    # Simulate input: batch_size=2, channels=64, spatial_size=256×256 (CPU tensor)
    x = torch.randn(2, 64, 256, 256)

    # Initialize GLCA-ResBlock (runs on CPU by default)
    glca_block = GLCA_ResBlock(in_channels=64, dilation=2)

    # Forward pass
    output = glca_block(x)

    # Verify output shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Shape consistency check: {x.shape == output.shape}")  # Should be True

    # Verify parameter count
    total_params = sum(p.numel() for p in glca_block.parameters())
    print(f"Total parameters: {total_params:,}")