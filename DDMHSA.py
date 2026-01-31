import torch
import torch.nn as nn
import torch.nn.functional as F


class DDMHSA(nn.Module):
    """
    Dilated Depthwise Multi-Head Self-Attention
    - Fuses dilated depthwise convolutions with multi-head self-attention
    - Expands receptive field while maintaining computational efficiency
    - Captures long-range dependencies for continuous feature representation
    """

    def __init__(self, in_channels, num_heads=8, dilation=2, dropout=0.1):
        super(DDMHSA, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.dilation = dilation
        self.dropout = dropout

        # Ensure channel dimension is divisible by number of heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.dim_per_head = in_channels // num_heads

        # 1×1 convolution for Q, K, V projection (same channel dimension)
        self.qkv_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=3 * in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        # Dilated depthwise convolution (expands receptive field to 7×7)
        self.dilated_depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            groups=in_channels,  # Depthwise: one convolution per channel
            bias=False
        )

        # Learnable temperature parameter for attention scaling
        self.temperature = nn.Parameter(torch.sqrt(torch.tensor(self.dim_per_head, dtype=torch.float32)))

        # Output projection and dropout
        self.out_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.dropout_layer = nn.Dropout(dropout)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming normalization and constant bias"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Parameter):
                nn.init.constant_(m, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W  # Total number of spatial pixels

        # Step 1: Generate Q, K, V and apply dilated depthwise convolution
        qkv = self.qkv_proj(x)  # Shape: B×(3C)×H×W
        q, k, v = torch.chunk(qkv, 3, dim=1)  # Split into Q, K, V: each B×C×H×W

        # Expand receptive field with dilated depthwise convolution
        q = self.dilated_depthwise(q)  # Shape: B×C×H×W
        k = self.dilated_depthwise(k)  # Shape: B×C×H×W
        v = self.dilated_depthwise(v)  # Shape: B×C×H×W

        # Step 2: Reshape for multi-head attention (B×C×H×W → B×num_heads×dim_per_head×N)
        q = q.reshape(B, self.num_heads, self.dim_per_head, N)  # B×h×d×N
        k = k.reshape(B, self.num_heads, self.dim_per_head, N)  # B×h×d×N
        v = v.reshape(B, self.num_heads, self.dim_per_head, N)  # B×h×d×N

        # Step 3: L2 normalization for Q and K (enhances numerical stability)
        q = F.normalize(q, p=2, dim=2)
        k = F.normalize(k, p=2, dim=2)

        # Step 4: Compute attention weights
        # Similarity matrix: B×h×N×N (Q×K^T / temperature)
        attn_raw = (q @ k.transpose(-2, -1)) / self.temperature.clamp(min=1e-5)
        attn = F.softmax(attn_raw, dim=-1)  # Normalize along last dimension
        attn = self.dropout_layer(attn)  # Apply dropout for regularization

        # Step 5: Aggregate values with attention weights
        out = attn @ v  # Shape: B×h×d×N

        # Step 6: Reshape back to spatial format (B×h×d×N → B×C×H×W)
        out = out.reshape(B, C, H, W)

        # Step 7: Output projection and residual connection (implicit in parent network)
        out = self.out_proj(out)

        return out


# Test the module (CPU-compatible)
if __name__ == "__main__":
    # Simulate input: batch_size=2, channels=64, spatial_size=256×256
    x = torch.randn(2, 64, 256, 256)

    # Initialize DDMHSA (num_heads=8 as default)
    ddmhsa = DDMHSA(in_channels=64, num_heads=8, dilation=2)

    # Forward pass
    output = ddmhsa(x)

    # Verify output shape (should match input shape)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Shape consistency check: {x.shape == output.shape}")  # Should be True