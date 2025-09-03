import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#Standard Convolution + BatchNorm + Activation block.
class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

#Area Attention Module
class AAt(nn.Module):
    #Note: If area_split_factor > 1, H*W must be divisible by it. When area_split_factor=1 means global attention.
    def __init__(self, dim, num_heads, area_split_factor=1): # Changed default name from area_size
        super().__init__()
        assert dim % num_heads == 0
        self.area_split_factor = area_split_factor
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.pe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)
        v_pe = v + self.pe(v)

        q = q.view(B, C, N)
        k = k.view(B, C, N)
        v = v_pe.view(B, C, N)

        B_eff, N_eff = B, N
        # Area Splitting Logic 
        can_split = (self.area_split_factor > 1 and N > 0 and N % self.area_split_factor == 0)
        if can_split:
            N_eff = N // self.area_split_factor
            q = q.view(B, C, self.area_split_factor, N_eff).permute(0, 2, 1, 3).reshape(B * self.area_split_factor, C, N_eff)
            k = k.view(B, C, self.area_split_factor, N_eff).permute(0, 2, 1, 3).reshape(B * self.area_split_factor, C, N_eff)
            v = v.view(B, C, self.area_split_factor, N_eff).permute(0, 2, 1, 3).reshape(B * self.area_split_factor, C, N_eff)
            B_eff = B * self.area_split_factor

        q = q.view(B_eff, self.num_heads, self.head_dim, N_eff).transpose(-2, -1) # B_eff, num_heads, N_eff, head_dim
        k = k.view(B_eff, self.num_heads, self.head_dim, N_eff)                   # B_eff, num_heads, head_dim, N_eff
        v = v.view(B_eff, self.num_heads, self.head_dim, N_eff).transpose(-2, -1) # B_eff, num_heads, N_eff, head_dim

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attended_v = attn @ v # B_eff, num_heads, N_eff, head_dim

        attended_v = attended_v.transpose(-2, -1).reshape(B_eff, N_eff, C) # B_eff, N_eff, C

        #Undo Area Splitting
        if can_split:
           attended_v = attended_v.view(B, self.area_split_factor, N_eff, C).permute(0, 2, 1, 3).reshape(B, N, C)

        attended_v = attended_v.transpose(-1, -2).view(B, C, H, W)

        output = self.proj(attended_v)
        output = self.proj_bn(output)
        return x + output # Internal residual

#channel attention module
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False), 
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out 
        return self.sigmoid(out)

#spatial attention module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7) 
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False) # 2 channels from avg+max pool
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) # Avg pool across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True) # Max pool across channels
        x_cat = torch.cat([avg_out, max_out], dim=1) # Concatenate
        out = self.conv1(x_cat) 
        return self.sigmoid(out)
#Convolutional Block Attention Module
class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x_out = self.ca(x) * x # Apply channel attention
        x_out = self.sa(x_out) * x_out # Apply spatial attention
        return x_out

#re-implemented RELAN block with mixed attention (AreaAttention and CBAM)
class RELANBlock(nn.Module):
    def __init__(self, c1, c2, c_hidden, num_branches=4,
                 attn_types=('cbam', 'area', 'cbam'), #list of attention types
                 num_heads=8, area_attn_split_factor=1, cbam_reduction=16, cbam_kernel_size=7, act=nn.SiLU):
        super().__init__()
        assert num_branches >= 2
        assert len(attn_types) == num_branches - 1
        self.num_branches = num_branches # Store for clarity in forward pass concatenation

        self.conv_start = ConvBNAct(c1, c_hidden, k=1, act=act())
        self.conv_main = ConvBNAct(c1, c_hidden, k=1, act=act())

        self.branches = nn.ModuleList()
        self.attentions = nn.ModuleList()

        current_c = c_hidden
        for attn_type in attn_types:
            self.branches.append(ConvBNAct(current_c, c_hidden, k=3, act=act())) # Standard 3x3 conv in branch

            if attn_type == 'cbam':
                self.attentions.append(CBAMBlock(c_hidden, reduction=cbam_reduction, kernel_size=cbam_kernel_size))
            elif attn_type == 'area':
                self.attentions.append(AAt(c_hidden, num_heads=num_heads, area_split_factor=area_attn_split_factor))
            else:
                raise ValueError(f"Unknown attention type: {attn_type}. Use 'cbam' or 'area'.")

            current_c = c_hidden

        concat_channels = c_hidden * num_branches
        self.conv_end = ConvBNAct(concat_channels, c2, k=1, act=act())

    def forward(self, x):
        x_residual = x
        out_start = self.conv_start(x)
        branch_input = self.conv_main(x)
        outputs = [out_start]

        for i in range(len(self.branches)):
            branch_conv_out = self.branches[i](branch_input)
            branch_attn_out = self.attentions[i](branch_conv_out)
            outputs.append(branch_attn_out)
            branch_input = branch_attn_out # Output feeds next branch's CONV

        # Ensure the correct number of outputs are concatenated
        # This depends on how many branches were actually processed
        concatenated = torch.cat(outputs, dim=1)

        fused_out = self.conv_end(concatenated)
        return fused_out + x_residual

# Used for downsampling between stages
class TransitionLayer(nn.Module):
    """Handles downsampling between stages."""
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = ConvBNAct(c1, c2, k=3, s=2, act=nn.SiLU()) # Using k=3 stride=2

    def forward(self, x):
        return self.conv(x)

#Stronger,smarter and better
class DynamicRELANClassifier(nn.Module):
    def __init__(self, num_classes, # Dynamic number of classes
                 backbone_channels=64,
                 stage_channels=(128, 256, 512, 1024),
                 stage_hidden_channels=(64, 128, 256, 512),
                 num_relan_blocks=(2, 2, 4, 2),
                 relan_branches=4,
                 attn_patterns=(('cbam', 'area', 'cbam'),
                                ('area', 'cbam', 'area'),
                                ('cbam', 'area', 'cbam'),
                                ('area', 'cbam', 'area')),
                 #attn_patterns=(('area', 'area', 'area'),
                                #('area', 'area', 'area'),
                                #('area', 'area', 'area'),
                                #('area', 'area', 'area')),
                 area_attn_heads=8,
                 area_attn_split_factor=1, 
                 cbam_reduction=16,
                 cbam_kernel_size=7,
                 act=nn.SiLU):
        super().__init__()
        self.num_classes = num_classes # Store num_classes for adaptive input size and output size

        #Backbone 
        self.backbone = nn.Sequential(
            ConvBNAct(3, backbone_channels // 2, k=3, s=2, act=act()), #stride 2 can be adjusted
            ConvBNAct(backbone_channels // 2, backbone_channels, k=3, s=1, act=act()),
            ConvBNAct(backbone_channels, backbone_channels, k=3, s=2, act=act()) #stride 2 can be adjusted
        )
        current_c = backbone_channels

        #RELAN Blocks loop
        self.stages = nn.ModuleList()
        num_stages = len(stage_channels)

        for i in range(num_stages):
            out_c = stage_channels[i]
            hidden_c = stage_hidden_channels[i]
            n_blocks = num_relan_blocks[i]
            attn_types = attn_patterns[i % len(attn_patterns)]
            print(f" Stage {i}: Using attention pattern: {attn_types}")

            stage = nn.Sequential()

            # Transition Layer for downsampling and channel adjustment. Add transition if not first stage OR if channel size changes
            if i > 0 or current_c != out_c: 
                stride = 2 if i > 0 else 1
                if i > 0:
                    stage.add_module(f"transition_{i}", TransitionLayer(current_c, out_c))
                else: # First stage, potential channel change only
                    stage.add_module(f"transition_{i}", ConvBNAct(current_c, out_c, k=1, s=1, act=act()))
                current_c = out_c

            # Add RELAN blocks
            for j in range(n_blocks):
                 stage.add_module(f"relan_{j}", RELANBlock(
                     c1=current_c,
                     c2=current_c, # c2 is the output channel of the RELAN block, same as input channel
                     c_hidden=hidden_c,
                     num_branches=relan_branches,
                     attn_types=attn_types,
                     num_heads=area_attn_heads,
                     area_attn_split_factor=area_attn_split_factor,
                     cbam_reduction=cbam_reduction,
                     cbam_kernel_size=cbam_kernel_size,
                     act=act
                 ))
            self.stages.append(stage)
            # current_c is now the output dimension of this stage (out_c)

        #Classifier Head
        #AdaptiveAvgPool2d enables dynamic input size handling before the Linear layer.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(current_c, self.num_classes) # Use dynamic num_classes here
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # or 'silu'
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input x can be of any size (H, W) due to AdaptiveAvgPool2d in head
        x = self.backbone(x)
        # Feature map size is now reduced, but still depends on original H, W
        for stage in self.stages:
            x = stage(x)
        # After last stage, x has shape (B, C_last, H_final, W_final) where H/W_final depend on input size
        x = self.head(x)
        # Output x has shape (B, num_classes) regardless of input H, W
        return x
