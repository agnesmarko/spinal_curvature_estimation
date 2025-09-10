# model implementation

import torch
import torch.nn as nn
from torch.nn import init

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):  
    # 2D convolution used in the original U-Net architecture  
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    # two ways of upsampling:
    # 1. using ConvTranspose2d - uses learnable parameters to upsample
    # 2. using nn.Upsample followed by a 1x1 convolution - no learnable parameters
    # (Upsample doubles the size using bilinear interpolation, then a 1x1 convolution adjusts the number of channels)

    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

    
def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConvBlock(nn.Module):
    # performs 2 convolutions (3x3 and ReLU) and 1 MaxPool(2x2) operation
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # store the output before pooling for skip connections
        before_pooling = x

        if self.pooling:
            x = self.pool(x)

        return x, before_pooling
    

class UpConvBlock(nn.Module):
    # performs 1 upsampling (2x2) and 2 convolutions (3x3 and ReLU) operations
    def __init__(self, in_channels, out_channels, mode='transpose'):
        super(UpConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.mode)
        self.conv1 = conv3x3(self.out_channels * 2, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, skip_connection):
        x = self.upconv(x)

        # concatenate the skip connection
        x = torch.cat((x, skip_connection), dim=1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x
    

class UNet(nn.Module):
    """U-Net model based on:  https://arxiv.org/abs/1505.04597"""

    def __init__(self, in_channels=1, out_channels=1, start_channels = 64, 
                 depth = 4, mode='transpose', use_final_conv=True):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_channels = start_channels
        self.use_final_conv = use_final_conv

        # the number of downsampling (with maxpool) and upsampling blocks
        self.depth = depth

        if mode not in ['transpose', 'upsample']:
            raise ValueError("mode must be either 'transpose' or 'upsample'")
        
        self.mode = mode

        # build the encoder and decoder parts of the U-Net
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        enc_out_chan = self._build_encoder()
        self.dec_out_chan = self._build_decoder(enc_out_chan)

        # final 1x1 convolution to map to the desired output channels (optional)
        if use_final_conv:
            self.final_conv = conv1x1(self.dec_out_chan, self.out_channels)

        # initialize weights
        for module in self.modules():
            self._initialize_weights(module)


    def _build_encoder(self):
        # build the encoder part of the U-Net
        
        for i in range(self.depth + 1):
            in_chan = self.in_channels if i == 0 else out_chan
            out_chan = self.start_channels * (2 ** i)

            pooling = (i < self.depth)  # no pooling in the last block
            down_block = DownConvBlock(in_chan, out_chan, pooling=pooling)
            self.down_blocks.append(down_block)

        return out_chan


    def _build_decoder(self, out_chan):
        # build the decoder part of the U-Net
        for _ in range(self.depth):
            in_chan = out_chan
            out_chan = in_chan // 2

            up_block = UpConvBlock(in_chan, out_chan, mode=self.mode)
            self.up_blocks.append(up_block)

        return out_chan


    @staticmethod
    def _initialize_weights(module):
        # initialize weights using Kaiming He initialization
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)


    def forward(self, x):
        # forward pass through the U-Net
        skip_connections = []

        # encoder part
        for down_block in self.down_blocks:
            x, before_pooling = down_block(x)
            skip_connections.append(before_pooling)

        # decoder part
        for i, up_block in enumerate(self.up_blocks):
            skip_connection = skip_connections[-(i + 2)] # the last one is not used as a skip connections, because it is the output of the last down block (no pooling)
            x = up_block(x, skip_connection)

        # final convolution to map to the desired output channels if enabled
        if self.use_final_conv:
            x = self.final_conv(x)
        
        return x
    

    def freeze_encoder(self):
        # freeze all encoder (down) blocks
        for block in self.down_blocks:
            for param in block.parameters():
                param.requires_grad = False
        print("Encoder frozen")


    def unfreeze_encoder(self):
        # unfreeze all encoder (down) blocks
        for block in self.down_blocks:
            for param in block.parameters():
                param.requires_grad = True
        print("Encoder unfrozen")


    def freeze_decoder(self):
        # freeze all decoder (up) blocks
        for block in self.up_blocks:
            for param in block.parameters():
                param.requires_grad = False
        print("Decoder frozen")


    def unfreeze_decoder(self):
        # unfreeze all decoder (up) blocks
        for block in self.up_blocks:
            for param in block.parameters():
                param.requires_grad = True
        print("Decoder unfrozen")


    def freeze_early_layers(self, num_layers=2):
        # freeze first N encoder blocks
        for i in range(min(num_layers, len(self.down_blocks))):
            for param in self.down_blocks[i].parameters():
                param.requires_grad = False
        print(f"First {num_layers} encoder blocks frozen")


    def get_trainable_params(self):
        # get count of trainable parameters
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total


    def print_layer_status(self):
        # print which layers are frozen/trainable
        print("\n=== Layer Status ===")
        for i, block in enumerate(self.down_blocks):
            frozen = not any(p.requires_grad for p in block.parameters())
            print(f"Encoder Block {i}: {'FROZEN' if frozen else 'TRAINABLE'}")

        for i, block in enumerate(self.up_blocks):
            frozen = not any(p.requires_grad for p in block.parameters())
            print(f"Decoder Block {i}: {'FROZEN' if frozen else 'TRAINABLE'}")

        trainable, total = self.get_trainable_params()
        print(f"Parameters: {trainable:,}/{total:,} trainable ({trainable/total*100:.1f}%)")


class RecurrentBlock(nn.Module):
    # recurrent block with t iterations
    def __init__(self, out_channels, t=2, dropout=0.0):
        super(RecurrentBlock, self).__init__()

        self.t = t
        self.out_channels = out_channels
        self.dropout = dropout


        self.conv = nn.Sequential(
            conv3x3(self.out_channels, self.out_channels),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

        # adding dropout if specified
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout2d(self.dropout)

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            else:
                x1 = self.conv(x + x1)

            # apply dropout if specified and in training mode
            if self.dropout > 0 and self.training:
                x1 = self.dropout_layer(x1)

        return x1
    

class RRCNNBlock(nn.Module):
    # recurrent residual convolutional neural network block (recurrent blocks with t iterations)
    def __init__(self, in_channels, out_channels, t=2, dropout=0.0):
        super(RRCNNBlock, self).__init__()

        self.dropout = dropout

        self.RCNN = nn.Sequential(
            RecurrentBlock(out_channels, t, dropout=dropout),
            RecurrentBlock(out_channels, t, dropout=dropout),
        )

        self.conv = conv1x1(in_channels, out_channels)

        # adding dropout after the residual connection
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout2d(self.dropout)
    
    def forward(self, x):
        x = self.conv(x)
        x1 = self.RCNN(x)

        # residual connection
        x = x + x1

        # apply dropout if specified and in training mode
        if self.dropout > 0 and self.training:
            x = self.dropout_layer(x)

        return x


class AttentionGate(nn.Module):
    # attention gate for skip connections
    # F_g = feature gating = number of features in the gating signal from the decoder path
    # F_l = feature local = number of features in the skip connections from the encoder 
    # F_int = feature intermediate =  reduced dimensionality used for efficient attention computation
    def __init__(self, F_g, F_l, F_int, dropout=0.0):
        super(AttentionGate, self).__init__()

        self.dropout = dropout

        # transform gating signal to intermediate dimension
        self.W_g = nn.Sequential(
            conv1x1(F_g, F_int),
            nn.BatchNorm2d(F_int)
        )

        # transform skip connection to intermediate dimension
        self.W_x = nn.Sequential(
            conv1x1(F_l, F_int),
            nn.BatchNorm2d(F_int)
        )

        # attention computation with optional dropout
        attention_layers = [
            conv1x1(F_int, 1),
            nn.BatchNorm2d(1),
        ]

        if self.dropout > 0:
            attention_layers.insert(-1, nn.Dropout2d(self.dropout))

        attention_layers.append(nn.Sigmoid())

        self.psi = nn.Sequential(*attention_layers)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # prevent attention from goin too close to zero
        psi = torch.clamp(psi, min=0.1, max=1.0)

        return x * psi


class DownRRConvBlock(nn.Module):
    # down-sampling block with RRCNN and optional pooling
    # performs a residual recurrent convolutional step with t iterations and 1 MaxPool(2x2) operation
    def __init__(self, in_channels, out_channels, t=2, pooling=True, dropout=0.0):
        super(DownRRConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.rrcnn = RRCNNBlock(in_channels, out_channels, t=t, dropout=dropout)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.rrcnn(x)
        before_pooling = x

        if self.pooling:
            x = self.pool(x)

        return x, before_pooling


class UpRRConvBlock(nn.Module):
    # up-sampling block with RRCNN and attention gate
    # performs an upsampling, attention-based concatenation 
    # and a residual recurrent convolutional step with t iterations
    def __init__(self, in_channels, out_channels, t=2, mode='transpose', use_attention=True, dropout=0.0):
        super(UpRRConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode
        self.use_attention = use_attention
        self.dropout = dropout

        self.upconv = upconv2x2(in_channels, out_channels, mode=mode)

        if self.use_attention:
            # F_int is typically F_g // 2 or F_l // 2 -> this can be 0 for small out_channels
            F_int = max(out_channels // 2, 1)
            self.attention = AttentionGate(F_g=out_channels, F_l=out_channels, F_int=F_int, dropout=dropout)

        # after concatenation: out_channels + out_channels = 2 * out_channels
        self.rrcnn = RRCNNBlock(out_channels * 2, out_channels, t=t, dropout=dropout)

    
    def forward(self, x, skip_connection):
        x = self.upconv(x)

        if self.use_attention:
            skip_connection = self.attention(g=x, x=skip_connection)

        x = torch.cat((x, skip_connection), dim=1)
        x = self.rrcnn(x)

        return x
    

class R2AttUNet(nn.Module):
    # recurrent residual attention u-net
    # recurrent connection: refines features over t iterations
    # residual connection: adds input to output for gradient flow

    def __init__(self, in_channels=1, out_channels=1, start_channels=64, 
                 depth=4, t=2, mode='transpose', use_attention=True, dropout=0.0):
        super(R2AttUNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_channels = start_channels
        self.depth = depth
        self.t = t
        self.mode = mode
        self.use_attention = use_attention
        self.dropout = dropout

        if mode not in ['transpose', 'upsample']:
            raise ValueError("mode must be either 'transpose' or 'upsample'")
        
        # build the encoder and decoder parts
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        enc_out_chan = self._build_encoder()
        dec_out_chan = self._build_decoder(enc_out_chan)

        # final 1x1 convolution to map to the desired output channels
        if self.dropout > 0:
            self.final_conv = nn.Sequential(
                nn.Dropout2d(self.dropout * 0.5),  # reduce dropout rate for final conv
                conv1x1(dec_out_chan, self.out_channels)
            )
        else:
            self.final_conv = conv1x1(dec_out_chan, self.out_channels)

        # initialize weights
        for module in self.modules():
            self._initialize_weights(module)


    def _build_encoder(self):
        # build encoder with RRCNN blocks
        for i in range(self.depth + 1):
            in_chan = self.in_channels if i == 0 else out_chan
            out_chan = self.start_channels * (2 ** i)

            pooling = (i < self.depth)  # no pooling in the last block

            # gradually increase  dropout in deeper layers
            layer_dropout = self.dropout * (i / self.depth) if self.dropout > 0 else 0.0

            down_block = DownRRConvBlock(in_chan, out_chan, t=self.t, pooling=pooling, dropout=layer_dropout)

            self.down_blocks.append(down_block)

        return out_chan
    

    def _build_decoder(self, out_chan):
        # build decoder with RRCNN blocks and attention gates
        for i in range(self.depth):
            in_chan = out_chan
            out_chan = in_chan // 2

            # gradually decrease dropout in decoder (less dropout towards the output)
            layer_dropout = self.dropout * ((self.depth - i -1) / self.depth) if self.dropout > 0 else 0.0

            up_block = UpRRConvBlock(in_chan, out_chan, t=self.t, 
                                mode=self.mode, use_attention=self.use_attention, dropout=layer_dropout)
            
            self.up_blocks.append(up_block)

        return out_chan
    

    @staticmethod
    def _initialize_weights(module):
        # initialize weights using Kaiming He initialization
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)


    def forward(self, x):
        # forward pass
        skip_connections = []

        # encoder part
        for down_block in self.down_blocks:
            x, before_pooling = down_block(x)
            skip_connections.append(before_pooling)

        # decoder part
        for i, up_block in enumerate(self.up_blocks):
            skip_connection = skip_connections[-(i + 2)] # the last one is not used as a skip connections, because it is the output of the last down block (no pooling)
            x = up_block(x, skip_connection)

        # final convolution to map to the desired output channels
        x = self.final_conv(x)
        return x



