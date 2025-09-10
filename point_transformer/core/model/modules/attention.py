# source: https://github.com/Pointcept

# Serialized Attention mechanism
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.structure import offset2bincount

try:
    import flash_attn
except ImportError:
    flash_attn = None


class RPE(nn.Module):
    """Relative Position Encoding."""

    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)
            + self.pos_bnd
            + torch.arange(3, device=coord.device) * self.rpe_num
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)
        return out


class SerializedAttention(nn.Module):
    # Serialized self-attention for point clouds

    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()

        # Validate patch_size
        if patch_size <= 0:
            patch_size = 48  


        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash and (flash_attn is not None)

        if self.enable_flash:
            assert not enable_rpe, "RPE not supported with Flash Attention"
            assert not upcast_attention
            assert not upcast_softmax
            assert flash_attn is not None, "flash_attn not installed"
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = nn.Dropout(attn_drop)


        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]
    
    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"

        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)

            # Calculate padded counts for each batch
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )

            # Only pad batches that have more points than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad

            # Create offset arrays
            _offset = F.pad(offset, (1, 0))
            _offset_pad = F.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))

            # Initialize pad and unpad arrays
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []

            for i in range(len(bincount)):
                # Original batch range
                start = _offset[i].item()
                end = _offset[i + 1].item()
                valid_count = end - start  

                # Padded batch range
                start_pad = _offset_pad[i].item()
                end_pad = _offset_pad[i + 1].item()
                pad_count = end_pad - start_pad

                # Update unpad: maps original indices to padded positions
                unpad[start:end] += start_pad - start

                # Handle padding for this batch
                if bincount[i] != bincount_pad[i] and bincount[i] > self.patch_size:
                    remainder = bincount[i] % self.patch_size
                    if remainder > 0:
                        # Copy pattern from previous patch to fill incomplete patch
                        last_patch_start = end_pad - self.patch_size
                        pad[last_patch_start + remainder:end_pad] = pad[
                            last_patch_start - self.patch_size + remainder:last_patch_start
                        ]

                # Adjust pad indices to be relative to the original batch start
                pad[start_pad:end_pad] -= start_pad - start

                # Generate cu_seqlens for this batch
                cu_seqlens.append(
                    torch.arange(
                        start_pad,
                        end_pad,
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )

            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = F.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )

        return point[pad_key], point[unpad_key], point[cu_seqlens_key]


    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )
        
        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # Get features
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # Check if we need to pad qkv for reshape
            total_points = qkv.shape[0]
            expected_points = (total_points // K) * K

            if total_points != expected_points:
                # Truncate to nearest multiple
                qkv = qkv[:expected_points]

            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H)
                .permute(2, 0, 3, 1, 4)
                .unbind(dim=0)
            )

            if self.upcast_attention:
                q = q.float()
                k = k.float()

            attn = (q * self.scale) @ k.transpose(-2, -1)

            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))

            if self.upcast_softmax:
                attn = attn.float()

            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            # Flash attention path
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)

        # Safety check and pad if needed
        required_size = inverse.max().item() + 1
        if feat.shape[0] < required_size:
            padding_size = required_size - feat.shape[0]
            feat = F.pad(feat, (0, 0, 0, padding_size), mode='constant', value=0)

        # Apply inverse mapping
        feat = feat[inverse]

        # Continue with projection and residual
        feat = self.proj_drop(self.proj(feat))
        point.feat = point.feat + feat

        return point