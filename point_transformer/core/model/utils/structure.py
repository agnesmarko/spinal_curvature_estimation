# source: https://github.com/Pointcept

import torch
import spconv.pytorch as spconv
from addict import Dict
from typing import Optional


def offset2batch(offset):
    # Convert offset to batch tensor
    batch = torch.zeros(offset[-1], dtype=torch.long, device=offset.device)
    for i in range(len(offset) - 1):
        batch[offset[i]:offset[i + 1]] = i
    return batch


def batch2offset(batch):
    # Convert batch tensor to offset.
    device = batch.device
    batch_size = batch[-1] + 1
    offset = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
    for i in range(batch_size):
        offset[i + 1] = offset[i] + (batch == i).sum()
    return offset


def offset2bincount(offset):
    # Convert offset to bincount.
    return offset[1:] - offset[:-1]


def encode(grid_coord, batch, depth, order="z"):
    # Encode grid coordinates with specified order.
    x, y, z = grid_coord[:, 0], grid_coord[:, 1], grid_coord[:, 2]

    if order == "z":
        # Z-order (Morton code) encoding
        code = 0
        for i in range(depth):
            code |= ((x >> i) & 1) << (3 * i)
            code |= ((y >> i) & 1) << (3 * i + 1)
            code |= ((z >> i) & 1) << (3 * i + 2)
    elif order == "z-trans":
        # Transposed Z-order
        code = 0
        for i in range(depth):
            code |= ((z >> i) & 1) << (3 * i)
            code |= ((y >> i) & 1) << (3 * i + 1)
            code |= ((x >> i) & 1) << (3 * i + 2)
    else:
        raise ValueError(f"Unknown order: {order}")

    # Add batch information to high bits
    code = code | (batch << (depth * 3))
    return code


class Point(Dict):
    # Point Structure for Point Transformer V3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Generate offset or batch if one is missing
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order=("z", "z-trans"), depth=None, shuffle_orders=False):
        # Serialize point cloud for efficient processing
        if isinstance(order, str):
            order = [order]

        assert "batch" in self.keys()

        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], 
                self.grid_size, 
                rounding_mode="trunc"
            ).int()

        if depth is None:
            depth = int(self.grid_coord.max() + 1).bit_length()

        self["serialized_depth"] = depth
        assert depth <= 16  # Maximum depth limitation

        # Generate serialization codes for each order
        code = [encode(self.grid_coord, self.batch, depth, order=o) for o in order]
        code = torch.stack(code)
        order_indices = torch.argsort(code)
        inverse = torch.zeros_like(order_indices).scatter_(
            dim=1,
            index=order_indices,
            src=torch.arange(0, code.shape[1], device=order_indices.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order_indices = order_indices[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order_indices
        self["serialized_inverse"] = inverse


    def sparsify(self, pad=96):
        # Convert to sparse representation for SpConv.
        assert {"feat", "batch"}.issubset(self.keys())

        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], 
                self.grid_size, 
                rounding_mode="trunc"
            ).int()

        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()

        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], 
                dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )

        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat
