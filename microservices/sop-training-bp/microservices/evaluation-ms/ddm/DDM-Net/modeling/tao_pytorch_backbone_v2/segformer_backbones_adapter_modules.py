# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapter Modules."""

import math
from functools import partial
import warnings
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.init import xavier_uniform_, constant_
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from timm.layers import DropPath, LayerScale



def load_ops(ops_dir, lib_name):
    """Load C++ Ops to PyTorch.

    Args:
        ops_dir (str): Path to the C++ src code directory.
        lib_name (str): Name of the library to load.
    """
    module_path = os.path.join(ops_dir, lib_name)
    torch.ops.load_library(module_path)


class MSDeformAttnFunction(Function):
    """MSDeformAttnFunction"""

    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        """Forward function.

        Args:
            value (torch.Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (torch.Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (torch.Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (torch.Tensor): The weight of sampling points
                used when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (torch.Tensor): The step used in image to column.

        Returns:
            torch.Tensor: has shape (bs, num_queries, embed_dims)

        """
        # import ipdb; ipdb.set_trace()
        ctx.im2col_step = im2col_step
        output = torch.ops.nvidia.MultiscaleDeformableAttnPlugin_TRT(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """Backward function.

        Args:
            grad_output (torch.Tensor): Gradient of output tensor of forward.

        Returns:
            tuple[Tensor]: Gradient of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            torch.ops.nvidia.DMHA_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def _is_power_of_2(n):
    """Check if n is power of 2.

    Args:
        n (int): input

    Returns:
        Boolean on if n is power of 2 or not.
    """
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0

def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).permute(
            0, 2, 1).reshape(bs * n_head, c, h, w)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(
            0, 2, 1, 3, 4).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention Module."""

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):
        """Multi-Scale Deformable Attention Constructor.

        Args:
            d_model (int): hidden dimension
            n_levels (int): number of feature levels
            n_heads (int): number of attention heads
            n_points (int): number of sampling points per attention head per feature level
            ratio (float): deformable ratio
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        self.output_proj = nn.Linear(int(d_model * ratio), d_model)

        self._reset_parameters()
        # load custom ops
        ops_dir = os.path.dirname(os.path.abspath(__file__)) + "/ops"
        lib_name = f"MultiScaleDeformableAttention.cpython-{sys.version_info.major}{sys.version_info.minor}-{os.uname().machine}-linux-gnu.so"
        load_ops(ops_dir, lib_name)

    def _reset_parameters(self):
        """Reset parameters."""
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, export=False):
        """Forward function.

        Args:
            query (torch.Tensor): (N, Length_{query}, C)
            reference_points (torch.Tensor): (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                             or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
            input_flatten (torch.Tensor): (N, sum_{l=0}^{L-1} H_l cdot W_l, C)
            input_spatial_shapes (torch.Tensor): (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            input_level_start_index (torch.Tensor): (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
            input_padding_mask (torch.Tensor): (N, sum_{l=0}^{L-1} H_l cdot W_l), True for padding elements, False for non-padding elements

        Returns:
            output (torch.Tensor): (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape

        # assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in, \
        #     f"{(input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum()} {Len_in}"

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads,
                           int(self.ratio * self.d_model) // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1],
                                            input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        input_spatial_shapes = input_spatial_shapes.long()
        input_level_start_index = input_level_start_index.long()

        if export:
            if torch.cuda.is_available() and value.is_cuda:
                output = torch.ops.nvidia.MultiscaleDeformableAttnPlugin_TRT(
                    value, input_spatial_shapes, input_level_start_index,
                    sampling_locations, attention_weights)
            else:
                # CPU implementation of multi-scale deformable attention
                # Note that this implementation uses GridSample operator which requires
                # opset version >= 16 and is much slower in TensorRT
                # warnings.warn("PyTorch native implementation of multi-scale deformable attention is being used. "
                #               "Expect slower inference performance until TensorRT further optimizes GridSample.")
                output = multi_scale_deformable_attn_pytorch(
                    value, input_spatial_shapes, sampling_locations, attention_weights
                )
        else:
            if torch.cuda.is_available() and value.is_cuda:
                # For mixed precision training
                half_float = False
                if value.dtype in [torch.float16, torch.bfloat16]:
                    half_float = value.dtype
                    value = value.float()
                    sampling_locations = sampling_locations.float()
                    attention_weights = attention_weights.float()

                output = MSDeformAttnFunction.apply(
                    value, input_spatial_shapes,
                    input_level_start_index, sampling_locations,
                    attention_weights, self.im2col_step)

                if half_float:
                    output = output.to(half_float)

            else:
                # CPU implementation of multi-scale deformable attention
                output = multi_scale_deformable_attn_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)

        output = output.view(N, Len_q, int(self.d_model * self.ratio))
        output = self.output_proj(output)
        return output


def get_reference_points(spatial_shapes, device):
    """Create reference points for Injector's and Extractor's MultiScaleDeformableAttention.

    Args:
        spatial_shapes (List[tuple]): (H, W) for different resolution reference points
        device (str): what device to use, ex: cpu or gpu

    Returns:
        torch.Tensor: reference points
    """
    reference_points_list = []
    for H_, W_ in spatial_shapes:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            # default value of torch.meshgrid indexing, to reduce warning.
            indexing="ij",
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x, patch_size=16):
    """Create deform inputs for InteractionBlock.

    Args:
        x (torch.Tensor): input features
        patch_size (int, optional): patch size. Defaults to 16.

    Returns:
        tuple: deformable inputs for Injector and Extractor in Adapter
    """
    h, w = x.shape[2:]

    # deform_inputs1 for Injector
    # the SPM use ResNet stem as CNN feature extractors and it has the downsampling steps for 4, 8, 16, 32.
    # we'll take c2, c3, c4 as input feat for InteractionBlocks. hence using 8, 16, 32 downsampling to get spatial
    # shapes
    spatial_shapes = torch.as_tensor(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], dtype=torch.long, device=x.device
    )
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(math.ceil(h / patch_size), math.ceil(w / patch_size))], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    # deform_inputs2 for Extractor
    spatial_shapes = torch.as_tensor(
        [(math.ceil(h / patch_size), math.ceil(w / patch_size))], dtype=torch.long, device=x.device
    )
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    """An implementation of ConvFFN in ViTAdapter.

    The differences between ConvFFN & FFN:
        1. ConvFFN introduces VitAdapterDWConv to encode positional
           information.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        """ConvFFN constructor.

        Args:
            in_features (int): The feature dimension. Same as
                `MultiheadAttention`.
            hidden_features (int): The hidden dimension of FFNs.
            out_features (int): The feature dimension. Same as
                `MultiheadAttention`.
            act_layer (nn.Module): activation layer.
            drop (float): dropout probability.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """Forward function.

        Args:
            x (torch.Tensor): input features
            H (int): height of stage feature
            W (int): width of stage feature

        Returns:
            torch.Tensor: layer forwarded features
        """
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    """An implementation of DWConv in VitAdapter.

    The differences between DWConv & regular DWConv:
        1. Split multi stage features then apply DWConv.
    """

    def __init__(self, dim=768, kernel_size=3, stride=1, padding=1):
        """DWConv constructor.

        Args:
            dim (int): The feature dimension.
            kernel_size (int): kernel size in Conv2d.
            stride (int): stride in Conv2d
            padding (int)L padding in Conv2d
        """
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            groups=dim,
        )

    def forward(self, x, H, W):
        """Forward function.

        Args:
            x (torch.Tensor): input features
            H (int): height of stage feature
            W (int): width of stage feature

        Returns:
            torch.Tensor: layer forwarded features
        """
        B, _, C = x.shape
        split_position = [H * 2 * W * 2, H * 2 * W * 2 + H * W]
        x1 = x[:, 0: split_position[0], :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, split_position[0]: split_position[1], :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, split_position[1]:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Extractor(nn.Module):
    """Multi Scale Feature Extractor in ViT-Adapter."""

    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        with_cp=False,
    ):
        """Extractor Constructor.

        Args:
            dims (int): The feature dimension.
            num_heads (int): Parallel attention heads. Defaults to 6.
            n_points (int): The number of sampling points for each query in each
                head of MultiScaleDeformableAttention. Defaults to 4.
            n_levels (int): The number of feature map used in
                Attention. Defaults to 1.
            deform_ratio (float): The expansion ratio of value_proj in DMHA.
                Defaults to 1.0.
            with_cffn (bool): The option to use ffn. If True, it use ffn.
                Default to True.
            cffn_ratio (float): The number of expansion ratio of feedforward
                network hidden layer channels. Default to 0.25.
            drop (float): Probability of an element to be zeroed
                after the feed forward layer. Defaults to 0.
            drop_path (float): stochastic depth rate. Defaults to 0.
            norm_layer (nn.Module): norm layer.
            with_cp (bool): Use checkpoint or not. Using checkpoint will save some
                memory while slowing down the training speed. Defaults to False.
        """
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points, ratio=deform_ratio
        )
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        """Forward function.

        Args:
            query (torch.Tensor): query features
            reference_points (torch.Tensor): reference point for extractor
            feat (torch.Tensor): input features
            spatial_shapes (torch.Tensor): spatial shapes of features
            level_start_index (torch.Tensor): level indicator
            H (int): feature height
            W (int): feature width

        Returns:
            torch.Tensor: forwarded features
        """

        def _inner_forward(query, feat):
            """Inner forward function.

            Args:
                query (torch.Tensor): query features
                feat (torch.Tensor): input features

            Returns:
                torch.Tensor: forwarded features
            """
            attn = self.attn(
                self.query_norm(query), reference_points, self.feat_norm(feat), spatial_shapes, level_start_index, None
            )
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad and not torch.onnx.is_in_onnx_export():
            query = checkpoint.checkpoint(_inner_forward, query, feat, use_reentrant=True)
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(nn.Module):
    """Injector in ViT-Adapter."""

    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.0,
        with_cp=False,
    ):
        """Injector Constructor

        Args:
            dim (int): The feature dimension.
            num_heads (int): Parallel attention heads. Defaults to 6.
            n_points (int): The number of sampling points for each query in each
                head of MultiScaleDeformableAttention. Defaults to 4.
            n_levels (int): The number of feature map used in
                Attention. Defaults to 1.
            deform_ratio (float): The expansion ratio of value_proj in DMHA.
                Defaults to 1.0.
            norm_layer (nn.Module): norm layer.
            init_values (float): initial value in LayerScale. If set to 0, LayerScale
                is not applied. Defaults to 0.0.
            with_cp (bool): Use checkpoint or not. Using checkpoint will save some
                memory while slowing down the training speed. Defaults to False.
        """
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points, ratio=deform_ratio
        )

        self.ls = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        """Forward function.

        Args:
            query (torch.Tensor): query features
            reference_points (torch.Tensor): reference point for injector
            feat (torch.Tensor): input features
            spatial_shapes (torch.Tensor): spatial shapes of features
            level_start_index (torch.Tensor): level indicator

        Returns:
            torch.Tensor: forwarded features
        """

        def _inner_forward(query, feat):
            """Inner forward function.

            Args:
                query (torch.Tensor): query features
                feat (torch.Tensor): input features

            Returns:
                torch.Tensor: forwarded features
            """
            attn = self.attn(
                self.query_norm(query), reference_points, self.feat_norm(feat), spatial_shapes, level_start_index, None
            )
            return query + self.ls(attn)

        if self.with_cp and query.requires_grad and not torch.onnx.is_in_onnx_export():
            query = checkpoint.checkpoint(_inner_forward, query, feat, use_reentrant=True)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlock(nn.Module):
    """InteractionBlock in ViT-Adapter."""

    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        deform_ratio=1.0,
        extra_extractor=False,
        with_cp=False,
    ):
        """InteractionBlock Constructor

        Args:
            dim (int): The feature dimension.
            num_heads (int): Parallel attention heads. Defaults to 6.
            n_points (int): The number of sampling points for each query in each
                head of MultiScaleDeformableAttention. Defaults to 4.
            norm_layer (nn.Module): norm layer.
            drop (float): Probability of an element to be zeroed
                after the feed forward layer. Defaults to 0.
            drop_path (float): stochastic depth rate. Defaults to 0.
            with_cffn (bool): The option to use ffn. If True, it use ffn.
                Default to True.
            cffn_ratio (float): The number of expansion ratio of feedforward
                network hidden layer channels. Default to 0.25.
            init_values (float): initial value in LayerScale. If set to 0, LayerScale
                is not applied. Defaults to 0.0.
            deform_ratio (float): The expansion ratio of value_proj in DMHA.
                Defaults to 1.0.
            extra_extractor (bool): The option to use extra Extractor in
                InteractionBlock. If True, it use extra Extractor.
                Default to False.
            with_cp (bool): Use checkpoint or not. Using checkpoint will save some
                memory while slowing down the training speed. Defaults to False.
        """
        super().__init__()

        self.injector = Injector(
            dim=dim,
            n_levels=3,
            num_heads=num_heads,
            init_values=init_values,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cp=with_cp,
        )
        self.extractor = Extractor(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            with_cp=with_cp,
        )
        if extra_extractor:
            self.extra_extractors = nn.Sequential(
                *[
                    Extractor(
                        dim=dim,
                        num_heads=num_heads,
                        n_points=n_points,
                        norm_layer=norm_layer,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                        deform_ratio=deform_ratio,
                        drop=drop,
                        drop_path=drop_path,
                        with_cp=with_cp,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W, batch_first=True):
        """Forward function.

        Args:
            x (torch.Tensor): query features for injector
            c (torch.Tensor): input features for injector
            blocks (nn.Module): ViT Transformer blocks module
            deform_inputs1 (torch.Tensor): deform inputs for InteractionBlock
            deform_inputs2 (torch.Tensor): deform inputs for InteractionBlock
            H (int): feature height
            W (int): feature width
            batch_first (bool, optional): use batch first format. Defaults to True.

        Returns:
            torch.Tensor: fowarded features
        """
        x = self.injector(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )

        x = x if batch_first else x.permute(1, 0, 2)  # [bs, seq_l, dim] -> [seq_l, bs, dim]
        for blk in blocks:
            x = blk(x)
        x = x if batch_first else x.permute(1, 0, 2)  # [seq_l, bs, dim] -> [bs, seq_l, dim]

        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H,
            W=W,
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H,
                    W=W,
                )
        return x, c


class RADIOInteractionBlock(nn.Module):
    """InteractionBlock in RADIO-Adapter."""

    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        deform_ratio=1.0,
        extra_extractor=False,
        with_cp=False,
    ):
        """InteractionBlock Constructor

        Args:
            dims (int): The feature dimension.
            num_heads (int): Parallel attention heads. Defaults to 6.
            n_points (int): The number of sampling points for each query in each
                head of MultiScaleDeformableAttention. Defaults to 4.
            norm_layer (nn.Module): norm layer.
            drop (float): Probability of an element to be zeroed
                after the feed forward layer. Defaults to 0.
            drop_path (float): stochastic depth rate. Defaults to 0.
            with_ffn (bool): The option to use ffn. If True, it use ffn.
                Default to True.
            cffn_ratio (float): The number of expansion ratio of feedforward
                network hidden layer channels. Default to 0.25.
            init_values (float): initial value in LayerScale. If set to 0, LayerScale
                is not applied. Defaults to 0.0.
            deform_ratio (float): The expansion ratio of value_proj in DMHA.
                Defaults to 1.0.
            extra_extractor (bool): The option to use extra Extractor in
                InteractionBlock. If True, it use extra Extractor.
                Default to False.
            with_cp (bool): Use checkpoint or not. Using checkpoint will save some
                memory while slowing down the training speed. Defaults to False.
        """
        super().__init__()

        self.injector = Injector(
            dim=dim,
            n_levels=3,
            num_heads=num_heads,
            init_values=init_values,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cp=with_cp,
        )
        self.extractor = Extractor(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            with_cp=with_cp,
        )
        if extra_extractor:
            self.extra_extractors = nn.Sequential(
                *[
                    Extractor(
                        dim=dim,
                        num_heads=num_heads,
                        n_points=n_points,
                        norm_layer=norm_layer,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                        deform_ratio=deform_ratio,
                        drop=drop,
                        drop_path=drop_path,
                        with_cp=with_cp,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W, num_summary=0):
        """Forward function."""
        x_summary = x[:, :num_summary]
        x_feat = x[:, num_summary:]
        x_feat = self.injector(
            query=x_feat,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )
        x = torch.cat([x_summary, x_feat], dim=1).contiguous()
        for blk in blocks:
            x = blk(x)
        x_feat = x[:, num_summary:]
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x_feat,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H,
            W=W,
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x_feat,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H,
                    W=W,
                )
        return x, c


class SpatialPriorModule(nn.Module):
    """SpatialPriorModule in ViT-Adapter."""

    def __init__(self, in_channel, patch_size, embed_dim, inplanes=64, out_indices=[0, 1, 2, 3], padding="corner"):
        """SpatialPriorModule Constructor.

        Args:
            in_channel (int): channel size of input.
            patch_size (int): The patch size in patch embedding.
            embed_dim (int): The feature dimension.
            inplanes (int): Hidden dimension. Defaults to 64.
            out_indices (list): List of block indices to return as feature.
            padding (str): Support "same" and "corner", "corner" mode
                would pad zero to bottom right, and "same" mode would
                pad zero around input. Default to "corner".
        """
        super().__init__()
        self.out_indices = out_indices

        self.stem = nn.Sequential(
            *[
                nn.Conv2d(in_channel, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(2 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv4 = nn.Sequential(
            *[
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        if len(out_indices) == 4:
            self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): input features

        Returns:
            torch.Tensor: forwarded features
        """
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        if len(self.out_indices) == 4:
            c1 = self.fc1(c1)

        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)
        bs, dim, _, _ = c2.shape

        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4
