import torch
import torch.nn as nn
import mmcv
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import ConvModule, DropPath, build_activation_layer
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from collections import Sequence

def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number down to the nearest value that can
    be divisible by the divisor.

    Args:
        value (int): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int, optional): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel
            number to the original channel number. Default: 0.9.
    Returns:
        int: The modified output channel number
    """

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value

class SELayer(BaseModule):
    """Squeeze-and-Excitation Module.

    Args:
        in_channels (int): The input (and output) in_channels of the SE layer.
        squeeze_channels (None or int): The intermediate channel number of
            SElayer. Default: None, means the value of ``squeeze_channels``
            is ``make_divisible(in_channels // ratio, divisor)``.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will
            be ``make_divisible(in_channels // ratio, divisor)``. Only used when
            ``squeeze_channels`` is None. Default: 16.
        divisor(int): The divisor to true divide the channel number. Only
            used when ``squeeze_channels`` is None. Default: 8.
        conv_cfg (None or dict): Config dict for convolution layer. Default:
            None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='Sigmoid'))
    """

    def __init__(self,
                 in_channels,
                 squeeze_channels=None,
                 ratio=16,
                 divisor=8,
                 bias='auto',
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid')),
                 init_cfg=None):
        super(SELayer, self).__init__(init_cfg)
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        # assert mmcv.is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        if squeeze_channels is None:
            squeeze_channels = make_divisible(in_channels // ratio, divisor)
        assert isinstance(squeeze_channels, int) and squeeze_channels > 0, \
            '"squeeze_channels" should be a positive integer, but get ' + \
            f'{squeeze_channels} instead.'
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=squeeze_channels,
            kernel_size=1,
            stride=1,
            bias=bias,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=squeeze_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=bias,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


def gather_map(data):
    # 如果 data["map_mask"] 是 list 或 tuple，则堆叠
    if isinstance(data["map_mask"], (list, tuple)):
        map_tensor = torch.stack(data["map_mask"], dim=0)
    else:
        map_tensor = data["map_mask"]
    map_tensor = map_tensor.transpose(dim0=-2, dim1=-1)  # NOTE: important!!
    return map_tensor.float()


@MODELS.register_module()
class ConvMapEncoder(BaseModule):
    """Convolution Backbone Map Encoder with Integrated Edge Residual Group.

    Args:
        conv_group_cfg (dict): The config of convolution group.
            - in_channels (int): The input channels of this module.
            - out_channels (list[int]): The output channels of this module.
            - mid_channels (list[int]): The input channels of the second convolution.
            - strides (list[int]): The stride of the first convolution. Defaults to 1.
            - kernel_size (int): The kernel size of the first convolution. Defaults to 3.
            - se_cfg (dict, optional): Config dict for se layer. Defaults to None.
            - with_residual (bool): Use residual connection. Defaults to True.
            - conv_cfg (dict, optional): Config dict for convolution layer. Defaults to None.
            - norm_cfg (dict): Config dict for normalization layer. Defaults to ``dict(type='BN')``.
            - act_cfg (dict): Config dict for activation layer. Defaults to ``dict(type='ReLU')``.
            - drop_path_rate (float): Stochastic depth rate. Defaults to 0.
            - with_cp (bool): Use checkpoint or not. Defaults to False.
            - init_cfg (dict | list[dict], optional): Initialization config dict.
        stream_name (str, optional): Stream name. Defaults to 'map'.
    """

    def __init__(
            self,
            conv_group_cfg=dict(
                in_channels=6,
                out_channels=[32, 64],
                mid_channels=[16, 32],
                strides=[2, 2],
                dilation=3,
                kernel_size=3,
                with_se=False,
                with_residual=True,
                conv_cfg=None,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU'),
                drop_path_rate=0.,
                with_cp=False,
                init_cfg=None
            ),
            stream_name="map"):
        super(ConvMapEncoder, self).__init__(stream_name)

        # Extract parameters from conv_group_cfg
        self.in_channels = conv_group_cfg['in_channels']
        self.out_channels = conv_group_cfg['out_channels']
        self.mid_channels = conv_group_cfg['mid_channels']
        self.strides = conv_group_cfg['strides']
        self.dilation = conv_group_cfg['dilation']
        self.kernel_size = conv_group_cfg['kernel_size']
        self.with_se = conv_group_cfg.get('with_se', None)
        self.with_residual = conv_group_cfg.get('with_residual', True)
        self.conv_cfg = conv_group_cfg.get('conv_cfg', None)
        self.norm_cfg = conv_group_cfg.get('norm_cfg', dict(type='BN'))
        self.act_cfg = conv_group_cfg.get('act_cfg', dict(type='ReLU'))
        self.drop_path_rate = conv_group_cfg.get('drop_path_rate', 0.)
        self.with_cp = conv_group_cfg.get('with_cp', False)
        self.init_cfg = conv_group_cfg.get('init_cfg', None)

        self.num_layers = len(self.mid_channels)
        if self.with_se:
            self.se_cfgs = [dict(in_channels=c) for c in self.mid_channels]
        else:
            self.se_cfgs = [None for _ in range(self.num_layers)]

        # Build the Edge Residual Group
        self.conv_group = self._build_edge_residual_group(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            mid_channels=self.mid_channels,
            strides=self.strides,
            dilation=self.dilation,
            kernel_size=self.kernel_size,
            se_cfgs=self.se_cfgs,
            with_residual=self.with_residual,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            drop_path_rate=self.drop_path_rate,
            with_cp=self.with_cp,
            init_cfg=self.init_cfg
        )

    def _build_edge_residual_group(self,
                                  in_channels,
                                  out_channels,
                                  mid_channels,
                                  strides,
                                  dilation,
                                  kernel_size=3,
                                  se_cfgs=None,
                                  with_residual=True,
                                  conv_cfg=None,
                                  norm_cfg=dict(type='BN'),
                                  act_cfg=dict(type='ReLU'),
                                  drop_path_rate=0.,
                                  with_cp=False,
                                  init_cfg=None):
        """Build the Edge Residual Group."""

        assert len(out_channels) == len(mid_channels) == len(strides)
        convs = nn.ModuleList()
        for i, (out, mid, stride, se_cfg) in enumerate(zip(out_channels, mid_channels, strides, se_cfgs)):
            convs.append(
                self._build_edge_residual_block(
                    in_channels,
                    out,
                    mid,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    se_cfg=se_cfg,
                    with_residual=with_residual,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    drop_path_rate=drop_path_rate,
                    with_cp=with_cp,
                    init_cfg=init_cfg))
            in_channels = out
        return convs

    def _build_edge_residual_block(self,
                                in_channels,
                                out_channels,
                                mid_channels,
                                kernel_size=3,
                                stride=1,
                                dilation=1,
                                se_cfg=None,
                                with_residual=True,
                                conv_cfg=None,
                                norm_cfg=dict(type='BN'),
                                act_cfg=dict(type='ReLU'),
                                drop_path_rate=0.,
                                with_cp=False,
                                init_cfg=None):
        """Build a single Edge Residual block with shortcut if needed."""
        # 如果需要下采样或者通道数不匹配，则使用1x1卷积作为shortcut
        if stride != 1 or in_channels != out_channels:
            shortcut = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
        else:
            shortcut = nn.Identity()
        
        with_se = se_cfg is not None

        conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            padding="same",
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        se_layer = SELayer(**se_cfg) if with_se else None

        conv2 = ConvModule(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            dilation=dilation,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        act = build_activation_layer(act_cfg)
        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        block = nn.ModuleDict({
            'conv1': conv1,
            'se_layer': se_layer,
            'conv2': conv2,
            'shortcut': shortcut,   # shortcut branch
            'act': act,
            'drop_path': drop_path,
        })
        return block


    def forward(self, data, metas=None):
        # 如果 data 不是 dict，则假设已经是 tensor
        if isinstance(data, dict):
            x = gather_map(data)
        else:
            x = data
        for conv_block in self.conv_group:
            def _inner_forward(x):
                identity = x  # 保存输入
                out = conv_block['conv1'](x)
                if conv_block['se_layer'] is not None:
                    out = conv_block['se_layer'](out)
                out = conv_block['conv2'](out)
                return conv_block['shortcut'](identity) + conv_block['drop_path'](out)
            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(_inner_forward, x)
            else:
                x = _inner_forward(x)
            x = conv_block['act'](x)
        return x
