# This implementation is modified from [1].
#
# This script adds the complex model extension.
#
# [1] https://github.com/lukemelas/EfficientNet-PyTorch
#     Apache License 2.0
#     https://opensource.org/licenses/Apache-2.0
"""Module for EfficientNet."""


import collections
from functools import partial
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F



__all__ = [
    'EfficientNetEncoder',
]


################################################################################
################################################################################
### original: /efficientnet_pytorch/utils.py
### https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
################################################################################
################################################################################

################################################################################
# GlobalParams and BlockArgs
################################################################################
# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
    'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
    'drop_connect_rate', 'depth_divisor', 'min_depth'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

# Set GlobalParams and BlockArgs's defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


################################################################################
# Helper functions
################################################################################
def drop_connect(inputs:torch.Tensor, p:float, training:bool) -> torch.Tensor:
    r"""Drop connect.

    Parameters
    ----------
    input : torch.Tensor [shape=(batch_size, channels, height, width)]
        Input image tensor.
    p : float
        Probability of drop connection.
        Must satisfy: ``0 <= p <= 1``.
    training : bool
        Whether the mode is training or not.
        If set to ``True``, apply drop connection.

    Returns
    -------
    torch.Tensor [shape=(batch_size, channels, height, width)]
        output: Output after drop connection.
    """
    assert 0 <= p <= 1, '``p`` must be in range of [0, 1]'

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    binary_tensor = inputs.new_empty(
        [batch_size, 1, 1, 1], dtype=torch.bool, requires_grad=False
    ).bernoulli_(keep_prob).float()

    output = inputs / keep_prob * binary_tensor
    return output


def get_width_and_height_from_size(x:Union[int, Tuple, List]
    ) -> Tuple[int, int]:
    r"""Obtain height and width from x.

    Parameters
    ----------
    x : Union[int, Tuple, List]
        Data size.

    Returns
    -------
    height : int
    width : int
    """
    if isinstance(x, int):
        return x, x
    if isinstance(x, (list, tuple)):
        return tuple(x)
    else:
        raise TypeError(type(x))


def calculate_output_image_size(
        input_image_size:Union[int, Tuple, List],
        stride:Union[int, Tuple, List]
    ) -> List[int]:
    r"""Calculates the output image size when using Conv2dSamePadding with a stride.
       Necessary for static padding. Thanks to mannatsingh for pointing this out.

    Parameters
    ----------
    input_image_size : Union[int, Tuple, List]
        Size of input image.
    stride : Union[int, Tuple, List]
        Conv2d operation's stride.

    Returns
    -------
    output_image_size : List[int]
        A list [H,W].
    """
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


################################################################################
# Swish with complex versions
################################################################################
# Swish activation function
Swish = nn.SiLU


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i:torch.Tensor) -> torch.Tensor:
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output:torch.Tensor) -> torch.Tensor:
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return SwishImplementation.apply(x)


################################################################################
# Same padding convolutions with complex versions
################################################################################
def _calculate_same_padding(ih:int, iw:int, kh:int, kw:int, sh:int, sw:int, dh:int, dw:int) -> List[int]:
    oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
    pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
    pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
    return [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2]


def get_same_padding_conv2d(image_size:Optional[Union[int, Tuple[int, int], List[int]]]=None) -> type:
    r"""Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.

    Parameters
    ----------
    image_size : Optional[Union[int, Tuple[int, int], List[int]]], optional
        Size of the image.
        By default, ``None``.

    Returns
    -------
    conv_cls : type
        ``Conv2dDynamicSamePadding``, ``Conv2dStaticSamePadding``.
    """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    r"""2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=0, dilation=dilation,
                         groups=groups, bias=bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        dh, dw = self.dilation
        padding = _calculate_same_padding(ih, iw, kh, kw, sh, sw, dh, dw)
        if any(p > 0 for p in padding):
            x = F.pad(x, padding)
        return F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    r"""2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True,
                 image_size=None):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=0, dilation=dilation,
                         groups=groups, bias=bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        dh, dw = self.dilation
        padding = _calculate_same_padding(ih, iw, kh, kw, sh, sw, dh, dw)
        if any(p > 0 for p in padding):
            self.static_padding = nn.ZeroPad2d(padding)
        else:
            self.static_padding = nn.Identity()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)
        return x


################################################################################
# Helper functions for loading model params
################################################################################
def round_filters(filters:int, global_params:GlobalParams) -> int:
    r"""Calculate and round number of filters based on width multiplier.
       Use width_coefficient, depth_divisor and min_depth of global_params.

    Parameters
    ----------
    filters : int
        Filters number to be calculated.
    global_params : GlobalParams (namedtuple)
        Global params of the model.

    Returns
    -------
    new_filters : int
        New filters number after calculating.
    """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    # TODO: modify the params names.
    #       maybe the names (width_divisor,min_width)
    #       are more suitable than (depth_divisor,min_depth).
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor  # pay attention to this line when using min_depth
    # follow the formula transferred from official TensorFlow implementation
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats:int, global_params:GlobalParams) -> int:
    r"""Calculate module's repeat number of a block based on depth multiplier.
       Use depth_coefficient of global_params.

    Parameters
    ----------
    repeats : int
        num_repeat to be calculated.
    global_params : GlobalParams (namedtuple)
        Global params of the model.

    Returns
    -------
    new repeat : int
        New repeat number after calculating.
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    # follow the formula transferred from official TensorFlow implementation
    return int(math.ceil(multiplier * repeats))


class BlockDecoder(object):
    r"""Block Decoder for readability,
       straight from the official TensorFlow repository.
    """

    @staticmethod
    def _decode_block_string(block_string:str) -> BlockArgs:
        r"""Get a block through a string notation of arguments.

        Parameters
        ----------
        block_string : str
            A string notation of arguments.
            Examples: ``'r1_k3_s11_e1_i32_o16_se0.25_noskip'``.

        Returns
        -------
        block_args : BlockArgs (namedtuple)
            The namedtuple defined at the top of this file.
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            num_repeat=int(options['r']),
            kernel_size=int(options['k']),
            stride=[int(options['s'][0])],
            expand_ratio=int(options['e']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            se_ratio=float(options['se']) if 'se' in options else None,
            id_skip=('noskip' not in block_string))

    @staticmethod
    def _encode_block_string(block:BlockArgs) -> str:
        r"""Encode a block to a string.

        Parameters
        ----------
        block : BlockArgs (namedtuple)
            A BlockArgs type argument.

        Returns
        -------
        block_string : str
            A String form of BlockArgs.
        """
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list:List[str]) -> BlockArgs:
        r"""Decode a list of string notations to specify blocks inside the network.

        Parameters
        ----------
        string_list : List[str]
            A list of strings, each string is a notation of block.

        Returns
        -------
        blocks_args : BlockArgs (namedtuple)
            A list of BlockArgs namedtuples of block args.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args:BlockArgs) -> List[str]:
        r"""Encode a list of BlockArgs to a list of strings.

        Parameters
        ----------
        blocks_args : BlockArgs (namedtuple)
            A list of BlockArgs namedtuples of block args.

        Returns
        -------
        block_strings : List[str]
            A list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet_params(model_name:str) -> Tuple[float, float, int, float]:
    r"""Map EfficientNet model name to parameter coefficients.

    Parameters
    ----------
    model_name : str
        Model name to be queried.

    Returns
    -------
    width : float
    depth : float
    res : int
    dropout : float
    """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


def efficientnet(width_coefficient:Optional[float]=None,
                 depth_coefficient:Optional[float]=None,
                 image_size:Optional[Union[int, Tuple[int, int], List[int]]]=None,
                 dropout_rate:float=0.2, drop_connect_rate:float=0.2, num_classes:int=1000,
                 ) -> Tuple[BlockArgs, GlobalParams]:
    r"""Create BlockArgs and GlobalParams for efficientnet model.

    Parameters
    ----------
    width_coefficient : Optional[float], optional
    depth_coefficient : Optional[float], optional
    image_size : Optional[Union[int, Tuple[int, int], List[int]]], optional
    dropout_rate : float, optional
    drop_connect_rate : float, optional
    num_classes : int, optional

    Returns
    -------
    blocks_args : BlockArgs (namedtuple)
    global_params : GlobalParams (namedtuple)
    """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        image_size=image_size,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        drop_connect_rate=drop_connect_rate,
        depth_divisor=8,
        min_depth=None,
    )

    return blocks_args, global_params


def get_model_params(model_name:str, override_params:Dict[str, Any]) -> Tuple[BlockArgs, GlobalParams]:
    r"""Get the block args and global params for a given model name.

    Parameters
    ----------
    model_name : str
        Model's name.
    override_params : Dict[str, Any]
        A dict to modify global_params.

    Returns
    -------
    blocks_args : BlockArgs (namedtuple)
    global_params : GlobalParams (namedtuple)
    """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s,
            )
    else:
        raise NotImplementedError('model name is not pre-defined: {}'.format(model_name))
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


################################################################################
################################################################################
### original: /efficientnet_pytorch/model.py
### https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
################################################################################
################################################################################
VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',

    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2'
)


class MBConvBlock(nn.Module):
    r"""Mobile Inverted Residual Bottleneck Block.

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)

    Parameters
    ----------
    block_args : BlockArgs (namedtuple)
    global_params : GlobalParam (namedtuple)
    image_size : Optional[Union[int, Tuple[int, int], List[int]]]
        [image_height, image_width].
    """

    def __init__(self, block_args:BlockArgs, global_params:GlobalParams,
                 image_size:Optional[Union[int, Tuple[int, int], List[int]]]=None):
        super().__init__()
        self._block_args = block_args
        Bn2d = nn.BatchNorm2d
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = Bn2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            self._swish0 = MemoryEfficientSwish()
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = Bn2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish1 = MemoryEfficientSwish()
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))  # Real convolution
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
            self._swish_se = MemoryEfficientSwish()

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = Bn2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs:torch.Tensor, drop_connect_rate:Optional[bool]=None) -> torch.Tensor:
        r"""MBConvBlock's forward function.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor.
        drop_connect_rate : Optional[bool], optional
            Drop connect rate (float, between 0 and 1).

        Returns
        -------
        outputs : torch.Tensor
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish0(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish1(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = x
            x_squeezed = F.adaptive_avg_pool2d(x_squeezed, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish_se(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x_squeezed = torch.sigmoid(x_squeezed)
            x = x_squeezed * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient:bool=True):
        r"""Sets swish function as memory efficient (for training) or standard (for export).

        Parameters
        ----------
        memory_efficient : bool, optional
            Whether to use memory-efficient version of swish.
            By default ``True``.
        """
        self._swish0 = MemoryEfficientSwish() if memory_efficient else Swish()
        self._swish1 = MemoryEfficientSwish() if memory_efficient else Swish()
        if self.has_se:
            self._swish_se = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNetEncoder(nn.Module):
    r"""EfficientNetEncoder model."""

    def __init__(self, blocks_args:Optional[BlockArgs]=None,
                 global_params:Optional[GlobalParams]=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        Bn2d = nn.BatchNorm2d
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        self.in_channels = in_channels
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = Bn2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._swish0 = MemoryEfficientSwish()
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = Bn2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._swish1 = MemoryEfficientSwish()

        # set activation to memory efficient swish by default
        self.out_channels = out_channels

    def set_swish(self, memory_efficient:bool=True):
        r"""Sets swish function as memory efficient (for training) or standard (for export).

        Parameters
        ----------
        memory_efficient : bool, optional
            Whether to use memory-efficient version of swish.
            By default ``True``.
        """
        self._swish0 = MemoryEfficientSwish() if memory_efficient else Swish()
        self._swish1 = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs:torch.Tensor) -> torch.Tensor:
        r"""use convolution layer to extract feature.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor.

        Returns
        -------
        outputs : torch.Tensor
            Output of the final convolution layer in the efficientnet model.
        """
        # Stem
        x = self._swish0(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish1(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        r"""EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor.

        Returns
        -------
        outputs : torch.Tensor
            Output of this model after processing.
        """
        # Convolution layers
        return self.extract_features(inputs)

    @classmethod
    def from_name(cls, model_name:str, in_channels:int=3, **override_params):
        r"""Create an efficientnet model according to name.

        Parameters
        ----------
        model_name : str
            Name for efficientnet.
        in_channels : int
            Input data's channel number, by default 3.
        override_params (other key word params)
            Params to override model's global_params.
            Optional key:
                ``'width_coefficient'``, ``'depth_coefficient'``,
                ``'image_size'``, ``'dropout_rate'``,
                ``'num_classes'``, ``'batch_norm_momentum'``,
                ``'batch_norm_epsilon'``, ``'drop_connect_rate'``,
                ``'depth_divisor'``, ``'min_depth'``

        Returns
        -------
        model : EfficientNetEncoder
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        model.in_channels = in_channels
        return model

    @classmethod
    def get_image_size(cls, model_name:str) -> int:
        r"""Get the input image size for a given efficientnet model.

        Parameters
        ----------
        model_name : str
            Name for efficientnet.

        Returns
        -------
        image_size : int
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name:str):
        r"""Validates model name.

        Parameters
        ----------
        model_name : str
            Name for efficientnet.
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels:int):
        r"""Adjust model's first convolution layer to in_channels, if in_channels not equals 3.

        Parameters
        ----------
        in_channels : int
            Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)  # number of output channels
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
