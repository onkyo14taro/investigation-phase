"""Module for complex neural networks."""


import torch
import torch.nn as nn

from complexfunctional import real_to_cmplx, cmplx_to_real


__all__ = [
    'ComplexBatchNorm',
]


####################################################################################################
####################################################################################################
### Batch Normalizations
####################################################################################################
####################################################################################################
# A modified version of the code at the following URL:
# https://github.com/IMLHF/SE_DCUNet/blob/f28bf1661121c8901ad38149ea827693f1830715/models/layers/complexnn.py#L55
# The changes are as follows:
# 1. Input and output native complex tensors.
# 2. Set the initial value of the non-diagonal component of the weight matrix of the affine transform to 0.
#    That is, initially, the affine transform is an identity transform.
# 3. Normalize so that the variance of the real and imaginary parts is 0.5,
#    that is, the variance in the meaning of complex number is 1
#    (If the variances of the real and imaginary parts are each set to 1,
#     the variance in the meaning of complex numbers will be 2).
class ComplexBatchNorm(nn.Module):
    r"""Applies Complex Batch Normalization [1] over a N-D input.

    [1] C. Trabelsi et al., “Deep Complex Networks,”
        6th International Conference on Learning Representations, ICLR, 2018, [Online].
        Available: http://arxiv.org/abs/1705.09792.

    Parameters
    ----------
    num_features : int
        Size of the dimension next to the batch size (``x.size(1)``).
    eps : float, optional
        Value added to the denominator for numerical stability, by default 5e-6
    momentum : float, optional
        Value used for the running_mean and running_var computation.
        Can be set to ``None`` for cumulative moving average (i.e. simple average).
        By default 0.1.
    affine : bool, optional
        Boolean value that when set to ``True``, this module has learnable
        affine parameters. By default ``True``.
    track_running_stats : bool, optional
        Boolean value that when set to ``True``, this module tracks the running mean and variance,
        and when set to ``False``, this module does not track such statistics, and initializes
        statistics buffers ``running_mean`` and ``running_var`` as ``None``.
        When these buffers are ``None``, this module always uses batch statistics
        in both training and eval modes. By default ``True``.
    """
    def __init__(self, num_features:int, eps:float=5e-6, momentum:float=0.1,
                 affine:bool=True, track_running_stats:bool=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats 

        if self.affine:
            self.W_rr = nn.Parameter(torch.Tensor(num_features))
            self.W_ri = nn.Parameter(torch.Tensor(num_features))
            self.W_ii = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(2*num_features))
        else:
            self.register_parameter('W_rr', None)
            self.register_parameter('W_ri', None)
            self.register_parameter('W_ii', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(2*num_features))
            self.register_buffer('running_V_rr', torch.ones(num_features))
            self.register_buffer('running_V_ri', torch.zeros(num_features))
            self.register_buffer('running_V_ii', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_V_rr', None)
            self.register_parameter('running_V_ri', None)
            self.register_parameter('running_V_ii', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_V_rr.fill_(0.5)
            self.running_V_ri.zero_()
            self.running_V_ii.fill_(0.5)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.W_rr)
            nn.init.ones_(self.W_ii)
            nn.init.zeros_(self.W_ri)
            nn.init.zeros_(self.bias)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        training = self.training or not self.track_running_stats
        redux = [i for i in range(x.dim()) if i != 1]
        vdim = [1] * x.dim()
        vdim[1] = x.size(1)
        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            mean = x.mean(redux, keepdim=True)
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean.lerp_(mean.squeeze(), exponential_average_factor)
        else:
            mean = self.running_mean.view(vdim)
        x = x - mean
        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        #
        x_r, x_i = cmplx_to_real(x, dim=1)
        vdim_half = vdim.copy()
        vdim_half[1] //= 2
        if training:
            V_rr = (x_r  ** 2).mean(redux, keepdim=True)
            V_ri = (x_r * x_i).mean(redux, keepdim=True)
            V_ii = (x_i  ** 2).mean(redux, keepdim=True)
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_V_rr.lerp_(V_rr.squeeze(), exponential_average_factor)
                    self.running_V_ri.lerp_(V_ri.squeeze(), exponential_average_factor)
                    self.running_V_ii.lerp_(V_ii.squeeze(), exponential_average_factor)
        else:
            V_rr = self.running_V_rr.view(vdim_half)
            V_ri = self.running_V_ri.view(vdim_half)
            V_ii = self.running_V_ii.view(vdim_half)
        V_rr = V_rr + self.eps
        V_ri = V_ri
        V_ii = V_ii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau = V_rr + V_ii
        delta = torch.addcmul(V_rr * V_ii, V_ri, V_ri, value=-1)
        s = torch.sqrt(delta)
        t = torch.sqrt(tau + 2*s)

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        corrected_rst = (s * t).reciprocal() * (2**-0.5)  # 2**-0.5 makes "complex variance" (V[X] = E[|X|^2]) one.
        U_rr = (s + V_ii) * corrected_rst
        U_ii = (s + V_rr) * corrected_rst
        U_ri = (  - V_ri) * corrected_rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [W_rr W_ri][U_rr U_ri] [x_r] + [B_r]
        #     [W_ir W_ii][U_ir U_ii] [x_i]   [B_i]
        #
        if self.affine:
            W_rr = self.W_rr.view(vdim_half)
            W_ri = self.W_ri.view(vdim_half)
            W_ii = self.W_ii.view(vdim_half)
            Z_rr = (W_rr * U_rr) + (W_ri * U_ri)
            Z_ri = (W_rr * U_ri) + (W_ri * U_ii)
            Z_ir = (W_ri * U_rr) + (W_ii * U_ri)
            Z_ii = (W_ri * U_ri) + (W_ii * U_ii)
        else:
            Z_rr, Z_ri, Z_ir, Z_ii = U_rr, U_ri, U_ri, U_ii

        y_r = (Z_rr * x_r) + (Z_ri * x_i)
        y_i = (Z_ir * x_r) + (Z_ii * x_i)
        y = real_to_cmplx(y_r, y_i, dim=1)

        if self.affine:
            y = y + self.bias.view(vdim)

        return y

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
                'track_running_stats={track_running_stats}'.format(**self.__dict__)
