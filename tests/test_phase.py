import itertools
import math
import os
import time
import sys

import numpy as np
import torch
import torch.nn.functional as F

import pytest


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import phase


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@pytest.mark.parametrize(['min', 'max'], [
    (None, None),
    (-0.5, None),
    (None, 0.5),
    (-0.5, 0.5),
])
def test_clamp_constraint(min, max):
    x = torch.randn(10000,).to(device).requires_grad_(True)
    y1 = phase.clamp_constraint(x, min=min, max=max)
    if min is None and max is None:
        y2 = x
    else:
        y2 = torch.clamp(x, min=min, max=max)
    y1.sum().backward()
    assert torch.all(y1 == y2)
    assert torch.all(x.grad == torch.ones_like(x))


@pytest.mark.parametrize(['min', 'max'], [
    (None, None),
    (-0.5, None),
    (None, 0.5),
    (-0.5, 0.5),
])
def test_clamp_constraint(min, max):
    x = torch.randn(10000).to(device).requires_grad_(True)
    y1 = phase.clamp_constraint(x, min=min, max=max)
    if min is None and max is None:
        y2 = x
    else:
        y2 = torch.clamp(x, min=min, max=max)
    y1.sum().backward()
    assert torch.all(y1 == y2)
    assert torch.all(x.grad == torch.ones_like(x))


def test_atan2_modified():
    input1 = torch.randn(10000).to(device)
    input1.masked_fill_(input1.abs() < torch.finfo(input1.dtype).tiny**0.5, 1.0).requires_grad_(True)
    other1 = torch.randn(10000).to(device)
    other1.masked_fill_(input1.abs() < torch.finfo(other1.dtype).tiny**0.5, 1.0).requires_grad_(True)
    y1 = torch.atan2(input1, other1)
    y1.sum().backward()
    input2 = input1.detach().requires_grad_(True)
    other2 = other1.detach().requires_grad_(True)
    y2 = phase.atan2_modified(input2, other2)
    y2.sum().backward()
    assert torch.all(y1 == y2)  # Both results are the same.
    assert torch.allclose(input1.grad, input2.grad)  # Both gradients are the same.
    assert torch.allclose(other1.grad, other2.grad)  # Both gradients are the same.

    # When the first and second arguments are both extremely small values,
    # correct the derivative value to be small so that it does not diverge.
    input1 = torch.tensor(1e-20).to(device).requires_grad_(True)
    other1 = torch.tensor(1e-20).to(device).requires_grad_(True)
    y1 = torch.atan2(input1, other1)
    y1.backward()
    input2 = input1.detach().requires_grad_(True)
    other2 = other1.detach().requires_grad_(True)
    y2 = phase.atan2_modified(input2, other2)
    y2.sum().backward()
    assert torch.all(y1 == y2)  # Both results are the same.
    assert torch.isinf(input1.grad) and torch.isinf(other1.grad)  # original: infinite
    assert not torch.isinf(input2.grad) and not torch.isinf(other2.grad)  # modified: finite

    # If atan2(0, 0), define the derivative to be 0.
    input1 = torch.tensor(0.0).to(device).requires_grad_(True)
    other1 = torch.tensor(0.0).to(device).requires_grad_(True)
    y1 = torch.atan2(input1, other1)
    y1.backward()
    input2 = input1.detach().requires_grad_(True)
    other2 = other1.detach().requires_grad_(True)
    y2 = phase.atan2_modified(input2, other2)
    y2.sum().backward()
    assert torch.all(y1 == y2)  # Both results are the same.
    assert torch.isnan(input1.grad) and torch.isnan(other1.grad)  # original: NaN
    assert input2.grad == other2.grad == 0.0  # modified: 0


def test_principal_angle():
    x1 = torch.tensor([-math.pi, math.pi, -4, -3.14, 0, 3.14, 4]).to(device).requires_grad_(True)
    y1 = (x1 + math.pi) % (2*math.pi) - math.pi
    y1.sum().backward()
    x2 = x1.detach().requires_grad_(True)
    y2 = phase.principal_angle(x2)
    y2.sum().backward()
    assert torch.all(y1[2:] == y2[2:])
    assert torch.all(y1[0] == -math.pi)
    assert torch.all(y2[0] == math.pi)
    assert torch.all(x1.grad == x2.grad)


@pytest.mark.parametrize('dim', [0, 1, 2])
def test_unwrap(dim):
    x = phase.principal_angle(torch.randn(100, 100, 100) * 10).to(device)  # Generate random phase.
    y1 = torch.from_numpy(np.unwrap(x.cpu().numpy(), axis=dim)).float().to(device)
    y2 = phase.unwrap(x, dim=dim)
    assert torch.allclose(y1, y2, atol=1e-3)


def naive_center_diff(input, dim:int):
    output = torch.zeros_like(input)
    length = input.size(dim)
    ndim = input.ndim
    output[phase._indice_along_dim(torch.arange(1,length-1), ndim, dim)] \
        = input[phase._indice_along_dim(torch.arange(2,length), ndim, dim)] \
        - input[phase._indice_along_dim(torch.arange(0,length-2), ndim, dim)]
    output[phase._indice_along_dim(torch.arange(1,length-1), ndim, dim)] *= 0.5
    output[phase._indice_along_dim(0, ndim, dim)] \
        = input[phase._indice_along_dim(1, ndim, dim)] \
        - input[phase._indice_along_dim(0, ndim, dim)]
    output[phase._indice_along_dim(length-1, ndim, dim)] \
        = input[phase._indice_along_dim(length-1, ndim, dim)] \
        - input[phase._indice_along_dim(length-2, ndim, dim)]
    return output


@pytest.mark.parametrize('dim', [0, 1, 2])
def test_unwrap_center_diff(dim):
    # Analytic test.
    if dim == 0:
        x = torch.tensor([-2., -1.,  1.,             2.,    math.pi,        -2.]).to(device)
        y1 = torch.tensor([1., 1.5, 1.5, (math.pi-1)/2., math.pi-2., math.pi-2.]).to(device)
        y2 = phase.unwrap_center_diff(x)
        assert torch.allclose(y1, y2)
    # Random test.
    for fn in (lambda x: x, F.leaky_relu):
        # Test up to 10 times until success, since it rarely produces different values at π boundaries.
        for _ in range(10):
            x1 = phase.principal_angle(torch.randn(100, 100, 100) * 10).to(device)  # Generate random phase.
            x1.requires_grad_(True)
            y1 = naive_center_diff(phase.unwrap(x1, dim=dim), dim=dim)
            fn(y1).sum().backward()
            x2 = x1.detach().requires_grad_(True)
            y2 = phase.unwrap_center_diff(x2, dim=dim)
            fn(y2).sum().backward()
            try:
                assert torch.allclose(y1, y2, atol=1e-5)
                assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
                return
            except AssertionError:
                pass
        assert False, "Congratulations! You've just had a rare phenomenon happen ten times in a row!"


def _naive_interp_nan_1d(input:np.ndarray):
    mask = np.isnan(input)
    x = np.nonzero(mask)[0]
    xp = np.nonzero(~mask)[0]
    if x.shape[0] == 0 or xp.shape[0] == 0:
        return input
    fp = input[xp]
    input[x] = np.interp(x, xp, fp)  # in-place
    return input


def naive_interp_nan_1d(input:torch.Tensor):
    device = input.device
    dtype = input.dtype
    input = input.detach().cpu().clone().numpy()
    output = np.apply_along_axis(_naive_interp_nan_1d, -1, input)
    return torch.from_numpy(output).to(device=device, dtype=dtype)


def test_interp_nan_1d():
    # Analytic test.
    for val in (0., 1., float('nan')):
        x = torch.tensor([
            [np.nan,    -1., np.nan,     1., np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [    1.,     2.,     3.,     4.,     5.,     6.],
            [np.nan,     2., np.nan, np.nan,     8., np.nan],
            [  -15., np.nan, np.nan, np.nan, np.nan,    15.],
        ]).to(device=device).requires_grad_(True)
        y1 = torch.tensor([
            [   -1.,    -1.,     0.,     1.,     1.,     1.],
            [   val,    val,    val,    val,    val,    val],
            [    1.,     2.,     3.,     4.,     5.,     6.],
            [    2.,     2.,     4.,     6.,     8.,     8.],
            [  -15.,    -9.,    -3.,     3.,     9.,    15.],
        ]).to(device=device)
        x_grad = torch.tensor([
            [    0.,    4.5,     0.,   16.5,     0.,     0.],
            [    0.,     0.,     0.,     0.,     0.,     0.],
            [    1.,     2.,     3.,     4.,     5.,     6.],
            [    0., 19./3.,     0.,     0., 44./3.,     0.],
            [    7.,     0.,     0.,     0.,     0.,    14.],
        ]).to(device=device)
        y2 = phase.interp_nan_1d(x, val_for_all_nan=val)
        (y2 * torch.arange(1., 1.+y2.size(1), device=device).view(1, -1)).sum().backward()
        assert torch.allclose(y1, y2, equal_nan=True)
        assert torch.allclose(x_grad, x.grad)

    # Random test.
    print(f'\ninterp_nan_1d speed test')
    for ndim in (1, 2, 3, 4):
        if ndim == 1:
            x = torch.rand(256 * 2 * 40 * 16000).to(device)
        elif ndim == 2:
            x = torch.rand(256 * 2 * 40, 16000).to(device)
        elif ndim == 3:
            x = torch.rand(256, 2 * 40, 16000).to(device)
        elif ndim == 4:
            x = torch.rand(256, 2, 40, 16000).to(device)
        x.masked_fill_(x >= 0.9, float('nan'))
        print(f'\nx.shape={x.shape}')
        start = time.time()
        y1 = naive_interp_nan_1d(x)
        end = time.time()
        print(f'ndim={ndim} Numpy (forward)             : {end-start:.6f} sec')
        start = time.time()
        y2 = phase.interp_nan_1d(x)
        end = time.time()
        print(f'ndim={ndim} Torch (forward; evaluation) : {end-start:.6f} sec')
        assert torch.allclose(y1, y2)
        x.requires_grad_(True)
        start = time.time()
        y2 = phase.interp_nan_1d(x)
        end = time.time()
        print(f'ndim={ndim} Torch (forward; training)   : {end-start:.6f} sec')
        assert torch.allclose(y1, y2)
        y2 = y2.sum()
        start = time.time()
        y2.backward()
        end = time.time()
        print(f'ndim={ndim} Torch (backward; training)  : {end-start:.6f} sec')
