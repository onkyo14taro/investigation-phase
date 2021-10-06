import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

import pytest


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import complexfunctional


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test_cmplx_to_real():
    for ndim in (1, 2, 3, 4):
        x = torch.rand(*[8, 6, 4, 2][:ndim], device=device)
        for dim in range(ndim):
            real1, imag1 = complexfunctional.cmplx_to_real(x, dim=dim)
            real2, imag2 = torch.chunk(x, 2, dim=dim)
            assert torch.all(real1 == real2)
            assert torch.all(imag1 == imag2)


@pytest.mark.parametrize(['ndim', 'dim'], [
    (1, 0), (2, 0), (2, 1), (3, 0), (3, 1),
    (3, 2), (4, 0), (4, 1), (4, 2), (4, 3),
])
@pytest.mark.xfail(raises=AssertionError)
def test_cmplx_to_real_fail(ndim, dim):
    for ndim in (1, 2, 3, 4):
        size = torch.tensor([8, 6, 4, 2][:ndim])
        for dim in range(ndim):
            size[dim] += 1
            x = torch.rand(*size, device=device)
            complexfunctional.cmplx_to_real(x, dim=dim)


def test_real_to_cmplx():
    for ndim in (1, 2, 3, 4):
        real = torch.rand(*[9, 7, 5, 3][:ndim], device=device)
        imag = torch.rand(*[9, 7, 5, 3][:ndim], device=device)
        for dim in range(ndim):
            cmplx1 = complexfunctional.real_to_cmplx(real, imag, dim=dim)
            cmplx2 = torch.cat([real, imag], dim=dim)
            assert torch.all(cmplx1 == cmplx2)


@pytest.mark.parametrize(['ndim', 'dim'], [
    (1, 0), (2, 0), (2, 1), (3, 0), (3, 1),
    (3, 2), (4, 0), (4, 1), (4, 2), (4, 3),
])
@pytest.mark.xfail(raises=AssertionError)
def test_real_to_cmplx_fail(ndim, dim):
    for ndim in (1, 2, 3, 4):
        size = torch.tensor([8, 6, 4, 2][:ndim])
        for dim in range(ndim):
            real = torch.rand(*size, device=device)
            size[torch.randint(0, ndim, (1,))[0]] += 1
            imag = torch.rand(*size, device=device)
            complexfunctional.real_to_cmplx(real, imag, dim=dim)
