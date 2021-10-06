# PyTorch implementation of TensorFlow's sparse_image_warp [1].
# This implementation is modified from [1, 2].
# 
# [1] https://github.com/tensorflow/tensorflow
#     Apache License 2.0
#     https://opensource.org/licenses/Apache-2.0
# [2] https://github.com/zcaceres/spec_augment
#     Copyright (c) 2019 Zach Caceres
#     MIT License
#     https://opensource.org/licenses/MIT

"""Module with functions related to sparse_image_warp."""


import einops
import torch


__all__ = [
    'sparse_image_warp',
    'sparse_image_warp_along_x',
    'sparse_image_warp_along_y',
]

################################################################################
################################################################################
### Computationally efficient functions for SpecAugment.
################################################################################
################################################################################
def sparse_image_warp_along_x(image:torch.Tensor,
                              source_control_point_locations:torch.Tensor,
                              dest_control_point_locations:torch.Tensor,
                              interpolation_order:int=2,
                              regularization_weight=0.0) -> torch.Tensor:
    """Image warping using correspondences between sparse control points along x-axis.

    See [1] for the original reference (Note that the tensor shape is different, etc.).

    [1] https://www.tensorflow.org/addons/api_docs/python/tfa/image/sparse_image_warp

    Parameters
    ----------
    image : torch.Tensor [shape=(batch_size, channels, height, width)]
    source_control_point_locations : torch.Tensor [shape=(batch_size, n_queries)]
    dest_control_point_locations : torch.Tensor [shape=(batch_size, n_queries)]
    interpolation_order : int, optional, by default 2
    regularization_weight : float, optional, by default 0.0

    Returns
    -------
    warped_img : torch.Tensor [shape=(batch_size, channels, height, width)]
    """
    return sparse_image_warp_along_axis(
        image, dim=3, source_control_point_locations=source_control_point_locations,
        dest_control_point_locations=dest_control_point_locations,
        interpolation_order=interpolation_order,
        regularization_weight=regularization_weight,
    )


def sparse_image_warp_along_y(image:torch.Tensor,
                              source_control_point_locations:torch.Tensor,
                              dest_control_point_locations:torch.Tensor,
                              interpolation_order:int=2,
                              regularization_weight=0.0) -> torch.Tensor:
    """Image warping using correspondences between sparse control points along y-axis.

    See [1] for the original reference (Note that the tensor shape is different, etc.).

    [1] https://www.tensorflow.org/addons/api_docs/python/tfa/image/sparse_image_warp

    Parameters
    ----------
    image : torch.Tensor [shape=(batch_size, channels, height, width)]
    source_control_point_locations : torch.Tensor [shape=(batch_size, n_queries)]
    dest_control_point_locations : torch.Tensor [shape=(batch_size, n_queries)]
    interpolation_order : int, optional, by default 2
    regularization_weight : float, optional, by default 0.0

    Returns
    -------
    warped_img : torch.Tensor [shape=(batch_size, channels, height, width)]
    """
    return sparse_image_warp_along_axis(
        image, dim=2, source_control_point_locations=source_control_point_locations,
        dest_control_point_locations=dest_control_point_locations,
        interpolation_order=interpolation_order,
        regularization_weight=regularization_weight,
    )


def sparse_image_warp_along_axis(image:torch.Tensor, dim:int,
                                 source_control_point_locations:torch.Tensor,
                                 dest_control_point_locations:torch.Tensor,
                                 interpolation_order:int=2,
                                 regularization_weight=0.0) -> torch.Tensor:
    """Image warping using correspondences between sparse control points along x- or y-axis.

    See [1] for the original reference (Note that the tensor shape is different, etc.).

    [1] https://www.tensorflow.org/addons/api_docs/python/tfa/image/sparse_image_warp

    Parameters
    ----------
    image : torch.Tensor [shape=(batch_size, channels, height, width)]
    source_control_point_locations : torch.Tensor [shape=(batch_size, n_queries)]
    dest_control_point_locations : torch.Tensor [shape=(batch_size, n_queries)]
    interpolation_order : int, optional, by default 2
    regularization_weight : float, optional, by default 0.0

    Returns
    -------
    warped_img : torch.Tensor [shape=(batch_size, channels, height, width)]
    """
    if dim == 2:
        other_dim = 3
    elif dim == 3:
        other_dim = 2
    else:
        raise ValueError('dim must be 2 (height) or 3 (width).')

    batch_size = image.size(0)
    dim_size = image.size(dim)
    other_dim_size = image.size(other_dim)

    control_point_flows = (dest_control_point_locations - source_control_point_locations).unsqueeze(2)  # shape=(batch_size, n_queries, 1)
    dest_control_point_locations = dest_control_point_locations.unsqueeze(2)  # shape=(batch_size, n_queries, 1)
    dim_locations = torch.arange(dim_size, device=image.device, requires_grad=False).float().unsqueeze(1)  # shape=(dim_size, 1)
    dim_locations = einops.repeat(dim_locations, 'd x -> b d x', b=batch_size)  # shape=(batch_size, dim_size, 1)
    dim_flows = interpolate_spline(
        dest_control_point_locations,  # shape=(batch_size, n=n_queries, d=1)
        control_point_flows,  # shape=(batch_size, n=n_queries, k=1)
        dim_locations,  # shape=(batch_size, m=dim_size, d=1)
        interpolation_order,
        regularization_weight
    )  # -> shape=(batch_size, m=dim_size, k=1)
    dim_flows = dim_flows.squeeze(2)  # shape=(batch_size, dim_size)

    other_dim_flows = torch.zeros(other_dim_size, device=image.device, requires_grad=False).float()  # shape=(other_dim_size, )
    other_dim_flows = einops.repeat(other_dim_flows, 'd -> b d', b=batch_size)  # shape=(batch_size, other_dim_size)

    y_flows = dim_flows if dim == 2 else other_dim_flows  # shape=(batch_size, height)
    x_flows = dim_flows if dim == 3 else other_dim_flows  # shape=(batch_size, width)
    height = dim_size if dim == 2 else other_dim_size
    width  = dim_size if dim == 3 else other_dim_size
    dense_flows = torch.stack((
        einops.repeat(y_flows, 'b h -> b h w', w=width),
        einops.repeat(x_flows, 'b w -> b h w', h=height)
    ), dim=1)  # shape=(batch_size, 2, height, width)

    warped_image = dense_image_warp(image, dense_flows)
    return warped_image, dense_flows


################################################################################
################################################################################
### /tensorflow_addons/image/sparse_image_warp.py
### https://github.com/tensorflow/addons/blob/v0.12.0/tensorflow_addons/image/sparse_image_warp.py
################################################################################
################################################################################
def sparse_image_warp(image:torch.Tensor,
                      source_control_point_locations:torch.Tensor,
                      dest_control_point_locations:torch.Tensor,
                      interpolation_order:int=2,
                      regularization_weight=0.0) -> torch.Tensor:
    """Image warping using correspondences between sparse control points.

    See [1] for the original reference (Note that the tensor shape is different, etc.).

    [1] https://www.tensorflow.org/addons/api_docs/python/tfa/image/sparse_image_warp

    Parameters
    ----------
    image : torch.Tensor [shape=(batch_size, channels, height, width)]
    source_control_point_locations : torch.Tensor [shape=(batch_size, n_queries, 2)]
    dest_control_point_locations : torch.Tensor [shape=(batch_size, n_queries, 2)]
    interpolation_order : int, optional, by default 2
    regularization_weight : float, optional, by default 0.0

    Returns
    -------
    warped_img : torch.Tensor [shape=(batch_size, channels, height, width)]
    """
    control_point_flows = (dest_control_point_locations - source_control_point_locations)  # shape=(batch_size, n_queries, 2)

    batch_size, image_channels, image_height, image_width = image.size()
    grid_locations = _get_grid_locations(image_height, image_width, image.device)  # shape=(height, width, 2)
    flattened_grid_locations = einops.rearrange(grid_locations, 'h w x -> (h w) x')  # shape=(height*width, 2)
    flattened_grid_locations = einops.repeat(flattened_grid_locations, 'hw x -> b hw x', b=batch_size)  # shape=(batch_size, height*width, 2)
    flattened_grid_locations = flattened_grid_locations.type(image.dtype)

    flattened_flows = interpolate_spline(
        dest_control_point_locations,  # shape=(batch_size, n=n_queries, d=2)
        control_point_flows,  # shape=(batch_size, n=n_queries, k=2)
        flattened_grid_locations,  # shape=(batch_size, m=height*width, d=2)
        interpolation_order,
        regularization_weight
    )  # -> # shape=(batch_size, m=height*width, k=2)

    # shape=(batch_size, height*width, 2) -> shape=(batch_size, 2, height, width)
    dense_flows = einops.rearrange(flattened_flows, 'b (h w) x -> b x h w')
    warped_image = dense_image_warp(image, dense_flows)
    return warped_image, dense_flows


def _get_grid_locations(image_height, image_width, device):
    """Wrapper for `torch.meshgrid`."""
    y_range = torch.arange(0., image_height, device=device, requires_grad=False)
    x_range = torch.arange(0., image_width, device=device, requires_grad=False)
    y_grid, x_grid = torch.meshgrid(y_range, x_range)
    return torch.stack((y_grid, x_grid), dim=-1)  # shape=(height, width, 2)


################################################################################
################################################################################
### /tensorflow_addons/image/interpolate_spline.py
### https://github.com/tensorflow/addons/blob/v0.12.0/tensorflow_addons/image/interpolate_spline.py
################################################################################
################################################################################
def interpolate_spline(train_points:torch.Tensor,
                       train_values:torch.Tensor,
                       query_points:torch.Tensor,
                       order:int,
                       regularization_weight=0.0) -> torch.Tensor:
    """Interpolate signal using polyharmonic interpolation.

    See [1] for the original reference (Note that the tensor shape is different, etc.).

    [1] https://www.tensorflow.org/addons/api_docs/python/tfa/image/interpolate_spline

    Parameters
    ----------
    train_points : torch.Tensor [shape=(batch_size, n, d)]
    train_values : torch.Tensor [shape=(batch_size, n, k)]
    query_points : torch.Tensor [shape=(batch_size, m, d)]
    order : int
    regularization_weight : float, optional

    Returns
    -------
    query_values : torch.Tensor [shape=(b, m, k)]
    """
    # First, fit the spline to the observed data.
    w, v = _solve_interpolation(train_points, train_values, order, regularization_weight)
    # Then, evaluate the spline at the query locations.
    query_values = _apply_interpolation(query_points, train_points, w, v, order)
    return query_values


def _cross_squared_distance_matrix(x:torch.Tensor, y:torch.Tensor):
    """Pairwise squared distance between two (batch) matrices' rows (2nd dim).

    Computes the pairwise distances between rows of x and rows of y.

    Args:
        x: torch.Tensor [shape=(b, n, d), float]
        y: torch.Tensor [shape=(b, m, d), float]
    Returns:
        squared_dists: torch.Tensor [shape=(b, n, m), float]
            `squared_dists[b,i,j] = ||x[b,i,:] - y[b,j,:]||^2`.
    """
    x_norm_squared = torch.sum(x**2, dim=2).unsqueeze(2)  # shape=(b, n, 1)
    y_norm_squared = torch.sum(y**2, dim=2).unsqueeze(1)  # shape=(b, 1, m)
    x_y_transpose = torch.einsum('bnd, bmd -> bnm', x, y)  # shape=(b, n, m)
    # squared_dists[b,i,j] = ||x_bi - y_bj||^2 = x_bi'x_bi- 2x_bi'x_bj + x_bj'x_bj
    squared_dists = x_norm_squared - 2*x_y_transpose + y_norm_squared
    return squared_dists.float()  # shape=(b, n, m)


def _pairwise_squared_distance_matrix(x:torch.Tensor) -> torch.Tensor:
    """Pairwise squared distance among a (batch) matrix's rows (2nd dim).

    This saves a bit of computation vs. using
    `_cross_squared_distance_matrix(x, x)`

    Args:
        x: torch.Tensor [shape=(b, n, d), float]
    Returns:
        squared_dists: torch.Tensor [shape=(b, n, n), float]
            `squared_dists[b,i,j] = ||x[b,i,:] - x[b,j,:]||^2`.
    """
    x_x_transpose = torch.einsum('bnd, bmd -> bnm', x, x)  # shape=(b, n, n)
    x_norm_squared = torch.diagonal(x_x_transpose, dim1=1, dim2=2).unsqueeze(2)  # shape=(b, n, 1)
    # squared_dists[b,i,j] = ||x_bi - x_bj||^2 = x_bi'x_bi- 2x_bi'x_bj + x_bj'x_bj
    squared_dists = x_norm_squared - 2*x_x_transpose + x_norm_squared.transpose(1, 2)
    return squared_dists  # shape=(b, n, n)


def _solve_interpolation(train_points:torch.Tensor,
                         train_values:torch.Tensor,
                         order:int,
                         regularization_weight:float):
    """Solve for interpolation coefficients.

    Computes the coefficients of the polyharmonic interpolant for the
    'training' data defined by `(train_points, train_values)` using the kernel
    $\phi$.

    Args:
        train_points: torch.Tensor [shape=(b, n, d)]; interpolation centers.
        train_values: torch.Tensor [shape=(b, n, k)]; function values.
        order: int; order of the interpolation.
        regularization_weight: float; weight to place on smoothness regularization term.
    Returns:
        w: torch.Tensor [shape=(b, n, k)]; weights on each interpolation center
        v: torch.Tensor [shape=(b, d, k)]; weights on each input dimension
    Raises:
      ValueError: if d or k is not fully specified.
    """
    b, n, d = train_points.size()
    k = train_values.size(-1)

    # First, rename variables so that the notation (c, f, w, v, A, B, etc.)
    # follows https://en.wikipedia.org/wiki/Polyharmonic_spline.
    # To account for python style guidelines we use
    # matrix_a for A and matrix_b for B.

    c = train_points  # [b, n, d]
    f = train_values.float()  # [b, n, k]

    matrix_a = _phi(_pairwise_squared_distance_matrix(c), order)  # [b, n, n]
    if regularization_weight > 0:
        batch_identity_matrix = torch.eye(n, dtype=c.dtype).unsqueeze(0)  # [1, n, n]
        matrix_a += regularization_weight * batch_identity_matrix

    # Append ones to the feature values for the bias term in the linear model.
    ones = torch.ones_like(c[..., :1])  # [b, n, 1]
    matrix_b = torch.cat((c, ones), 2).float()  # [b, n, d + 1]
    left_block = torch.cat((matrix_a, matrix_b.transpose(2, 1)), 1)  # [b, n + d + 1, n]

    num_b_cols = matrix_b.size(2)  # d + 1
    # In Tensorflow, zeros are used here. Pytorch gesv fails with zeros for some reason we don't understand.
    # So instead we use very tiny randn values (variance of one, zero mean) on one side of our multiplication.
    lhs_zeros = torch.randn((b, num_b_cols, num_b_cols)) * 1e-10
    right_block = torch.cat((matrix_b, lhs_zeros), 1)  # [b, n + d + 1, d + 1]

    lhs = torch.cat((left_block, right_block), 2)  # [b, n + d + 1, n + d + 1]

    rhs_zeros = torch.zeros((b, d + 1, k), dtype=train_points.dtype).float()
    rhs = torch.cat((f, rhs_zeros), 1)  # [b, n + d + 1, k]

    # Then, solve the linear system and unpack the results.
    X, LU = torch.solve(rhs, lhs)  # X.shape=(b, n + d + 1, k)
    w = X[:, :n, :]  # [b, n, k]
    v = X[:, n:, :]  # [b, d + 1, k]

    return w, v


def _phi(r:torch.Tensor, order:int) -> torch.Tensor:
    """Coordinate-wise nonlinearity used to define the order of the
    interpolation.

    See https://en.wikipedia.org/wiki/Polyharmonic_spline for the definition.

    Args:
        r: torch.Tensor; input op.
        order : int; interpolation order.
    Returns:
        `phi_k` evaluated coordinate-wise on `r`, for `k = r`.
    """
    EPSILON = torch.tensor(1e-10)
    # using EPSILON prevents log(0), sqrt0), etc.
    # sqrt(0) is well-defined, but its gradient is not
    if order == 1:
        r = torch.max(r, EPSILON)
        r = torch.sqrt(r)
        return r
    elif order == 2:
        return 0.5 * r * torch.log(torch.max(r, EPSILON))
    elif order == 4:
        return 0.5 * torch.square(r) * torch.log(torch.max(r, EPSILON))
    elif order % 2 == 0:
        r = torch.max(r, EPSILON)
        return 0.5 * torch.pow(r, 0.5 * order) * torch.log(r)
    else:
        r = torch.max(r, EPSILON)
        return torch.pow(r, 0.5 * order)


def _apply_interpolation(query_points:torch.Tensor,
                         train_points:torch.Tensor,
                         w:torch.Tensor, v:torch.Tensor, order:int) -> torch.Tensor:
    """Apply polyharmonic interpolation model to data.

    Given coefficients w and v for the interpolation model, we evaluate
    interpolated function values at query_points.
    Args:
        query_points: torch:Tensor [shape=(b, m, d)]
            x values to evaluate the interpolation at.
        train_points: torch:Tensor [shape=(b, n, d)]
            x values that act as the interpolation centers
            (the c variables in the wikipedia article).
        w: torch:Tensor [shape=(b, n, k)]
            weights on each interpolation center.
        v: torch:Tensor [shape=(b, d + 1, k)]
            weights on each input dimension.
        order: order of the interpolation.
    Returns:
        torch:Tensor [shape=(b, m, k)]
            Polyharmonic interpolation evaluated at points defined in `query_points`.
    """
    # First, compute the contribution from the rbf term.
    pairwise_dists = _cross_squared_distance_matrix(query_points.float(), train_points.float())  # shape=(b, m, n)
    phi_pairwise_dists = _phi(pairwise_dists, order)  # shape=(b, m, n)
    rbf_term = torch.einsum('bmn, bnk -> bmk', phi_pairwise_dists, w)  # shape=(b, m, k)

    # Then, compute the contribution from the linear term.
    # Pad query_points with ones, for the bias term in the linear model.
    ones = torch.ones_like(query_points[..., :1])
    query_points_pad = torch.cat((query_points, ones), 2).float()  # shape=(b, m, d+1)
    linear_term = torch.einsum('bmd, bdk -> bmk', query_points_pad, v)  # shape=(b, m, k)

    return rbf_term + linear_term


################################################################################
################################################################################
### /tensorflow_addons/image/dense_image_warp.py
### https://github.com/tensorflow/addons/blob/v0.12.0/tensorflow_addons/image/dense_image_warp.py
################################################################################
################################################################################
def dense_image_warp(image:torch.Tensor, flow:torch.Tensor) -> torch.Tensor:
    """Image warping using per-pixel flow vectors.

    See [1] for the original reference (Note that the tensor shape is different, etc.).

    [1] https://www.tensorflow.org/addons/api_docs/python/tfa/image/dense_image_warp

    Parameters
    ----------
    image : torch.Tensor [shape=(batch, channels, height, width)]
    flow : torch.Tensor [shape=(batch, 2, height, width)]

    Returns
    -------
    warped_image : torch.Tensor [shape=(batch, channels, height, width)]
    """
    batch_size, channels, height, width = image.shape
    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    y_range = torch.arange(0., height, device=image.device, requires_grad=False)
    x_range = torch.arange(0., width, device=image.device, requires_grad=False)
    y_grid, x_grid = torch.meshgrid(y_range, x_range)
    stacked_grid = torch.stack((y_grid, x_grid), dim=0)  # shape=(2, height, width)
    batched_grid = stacked_grid.unsqueeze(0)  # shape=(1, 2, height, width)
    query_points_on_grid = batched_grid - flow  # shape=(batch_size, 2, height, width)
    query_points_flattened = einops.rearrange(query_points_on_grid, 'b x h w -> b (h w) x')  # shape=(batch_size, height * width, 2)
    # Compute values at the query points, then reshape the result back to the
    # image grid.
    interpolated = interpolate_bilinear(image, query_points_flattened)  # shape=(batch_size, channels, n_queries)
    interpolated = einops.rearrange(interpolated, 'b c (h w) -> b c h w', h=height, w=width)
    return interpolated


################################################################################
################################################################################
### /tensorflow_addons/image/interpolate_bilinear.py
### https://github.com/tensorflow/addons/blob/v0.12.0/tensorflow_addons/image/interpolate_bilinear.py
################################################################################
################################################################################
def interpolate_bilinear(grid:torch.Tensor,
                         query_points:torch.Tensor,
                         indexing:str="ij") -> torch.Tensor:
    """Similar to Matlab's interp2 function.

    Finds values for query points on a grid using bilinear interpolation.
    See [1] for the original reference (Note that the tensor shape is different, etc.).

    [1] https://www.tensorflow.org/addons/api_docs/python/tfa/image/interpolate_bilinear

    Parameters
    ----------
    grid : torch.Tensor [shape=(batch_size, channels, height, width)]
    query_points : torch.Tensor [shape=(batch_size, n_queries, 2)]
    indexing : str
        whether the query points are specified as row and column (ij),
        or Cartesian coordinates (xy).

    Returns
    -------
    query_values : torch.Tensor [shape=(batch_size, channels, n_queries)]
    """
    if indexing != "ij" and indexing != "xy":
        raise ValueError("Indexing mode must be 'ij' or 'xy'")
    if grid.ndim != 4:
        raise ValueError("grid must be 4D Tensor")
    if query_points.ndim != 3:
        raise ValueError("query_points must be 3 dimensional.")

    n_queries = query_points.size(1)

    alphas = []
    floors = []
    ceils = []
    index_order = [0, 1] if indexing == 'ij' else [1, 0]
    unstacked_query_points = query_points.unbind(2)

    for i, dim in enumerate(index_order):  # height -> width
        queries = unstacked_query_points[dim]  # shape=(batch_size, n_queries)
        size_in_indexing_dimension = grid.size(i+2)  # height or width

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = torch.tensor(size_in_indexing_dimension - 2, dtype=query_points.dtype)
        min_floor = torch.tensor(0.0, dtype=query_points.dtype)
        floor = torch.min(torch.max(min_floor, torch.floor(queries)), max_floor).long()
        floors.append(floor.view(-1))  # shape=(batch_size * n_queries)
        ceil = floor + 1
        ceils.append(ceil.view(-1))  # shape=(batch_size * n_queries)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = (queries - floor).type(grid.dtype)
        min_alpha = torch.tensor(0.0, dtype=grid.dtype)
        max_alpha = torch.tensor(1.0, dtype=grid.dtype)
        alpha = torch.min(torch.max(min_alpha, alpha), max_alpha)  # shape=(batch_size, n_queries)

        # Expand alpha to [b, 1, n] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = torch.unsqueeze(alpha, 1)  # shape=(batch_size, 1, n_queries)
        alphas.append(alpha)

    batch_size, channels, height, width = grid.size()
    flattened_grid = einops.rearrange(grid, 'b c h w -> (b h w) c')
    batch_indice = torch.arange(batch_size).repeat(n_queries, 1).t().reshape(-1)  # [0, ..., 0, 1, ..., 1, 2, ...]

    # This wraps array_ops.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using array_ops.gather_nd.
    def gather(y_coords:torch.Tensor, x_coords:torch.Tensor):
        gathered_values = grid[batch_indice, :, y_coords, x_coords]  # shape=(batch_size * n_queries, channels)
        return einops.rearrange(gathered_values, '(b q) c -> b c q', b=batch_size, q=n_queries)  # shape=(batch_size, channels, n_queries)

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1])
    top_right = gather(floors[0], ceils[1])
    bottom_left = gather(ceils[0], floors[1])
    bottom_right = gather(ceils[0], ceils[1])

    interp_top = alphas[1] * (top_right - top_left) + top_left
    interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
    interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp
