"""
ONNX export for prosemble prototype-based models.

Converts a fitted model's predict function into an ONNX graph.
Only supports models whose distance function can be expressed with
standard ONNX operators.  Unsupported models raise
``NotImplementedError`` with a clear message.

Supported distance functions:

- ``squared_euclidean_distance_matrix``
- ``euclidean_distance_matrix``
- ``manhattan_distance_matrix``
- ``omega_distance_matrix`` (global projection matrix)
- ``lomega_distance_matrix`` (per-prototype local matrices)
- ``tangent_distance_matrix`` (per-prototype tangent subspace)
- ``relevance_weighted`` (per-feature relevance weighting)

Supported decision patterns:

- WTAC (supervised classification)
- ArgMin (unsupervised clustering)
- One-class hard nearest (OCGLVQ family)
- One-class Gaussian soft (OCRSLVQ family)
- One-class Gaussian+NG soft (OCRSLVQ_NG family)
- SVQ-OCC response model (SVQOCC family)
- CBC reasoning (CBC, ImageCBC)
- PLVQ Gaussian mixture soft assignment

Supported encoder models:

- MLP encoder (SiameseGLVQ, SiameseGMLVQ, SiameseGTLVQ, LVQMLN, PLVQ)
- CNN encoder (ImageGLVQ, ImageGMLVQ, ImageGTLVQ, ImageCBC)

Not supported:

- ``gaussian_kernel_matrix``, ``polynomial_kernel_matrix``
- Riemannian manifold distances (logm, expm have no ONNX equivalent)
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np


def _check_onnx_installed():
    """Raise ImportError with clear message if onnx is not installed."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise ImportError(
            "ONNX export requires the 'onnx' package. "
            "Install with: pip install prosemble[onnx]"
        )


# ---------------------------------------------------------------------------
# Numpy forward functions (for pre-computing latent prototypes at export time)
# ---------------------------------------------------------------------------

def _get_activation_np(name):
    """Return a numpy activation function by name."""
    if name == 'sigmoid':
        return lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    elif name == 'relu':
        return lambda z: np.maximum(0, z)
    elif name == 'tanh':
        return np.tanh
    elif name == 'leaky_relu':
        return lambda z: np.where(z > 0, z, 0.01 * z)
    elif name == 'selu':
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        return lambda z: scale * np.where(z > 0, z, alpha * (np.exp(z) - 1))
    else:
        raise ValueError(f"Unknown activation: {name}")


def _mlp_forward_np(params, x, activation='sigmoid'):
    """Numpy MLP forward pass (for pre-computing latent prototypes).

    Parameters
    ----------
    params : list of (weight, bias) tuples
        MLP parameters (JAX or numpy arrays).
    x : array of shape (n, d_in)
    activation : str

    Returns
    -------
    numpy array of shape (n, d_out)
    """
    act_fn = _get_activation_np(activation)
    x = np.asarray(x, dtype=np.float32)
    for w, b in params:
        x = act_fn(x @ np.asarray(w, dtype=np.float32)
                    + np.asarray(b, dtype=np.float32))
    return x


def _cnn_forward_np(params, x, activation='relu'):
    """Numpy CNN forward pass (for pre-computing latent prototypes).

    Parameters
    ----------
    params : dict with 'conv_layers' and 'linear'
    x : array of shape (N, H, W, C) — NHWC format
    activation : str

    Returns
    -------
    numpy array of shape (N, latent_dim)
    """
    act_fn = _get_activation_np(activation)
    x = np.asarray(x, dtype=np.float32)

    for kernel, bias in params['conv_layers']:
        kernel = np.asarray(kernel, dtype=np.float32)  # (kH, kW, C_in, C_out)
        bias = np.asarray(bias, dtype=np.float32)       # (C_out,)
        N, H, W, C_in = x.shape
        kH, kW = kernel.shape[:2]
        pH, pW = kH // 2, kW // 2
        # SAME padding
        x_pad = np.pad(x, ((0, 0), (pH, kH - 1 - pH),
                            (pW, kW - 1 - pW), (0, 0)))
        out = np.zeros((N, H, W, kernel.shape[3]), dtype=np.float32)
        for i in range(kH):
            for j in range(kW):
                out += np.einsum('nhwi,io->nhwo',
                                 x_pad[:, i:i + H, j:j + W, :],
                                 kernel[i, j])
        out += bias
        x = act_fn(out)

    # Global average pooling: (N, H, W, C) -> (N, C)
    x = np.mean(x, axis=(1, 2))

    # Linear head
    w, b = params['linear']
    x = act_fn(x @ np.asarray(w, dtype=np.float32)
               + np.asarray(b, dtype=np.float32))
    return x


# ---------------------------------------------------------------------------
# ONNX encoder builders
# ---------------------------------------------------------------------------

def _activation_node(input_name, output_name, activation):
    """Create a single ONNX activation node."""
    import onnx.helper as oh

    if activation == 'sigmoid':
        return oh.make_node('Sigmoid', [input_name], [output_name])
    elif activation == 'relu':
        return oh.make_node('Relu', [input_name], [output_name])
    elif activation == 'tanh':
        return oh.make_node('Tanh', [input_name], [output_name])
    elif activation == 'leaky_relu':
        return oh.make_node('LeakyRelu', [input_name], [output_name],
                            alpha=0.01)
    elif activation == 'selu':
        return oh.make_node('Selu', [input_name], [output_name])
    else:
        raise ValueError(f"Unknown activation for ONNX: {activation}")


def _mlp_encoder_onnx(input_name, params, activation):
    """Build ONNX nodes for an MLP encoder.

    Parameters
    ----------
    input_name : str
        Name of the input tensor (e.g. 'X').
    params : list of (weight, bias) tuples
    activation : str

    Returns
    -------
    nodes : list of onnx.NodeProto
    initializers : list of onnx.TensorProto
    output_name : str
        Name of the encoder's output tensor.
    """
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []
    inits = []
    prev = input_name

    for i, (w, b) in enumerate(params):
        w_np = np.asarray(w, dtype=np.float32)
        b_np = np.asarray(b, dtype=np.float32)
        w_name = f'_enc_w_{i}'
        b_name = f'_enc_b_{i}'
        mm_name = f'_enc_mm_{i}'
        add_name = f'_enc_add_{i}'
        is_last = (i == len(params) - 1)
        act_name = '_enc_out' if is_last else f'_enc_act_{i}'

        inits.append(oh.make_tensor(
            w_name, TensorProto.FLOAT, list(w_np.shape),
            w_np.flatten().tolist(),
        ))
        inits.append(oh.make_tensor(
            b_name, TensorProto.FLOAT, list(b_np.shape),
            b_np.flatten().tolist(),
        ))
        nodes.append(oh.make_node('MatMul', [prev, w_name], [mm_name]))
        nodes.append(oh.make_node('Add', [mm_name, b_name], [add_name]))
        nodes.append(_activation_node(add_name, act_name, activation))
        prev = act_name

    return nodes, inits, '_enc_out'


def _cnn_encoder_onnx(input_name, params, input_shape, activation):
    """Build ONNX nodes for a CNN encoder.

    Parameters
    ----------
    input_name : str
        Name of the flat input tensor (batch, H*W*C).
    params : dict with 'conv_layers' and 'linear'
    input_shape : tuple
        (H, W, C) of the original images.
    activation : str

    Returns
    -------
    nodes : list of onnx.NodeProto
    initializers : list of onnx.TensorProto
    output_name : str
    """
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []
    inits = []

    H, W, C = input_shape

    # Reshape flat input to (batch, H, W, C)
    nhwc_shape = np.array([-1, H, W, C], dtype=np.int64)
    inits.append(oh.make_tensor(
        '_enc_nhwc_shape', TensorProto.INT64, [4], nhwc_shape.tolist(),
    ))
    nodes.append(oh.make_node(
        'Reshape', [input_name, '_enc_nhwc_shape'], ['_enc_nhwc'],
    ))

    # Transpose NHWC -> NCHW for ONNX Conv
    nodes.append(oh.make_node(
        'Transpose', ['_enc_nhwc'], ['_enc_nchw'],
        perm=[0, 3, 1, 2],
    ))

    prev = '_enc_nchw'
    for i, (kernel, bias) in enumerate(params['conv_layers']):
        # JAX kernel: (kH, kW, C_in, C_out) -> ONNX: (C_out, C_in, kH, kW)
        k_np = np.asarray(kernel, dtype=np.float32).transpose(3, 2, 0, 1)
        b_np = np.asarray(bias, dtype=np.float32)
        k_name = f'_enc_ck_{i}'
        b_name = f'_enc_cb_{i}'
        conv_name = f'_enc_conv_{i}'
        act_name = f'_enc_conv_act_{i}'

        inits.append(oh.make_tensor(
            k_name, TensorProto.FLOAT, list(k_np.shape),
            k_np.flatten().tolist(),
        ))
        inits.append(oh.make_tensor(
            b_name, TensorProto.FLOAT, list(b_np.shape),
            b_np.flatten().tolist(),
        ))
        nodes.append(oh.make_node(
            'Conv', [prev, k_name, b_name], [conv_name],
            auto_pad='SAME_UPPER', strides=[1, 1],
        ))
        nodes.append(_activation_node(conv_name, act_name, activation))
        prev = act_name

    # GlobalAveragePool -> (N, C_last, 1, 1)
    nodes.append(oh.make_node(
        'GlobalAveragePool', [prev], ['_enc_gap'],
    ))

    # Flatten -> (N, C_last)
    nodes.append(oh.make_node(
        'Flatten', ['_enc_gap'], ['_enc_flat'],
        axis=1,
    ))

    # Linear head: MatMul + Add + Activation
    w_lin, b_lin = params['linear']
    w_np = np.asarray(w_lin, dtype=np.float32)
    b_np = np.asarray(b_lin, dtype=np.float32)
    inits.append(oh.make_tensor(
        '_enc_lin_w', TensorProto.FLOAT, list(w_np.shape),
        w_np.flatten().tolist(),
    ))
    inits.append(oh.make_tensor(
        '_enc_lin_b', TensorProto.FLOAT, list(b_np.shape),
        b_np.flatten().tolist(),
    ))
    nodes.append(oh.make_node('MatMul', ['_enc_flat', '_enc_lin_w'],
                              ['_enc_lin_mm']))
    nodes.append(oh.make_node('Add', ['_enc_lin_mm', '_enc_lin_b'],
                              ['_enc_lin_add']))
    nodes.append(_activation_node('_enc_lin_add', '_enc_out', activation))

    return nodes, inits, '_enc_out'


# ---------------------------------------------------------------------------
# Distance function -> ONNX subgraph builders
# ---------------------------------------------------------------------------

def _squared_euclidean_onnx(builder, X_name, proto_name):
    """Add squared Euclidean distance nodes: ||X - W||^2.

    Uses the expansion: ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x*y^T
    """
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []

    # X_sq = sum(X**2, axis=1, keepdims=True)  -> (batch, 1)
    nodes.append(oh.make_node('Mul', [X_name, X_name], ['_x_sq_full']))
    nodes.append(oh.make_node(
        'ReduceSum', ['_x_sq_full', '_axis1_const'],
        ['_x_sq'], keepdims=1,
    ))

    # W_sq = sum(W**2, axis=1, keepdims=True)  -> (n_proto, 1)
    nodes.append(oh.make_node('Mul', [proto_name, proto_name], ['_w_sq_full']))
    nodes.append(oh.make_node(
        'ReduceSum', ['_w_sq_full', '_axis1_const'],
        ['_w_sq'], keepdims=1,
    ))

    # W_sq_T -> (1, n_proto)
    nodes.append(oh.make_node('Transpose', ['_w_sq'], ['_w_sq_t']))

    # XW = X @ W^T -> (batch, n_proto)
    nodes.append(oh.make_node('MatMul', [X_name, '_w_t'], ['_xw']))

    # D = X_sq + W_sq_T - 2*XW
    nodes.append(oh.make_node('Mul', ['_xw', '_two_const'], ['_2xw']))
    nodes.append(oh.make_node('Add', ['_x_sq', '_w_sq_t'], ['_xsq_wsq']))
    nodes.append(oh.make_node('Sub', ['_xsq_wsq', '_2xw'], ['_dist_raw']))

    # Clip to >= 0
    nodes.append(oh.make_node('Relu', ['_dist_raw'], ['distances']))

    # Need: W^T, constants
    extra_initializers = [
        oh.make_tensor('_two_const', TensorProto.FLOAT, [], [2.0]),
        oh.make_tensor('_axis1_const', TensorProto.INT64, [1], [1]),
    ]
    # W^T computed as a Transpose node
    nodes.insert(0, oh.make_node('Transpose', [proto_name], ['_w_t']))

    return nodes, extra_initializers, 'distances'


def _euclidean_onnx(builder, X_name, proto_name):
    """Euclidean distance = sqrt(squared_euclidean)."""
    import onnx.helper as oh

    nodes, inits, dist_name = _squared_euclidean_onnx(builder, X_name, proto_name)
    nodes.append(oh.make_node('Sqrt', [dist_name], ['distances_eucl']))
    return nodes, inits, 'distances_eucl'


def _manhattan_onnx(builder, X_name, proto_name):
    """Manhattan distance: sum(|X - W|, axis=-1)."""
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []

    # Unsqueeze X -> (batch, 1, features)
    nodes.append(oh.make_node('Unsqueeze', [X_name, '_axis1_const'], ['_x_exp']))
    # Unsqueeze W -> (1, n_proto, features)
    nodes.append(oh.make_node('Unsqueeze', [proto_name, '_axis0_const'], ['_w_exp']))

    # |X - W|
    nodes.append(oh.make_node('Sub', ['_x_exp', '_w_exp'], ['_diff']))
    nodes.append(oh.make_node('Abs', ['_diff'], ['_abs_diff']))

    # Sum over features axis=2
    nodes.append(oh.make_node(
        'ReduceSum', ['_abs_diff', '_axis2_const'],
        ['distances_manh'], keepdims=0,
    ))

    extra_initializers = [
        oh.make_tensor('_axis0_const', TensorProto.INT64, [1], [0]),
        oh.make_tensor('_axis1_const', TensorProto.INT64, [1], [1]),
        oh.make_tensor('_axis2_const', TensorProto.INT64, [1], [2]),
    ]
    return nodes, extra_initializers, 'distances_manh'


def _omega_onnx(builder, X_name, proto_name, omega_name):
    """Omega distance: ||X@omega - W@omega||^2."""
    import onnx.helper as oh

    nodes = []
    # Project: X_proj = X @ omega, W_proj = W @ omega
    x_proj = '_om_x_proj'
    w_proj = '_om_w_proj'
    nodes.append(oh.make_node('MatMul', [X_name, omega_name], [x_proj]))
    nodes.append(oh.make_node('MatMul', [proto_name, omega_name], [w_proj]))

    # Squared euclidean on projected space
    sq_nodes, sq_inits, sq_dist = _squared_euclidean_onnx(
        builder, x_proj, w_proj
    )
    # Rename all internal names to avoid collision with top-level graph,
    # but preserve references to the projection outputs.
    preserve = {x_proj, w_proj}
    for n in sq_nodes:
        for i, out in enumerate(n.output):
            n.output[i] = '_om' + out
        for i, inp in enumerate(n.input):
            if inp in preserve:
                pass  # keep as-is
            elif inp.startswith('_'):
                n.input[i] = '_om' + inp

    # Rename initializer tensor names to match renamed node inputs
    renamed_inits = []
    for init in sq_inits:
        new_name = '_om' + init.name
        renamed_inits.append(oh.make_tensor(
            new_name, init.data_type, list(init.dims),
            list(init.int64_data or init.float_data),
        ))

    nodes.extend(sq_nodes)
    out_name = '_om' + sq_dist

    return nodes, renamed_inits, out_name


def _relevance_weighted_onnx(builder, X_name, proto_name, relevances_name):
    r"""Relevance-weighted squared Euclidean: sum(lambda_j * (x_j - w_j)^2).

    Parameters
    ----------
    relevances_name : str
        Name of the (d,) relevance vector initializer.

    Returns
    -------
    nodes, initializers, output_name
        Output shape: (batch, n_proto)
    """
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []

    # Unsqueeze X -> (batch, 1, features)
    nodes.append(oh.make_node(
        'Unsqueeze', [X_name, '_rw_axis1'], ['_rw_x_exp'],
    ))
    # Unsqueeze W -> (1, n_proto, features)
    nodes.append(oh.make_node(
        'Unsqueeze', [proto_name, '_rw_axis0'], ['_rw_w_exp'],
    ))

    # diff = X - W -> (batch, n_proto, features)
    nodes.append(oh.make_node('Sub', ['_rw_x_exp', '_rw_w_exp'], ['_rw_diff']))

    # diff^2
    nodes.append(oh.make_node('Mul', ['_rw_diff', '_rw_diff'], ['_rw_diff_sq']))

    # weighted = relevances * diff^2  (relevances broadcasts as (1, 1, d))
    nodes.append(oh.make_node(
        'Mul', [relevances_name, '_rw_diff_sq'], ['_rw_weighted'],
    ))

    # sum over features axis=2
    nodes.append(oh.make_node(
        'ReduceSum', ['_rw_weighted', '_rw_axis2'],
        ['distances_rw'], keepdims=0,
    ))

    extra_initializers = [
        oh.make_tensor('_rw_axis0', TensorProto.INT64, [1], [0]),
        oh.make_tensor('_rw_axis1', TensorProto.INT64, [1], [1]),
        oh.make_tensor('_rw_axis2', TensorProto.INT64, [1], [2]),
    ]
    return nodes, extra_initializers, 'distances_rw'


def _local_omega_onnx(builder, X_name, proto_name, omegas_name):
    r"""Local omega distance: ||Omega_k (x - w_k)||^2 for each prototype k.

    Uses ONNX batched MatMul: (p, n, d) @ (p, d, l) -> (p, n, l).

    Parameters
    ----------
    omegas_name : str
        Name of the (p, d, l) per-prototype omega matrices initializer.

    Returns
    -------
    nodes, initializers, output_name
        Output shape: (batch, n_proto)
    """
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []

    # Unsqueeze X -> (batch, 1, features)
    nodes.append(oh.make_node(
        'Unsqueeze', [X_name, '_lo_axis1'], ['_lo_x_exp'],
    ))
    # Unsqueeze W -> (1, n_proto, features)
    nodes.append(oh.make_node(
        'Unsqueeze', [proto_name, '_lo_axis0'], ['_lo_w_exp'],
    ))

    # diff = X - W -> (batch, n_proto, features)
    nodes.append(oh.make_node(
        'Sub', ['_lo_x_exp', '_lo_w_exp'], ['_lo_diff'],
    ))

    # Transpose diff -> (n_proto, batch, features) for batched MatMul
    nodes.append(oh.make_node(
        'Transpose', ['_lo_diff'], ['_lo_diff_t'],
        perm=[1, 0, 2],
    ))

    # Batched MatMul: (p, n, d) @ (p, d, l) -> (p, n, l)
    nodes.append(oh.make_node(
        'MatMul', ['_lo_diff_t', omegas_name], ['_lo_projected'],
    ))

    # projected^2
    nodes.append(oh.make_node(
        'Mul', ['_lo_projected', '_lo_projected'], ['_lo_proj_sq'],
    ))

    # ReduceSum over latent axis=2 -> (p, n)
    nodes.append(oh.make_node(
        'ReduceSum', ['_lo_proj_sq', '_lo_axis2'],
        ['_lo_dist_t'], keepdims=0,
    ))

    # Transpose -> (n, p) = (batch, n_proto)
    nodes.append(oh.make_node(
        'Transpose', ['_lo_dist_t'], ['distances_lo'],
        perm=[1, 0],
    ))

    extra_initializers = [
        oh.make_tensor('_lo_axis0', TensorProto.INT64, [1], [0]),
        oh.make_tensor('_lo_axis1', TensorProto.INT64, [1], [1]),
        oh.make_tensor('_lo_axis2', TensorProto.INT64, [1], [2]),
    ]
    return nodes, extra_initializers, 'distances_lo'


def _tangent_onnx(builder, X_name, proto_name, omegas_name):
    r"""Tangent distance: ||(I - Omega_k Omega_k^T)(x - w_k)||^2.

    Computes the squared norm of the component of (x - w_k) orthogonal
    to prototype k's tangent subspace.

    Uses ONNX batched MatMul:
      proj = (p,n,d) @ (p,d,s) -> (p,n,s)
      recon = (p,n,s) @ (p,s,d) -> (p,n,d)
      tang_diff = diff - recon

    Parameters
    ----------
    omegas_name : str
        Name of the (p, d, s) per-prototype orthonormal subspace bases.

    Returns
    -------
    nodes, initializers, output_name
        Output shape: (batch, n_proto)
    """
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []

    # Unsqueeze X -> (batch, 1, features)
    nodes.append(oh.make_node(
        'Unsqueeze', [X_name, '_tg_axis1'], ['_tg_x_exp'],
    ))
    # Unsqueeze W -> (1, n_proto, features)
    nodes.append(oh.make_node(
        'Unsqueeze', [proto_name, '_tg_axis0'], ['_tg_w_exp'],
    ))

    # diff = X - W -> (batch, n_proto, features)
    nodes.append(oh.make_node(
        'Sub', ['_tg_x_exp', '_tg_w_exp'], ['_tg_diff'],
    ))

    # Transpose diff -> (n_proto, batch, features)
    nodes.append(oh.make_node(
        'Transpose', ['_tg_diff'], ['_tg_diff_t'],
        perm=[1, 0, 2],
    ))

    # Step 1: Project onto subspace
    # proj = (p, n, d) @ (p, d, s) -> (p, n, s)
    nodes.append(oh.make_node(
        'MatMul', ['_tg_diff_t', omegas_name], ['_tg_proj'],
    ))

    # Step 2: Transpose omegas -> (p, s, d) for reconstruction
    nodes.append(oh.make_node(
        'Transpose', [omegas_name], ['_tg_omegas_T'],
        perm=[0, 2, 1],
    ))

    # recon = (p, n, s) @ (p, s, d) -> (p, n, d)
    nodes.append(oh.make_node(
        'MatMul', ['_tg_proj', '_tg_omegas_T'], ['_tg_recon'],
    ))

    # Step 3: tang_diff = diff - recon (orthogonal complement)
    nodes.append(oh.make_node(
        'Sub', ['_tg_diff_t', '_tg_recon'], ['_tg_tang_diff'],
    ))

    # tang_diff^2
    nodes.append(oh.make_node(
        'Mul', ['_tg_tang_diff', '_tg_tang_diff'], ['_tg_tang_sq'],
    ))

    # ReduceSum over features axis=2 -> (p, n)
    nodes.append(oh.make_node(
        'ReduceSum', ['_tg_tang_sq', '_tg_axis2'],
        ['_tg_dist_t'], keepdims=0,
    ))

    # Transpose -> (n, p) = (batch, n_proto)
    nodes.append(oh.make_node(
        'Transpose', ['_tg_dist_t'], ['distances_tg'],
        perm=[1, 0],
    ))

    extra_initializers = [
        oh.make_tensor('_tg_axis0', TensorProto.INT64, [1], [0]),
        oh.make_tensor('_tg_axis1', TensorProto.INT64, [1], [1]),
        oh.make_tensor('_tg_axis2', TensorProto.INT64, [1], [2]),
    ]
    return nodes, extra_initializers, 'distances_tg'


# ---------------------------------------------------------------------------
# Riemannian model builders
# ---------------------------------------------------------------------------


def _riemannian_so_chordal_onnx(X_name, proto_name, n):
    r"""Chordal distance on SO(n): d^2(R,S) = ||R - S||^2_F.

    Input X and prototypes are flattened (batch, n*n) and (p, n*n).
    Reshapes to 3D, broadcasts, computes Frobenius distance matrix.

    Returns
    -------
    nodes, initializers, output_name
        Output shape: (batch, n_proto)
    """
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []

    # Unsqueeze X -> (batch, 1, n*n)
    nodes.append(oh.make_node(
        'Unsqueeze', [X_name, '_rsc_axis1'], ['_rsc_x_exp'],
    ))
    # Unsqueeze W -> (1, p, n*n)
    nodes.append(oh.make_node(
        'Unsqueeze', [proto_name, '_rsc_axis0'], ['_rsc_w_exp'],
    ))

    # diff = X - W -> (batch, p, n*n)
    nodes.append(oh.make_node(
        'Sub', ['_rsc_x_exp', '_rsc_w_exp'], ['_rsc_diff'],
    ))

    # diff^2
    nodes.append(oh.make_node(
        'Mul', ['_rsc_diff', '_rsc_diff'], ['_rsc_diff_sq'],
    ))

    # ReduceSum over last axis -> (batch, p)
    nodes.append(oh.make_node(
        'ReduceSum', ['_rsc_diff_sq', '_rsc_axis2'],
        ['distances_rsc'], keepdims=0,
    ))

    extra_initializers = [
        oh.make_tensor('_rsc_axis0', TensorProto.INT64, [1], [0]),
        oh.make_tensor('_rsc_axis1', TensorProto.INT64, [1], [1]),
        oh.make_tensor('_rsc_axis2', TensorProto.INT64, [1], [2]),
    ]
    return nodes, extra_initializers, 'distances_rsc'


def _riemannian_so_tangent_onnx(X_name, proto_name, n):
    r"""Compute SO(n) tangent vectors: Log_W(X) = W @ skew(W^T @ X).

    skew(A) = (A - A^T) / 2

    Input X: (batch, n*n), prototypes W: (p, n*n).
    Output tangent: (p, batch, n*n) — ready for downstream metric.

    Returns
    -------
    nodes, initializers, output_name
    """
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []

    # Reshape X -> (batch, n, n)
    nodes.append(oh.make_node(
        'Reshape', [X_name, '_rst_x_shape'], ['_rst_X3'],
    ))
    # Reshape W -> (p, n, n)
    nodes.append(oh.make_node(
        'Reshape', [proto_name, '_rst_w_shape'], ['_rst_W3'],
    ))

    # Unsqueeze X -> (batch, 1, n, n)
    nodes.append(oh.make_node(
        'Unsqueeze', ['_rst_X3', '_rst_axis1'], ['_rst_X4'],
    ))
    # Unsqueeze W -> (1, p, n, n)
    nodes.append(oh.make_node(
        'Unsqueeze', ['_rst_W3', '_rst_axis0'], ['_rst_W4'],
    ))

    # W^T: transpose last two dims of W -> (1, p, n, n)
    nodes.append(oh.make_node(
        'Transpose', ['_rst_W4'], ['_rst_Wt'],
        perm=[0, 1, 3, 2],
    ))

    # RtS = W^T @ X -> (batch, p, n, n) via broadcasting
    nodes.append(oh.make_node(
        'MatMul', ['_rst_Wt', '_rst_X4'], ['_rst_RtS'],
    ))

    # RtS^T: transpose last two dims
    nodes.append(oh.make_node(
        'Transpose', ['_rst_RtS'], ['_rst_RtS_T'],
        perm=[0, 1, 3, 2],
    ))

    # skew = (RtS - RtS^T) / 2
    nodes.append(oh.make_node(
        'Sub', ['_rst_RtS', '_rst_RtS_T'], ['_rst_skew_raw'],
    ))
    nodes.append(oh.make_node(
        'Div', ['_rst_skew_raw', '_rst_two'], ['_rst_skew'],
    ))

    # tangent = W @ skew -> (batch, p, n, n)
    nodes.append(oh.make_node(
        'MatMul', ['_rst_W4', '_rst_skew'], ['_rst_tangent4d'],
    ))

    # Reshape tangent -> (batch, p, n*n)
    nodes.append(oh.make_node(
        'Reshape', ['_rst_tangent4d', '_rst_tang_shape'], ['_rst_tangent3d'],
    ))

    # Transpose -> (p, batch, n*n) for downstream metric ops
    nodes.append(oh.make_node(
        'Transpose', ['_rst_tangent3d'], ['tangent_so'],
        perm=[1, 0, 2],
    ))

    extra_initializers = [
        oh.make_tensor('_rst_axis0', TensorProto.INT64, [1], [0]),
        oh.make_tensor('_rst_axis1', TensorProto.INT64, [1], [1]),
        oh.make_tensor('_rst_x_shape', TensorProto.INT64, [3], [-1, n, n]),
        oh.make_tensor('_rst_w_shape', TensorProto.INT64, [3], [-1, n, n]),
        oh.make_tensor('_rst_tang_shape', TensorProto.INT64, [3],
                       [0, 0, n * n]),
        oh.make_tensor('_rst_two', TensorProto.FLOAT, [], [2.0]),
    ]
    return nodes, extra_initializers, 'tangent_so'


def _riemannian_gr_tangent_onnx(X_name, proto_name, n, k):
    r"""Compute Grassmannian tangent vectors: Log_{Q1}(Q2) = Q2 - Q1(Q1^T Q2).

    Input X: (batch, n*k), prototypes W: (p, n*k).
    Output tangent: (p, batch, n*k) — ready for downstream metric.

    Returns
    -------
    nodes, initializers, output_name
    """
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []

    # Reshape X -> (batch, n, k)
    nodes.append(oh.make_node(
        'Reshape', [X_name, '_rgt_x_shape'], ['_rgt_X3'],
    ))
    # Reshape W -> (p, n, k)
    nodes.append(oh.make_node(
        'Reshape', [proto_name, '_rgt_w_shape'], ['_rgt_W3'],
    ))

    # Unsqueeze X -> (batch, 1, n, k)
    nodes.append(oh.make_node(
        'Unsqueeze', ['_rgt_X3', '_rgt_axis1'], ['_rgt_X4'],
    ))
    # Unsqueeze W -> (1, p, n, k)
    nodes.append(oh.make_node(
        'Unsqueeze', ['_rgt_W3', '_rgt_axis0'], ['_rgt_W4'],
    ))

    # W^T: transpose last two dims -> (1, p, k, n)
    nodes.append(oh.make_node(
        'Transpose', ['_rgt_W4'], ['_rgt_Wt'],
        perm=[0, 1, 3, 2],
    ))

    # Q1tQ2 = W^T @ X -> (batch, p, k, k)
    nodes.append(oh.make_node(
        'MatMul', ['_rgt_Wt', '_rgt_X4'], ['_rgt_Q1tQ2'],
    ))

    # proj = W @ Q1tQ2 -> (batch, p, n, k)
    nodes.append(oh.make_node(
        'MatMul', ['_rgt_W4', '_rgt_Q1tQ2'], ['_rgt_proj'],
    ))

    # tangent = X - proj -> (batch, p, n, k)
    nodes.append(oh.make_node(
        'Sub', ['_rgt_X4', '_rgt_proj'], ['_rgt_tangent4d'],
    ))

    # Reshape -> (batch, p, n*k)
    nodes.append(oh.make_node(
        'Reshape', ['_rgt_tangent4d', '_rgt_tang_shape'], ['_rgt_tangent3d'],
    ))

    # Transpose -> (p, batch, n*k)
    nodes.append(oh.make_node(
        'Transpose', ['_rgt_tangent3d'], ['tangent_gr'],
        perm=[1, 0, 2],
    ))

    extra_initializers = [
        oh.make_tensor('_rgt_axis0', TensorProto.INT64, [1], [0]),
        oh.make_tensor('_rgt_axis1', TensorProto.INT64, [1], [1]),
        oh.make_tensor('_rgt_x_shape', TensorProto.INT64, [3], [-1, n, k]),
        oh.make_tensor('_rgt_w_shape', TensorProto.INT64, [3], [-1, n, k]),
        oh.make_tensor('_rgt_tang_shape', TensorProto.INT64, [3],
                       [0, 0, n * k]),
    ]
    return nodes, extra_initializers, 'tangent_gr'


def _riemannian_global_omega_onnx(tangent_name, omega_name):
    r"""Global omega metric on pre-computed tangents: d^2 = ||tangent @ Omega||^2.

    Input tangent: (p, batch, d_flat), omega: (d_flat, latent_dim).
    Output: (batch, p) distances.
    """
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []

    # MatMul: (p, batch, d) @ (d, l) -> (p, batch, l)
    nodes.append(oh.make_node(
        'MatMul', [tangent_name, omega_name], ['_rgo_projected'],
    ))

    # projected^2
    nodes.append(oh.make_node(
        'Mul', ['_rgo_projected', '_rgo_projected'], ['_rgo_proj_sq'],
    ))

    # ReduceSum over latent axis=2 -> (p, batch)
    nodes.append(oh.make_node(
        'ReduceSum', ['_rgo_proj_sq', '_rgo_axis2'],
        ['_rgo_dist_t'], keepdims=0,
    ))

    # Transpose -> (batch, p)
    nodes.append(oh.make_node(
        'Transpose', ['_rgo_dist_t'], ['distances_rgo'],
        perm=[1, 0],
    ))

    extra_initializers = [
        oh.make_tensor('_rgo_axis2', TensorProto.INT64, [1], [2]),
    ]
    return nodes, extra_initializers, 'distances_rgo'


def _riemannian_local_omega_onnx(tangent_name, omegas_name):
    r"""Per-prototype omega metric: d^2_k = ||tangent_k @ Omega_k||^2.

    Input tangent: (p, batch, d_flat), omegas: (p, d_flat, latent_dim).
    Output: (batch, p) distances.
    """
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []

    # Batched MatMul: (p, batch, d) @ (p, d, l) -> (p, batch, l)
    nodes.append(oh.make_node(
        'MatMul', [tangent_name, omegas_name], ['_rlo_projected'],
    ))

    # projected^2
    nodes.append(oh.make_node(
        'Mul', ['_rlo_projected', '_rlo_projected'], ['_rlo_proj_sq'],
    ))

    # ReduceSum over latent axis=2 -> (p, batch)
    nodes.append(oh.make_node(
        'ReduceSum', ['_rlo_proj_sq', '_rlo_axis2'],
        ['_rlo_dist_t'], keepdims=0,
    ))

    # Transpose -> (batch, p)
    nodes.append(oh.make_node(
        'Transpose', ['_rlo_dist_t'], ['distances_rlo'],
        perm=[1, 0],
    ))

    extra_initializers = [
        oh.make_tensor('_rlo_axis2', TensorProto.INT64, [1], [2]),
    ]
    return nodes, extra_initializers, 'distances_rlo'


def _riemannian_tangent_subspace_onnx(tangent_name, omegas_name):
    r"""Tangent subspace distance: d^2 = ||(I - Omega_k Omega_k^T) tangent_k||^2.

    Input tangent: (p, batch, d_flat), omegas: (p, d_flat, s).
    Output: (batch, p) distances.
    """
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []

    # Step 1: Project onto subspace
    # proj = (p, batch, d) @ (p, d, s) -> (p, batch, s)
    nodes.append(oh.make_node(
        'MatMul', [tangent_name, omegas_name], ['_rts_proj'],
    ))

    # Step 2: Transpose omegas -> (p, s, d)
    nodes.append(oh.make_node(
        'Transpose', [omegas_name], ['_rts_omegas_T'],
        perm=[0, 2, 1],
    ))

    # recon = (p, batch, s) @ (p, s, d) -> (p, batch, d)
    nodes.append(oh.make_node(
        'MatMul', ['_rts_proj', '_rts_omegas_T'], ['_rts_recon'],
    ))

    # Step 3: residual = tangent - recon
    nodes.append(oh.make_node(
        'Sub', [tangent_name, '_rts_recon'], ['_rts_residual'],
    ))

    # residual^2
    nodes.append(oh.make_node(
        'Mul', ['_rts_residual', '_rts_residual'], ['_rts_res_sq'],
    ))

    # ReduceSum over features axis=2 -> (p, batch)
    nodes.append(oh.make_node(
        'ReduceSum', ['_rts_res_sq', '_rts_axis2'],
        ['_rts_dist_t'], keepdims=0,
    ))

    # Transpose -> (batch, p)
    nodes.append(oh.make_node(
        'Transpose', ['_rts_dist_t'], ['distances_rts'],
        perm=[1, 0],
    ))

    extra_initializers = [
        oh.make_tensor('_rts_axis2', TensorProto.INT64, [1], [2]),
    ]
    return nodes, extra_initializers, 'distances_rts'


def _export_riemannian_onnx(model, batch_size, opset_version, path):
    """Export a Riemannian supervised model to ONNX.

    Handles RiemannianSRNG (chordal distance on SO(n)) and
    RiemannianSMNG/SLNG/STNG (tangent-space metric on SO(n) or Grassmannian).
    """
    import onnx
    import onnx.helper as oh
    from onnx import TensorProto

    from prosemble.core.manifolds import SO, Grassmannian

    # Determine manifold and model variant
    manifold = model.manifold
    is_so = isinstance(manifold, SO)
    is_gr = isinstance(manifold, Grassmannian)
    model_name = type(model).__name__

    # Determine point shape
    if is_so:
        n = manifold.n
        n_features = n * n
        point_shape = (n, n)
    else:  # Grassmannian
        n, k = manifold.n, manifold.k
        n_features = n * k
        point_shape = (n, k)

    batch_dim = batch_size if batch_size > 0 else 'batch'
    input_shape = [batch_dim, n_features]

    all_nodes = []
    initializers = []

    # Prototypes (flattened manifold points)
    prototypes = np.asarray(model.prototypes_, dtype=np.float32)
    n_proto = prototypes.shape[0]
    initializers.append(
        oh.make_tensor(
            'prototypes', TensorProto.FLOAT,
            list(prototypes.shape), prototypes.flatten().tolist(),
        ),
    )

    # Prototype labels
    proto_labels = np.asarray(model.prototype_labels_).astype(np.int64)
    initializers.append(
        oh.make_tensor(
            'proto_labels', TensorProto.INT64,
            list(proto_labels.shape), proto_labels.flatten().tolist(),
        ),
    )

    # --- Distance computation ---
    if model_name == 'RiemannianSRNG':
        # Chordal distance: ||X - W||^2_F (works on flattened vectors directly)
        nodes, extra_inits, dist_out = _riemannian_so_chordal_onnx(
            'X', 'prototypes', n,
        )
        all_nodes.extend(nodes)
        initializers.extend(extra_inits)

    else:
        # SMNG, SLNG, STNG: compute tangent vectors first, then apply metric
        if is_so:
            tang_nodes, tang_inits, tangent_out = _riemannian_so_tangent_onnx(
                'X', 'prototypes', n,
            )
        else:  # Grassmannian
            tang_nodes, tang_inits, tangent_out = _riemannian_gr_tangent_onnx(
                'X', 'prototypes', n, k,
            )
        all_nodes.extend(tang_nodes)
        initializers.extend(tang_inits)

        # Apply metric based on model variant
        if model_name == 'RiemannianSMNG':
            # Global omega
            omega = np.asarray(model.omega_, dtype=np.float32)
            initializers.append(
                oh.make_tensor(
                    'omega', TensorProto.FLOAT,
                    list(omega.shape), omega.flatten().tolist(),
                ),
            )
            met_nodes, met_inits, dist_out = _riemannian_global_omega_onnx(
                tangent_out, 'omega',
            )

        elif model_name == 'RiemannianSLNG':
            # Per-prototype local omega
            omegas = np.asarray(model.omegas_, dtype=np.float32)
            initializers.append(
                oh.make_tensor(
                    'omegas', TensorProto.FLOAT,
                    list(omegas.shape), omegas.flatten().tolist(),
                ),
            )
            met_nodes, met_inits, dist_out = _riemannian_local_omega_onnx(
                tangent_out, 'omegas',
            )

        elif model_name == 'RiemannianSTNG':
            # Tangent subspace
            omegas = np.asarray(model.omegas_, dtype=np.float32)
            initializers.append(
                oh.make_tensor(
                    'omegas', TensorProto.FLOAT,
                    list(omegas.shape), omegas.flatten().tolist(),
                ),
            )
            met_nodes, met_inits, dist_out = _riemannian_tangent_subspace_onnx(
                tangent_out, 'omegas',
            )
        else:
            raise NotImplementedError(
                f"Unknown Riemannian model variant: {model_name}"
            )

        all_nodes.extend(met_nodes)
        initializers.extend(met_inits)

    # --- WTAC decision ---
    comp_nodes, comp_inits = _wtac_onnx_nodes(dist_out, 'proto_labels')
    all_nodes.extend(comp_nodes)
    initializers.extend(comp_inits)

    # --- Build graph ---
    X_input = oh.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    Y_output = oh.make_tensor_value_info(
        'predictions', TensorProto.INT64, [batch_dim],
    )

    graph = oh.make_graph(
        all_nodes,
        'prosemble_riemannian_predict',
        [X_input],
        [Y_output],
        initializer=initializers,
    )

    onnx_model = oh.make_model(graph, opset_imports=[
        oh.make_opsetid('', opset_version),
    ])
    onnx_model.ir_version = 8
    onnx.checker.check_model(onnx_model)

    if path is not None:
        onnx.save(onnx_model, path)

    return onnx_model


# ---------------------------------------------------------------------------
# Competition / decision builders
# ---------------------------------------------------------------------------

def _wtac_onnx_nodes(dist_name, proto_labels_name):
    """WTAC: predictions = proto_labels[argmin(distances, axis=1)]."""
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []
    # ArgMin over prototypes axis
    nodes.append(oh.make_node(
        'ArgMin', [dist_name], ['_winners'],
        axis=1, keepdims=0,
    ))
    # Flatten winners for Gather
    nodes.append(oh.make_node('Cast', ['_winners'], ['_winners_i64'], to=TensorProto.INT64))
    # Gather labels
    nodes.append(oh.make_node(
        'Gather', [proto_labels_name, '_winners_i64'], ['predictions'],
        axis=0,
    ))
    return nodes, []


def _argmin_onnx_nodes(dist_name):
    """Unsupervised: predictions = argmin(distances, axis=1)."""
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []
    nodes.append(oh.make_node(
        'ArgMin', [dist_name], ['_predictions_raw'],
        axis=1, keepdims=0,
    ))
    nodes.append(oh.make_node(
        'Cast', ['_predictions_raw'], ['predictions'],
        to=TensorProto.INT64,
    ))
    return nodes, []


def _oc_hard_nearest_onnx(dist_name, model):
    """One-class hard nearest decision.

    decision_function:
        nearest_idx = argmin(distances)
        d_nearest = distances[nearest_idx]
        theta_nearest = thetas[nearest_idx]
        mu = (d - theta) / (d + theta + eps)
        score = 1 - sigmoid(beta * mu)
    predict:
        score >= 0.5 -> target_label, else non_target_label
    """
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []
    inits = []

    beta = float(model.beta)
    target = int(model._target_label)
    non_target = int(model._non_target_label)
    thetas = np.asarray(model.thetas_).astype(np.float32)

    # Constants
    inits.extend([
        oh.make_tensor('_oc_thetas', TensorProto.FLOAT,
                        list(thetas.shape), thetas.flatten().tolist()),
        oh.make_tensor('_oc_beta', TensorProto.FLOAT, [], [beta]),
        oh.make_tensor('_oc_eps', TensorProto.FLOAT, [], [1e-10]),
        oh.make_tensor('_oc_one', TensorProto.FLOAT, [], [1.0]),
        oh.make_tensor('_oc_half', TensorProto.FLOAT, [], [0.5]),
        oh.make_tensor('_oc_target', TensorProto.INT32, [], [target]),
        oh.make_tensor('_oc_non_target', TensorProto.INT32, [], [non_target]),
        oh.make_tensor('_oc_axis1', TensorProto.INT64, [1], [1]),
    ])

    # nearest_idx = ArgMin(distances, axis=1)  -> (n,)
    nodes.append(oh.make_node(
        'ArgMin', [dist_name], ['_oc_nearest_idx'],
        axis=1, keepdims=0,
    ))

    # Cast to int64 for indexing
    nodes.append(oh.make_node(
        'Cast', ['_oc_nearest_idx'], ['_oc_nearest_i64'],
        to=TensorProto.INT64,
    ))

    # d_nearest via GatherElements: need indices as (n, 1) for axis=1
    nodes.append(oh.make_node(
        'Unsqueeze', ['_oc_nearest_i64', '_oc_axis1'], ['_oc_idx_2d'],
    ))
    nodes.append(oh.make_node(
        'GatherElements', [dist_name, '_oc_idx_2d'], ['_oc_d_2d'],
        axis=1,
    ))
    # Squeeze back to (n,)
    nodes.append(oh.make_node(
        'Squeeze', ['_oc_d_2d', '_oc_axis1'], ['_oc_d_nearest'],
    ))

    # theta_nearest = Gather(thetas, nearest_idx)  -> (n,)
    nodes.append(oh.make_node(
        'Gather', ['_oc_thetas', '_oc_nearest_i64'], ['_oc_theta_nearest'],
        axis=0,
    ))

    # mu = (d - theta) / (d + theta + eps)
    nodes.append(oh.make_node(
        'Sub', ['_oc_d_nearest', '_oc_theta_nearest'], ['_oc_num'],
    ))
    nodes.append(oh.make_node(
        'Add', ['_oc_d_nearest', '_oc_theta_nearest'], ['_oc_den_raw'],
    ))
    nodes.append(oh.make_node(
        'Add', ['_oc_den_raw', '_oc_eps'], ['_oc_den'],
    ))
    nodes.append(oh.make_node(
        'Div', ['_oc_num', '_oc_den'], ['_oc_mu'],
    ))

    # score = 1 - sigmoid(beta * mu)
    nodes.append(oh.make_node(
        'Mul', ['_oc_beta', '_oc_mu'], ['_oc_beta_mu'],
    ))
    nodes.append(oh.make_node(
        'Sigmoid', ['_oc_beta_mu'], ['_oc_sig'],
    ))
    nodes.append(oh.make_node(
        'Sub', ['_oc_one', '_oc_sig'], ['_oc_score'],
    ))

    # predictions = Where(score >= 0.5, target, non_target)
    nodes.append(oh.make_node(
        'GreaterOrEqual', ['_oc_score', '_oc_half'], ['_oc_mask'],
    ))
    nodes.append(oh.make_node(
        'Where', ['_oc_mask', '_oc_target', '_oc_non_target'], ['predictions'],
    ))

    return nodes, inits


def _oc_gaussian_soft_onnx(dist_name, model):
    """One-class Gaussian soft-weighted decision.

    decision_function:
        weights = softmax(-distances / (2*sigma^2))
        mu_k = (distances - thetas) / (distances + thetas + eps)
        weighted_mu = sum(weights * mu_k, axis=1)
        score = 1 - sigmoid(beta * weighted_mu)
    predict:
        score >= 0.5 -> target_label, else non_target_label
    """
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []
    inits = []

    beta = float(model.beta)
    sigma = float(model.sigma)
    target = int(model._target_label)
    non_target = int(model._non_target_label)
    thetas = np.asarray(model.thetas_).astype(np.float32)
    two_sigma_sq = 2.0 * sigma * sigma

    inits.extend([
        oh.make_tensor('_gs_thetas', TensorProto.FLOAT,
                        list(thetas.shape), thetas.flatten().tolist()),
        oh.make_tensor('_gs_beta', TensorProto.FLOAT, [], [beta]),
        oh.make_tensor('_gs_eps', TensorProto.FLOAT, [], [1e-10]),
        oh.make_tensor('_gs_one', TensorProto.FLOAT, [], [1.0]),
        oh.make_tensor('_gs_half', TensorProto.FLOAT, [], [0.5]),
        oh.make_tensor('_gs_neg_inv_2s2', TensorProto.FLOAT, [],
                        [-1.0 / two_sigma_sq]),
        oh.make_tensor('_gs_target', TensorProto.INT32, [], [target]),
        oh.make_tensor('_gs_non_target', TensorProto.INT32, [], [non_target]),
        oh.make_tensor('_gs_axis1', TensorProto.INT64, [1], [1]),
    ])

    # Gaussian weights = softmax(-d / (2*sigma^2), axis=1)
    nodes.append(oh.make_node(
        'Mul', [dist_name, '_gs_neg_inv_2s2'], ['_gs_logits'],
    ))
    nodes.append(oh.make_node(
        'Softmax', ['_gs_logits'], ['_gs_weights'],
        axis=1,
    ))

    # Per-prototype mu_k = (d - theta) / (d + theta + eps)
    # thetas broadcast: (K,) -> (1, K) via Unsqueeze
    nodes.append(oh.make_node(
        'Unsqueeze', ['_gs_thetas', '_gs_axis1'], ['_gs_thetas_r1'],
    ))
    # Reshape to row: remove batch dim from Unsqueeze result
    # Actually Unsqueeze at axis=0 gives (1, K) — let's use axis=0
    # We need (1, K) for broadcasting with (n, K) distances
    # Fix: Unsqueeze thetas at axis=0 instead
    inits.append(
        oh.make_tensor('_gs_axis0', TensorProto.INT64, [1], [0]),
    )
    # Remove the axis=1 unsqueeze for thetas, use axis=0
    nodes.pop()  # remove the wrong Unsqueeze
    nodes.append(oh.make_node(
        'Unsqueeze', ['_gs_thetas', '_gs_axis0'], ['_gs_thetas_2d'],
    ))

    nodes.append(oh.make_node(
        'Sub', [dist_name, '_gs_thetas_2d'], ['_gs_num'],
    ))
    nodes.append(oh.make_node(
        'Add', [dist_name, '_gs_thetas_2d'], ['_gs_den_raw'],
    ))
    nodes.append(oh.make_node(
        'Add', ['_gs_den_raw', '_gs_eps'], ['_gs_den'],
    ))
    nodes.append(oh.make_node(
        'Div', ['_gs_num', '_gs_den'], ['_gs_mu_k'],
    ))

    # weighted_mu = sum(weights * mu_k, axis=1)
    nodes.append(oh.make_node(
        'Mul', ['_gs_weights', '_gs_mu_k'], ['_gs_w_mu'],
    ))
    nodes.append(oh.make_node(
        'ReduceSum', ['_gs_w_mu', '_gs_axis1'], ['_gs_weighted_mu'],
        keepdims=0,
    ))

    # score = 1 - sigmoid(beta * weighted_mu)
    nodes.append(oh.make_node(
        'Mul', ['_gs_beta', '_gs_weighted_mu'], ['_gs_beta_mu'],
    ))
    nodes.append(oh.make_node(
        'Sigmoid', ['_gs_beta_mu'], ['_gs_sig'],
    ))
    nodes.append(oh.make_node(
        'Sub', ['_gs_one', '_gs_sig'], ['_gs_score'],
    ))

    # predictions
    nodes.append(oh.make_node(
        'GreaterOrEqual', ['_gs_score', '_gs_half'], ['_gs_mask'],
    ))
    nodes.append(oh.make_node(
        'Where', ['_gs_mask', '_gs_target', '_gs_non_target'], ['predictions'],
    ))

    return nodes, inits


def _oc_gaussian_ng_onnx(dist_name, model, n_proto):
    """One-class Gaussian+NG soft-weighted decision.

    Same as Gaussian Soft, but weights = Gaussian × NG rank weights
    (normalized), using the converged gamma_.

    NG rank weights:
        order = argsort(distances)
        ranks = argsort(order)  # inverse permutation
        h = exp(-ranks / gamma)
        h_norm = h / sum(h)
    Combined: combined = gauss * h_norm (re-normalized)
    """
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []
    inits = []

    beta = float(model.beta)
    sigma = float(model.sigma)
    gamma = float(model.gamma_)
    target = int(model._target_label)
    non_target = int(model._non_target_label)
    thetas = np.asarray(model.thetas_).astype(np.float32)
    two_sigma_sq = 2.0 * sigma * sigma
    K = n_proto

    # Range [0, 1, ..., K-1] for ScatterElements
    range_K = np.arange(K, dtype=np.int64)

    inits.extend([
        oh.make_tensor('_gn_thetas', TensorProto.FLOAT,
                        list(thetas.shape), thetas.flatten().tolist()),
        oh.make_tensor('_gn_beta', TensorProto.FLOAT, [], [beta]),
        oh.make_tensor('_gn_eps', TensorProto.FLOAT, [], [1e-10]),
        oh.make_tensor('_gn_one', TensorProto.FLOAT, [], [1.0]),
        oh.make_tensor('_gn_half', TensorProto.FLOAT, [], [0.5]),
        oh.make_tensor('_gn_neg_inv_2s2', TensorProto.FLOAT, [],
                        [-1.0 / two_sigma_sq]),
        oh.make_tensor('_gn_neg_inv_gamma', TensorProto.FLOAT, [],
                        [-1.0 / (gamma + 1e-10)]),
        oh.make_tensor('_gn_target', TensorProto.INT32, [], [target]),
        oh.make_tensor('_gn_non_target', TensorProto.INT32, [], [non_target]),
        oh.make_tensor('_gn_axis0', TensorProto.INT64, [1], [0]),
        oh.make_tensor('_gn_axis1', TensorProto.INT64, [1], [1]),
        oh.make_tensor('_gn_K', TensorProto.INT64, [1], [K]),
        oh.make_tensor('_gn_range_K', TensorProto.INT64, [K],
                        range_K.tolist()),
    ])

    # --- Gaussian weights ---
    nodes.append(oh.make_node(
        'Mul', [dist_name, '_gn_neg_inv_2s2'], ['_gn_logits'],
    ))
    nodes.append(oh.make_node(
        'Softmax', ['_gn_logits'], ['_gn_gauss'],
        axis=1,
    ))

    # --- NG rank weights ---
    # TopK to get argsort (ascending = smallest first)
    nodes.append(oh.make_node(
        'TopK', [dist_name, '_gn_K'], ['_gn_sorted_vals', '_gn_sorted_idx'],
        largest=0, sorted=1,
    ))

    # Compute ranks via ScatterElements:
    # ranks[i, sorted_idx[i,j]] = j
    # Create (n, K) of zeros, then scatter range_K into it
    # First, get the shape of distances for creating zeros
    nodes.append(oh.make_node(
        'Shape', [dist_name], ['_gn_dist_shape'],
    ))
    nodes.append(oh.make_node(
        'ConstantOfShape', ['_gn_dist_shape'], ['_gn_zeros'],
        value=oh.make_tensor('', TensorProto.INT64, [1], [0]),
    ))

    # Expand range_K to (n, K): first Unsqueeze to (1, K), then Expand
    nodes.append(oh.make_node(
        'Unsqueeze', ['_gn_range_K', '_gn_axis0'], ['_gn_range_2d'],
    ))
    nodes.append(oh.make_node(
        'Expand', ['_gn_range_2d', '_gn_dist_shape'], ['_gn_range_expanded'],
    ))

    # ScatterElements: zeros[i, sorted_idx[i,j]] = range_expanded[i,j] = j
    nodes.append(oh.make_node(
        'ScatterElements', ['_gn_zeros', '_gn_sorted_idx', '_gn_range_expanded'],
        ['_gn_ranks_i64'],
        axis=1,
    ))

    # Cast ranks to float
    nodes.append(oh.make_node(
        'Cast', ['_gn_ranks_i64'], ['_gn_ranks'],
        to=TensorProto.FLOAT,
    ))

    # h = exp(-ranks / gamma) = exp(ranks * neg_inv_gamma)
    nodes.append(oh.make_node(
        'Mul', ['_gn_ranks', '_gn_neg_inv_gamma'], ['_gn_h_logits'],
    ))
    nodes.append(oh.make_node(
        'Exp', ['_gn_h_logits'], ['_gn_h'],
    ))

    # h_norm = h / sum(h, axis=1, keepdims=1)
    nodes.append(oh.make_node(
        'ReduceSum', ['_gn_h', '_gn_axis1'], ['_gn_h_sum'],
        keepdims=1,
    ))
    nodes.append(oh.make_node(
        'Add', ['_gn_h_sum', '_gn_eps'], ['_gn_h_sum_eps'],
    ))
    nodes.append(oh.make_node(
        'Div', ['_gn_h', '_gn_h_sum_eps'], ['_gn_h_norm'],
    ))

    # --- Combined weights ---
    nodes.append(oh.make_node(
        'Mul', ['_gn_gauss', '_gn_h_norm'], ['_gn_combined_raw'],
    ))
    nodes.append(oh.make_node(
        'ReduceSum', ['_gn_combined_raw', '_gn_axis1'], ['_gn_comb_sum'],
        keepdims=1,
    ))
    nodes.append(oh.make_node(
        'Add', ['_gn_comb_sum', '_gn_eps'], ['_gn_comb_sum_eps'],
    ))
    nodes.append(oh.make_node(
        'Div', ['_gn_combined_raw', '_gn_comb_sum_eps'], ['_gn_combined'],
    ))

    # --- Per-prototype mu_k ---
    nodes.append(oh.make_node(
        'Unsqueeze', ['_gn_thetas', '_gn_axis0'], ['_gn_thetas_2d'],
    ))
    nodes.append(oh.make_node(
        'Sub', [dist_name, '_gn_thetas_2d'], ['_gn_num'],
    ))
    nodes.append(oh.make_node(
        'Add', [dist_name, '_gn_thetas_2d'], ['_gn_den_raw'],
    ))
    nodes.append(oh.make_node(
        'Add', ['_gn_den_raw', '_gn_eps'], ['_gn_den'],
    ))
    nodes.append(oh.make_node(
        'Div', ['_gn_num', '_gn_den'], ['_gn_mu_k'],
    ))

    # weighted_mu = sum(combined * mu_k, axis=1)
    nodes.append(oh.make_node(
        'Mul', ['_gn_combined', '_gn_mu_k'], ['_gn_w_mu'],
    ))
    nodes.append(oh.make_node(
        'ReduceSum', ['_gn_w_mu', '_gn_axis1'], ['_gn_weighted_mu'],
        keepdims=0,
    ))

    # score = 1 - sigmoid(beta * weighted_mu)
    nodes.append(oh.make_node(
        'Mul', ['_gn_beta', '_gn_weighted_mu'], ['_gn_beta_mu'],
    ))
    nodes.append(oh.make_node(
        'Sigmoid', ['_gn_beta_mu'], ['_gn_sig'],
    ))
    nodes.append(oh.make_node(
        'Sub', ['_gn_one', '_gn_sig'], ['_gn_score'],
    ))

    # predictions
    nodes.append(oh.make_node(
        'GreaterOrEqual', ['_gn_score', '_gn_half'], ['_gn_mask'],
    ))
    nodes.append(oh.make_node(
        'Where', ['_gn_mask', '_gn_target', '_gn_non_target'], ['predictions'],
    ))

    return nodes, inits


def _svqocc_onnx(dist_name, model, n_proto):
    """SVQ-OCC decision: response probability × Heaviside sigmoid.

    decision_function:
        if gaussian: p_k = softmax(-gamma_resp * distances)
        elif student_t: p_k = normalize((1 + d/nu)^(-(nu+1)/2))
        else: p_k = 1/K
        heaviside = sigmoid((thetas - distances) / sigma)
        score = clip(sum(p_k * heaviside), 0, 1)
    predict:
        score >= 0.5 -> target_label, else non_target_label
    """
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []
    inits = []

    sigma = float(model.sigma)
    target = int(model._target_label)
    non_target = int(model._non_target_label)
    thetas = np.asarray(model.thetas_).astype(np.float32)
    response_type = model.response_type
    K = n_proto

    inits.extend([
        oh.make_tensor('_sv_thetas', TensorProto.FLOAT,
                        list(thetas.shape), thetas.flatten().tolist()),
        oh.make_tensor('_sv_inv_sigma', TensorProto.FLOAT, [],
                        [1.0 / (sigma + 1e-10)]),
        oh.make_tensor('_sv_zero', TensorProto.FLOAT, [], [0.0]),
        oh.make_tensor('_sv_one', TensorProto.FLOAT, [], [1.0]),
        oh.make_tensor('_sv_half', TensorProto.FLOAT, [], [0.5]),
        oh.make_tensor('_sv_target', TensorProto.INT32, [], [target]),
        oh.make_tensor('_sv_non_target', TensorProto.INT32, [], [non_target]),
        oh.make_tensor('_sv_axis0', TensorProto.INT64, [1], [0]),
        oh.make_tensor('_sv_axis1', TensorProto.INT64, [1], [1]),
    ])

    # --- Response probability p_k ---
    if response_type == 'gaussian':
        gamma_resp = float(model.gamma_resp)
        inits.append(
            oh.make_tensor('_sv_neg_gamma', TensorProto.FLOAT, [],
                            [-gamma_resp]),
        )
        nodes.append(oh.make_node(
            'Mul', [dist_name, '_sv_neg_gamma'], ['_sv_resp_logits'],
        ))
        nodes.append(oh.make_node(
            'Softmax', ['_sv_resp_logits'], ['_sv_p_k'],
            axis=1,
        ))
    elif response_type == 'student_t':
        nu = float(model.nu)
        exponent = -(nu + 1.0) / 2.0
        inits.extend([
            oh.make_tensor('_sv_inv_nu', TensorProto.FLOAT, [], [1.0 / nu]),
            oh.make_tensor('_sv_exponent', TensorProto.FLOAT, [], [exponent]),
            oh.make_tensor('_sv_eps', TensorProto.FLOAT, [], [1e-10]),
        ])
        # (1 + d/nu)
        nodes.append(oh.make_node(
            'Mul', [dist_name, '_sv_inv_nu'], ['_sv_d_over_nu'],
        ))
        nodes.append(oh.make_node(
            'Add', ['_sv_one', '_sv_d_over_nu'], ['_sv_base'],
        ))
        # base^exponent
        nodes.append(oh.make_node(
            'Pow', ['_sv_base', '_sv_exponent'], ['_sv_p_unnorm'],
        ))
        # normalize
        nodes.append(oh.make_node(
            'ReduceSum', ['_sv_p_unnorm', '_sv_axis1'], ['_sv_p_sum'],
            keepdims=1,
        ))
        nodes.append(oh.make_node(
            'Add', ['_sv_p_sum', '_sv_eps'], ['_sv_p_sum_eps'],
        ))
        nodes.append(oh.make_node(
            'Div', ['_sv_p_unnorm', '_sv_p_sum_eps'], ['_sv_p_k'],
        ))
    else:  # uniform
        inv_K = 1.0 / K
        inits.append(
            oh.make_tensor('_sv_inv_K', TensorProto.FLOAT, [], [inv_K]),
        )
        # Broadcast scalar to (n, K) via Mul with ones-like distances
        # Use: p_k = distances * 0 + inv_K (broadcasts correctly)
        nodes.append(oh.make_node(
            'Mul', [dist_name, '_sv_zero'], ['_sv_zeros_nk'],
        ))
        nodes.append(oh.make_node(
            'Add', ['_sv_zeros_nk', '_sv_inv_K'], ['_sv_p_k'],
        ))

    # --- Heaviside sigmoid ---
    # heaviside = sigmoid((thetas - distances) / sigma)
    nodes.append(oh.make_node(
        'Unsqueeze', ['_sv_thetas', '_sv_axis0'], ['_sv_thetas_2d'],
    ))
    nodes.append(oh.make_node(
        'Sub', ['_sv_thetas_2d', dist_name], ['_sv_theta_minus_d'],
    ))
    nodes.append(oh.make_node(
        'Mul', ['_sv_theta_minus_d', '_sv_inv_sigma'], ['_sv_heav_input'],
    ))
    nodes.append(oh.make_node(
        'Sigmoid', ['_sv_heav_input'], ['_sv_heaviside'],
    ))

    # --- Responsibility = p_k * heaviside ---
    nodes.append(oh.make_node(
        'Mul', ['_sv_p_k', '_sv_heaviside'], ['_sv_responsibility'],
    ))

    # --- Score = clip(sum(responsibility, axis=1), 0, 1) ---
    nodes.append(oh.make_node(
        'ReduceSum', ['_sv_responsibility', '_sv_axis1'], ['_sv_raw_score'],
        keepdims=0,
    ))
    nodes.append(oh.make_node(
        'Clip', ['_sv_raw_score', '_sv_zero', '_sv_one'], ['_sv_score'],
    ))

    # --- Predictions ---
    nodes.append(oh.make_node(
        'GreaterOrEqual', ['_sv_score', '_sv_half'], ['_sv_mask'],
    ))
    nodes.append(oh.make_node(
        'Where', ['_sv_mask', '_sv_target', '_sv_non_target'], ['predictions'],
    ))

    return nodes, inits


def _cbc_onnx(dist_name, model):
    """CBC reasoning decision.

    decision:
        detections = exp(-distances / (2 * sigma^2))
        A = clip(reasonings[:, :, 0], 0, 1)
        B = clip(reasonings[:, :, 1], 0, 1)
        pk = A, nk = (1 - A) * B
        probs = (detections @ (pk - nk) + sum(nk)) / (sum(pk + nk) + eps)
        predictions = argmax(probs, axis=1)
    """
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []
    inits = []

    sigma_sq = float(model.sigma) ** 2

    # Pre-compute CBC reasoning constants
    reasonings = np.asarray(model.reasonings_, dtype=np.float32)
    A = np.clip(reasonings[:, :, 0], 0, 1)
    B = np.clip(reasonings[:, :, 1], 0, 1)
    pk = A
    nk = (1.0 - A) * B
    pk_minus_nk = (pk - nk).astype(np.float32)      # (n_comp, n_classes)
    sum_nk = np.sum(nk, axis=0).astype(np.float32)   # (n_classes,)
    denom = (np.sum(pk + nk, axis=0) + 1e-8).astype(np.float32)

    inits.extend([
        oh.make_tensor('_cbc_scale', TensorProto.FLOAT, [],
                        [-1.0 / (2.0 * sigma_sq)]),
        oh.make_tensor('_cbc_pk_nk', TensorProto.FLOAT,
                        list(pk_minus_nk.shape),
                        pk_minus_nk.flatten().tolist()),
        oh.make_tensor('_cbc_sum_nk', TensorProto.FLOAT,
                        list(sum_nk.shape), sum_nk.flatten().tolist()),
        oh.make_tensor('_cbc_denom', TensorProto.FLOAT,
                        list(denom.shape), denom.flatten().tolist()),
    ])

    # Gaussian similarity: detections = exp(-d / (2*sigma^2))
    nodes.append(oh.make_node(
        'Mul', [dist_name, '_cbc_scale'], ['_cbc_logits'],
    ))
    nodes.append(oh.make_node(
        'Exp', ['_cbc_logits'], ['_cbc_detections'],
    ))

    # numerator = detections @ (pk - nk) + sum(nk)
    nodes.append(oh.make_node(
        'MatMul', ['_cbc_detections', '_cbc_pk_nk'], ['_cbc_matmul'],
    ))
    nodes.append(oh.make_node(
        'Add', ['_cbc_matmul', '_cbc_sum_nk'], ['_cbc_numerator'],
    ))

    # probs = numerator / denominator
    nodes.append(oh.make_node(
        'Div', ['_cbc_numerator', '_cbc_denom'], ['_cbc_probs'],
    ))

    # predictions = argmax(probs, axis=1)
    nodes.append(oh.make_node(
        'ArgMax', ['_cbc_probs'], ['_cbc_preds_raw'],
        axis=1, keepdims=0,
    ))
    nodes.append(oh.make_node(
        'Cast', ['_cbc_preds_raw'], ['predictions'],
        to=TensorProto.INT64,
    ))

    return nodes, inits


def _plvq_onnx(dist_name, model):
    """PLVQ Gaussian mixture soft assignment decision.

    decision:
        logits = -distances / (2 * sigma^2)
        probs = softmax(logits, axis=1)
        class_probs = probs @ class_mask   (aggregate per class)
        predictions = argmax(class_probs, axis=1)
    """
    import onnx.helper as oh
    from onnx import TensorProto

    nodes = []
    inits = []

    sigma_sq = float(model.sigma) ** 2
    proto_labels = np.asarray(model.prototype_labels_, dtype=np.int64)
    n_proto = len(proto_labels)
    n_classes = int(model.n_classes_)

    # Pre-compute class mask: M[j, c] = 1 if prototype j belongs to class c
    class_mask = np.zeros((n_proto, n_classes), dtype=np.float32)
    for j, c in enumerate(proto_labels):
        class_mask[j, int(c)] = 1.0

    inits.extend([
        oh.make_tensor('_plvq_scale', TensorProto.FLOAT, [],
                        [-1.0 / (2.0 * sigma_sq)]),
        oh.make_tensor('_plvq_mask', TensorProto.FLOAT,
                        list(class_mask.shape),
                        class_mask.flatten().tolist()),
    ])

    # logits = -d / (2*sigma^2)
    nodes.append(oh.make_node(
        'Mul', [dist_name, '_plvq_scale'], ['_plvq_logits'],
    ))

    # probs = softmax(logits, axis=1)
    nodes.append(oh.make_node(
        'Softmax', ['_plvq_logits'], ['_plvq_probs'],
        axis=1,
    ))

    # class_probs = probs @ class_mask  -> (batch, n_classes)
    nodes.append(oh.make_node(
        'MatMul', ['_plvq_probs', '_plvq_mask'], ['_plvq_class_probs'],
    ))

    # predictions = argmax(class_probs, axis=1)
    nodes.append(oh.make_node(
        'ArgMax', ['_plvq_class_probs'], ['_plvq_preds_raw'],
        axis=1, keepdims=0,
    ))
    nodes.append(oh.make_node(
        'Cast', ['_plvq_preds_raw'], ['predictions'],
        to=TensorProto.INT64,
    ))

    return nodes, inits


# ---------------------------------------------------------------------------
# Model type and distance identification
# ---------------------------------------------------------------------------

def _identify_model_type(model):
    """Identify the model type and distance function for ONNX export.

    Returns
    -------
    model_type : str
        One of 'supervised', 'unsupervised', 'oc_hard_nearest',
        'oc_gaussian_soft', 'oc_gaussian_ng', 'svqocc', 'cbc', 'plvq'.
    dist_type : str
        One of 'squared_euclidean', 'euclidean', 'manhattan', 'omega',
        'relevance', 'local_omega', 'tangent'.
    """
    # --- Determine model type ---

    has_encoder = (
        hasattr(model, 'backbone_params_')
        and model.backbone_params_ is not None
    )

    # Encoder models: detect first (before OC/supervised checks)
    if has_encoder:
        # ImageCBC: encoder + CBC reasoning
        if (hasattr(model, 'components_') and model.components_ is not None
                and hasattr(model, 'reasonings_')
                and model.reasonings_ is not None):
            model_type = 'cbc'
        # PLVQ: encoder + Gaussian mixture (unique: has loss_type)
        elif hasattr(model, 'loss_type'):
            model_type = 'plvq'
        # Siamese*, Image*, LVQMLN: encoder + WTAC
        else:
            model_type = 'supervised'
        dist_type = _identify_distance_fn(model, model_type)
        return model_type, dist_type

    # CBC (non-encoder): has reasonings_ but no backbone_params_
    if (hasattr(model, 'reasonings_') and model.reasonings_ is not None
            and hasattr(model, 'components_')
            and model.components_ is not None):
        dist_type = _identify_distance_fn(model, 'cbc')
        return 'cbc', dist_type

    # SVQ-OCC: unique response_type attribute
    is_svqocc = hasattr(model, 'response_type') and hasattr(model, 'gamma_resp')

    # One-class models: all have thetas_ (learned thresholds)
    has_thetas = (
        hasattr(model, 'thetas_') and model.thetas_ is not None
    )

    if is_svqocc:
        model_type = 'svqocc'
    elif has_thetas:
        # Distinguish OC decision patterns by sigma and gamma_
        has_sigma = hasattr(model, 'sigma') and not is_svqocc
        has_gamma = hasattr(model, 'gamma_') and model.gamma_ is not None
        if has_sigma and has_gamma:
            model_type = 'oc_gaussian_ng'
        elif has_sigma:
            model_type = 'oc_gaussian_soft'
        else:
            model_type = 'oc_hard_nearest'
    else:
        # Supervised or unsupervised (existing logic)
        has_labels = (
            hasattr(model, 'prototype_labels_')
            and model.prototype_labels_ is not None
        )
        model_type = 'supervised' if has_labels else 'unsupervised'

    # --- Determine distance type ---
    dist_type = _identify_distance_fn(model, model_type)

    return model_type, dist_type


def _identify_distance_fn(model, model_type='supervised') -> str:
    """Identify which distance function a model uses.

    For OC and SVQ-OCC models, distance is detected from learned
    metric parameters (omega_, omegas_, relevances_).  For supervised
    and unsupervised models, distance is detected from the distance_fn
    attribute.
    """
    # --- Attribute-based detection (OC, SVQ-OCC, and supervised metric models) ---

    # Tangent vs local omega: tangent models have subspace_dim
    has_omegas = hasattr(model, 'omegas_') and model.omegas_ is not None
    is_tangent = has_omegas and hasattr(model, 'subspace_dim')

    if is_tangent:
        return 'tangent'
    if has_omegas:
        return 'local_omega'

    # Global omega
    if hasattr(model, 'omega_') and model.omega_ is not None:
        return 'omega'

    # Relevance-weighted (GRLVQ family)
    if hasattr(model, 'relevances_') and model.relevances_ is not None:
        return 'relevance'

    # For OC/SVQ-OCC/CBC/PLVQ models without metric params,
    # default to squared_euclidean
    if model_type in ('oc_hard_nearest', 'oc_gaussian_soft',
                       'oc_gaussian_ng', 'svqocc', 'cbc', 'plvq'):
        return 'squared_euclidean'

    # Encoder models without explicit distance_fn: default to squared_euclidean
    if (hasattr(model, 'backbone_params_')
            and model.backbone_params_ is not None):
        return 'squared_euclidean'

    # --- Function-based detection (supervised/unsupervised models) ---
    from prosemble.core.distance import (
        squared_euclidean_distance_matrix,
        euclidean_distance_matrix,
        manhattan_distance_matrix,
    )

    fn = model.distance_fn

    _known = {
        squared_euclidean_distance_matrix: 'squared_euclidean',
        euclidean_distance_matrix: 'euclidean',
        manhattan_distance_matrix: 'manhattan',
    }

    for known_fn, name in _known.items():
        if fn is known_fn:
            return name

    if hasattr(fn, 'func'):
        from prosemble.core.distance import omega_distance_matrix
        if fn.func is omega_distance_matrix:
            return 'omega'

    fn_name = getattr(fn, '__name__', '') or getattr(fn, '__wrapped__', '')
    if 'squared_euclidean' in str(fn_name):
        return 'squared_euclidean'
    if 'euclidean' in str(fn_name) and 'squared' not in str(fn_name):
        return 'euclidean'
    if 'manhattan' in str(fn_name):
        return 'manhattan'
    if 'omega_distance_matrix' in str(fn_name):
        return 'omega'

    raise NotImplementedError(
        f"ONNX export is not supported for distance function "
        f"'{fn_name or fn}'. Supported: squared_euclidean, euclidean, "
        f"manhattan, omega, relevance, local_omega, tangent."
    )


def _check_model_exportable(model):
    """Check if a model can be exported to ONNX.

    Raises NotImplementedError for models that cannot be converted.
    """
    # Riemannian models: check manifold-specific exportability
    from prosemble.models.riemannian_srng import RiemannianSRNG
    if isinstance(model, RiemannianSRNG):
        from prosemble.core.manifolds import SO, Grassmannian, SPD
        if isinstance(model.manifold, SPD):
            raise NotImplementedError(
                "ONNX export is not supported for Riemannian models with "
                "SPD(n) manifold. Eigendecomposition (eigh) has no ONNX "
                "equivalent."
            )
        model_name = type(model).__name__
        if (isinstance(model.manifold, Grassmannian)
                and model_name == 'RiemannianSRNG'):
            raise NotImplementedError(
                "ONNX export is not supported for RiemannianSRNG with "
                "Grassmannian manifold. SVD-based geodesic distance has "
                "no ONNX equivalent."
            )
        # SO(n) for all 4 models, and Grassmannian for SMNG/SLNG/STNG are OK
        return

    try:
        from prosemble.models.riemannian_neural_gas import RiemannianNeuralGas
        if isinstance(model, RiemannianNeuralGas):
            raise NotImplementedError(
                "ONNX export is not supported for RiemannianNeuralGas. "
                "Manifold operations have no ONNX equivalent."
            )
    except ImportError:
        pass

    # Kernel fuzzy clustering models
    if hasattr(model, 'sigma') and hasattr(model, 'centroids_'):
        raise NotImplementedError(
            "ONNX export is not supported for kernel fuzzy clustering "
            "models. Kernel distance (Gaussian kernel) has no standard "
            "ONNX equivalent."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_onnx(
    model: Any,
    batch_size: int = 1,
    opset_version: int = 17,
    path: str | None = None,
):
    """Export a fitted model's predict function to ONNX format.

    Builds an ONNX graph that reproduces the model's ``predict()``
    output.  Supports supervised (WTAC), unsupervised (ArgMin),
    one-class (threshold-based), SVQ-OCC (response model), CBC
    (reasoning matrices), PLVQ (Gaussian mixture), encoder models
    (MLP/CNN backbone), and Riemannian models on SO(n)/Grassmannian
    manifolds (75 of 87 models total).

    Parameters
    ----------
    model : SupervisedPrototypeModel or UnsupervisedPrototypeModel
        A fitted prosemble model.
    batch_size : int
        Fixed batch dimension for the input.  Use ``-1`` for dynamic
        batch size (ONNX symbolic dimension).
    opset_version : int
        ONNX opset version.  Default: 17.
    path : str, optional
        If provided, save the ONNX model to this file path.

    Returns
    -------
    onnx.ModelProto
        The exported ONNX model.

    Raises
    ------
    NotImplementedError
        If the model's distance function or decision pattern is not
        supported.
    ImportError
        If the ``onnx`` package is not installed.
    """
    _check_onnx_installed()
    import onnx
    import onnx.helper as oh
    from onnx import TensorProto

    model._check_fitted()
    _check_model_exportable(model)

    # Riemannian models: use dedicated export path
    from prosemble.models.riemannian_srng import RiemannianSRNG
    if isinstance(model, RiemannianSRNG):
        return _export_riemannian_onnx(model, batch_size, opset_version, path)

    # Identify model type and distance
    model_type, dist_type = _identify_model_type(model)

    # Check for encoder (MLP/CNN backbone)
    has_encoder = (
        hasattr(model, 'backbone_params_')
        and model.backbone_params_ is not None
    )
    is_cnn = has_encoder and isinstance(model.backbone_params_, dict)

    # Input shape
    batch_dim = batch_size if batch_size > 0 else 'batch'

    all_nodes = []
    initializers = []

    if has_encoder:
        # --- Encoder model path ---

        # Determine input dimension
        if is_cnn:
            n_input_features = int(np.prod(model.input_shape))
        else:
            n_input_features = int(
                np.asarray(model.backbone_params_[0][0]).shape[0]
            )
        input_shape = [batch_dim, n_input_features]

        # Build encoder ONNX nodes
        if is_cnn:
            enc_nodes, enc_inits, enc_out = _cnn_encoder_onnx(
                'X', model.backbone_params_, model.input_shape,
                model.activation,
            )
        else:
            enc_nodes, enc_inits, enc_out = _mlp_encoder_onnx(
                'X', model.backbone_params_, model.activation,
            )
        all_nodes.extend(enc_nodes)
        initializers.extend(enc_inits)

        # Pre-compute latent prototypes/components
        model_name = type(model).__name__
        proto_need_encoding = (
            model_name.startswith('Siamese')
            or model_name.startswith('Image')
        )

        if model_type == 'cbc':
            # CBC models use components_
            comps = np.asarray(model.components_, dtype=np.float32)
            if proto_need_encoding:
                if is_cnn:
                    comps_img = comps.reshape(-1, *model.input_shape)
                    latent_protos = _cnn_forward_np(
                        model.backbone_params_, comps_img, model.activation,
                    )
                else:
                    latent_protos = _mlp_forward_np(
                        model.backbone_params_, comps, model.activation,
                    )
            else:
                latent_protos = comps
        elif proto_need_encoding:
            protos = np.asarray(model.prototypes_, dtype=np.float32)
            if is_cnn:
                protos_img = protos.reshape(-1, *model.input_shape)
                latent_protos = _cnn_forward_np(
                    model.backbone_params_, protos_img, model.activation,
                )
            else:
                latent_protos = _mlp_forward_np(
                    model.backbone_params_, protos, model.activation,
                )
        else:
            # LVQMLN/PLVQ: prototypes already in latent space
            latent_protos = np.asarray(model.prototypes_, dtype=np.float32)

        latent_protos = latent_protos.astype(np.float32)
        n_proto = latent_protos.shape[0]
        X_for_distance = enc_out

    else:
        # --- Non-encoder model path ---
        prototypes = np.asarray(model.prototypes_, dtype=np.float32)
        n_proto, n_features = prototypes.shape
        input_shape = [batch_dim, n_features]
        latent_protos = prototypes
        X_for_distance = 'X'

    # --- Prototypes initializer ---
    initializers.append(
        oh.make_tensor(
            'prototypes', TensorProto.FLOAT,
            list(latent_protos.shape), latent_protos.flatten().tolist(),
        ),
    )

    # Prototype labels for supervised models
    has_labels = (
        model_type == 'supervised'
        and hasattr(model, 'prototype_labels_')
        and model.prototype_labels_ is not None
    )
    if has_labels:
        proto_labels = np.asarray(model.prototype_labels_).astype(np.int64)
        initializers.append(
            oh.make_tensor(
                'proto_labels', TensorProto.INT64,
                list(proto_labels.shape), proto_labels.flatten().tolist(),
            ),
        )

    # --- Distance nodes ---
    if dist_type == 'squared_euclidean':
        nodes, extra_inits, dist_out = _squared_euclidean_onnx(
            None, X_for_distance, 'prototypes',
        )
    elif dist_type == 'euclidean':
        nodes, extra_inits, dist_out = _euclidean_onnx(
            None, X_for_distance, 'prototypes',
        )
    elif dist_type == 'manhattan':
        nodes, extra_inits, dist_out = _manhattan_onnx(
            None, X_for_distance, 'prototypes',
        )
    elif dist_type == 'omega':
        omega = np.asarray(model.omega_, dtype=np.float32)
        initializers.append(
            oh.make_tensor(
                'omega', TensorProto.FLOAT,
                list(omega.shape), omega.flatten().tolist(),
            ),
        )
        nodes, extra_inits, dist_out = _omega_onnx(
            None, X_for_distance, 'prototypes', 'omega',
        )
    elif dist_type == 'relevance':
        relevances = np.asarray(model.relevances_).astype(np.float32)
        initializers.append(
            oh.make_tensor(
                'relevances', TensorProto.FLOAT,
                list(relevances.shape), relevances.flatten().tolist(),
            ),
        )
        nodes, extra_inits, dist_out = _relevance_weighted_onnx(
            None, X_for_distance, 'prototypes', 'relevances',
        )
    elif dist_type == 'local_omega':
        omegas = np.asarray(model.omegas_).astype(np.float32)
        initializers.append(
            oh.make_tensor(
                'omegas', TensorProto.FLOAT,
                list(omegas.shape), omegas.flatten().tolist(),
            ),
        )
        nodes, extra_inits, dist_out = _local_omega_onnx(
            None, X_for_distance, 'prototypes', 'omegas',
        )
    elif dist_type == 'tangent':
        omegas = np.asarray(model.omegas_).astype(np.float32)
        initializers.append(
            oh.make_tensor(
                'omegas', TensorProto.FLOAT,
                list(omegas.shape), omegas.flatten().tolist(),
            ),
        )
        nodes, extra_inits, dist_out = _tangent_onnx(
            None, X_for_distance, 'prototypes', 'omegas',
        )
    else:
        raise NotImplementedError(f"Unknown distance type: {dist_type}")

    all_nodes.extend(nodes)
    initializers.extend(extra_inits)

    # --- Decision / competition nodes ---
    if model_type == 'supervised':
        comp_nodes, comp_inits = _wtac_onnx_nodes(dist_out, 'proto_labels')
    elif model_type == 'unsupervised':
        comp_nodes, comp_inits = _argmin_onnx_nodes(dist_out)
    elif model_type == 'oc_hard_nearest':
        comp_nodes, comp_inits = _oc_hard_nearest_onnx(dist_out, model)
    elif model_type == 'oc_gaussian_soft':
        comp_nodes, comp_inits = _oc_gaussian_soft_onnx(dist_out, model)
    elif model_type == 'oc_gaussian_ng':
        comp_nodes, comp_inits = _oc_gaussian_ng_onnx(
            dist_out, model, n_proto,
        )
    elif model_type == 'svqocc':
        comp_nodes, comp_inits = _svqocc_onnx(dist_out, model, n_proto)
    elif model_type == 'cbc':
        comp_nodes, comp_inits = _cbc_onnx(dist_out, model)
    elif model_type == 'plvq':
        comp_nodes, comp_inits = _plvq_onnx(dist_out, model)
    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")

    all_nodes.extend(comp_nodes)
    initializers.extend(comp_inits)

    # --- Output dtype ---
    # OC and SVQ-OCC models output int32 (matching JAX astype(jnp.int32))
    if model_type in ('oc_hard_nearest', 'oc_gaussian_soft',
                       'oc_gaussian_ng', 'svqocc'):
        output_dtype = TensorProto.INT32
    else:
        output_dtype = TensorProto.INT64

    # --- Build graph ---
    X_input = oh.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    Y_output = oh.make_tensor_value_info(
        'predictions', output_dtype, [batch_dim],
    )

    graph = oh.make_graph(
        all_nodes,
        'prosemble_predict',
        [X_input],
        [Y_output],
        initializer=initializers,
    )

    onnx_model = oh.make_model(graph, opset_imports=[
        oh.make_opsetid('', opset_version),
    ])
    onnx_model.ir_version = 8

    onnx_model.doc_string = (
        f"Prosemble {type(model).__name__} predict function. "
        f"Distance: {dist_type}, decision: {model_type}."
    )

    onnx.checker.check_model(onnx_model)

    if path is not None:
        onnx.save(onnx_model, path)

    return onnx_model
