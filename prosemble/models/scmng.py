"""
Supervised Class-wise Matrix Neural Gas (SCMNG).

Combines per-class linear transformations :math:`\\Omega_c` with Neural Gas
neighborhood cooperation. Each class shares a single :math:`\\Omega_c` matrix,
so cooperating same-class prototypes contribute aligned gradients to their
shared metric — avoiding the gradient dilution of global :math:`\\Omega` (SMNG)
while using fewer parameters than per-prototype :math:`\\Omega_k` (SLNG).

Cost function:

.. math::

    E_{\\text{SCMNG}} = \\frac{1}{N} \\sum_\\mu \\sum_{r: c(w_r)=c(x_\\mu)}
        \\frac{h(\\text{rank}_r, \\gamma)}{C(\\gamma)} \\cdot \\Phi(\\mu_r)

where:

.. math::

    d(x, w_r) = \\|\\Omega_{c(w_r)}(x - w_r)\\|^2 \\quad \\text{(class-wise projection)}

.. math::

    \\mu_r = \\frac{d_r - d_r^-}{d_r + d_r^-}

.. math::

    h(\\text{rank}, \\gamma) = \\exp(-\\text{rank} / \\gamma)
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.competitions import wtac
from prosemble.core.initializers import identity_omega_init


@jit
def _predict_scmng_jit(X, prototypes, omegas, proto_labels):
    """JIT-compiled SCMNG prediction with class-wise Omega metrics."""
    diff = X[:, None, :] - prototypes[None, :, :]
    omega_per_proto = omegas[proto_labels]
    projected = jnp.einsum('npd,pdl->npl', diff, omega_per_proto)
    distances = jnp.sum(projected ** 2, axis=2)
    return wtac(distances, proto_labels)


class SCMNG(SupervisedPrototypeModel):
    """Supervised Class-wise Matrix Neural Gas.

    Each class c has its own :math:`\\Omega_c` matrix. Cooperating prototypes
    within the same class share :math:`\\Omega_c`, so their gradients are
    aligned (all point through the same class-specific metric).

    Parameters
    ----------
    latent_dim : int, optional
        Dimensionality of each class :math:`\\Omega_c` projection space.
        If None, uses input dim.
    beta : float
        Transfer function steepness parameter.
    gamma_init : float, optional
        Initial neighborhood range. Default: max prototypes per class / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay for gamma.
    lr_ratio : float
        Ratio of wrong-class to correct-class learning rate.
        Default: 1.0 (equal).
    omega_lr : float, optional
        Separate learning rate for omega matrices. Default: None (use lr).
    n_prototypes_per_class : int
        Number of prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    """

    def __init__(self, latent_dim=None, beta=10.0, gamma_init=None,
                 gamma_final=0.01, gamma_decay=None, lr_ratio=1.0,
                 omega_lr=None, n_prototypes_per_class=1, max_iter=100,
                 lr=0.01, epsilon=1e-6, random_seed=42, distance_fn=None,
                 optimizer='adam', transfer_fn=None, margin=0.0,
                 callbacks=None, use_scan=True, batch_size=None,
                 lr_scheduler=None, lr_scheduler_kwargs=None,
                 prototypes_initializer=None, patience=None,
                 restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None,
                 mixed_precision=None):
        super().__init__(
            n_prototypes_per_class=n_prototypes_per_class,
            max_iter=max_iter, lr=lr, epsilon=epsilon,
            random_seed=random_seed, distance_fn=distance_fn,
            optimizer=optimizer, transfer_fn=transfer_fn, margin=margin,
            callbacks=callbacks, use_scan=use_scan, batch_size=batch_size,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            prototypes_initializer=prototypes_initializer,
            patience=patience, restore_best=restore_best,
            class_weight=class_weight,
            gradient_accumulation_steps=gradient_accumulation_steps,
            ema_decay=ema_decay, freeze_params=freeze_params,
            lookahead=lookahead, mixed_precision=mixed_precision,
        )
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.gamma_decay = gamma_decay
        self.lr_ratio = lr_ratio
        self.omega_lr = omega_lr
        self.omegas_ = None
        self.gamma_ = None

        if omega_lr is not None and isinstance(optimizer, str):
            import optax
            proto_opt = self._build_optimizer(optimizer, lr)
            metric_opt = self._build_optimizer(optimizer, omega_lr)
            self._optimizer = optax.multi_transform(
                {'prototypes': proto_opt, 'omegas': metric_opt, 'gamma': proto_opt},
                param_labels=lambda params: {k: k for k in params},
            )

        if self.freeze_params is None:
            self.freeze_params = ['gamma']
        elif 'gamma' not in self.freeze_params:
            self.freeze_params = list(self.freeze_params) + ['gamma']

    def _get_resume_params(self, params):
        params['omegas'] = self.omegas_
        gamma = self.gamma_ if self.gamma_ is not None else (
            self._gamma_init_actual if hasattr(self, '_gamma_init_actual') else 1.0
        )
        params['gamma'] = jnp.array(gamma, dtype=jnp.float32)
        return params

    def _init_state(self, X, y, key):
        n_features = X.shape[1]
        latent_dim = self.latent_dim or n_features
        key1, key2 = jax.random.split(key)

        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )

        n_classes = len(jnp.unique(y))
        omega_single = identity_omega_init(n_features, latent_dim)
        omegas = jnp.tile(omega_single[None, :, :], (n_classes, 1, 1))

        if isinstance(self.n_prototypes_per_class, int):
            max_per_class = self.n_prototypes_per_class
        elif isinstance(self.n_prototypes_per_class, dict):
            max_per_class = max(self.n_prototypes_per_class.values())
        else:
            max_per_class = max(self.n_prototypes_per_class)
        gamma_init = self.gamma_init if self.gamma_init is not None else max_per_class / 2.0
        gamma_init = max(gamma_init, self.gamma_final + 1e-6)
        self._gamma_init_actual = gamma_init

        if self.gamma_decay is not None:
            self._gamma_decay = self.gamma_decay
        else:
            self._gamma_decay = (self.gamma_final / gamma_init) ** (1.0 / self.max_iter)

        params = {
            'prototypes': prototypes,
            'omegas': omegas,
            'gamma': jnp.array(gamma_init, dtype=jnp.float32),
        }
        opt_state = self._optimizer.init(params)
        from prosemble.models.prototype_base import SupervisedState
        state = SupervisedState(
            prototypes=prototypes,
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        prototypes = params['prototypes']
        omegas = params['omegas']  # (C, d, l)
        gamma = params['gamma']

        # 1. Class-wise Omega distance: d(x, w_r) = ||Omega_{c(w_r)}(x - w_r)||^2
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, p, d)
        omega_per_proto = omegas[proto_labels]  # (p, d, l)
        projected = jnp.einsum('npd,pdl->npl', diff, omega_per_proto)  # (n, p, l)
        distances = jnp.sum(projected ** 2, axis=2)  # (n, p)

        # 2. Rank same-class prototypes by ascending distance
        same_class = (y[:, None] == proto_labels[None, :])  # (n, p)
        INF = jnp.finfo(distances.dtype).max
        d_same = jnp.where(same_class, distances, INF)  # (n, p)
        order = jnp.argsort(d_same, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)  # (n, p)

        # 3. Neighborhood function h = exp(-rank / gamma)
        h = jnp.exp(-ranks / (gamma + 1e-10))  # (n, p)
        h = jnp.where(same_class, h, 0.0)

        # 4. Normalize cooperation weights
        C = jnp.sum(h, axis=1, keepdims=True)  # (n, 1)
        h_normalized = h / (C + 1e-10)  # (n, p)

        # 5. Closest different-class prototype distance
        d_diff = jnp.where(~same_class, distances, INF)
        dm = jnp.min(d_diff, axis=1)  # (n,)

        # 5b. Optional separate learning rates
        if self.lr_ratio != 1.0:
            dm = jax.lax.stop_gradient(dm) + self.lr_ratio * (
                dm - jax.lax.stop_gradient(dm))

        # 6. GLVQ margin for each (sample, same-class prototype) pair
        mu = (distances - dm[:, None]) / (distances + dm[:, None] + 1e-10)  # (n, p)

        # 7. Apply transfer function
        from prosemble.core.activations import sigmoid_beta
        transfer = self.transfer_fn or sigmoid_beta
        cost = transfer(mu + self.margin, self.beta)  # (n, p)

        # 8. Rank-weighted sum over same-class prototypes, then mean over samples
        weighted_cost = jnp.sum(h_normalized * cost, axis=1)  # (n,)
        return jnp.mean(weighted_cost)

    def _post_update(self, params):
        new_gamma = params['gamma'] * self._gamma_decay
        new_gamma = jnp.maximum(new_gamma, self.gamma_final)
        return {**params, 'gamma': new_gamma}

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.omegas_ = params['omegas']
        self.gamma_ = float(params['gamma'])

    def predict(self, X):
        """Predict using class-wise :math:`\\Omega_c` distances."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_scmng_jit(
            X, self.prototypes_, self.omegas_, self.prototype_labels_
        )

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.omegas_ is not None:
            attrs.append('omegas_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.omegas_ is not None:
            arrays['omegas_'] = np.asarray(self.omegas_)
        if self.gamma_ is not None:
            arrays['gamma_'] = np.asarray(self.gamma_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'omegas_' in arrays:
            self.omegas_ = jnp.asarray(arrays['omegas_'])
        if 'gamma_' in arrays:
            self.gamma_ = float(arrays['gamma_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['beta'] = self.beta
        hp['gamma_init'] = self.gamma_init
        hp['gamma_final'] = self.gamma_final
        hp['gamma_decay'] = self.gamma_decay
        hp['lr_ratio'] = self.lr_ratio
        hp['omega_lr'] = self.omega_lr
        if self.latent_dim is not None:
            hp['latent_dim'] = self.latent_dim
        return hp
