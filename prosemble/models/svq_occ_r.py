"""
Relevance-Weighted SVQ-OCC (SVQ-OCC-R).

Extends SVQ-OCC with per-feature adaptive relevance weighting,
following the GRLVQ pattern. The distance becomes:

    d_λ(x, w_k) = Σ_j λ_j (x_j - w_{k,j})²

where λ = softmax(relevances) are learned per-feature weights.
This enables the model to identify which features are most
discriminative for one-class classification.

References
----------
.. [1] Staps et al. (2022). Prototype-based One-Class-Classification
       Learning Using Local Representations. IEEE WSOM+ 2022.
.. [2] Hammer, Villmann (2002). Generalized Relevance Learning Vector
       Quantization. Neural Networks, 15(8-9), 1059-1068.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.svq_occ import SVQOCC


class SVQOCC_R(SVQOCC):
    """Relevance-Weighted SVQ-OCC.

    Extends SVQ-OCC with per-feature relevance weighting (like GRLVQ).
    Learns which features are most important for distinguishing
    target from non-target data.

    Parameters
    ----------
    n_prototypes : int
        Number of prototypes for the target class.
    target_label : int, optional
        Target (normal) class label. Default: auto-detect.
    alpha : float
        Balance between representation and classification. Default: 0.5.
    cost_function : str
        'contrastive', 'brier', or 'cross_entropy'. Default: 'contrastive'.
    response_type : str
        'gaussian', 'student_t', or 'uniform'. Default: 'gaussian'.
    sigma : float
        Sigmoid sharpness. Default: 0.1.
    gamma_resp : float
        Response bandwidth. Default: 1.0.
    nu : float
        Student-t degrees of freedom. Default: 1.0.
    lambda_init : float, optional
        Initial NG neighborhood range.
    lambda_final : float
        Final NG neighborhood range. Default: 0.01.
    lambda_decay : float, optional
        Per-step decay for lambda.

    Attributes
    ----------
    relevances_ : array of shape (n_features,)
        Learned per-feature relevance weights (softmax-normalized).

    See Also
    --------
    SVQOCC : Full SVQ-OCC parameter documentation.
    SupervisedPrototypeModel : Full list of base parameters.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.relevances_ = None

    def _get_resume_params(self, params):
        params = super()._get_resume_params(params)
        params['relevances'] = self._raw_relevances
        return params

    def _init_state(self, X, y, key):
        state, params, proto_labels = super()._init_state(X, y, key)
        n_features = X.shape[1]
        # Initialize uniform relevances (softmax of zeros = uniform)
        params['relevances'] = jnp.ones(n_features) / n_features
        # Reinitialize optimizer with the added parameter
        opt_state = self._optimizer.init(params)
        from prosemble.models.prototype_base import SupervisedState
        state = SupervisedState(
            prototypes=params['prototypes'],
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        prototypes = params['prototypes']
        thetas = params['thetas']
        lambda_ng = params['lambda_ng']
        relevances = params['relevances']

        n_protos = prototypes.shape[0]

        # Relevance-weighted squared Euclidean distances
        lam = jax.nn.softmax(relevances)  # (d,)
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, K, d)
        sq_distances = jnp.sum(lam[None, None, :] * diff ** 2, axis=2)  # (n, K)

        # Target / non-target masks
        target_mask = (y == self._target_label)

        # ===== Representation cost R (target data only) =====
        order = jnp.argsort(sq_distances, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)

        h_ng = jnp.exp(-ranks / (lambda_ng + 1e-10))
        R_per_sample = jnp.sum(h_ng * sq_distances, axis=1)
        R_per_sample = jnp.where(target_mask, R_per_sample, 0.0)
        n_target = jnp.sum(target_mask) + 1e-10
        R = jnp.sum(R_per_sample) / n_target

        # ===== Classification cost C =====
        if self.response_type == 'gaussian':
            logits = -self.gamma_resp * sq_distances
            p_k = jax.nn.softmax(logits, axis=1)
        elif self.response_type == 'student_t':
            p_unnorm = (1.0 + sq_distances / self.nu) ** (-(self.nu + 1) / 2)
            p_k = p_unnorm / (jnp.sum(p_unnorm, axis=1, keepdims=True) + 1e-10)
        else:  # uniform
            p_k = jnp.ones_like(sq_distances) / n_protos

        thetas_pos = jnp.maximum(thetas, 1e-6)
        heaviside = jax.nn.sigmoid(
            (thetas_pos[None, :] - sq_distances) / (self.sigma + 1e-10)
        )

        responsibility = p_k * heaviside
        total_resp = jnp.sum(responsibility, axis=1)
        total_resp = jnp.clip(total_resp, 1e-10, 1.0 - 1e-10)

        y_binary = target_mask.astype(jnp.float32)

        if self.cost_function == 'contrastive':
            TP = jnp.sum(y_binary * total_resp)
            FN = jnp.sum(y_binary) - TP
            FP = jnp.sum((1.0 - y_binary) * total_resp)
            TN = jnp.sum(1.0 - y_binary) - FP
            numerator = TP * TN - FP * FN
            denominator = (TP + FP + 1e-10) * (TN + FN + 1e-10)
            C = 1.0 - numerator / denominator
        elif self.cost_function == 'brier':
            C = jnp.mean((y_binary - total_resp) ** 2)
        else:  # cross_entropy
            C = -jnp.mean(
                y_binary * jnp.log(total_resp) +
                (1.0 - y_binary) * jnp.log(1.0 - total_resp)
            )

        return self.alpha * R + (1.0 - self.alpha) * C

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.relevances_ = jax.nn.softmax(params['relevances'])
        self._raw_relevances = params['relevances']

    def decision_function(self, X):
        """Compute scores using relevance-weighted distances."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)

        diff = X[:, None, :] - self.prototypes_[None, :, :]
        sq_distances = jnp.sum(
            self.relevances_[None, None, :] * diff ** 2, axis=2
        )
        n_protos = self.prototypes_.shape[0]

        if self.response_type == 'gaussian':
            logits = -self.gamma_resp * sq_distances
            p_k = jax.nn.softmax(logits, axis=1)
        elif self.response_type == 'student_t':
            p_unnorm = (
                (1.0 + sq_distances / self.nu) ** (-(self.nu + 1) / 2)
            )
            p_k = p_unnorm / (
                jnp.sum(p_unnorm, axis=1, keepdims=True) + 1e-10
            )
        else:
            p_k = jnp.ones_like(sq_distances) / n_protos

        heaviside = jax.nn.sigmoid(
            (self.thetas_[None, :] - sq_distances) / (self.sigma + 1e-10)
        )

        responsibility = p_k * heaviside
        return jnp.clip(jnp.sum(responsibility, axis=1), 0.0, 1.0)

    @property
    def relevance_profile(self):
        """Return the learned per-feature relevance weights (normalized)."""
        if self.relevances_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.relevances_

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.relevances_ is not None:
            attrs.append('relevances_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.relevances_ is not None:
            arrays['relevances_'] = np.asarray(self.relevances_)
        if hasattr(self, '_raw_relevances') and self._raw_relevances is not None:
            arrays['_raw_relevances'] = np.asarray(self._raw_relevances)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'relevances_' in arrays:
            self.relevances_ = jnp.asarray(arrays['relevances_'])
        if '_raw_relevances' in arrays:
            self._raw_relevances = jnp.asarray(arrays['_raw_relevances'])
