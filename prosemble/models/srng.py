"""
Supervised Relevance Neural Gas (SRNG).

Combines GLVQ's classification loss with Neural Gas neighborhood
cooperation and per-feature adaptive relevance weighting. Unlike
standard GLVQ, which only adapts the closest correct/wrong prototypes,
SRNG updates all same-class prototypes weighted by their rank, making
it robust to initialization and capable of handling highly multimodal
data distributions.

References
----------
.. [1] Hammer, B., Strickert, M., & Villmann, T. (2003). Supervised
       Neural Gas with General Similarity Measure. Neural Processing
       Letters.
.. [2] Hammer, B., Strickert, M., & Villmann, T. (2004). Relevance
       LVQ versus SVM. Lecture Notes in Computer Science.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel


class SRNG(SupervisedPrototypeModel):
    """Supervised Relevance Neural Gas.

    Combines three key ideas:

    - GLVQ loss: ``(d+ - d-) / (d+ + d-)`` for margin-based classification
    - Neural Gas cooperation: all same-class prototypes participate in
      the loss, weighted by rank via ``exp(-rank / gamma)``
    - Relevance weighting: per-feature ``lambda_j`` learned during training

    The neighborhood range gamma decays during training from gamma_init
    to gamma_final. When gamma -> 0, SRNG recovers standard GRLVQ.

    Parameters
    ----------
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    beta : float
        Transfer function steepness parameter.
    gamma_init : float, optional
        Initial neighborhood range. Default: max prototypes per class / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay factor for gamma.
        Default: computed from max_iter so gamma reaches gamma_final.
    """

    def __init__(self, beta=10.0, gamma_init=None, gamma_final=0.01,
                 gamma_decay=None, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.gamma_decay = gamma_decay
        self.relevances_ = None
        self.gamma_ = None

        # Ensure gamma is frozen from optimizer (not trainable)
        if self.freeze_params is None:
            self.freeze_params = ['gamma']
        elif 'gamma' not in self.freeze_params:
            self.freeze_params = list(self.freeze_params) + ['gamma']

    def _get_resume_params(self, params):
        params['relevances'] = self.relevances_
        gamma = self.gamma_ if self.gamma_ is not None else (
            self._gamma_init_actual if hasattr(self, '_gamma_init_actual') else 1.0
        )
        params['gamma'] = jnp.array(gamma, dtype=jnp.float32)
        return params

    def _init_state(self, X, y, key):
        n_features = X.shape[1]
        key1, key2 = jax.random.split(key)
        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )

        # Compute gamma_init from prototype count if not set
        if isinstance(self.n_prototypes_per_class, int):
            max_per_class = self.n_prototypes_per_class
        elif isinstance(self.n_prototypes_per_class, dict):
            max_per_class = max(self.n_prototypes_per_class.values())
        else:
            max_per_class = max(self.n_prototypes_per_class)
        gamma_init = self.gamma_init if self.gamma_init is not None else max_per_class / 2.0
        gamma_init = max(gamma_init, self.gamma_final + 1e-6)
        self._gamma_init_actual = gamma_init

        # Compute decay factor
        if self.gamma_decay is not None:
            self._gamma_decay = self.gamma_decay
        else:
            self._gamma_decay = (self.gamma_final / gamma_init) ** (1.0 / self.max_iter)

        relevances = jnp.ones(n_features) / n_features
        params = {
            'prototypes': prototypes,
            'relevances': relevances,
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
        relevances = params['relevances']
        gamma = params['gamma']

        # 1. Relevance-weighted squared Euclidean distance
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, p, d)
        lam = jax.nn.softmax(relevances)  # (d,)
        distances = jnp.sum(lam[None, None, :] * diff ** 2, axis=2)  # (n, p)

        # 2. Compute ranks within same-class prototypes
        same_class = (y[:, None] == proto_labels[None, :])  # (n, p)
        INF = jnp.finfo(distances.dtype).max
        d_same = jnp.where(same_class, distances, INF)  # (n, p)

        # Double argsort for ranks (same pattern as neural_gas.py)
        order = jnp.argsort(d_same, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)  # (n, p)

        # 3. Neighborhood function h = exp(-rank / gamma)
        h = jnp.exp(-ranks / (gamma + 1e-10))  # (n, p)
        h = jnp.where(same_class, h, 0.0)  # zero for wrong-class

        # 4. Normalize: C = sum of h over same-class prototypes per sample
        C = jnp.sum(h, axis=1, keepdims=True)  # (n, 1)
        h_normalized = h / (C + 1e-10)  # (n, p)

        # 5. Closest different-class prototype distance
        d_diff = jnp.where(~same_class, distances, INF)
        dm = jnp.min(d_diff, axis=1)  # (n,)

        # 6. GLVQ mu for each (sample, same-class prototype) pair
        mu = (distances - dm[:, None]) / (distances + dm[:, None] + 1e-10)  # (n, p)

        # 7. Apply transfer function
        from prosemble.core.activations import sigmoid_beta
        transfer = self.transfer_fn or sigmoid_beta
        cost = transfer(mu + self.margin, self.beta)  # (n, p)

        # 8. Rank-weighted sum over same-class prototypes, then mean over samples
        weighted_cost = jnp.sum(h_normalized * cost, axis=1)  # (n,)
        return jnp.mean(weighted_cost)

    def _post_update(self, params):
        # Decay gamma (neighborhood range) each step
        new_gamma = params['gamma'] * self._gamma_decay
        new_gamma = jnp.maximum(new_gamma, self.gamma_final)
        return {**params, 'gamma': new_gamma}

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.relevances_ = jax.nn.softmax(params['relevances'])
        self.gamma_ = float(params['gamma'])

    @property
    def relevance_profile(self):
        """Return the learned relevance weights (normalized)."""
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
        if self.gamma_ is not None:
            arrays['gamma_'] = np.asarray(self.gamma_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'relevances_' in arrays:
            self.relevances_ = jnp.asarray(arrays['relevances_'])
        if 'gamma_' in arrays:
            self.gamma_ = float(arrays['gamma_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['beta'] = self.beta
        hp['gamma_init'] = self.gamma_init
        hp['gamma_final'] = self.gamma_final
        hp['gamma_decay'] = self.gamma_decay
        return hp
