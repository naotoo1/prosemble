"""
Cross-Entropy LVQ with Neural Gas cooperation (CELVQ-NG).

Combines CELVQ's cross-entropy loss over all-class softmax logits with
Neural Gas rank-based neighborhood cooperation. Instead of using the
hard per-class minimum distance (as in CELVQ), prototypes within each
class are weighted by their NG rank: h_k = exp(-rank / gamma). This
replaces the hard min pooling with a soft NG-weighted pooling.

When gamma -> 0, only the nearest prototype per class dominates and
CELVQ-NG recovers standard CELVQ.

References
----------
.. [1] Villmann, T., et al. (2019). Analysis of variants of
       classification learning vector quantization by a stochastic
       setting.
.. [2] Hammer, B., Strickert, M., & Villmann, T. (2003). Supervised
       Neural Gas with General Similarity Measure. Neural Processing
       Letters.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel


class CELVQ_NG(SupervisedPrototypeModel):
    """Cross-Entropy LVQ with Neural Gas neighborhood cooperation.

    For each class, prototypes are ranked by distance and weighted
    by exp(-rank / gamma). The NG-weighted class distances become
    logits for cross-entropy loss over all classes simultaneously.

    Parameters
    ----------
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    gamma_init : float, optional
        Initial neighborhood range. Default: max prototypes per class / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay factor for gamma.
        Default: computed from max_iter so gamma reaches gamma_final.

    Attributes
    ----------
    gamma_ : float
        Final gamma value after training.
    """

    def __init__(self, gamma_init=None, gamma_final=0.01,
                 gamma_decay=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.gamma_decay = gamma_decay
        self.gamma_ = None

        # Ensure gamma is frozen from optimizer (not trainable)
        if self.freeze_params is None:
            self.freeze_params = ['gamma']
        elif 'gamma' not in self.freeze_params:
            self.freeze_params = list(self.freeze_params) + ['gamma']

    def _get_resume_params(self, params):
        gamma = self.gamma_ if self.gamma_ is not None else (
            self._gamma_init_actual if hasattr(self, '_gamma_init_actual') else 1.0
        )
        params['gamma'] = jnp.array(gamma, dtype=jnp.float32)
        return params

    def _init_state(self, X, y, key):
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

        params = {
            'prototypes': prototypes,
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
        gamma = params['gamma']
        n_classes = self.n_classes_

        # 1. Compute distances (n, p)
        distances = self.distance_fn(X, prototypes)

        # 2. NG-weighted per-class distance pooling
        INF = jnp.finfo(distances.dtype).max
        class_dists_list = []

        for c in range(n_classes):
            # Mask: which prototypes belong to class c
            class_mask = (proto_labels == c)  # (p,)

            # Distances to class c prototypes (INF for non-class)
            d_class = jnp.where(class_mask[None, :], distances, INF)  # (n, p)

            # Rank within class c (double argsort)
            order = jnp.argsort(d_class, axis=1)
            ranks = jnp.argsort(order, axis=1).astype(jnp.float32)  # (n, p)

            # NG neighborhood function
            h = jnp.exp(-ranks / (gamma + 1e-10))  # (n, p)
            h = jnp.where(class_mask[None, :], h, 0.0)  # zero non-class

            # Normalize within class
            C = jnp.sum(h, axis=1, keepdims=True)  # (n, 1)
            h_normalized = h / (C + 1e-10)  # (n, p)

            # NG-weighted class distance
            weighted_dist = jnp.sum(h_normalized * distances, axis=1)  # (n,)
            class_dists_list.append(weighted_dist)

        # 3. Stack into (n, n_classes)
        class_dists = jnp.stack(class_dists_list, axis=1)  # (n, n_classes)

        # 4. Cross-entropy loss
        logits = -class_dists  # negate: smaller distance = larger logit
        log_probs = jax.nn.log_softmax(logits, axis=1)
        target_one_hot = jax.nn.one_hot(y, n_classes)
        return -jnp.mean(jnp.sum(target_one_hot * log_probs, axis=1))

    def _post_update(self, params):
        # Decay gamma (neighborhood range) each step
        new_gamma = params['gamma'] * self._gamma_decay
        new_gamma = jnp.maximum(new_gamma, self.gamma_final)
        return {**params, 'gamma': new_gamma}

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.gamma_ = float(params['gamma'])

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.gamma_ is not None:
            arrays['gamma_'] = np.asarray(self.gamma_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'gamma_' in arrays:
            self.gamma_ = float(arrays['gamma_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['gamma_init'] = self.gamma_init
        hp['gamma_final'] = self.gamma_final
        hp['gamma_decay'] = self.gamma_decay
        return hp
