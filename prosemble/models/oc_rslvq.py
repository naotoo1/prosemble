"""
One-Class Robust Soft LVQ (OC-RSLVQ).

Extends OC-GLVQ by replacing hard nearest-prototype assignment with
probabilistic soft-weighting via Gaussian mixture responsibilities:

    p(k|x) = exp(-d_k / 2σ²) / Σ_j exp(-d_j / 2σ²)
    μ_k = s · (d_k - θ_k) / (d_k + θ_k)
    loss = mean(Σ_k p(k|x) · sigmoid(μ_k + margin, β))

Unlike OC-GLVQ which uses only the nearest prototype for each sample,
OC-RSLVQ distributes evidence across all prototypes weighted by Gaussian
proximity. This provides smoother decision boundaries and natural
uncertainty quantification.

References
----------
.. [1] Seo, S., & Obermayer, K. (2003). Soft Nearest Prototype
       Classification. IEEE Trans. Neural Networks, 15(7):1589-1604.
.. [2] Seo, S., & Obermayer, K. (2007). Soft Learning Vector
       Quantization. Neural Computation, 19(6):1589-1604.
.. [3] Staps et al. (2022). Prototype-based One-Class-Classification
       Learning Using Local Representations. IJCNN 2022.
"""

import jax
import jax.numpy as jnp

from prosemble.models.oc_glvq import OCGLVQ
from prosemble.core.activations import sigmoid_beta


class OCRSLVQ(OCGLVQ):
    """One-Class Robust Soft LVQ.

    Combines one-class threshold detection with probabilistic soft-weighting
    of all prototypes via Gaussian mixture responsibilities.

    All prototypes contribute to the one-class decision via Gaussian
    proximity weights, with standard Euclidean distances.

    Parameters
    ----------
    sigma : float
        Bandwidth of Gaussian mixture for prototype weighting.
    n_prototypes : int
        Number of prototypes for the target class. Default: 3.
    target_label : int, optional
        Target (normal) class label. Default: auto-detect.
    beta : float
        Sigmoid steepness. Default: 10.0.

    Attributes
    ----------
    thetas_ : array of shape (n_prototypes,)
        Learned per-prototype acceptance thresholds.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    def __init__(self, sigma=1.0, n_prototypes=3, target_label=None,
                 beta=10.0, max_iter=100, lr=0.01, epsilon=1e-6,
                 random_seed=42, distance_fn=None, optimizer='adam',
                 transfer_fn=None, margin=0.0, callbacks=None,
                 use_scan=True, batch_size=None, lr_scheduler=None,
                 lr_scheduler_kwargs=None, prototypes_initializer=None,
                 patience=None, restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None,
                 mixed_precision=None):
        super().__init__(
            n_prototypes=n_prototypes, target_label=target_label,
            beta=beta, max_iter=max_iter, lr=lr, epsilon=epsilon,
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
        self.sigma = sigma

    def _compute_loss(self, params, X, y, proto_labels):
        prototypes = params['prototypes']
        thetas = params['thetas']

        # Squared Euclidean distances: (n, K)
        distances = self.distance_fn(X, prototypes)

        # Gaussian weights: p(k|x) for all prototypes
        log_probs = -distances / (2.0 * self.sigma ** 2)
        log_norm = jnp.max(log_probs, axis=1, keepdims=True)
        weights = jnp.exp(log_probs - log_norm)
        weights = weights / jnp.sum(weights, axis=1, keepdims=True)

        # Per-prototype OC mu
        s = jnp.where(y == self._target_label, 1.0, -1.0)
        mu = s[:, None] * (distances - thetas[None, :]) / (
            distances + thetas[None, :] + 1e-10
        )

        # Weighted sigmoid loss
        transfer = self.transfer_fn or sigmoid_beta
        cost = transfer(mu + self.margin, self.beta)
        return jnp.mean(jnp.sum(weights * cost, axis=1))

    def decision_function(self, X):
        """Compute target-likeness scores using soft-weighted distances.

        Scores near 1.0 indicate target class, near 0.0 indicate outlier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : array of shape (n_samples,)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)

        distances = self.distance_fn(X, self.prototypes_)

        # Gaussian weights
        log_probs = -distances / (2.0 * self.sigma ** 2)
        log_norm = jnp.max(log_probs, axis=1, keepdims=True)
        weights = jnp.exp(log_probs - log_norm)
        weights = weights / jnp.sum(weights, axis=1, keepdims=True)

        # Per-prototype mu (from target perspective)
        mu = (distances - self.thetas_[None, :]) / (
            distances + self.thetas_[None, :] + 1e-10
        )
        # Weighted score
        weighted_mu = jnp.sum(weights * mu, axis=1)
        return 1.0 - jax.nn.sigmoid(self.beta * weighted_mu)

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['sigma'] = self.sigma
        return hp
