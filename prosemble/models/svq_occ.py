"""
Supervised Vector Quantization One-Class Classification (SVQ-OCC).

Combines Neural Gas representation learning with one-class classification
using per-prototype visibility parameters. Each prototype is equipped with
a local visibility range θ_k — data within θ_k of prototype w_k is
classified as target.

The overall cost function is:

    E(X, W) = α · R(X⁺, W) + (1 − α) · C(X, W, Θ)

where R is the Neural Gas representation cost over target data X⁺, and
C is a classification cost using per-prototype responsibilities:

    r(x, w_k, γ_k, θ_k) = p(w_k, γ_k | x) · sgd_σ(d(x, w_k), θ_k)

Three classification costs are available:
- Contrastive Score (CS): 1 − (TP·TN − FP·FN) / ((TP+FP)(TN+FN))
- Brier Score (BS): mean (y − Σ_k r)²
- Cross Entropy (CE): −mean [y·log(Σr) + (1−y)·log(1−Σr)]

Three response probability models p(w_k|x) are supported:
- Gaussian: softmax(−γ·d(x, w_k))
- Student-t: (1 + d/ν)^(−(ν+1)/2), normalized
- Uniform: 1/K

References
----------
.. [1] Staps, C., Schubert, L., Kaden, M., Lampe, B., Hermann, W.,
       & Villmann, T. (2022). Prototype-based One-Class-Classification
       Learning Using Local Representations. IEEE WSOM+ 2022.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.initializers import stratified_selection_init


class SVQOCC(SupervisedPrototypeModel):
    """Supervised Vector Quantization One-Class Classification.

    Combines Neural Gas representation learning with per-prototype
    visibility parameters θ_k for one-class classification.

    Parameters
    ----------
    n_prototypes : int
        Number of prototypes for the target class.
    target_label : int, optional
        Which label is the target (normal) class. Default: auto-detect
        as the most frequent class.
    alpha : float
        Balance between representation (R) and classification (C) cost.
        E = α·R + (1−α)·C. Default: 0.5.
    cost_function : str
        Classification cost variant: 'contrastive', 'brier', 'cross_entropy'.
        Default: 'contrastive'.
    response_type : str
        Response probability model: 'gaussian', 'student_t', 'uniform'.
        Default: 'gaussian'.
    sigma : float
        Sigmoid sharpness for differentiable Heaviside approximation.
        Smaller = sharper boundary. Default: 0.1.
    gamma_resp : float
        Response bandwidth for Gaussian probabilistic assignment. Default: 1.0.
    nu : float
        Degrees of freedom for Student-t response. Default: 1.0.
    lambda_init : float, optional
        Initial NG neighborhood range. Default: n_prototypes / 2.
    lambda_final : float
        Final NG neighborhood range. Default: 0.01.
    lambda_decay : float, optional
        Per-step multiplicative decay for lambda. Default: computed from max_iter.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    _valid_costs = ('contrastive', 'brier', 'cross_entropy')
    _valid_responses = ('gaussian', 'student_t', 'uniform')

    def __init__(self, n_prototypes=3, target_label=None, alpha=0.5,
                 cost_function='contrastive', response_type='gaussian',
                 sigma=0.1, gamma_resp=1.0, nu=1.0,
                 lambda_init=None, lambda_final=0.01, lambda_decay=None,
                 **kwargs):
        super().__init__(n_prototypes_per_class=n_prototypes, **kwargs)
        self.n_prototypes = n_prototypes
        self.target_label = target_label
        self.alpha = alpha
        self.sigma = sigma
        self.gamma_resp = gamma_resp
        self.nu = nu
        self.lambda_init = lambda_init
        self.lambda_final = lambda_final
        self.lambda_decay = lambda_decay

        if cost_function not in self._valid_costs:
            raise ValueError(
                f"cost_function must be one of {self._valid_costs}, "
                f"got '{cost_function}'"
            )
        self.cost_function = cost_function

        if response_type not in self._valid_responses:
            raise ValueError(
                f"response_type must be one of {self._valid_responses}, "
                f"got '{response_type}'"
            )
        self.response_type = response_type

        # Fitted attributes
        self.thetas_ = None
        self.lambda_ = None
        self._target_label = None
        self._non_target_label = None

        # Freeze lambda_ng from optimizer (decayed manually)
        if self.freeze_params is None:
            self.freeze_params = ['lambda_ng']
        elif 'lambda_ng' not in self.freeze_params:
            self.freeze_params = list(self.freeze_params) + ['lambda_ng']

    def _get_resume_params(self, params):
        params['thetas'] = self.thetas_
        lam = self.lambda_ if self.lambda_ is not None else (
            self._lambda_init_actual if hasattr(self, '_lambda_init_actual')
            else 1.0
        )
        params['lambda_ng'] = jnp.array(lam, dtype=jnp.float32)
        return params

    def _init_state(self, X, y, key):
        # Determine target and non-target labels
        classes = jnp.unique(y)
        if self.target_label is not None:
            self._target_label = int(self.target_label)
        else:
            # Auto-detect: most frequent class is target
            counts = jnp.array([jnp.sum(y == c) for c in classes])
            self._target_label = int(classes[jnp.argmax(counts)])

        non_target = classes[classes != self._target_label]
        self._non_target_label = int(non_target[0]) if len(non_target) > 0 else (
            1 - self._target_label
        )

        # Filter target class data for prototype initialization
        target_mask = (y == self._target_label)
        X_target = X[target_mask]
        y_target = jnp.full(X_target.shape[0], self._target_label, dtype=jnp.int32)

        key1, key2 = jax.random.split(key)

        # Initialize prototypes from target class data
        if self.prototypes_initializer is not None:
            prototypes, _ = self._init_prototypes(
                X_target, y_target, self.n_prototypes, key1
            )
        else:
            prototypes, _ = stratified_selection_init(
                X_target, y_target, self.n_prototypes, key1
            )
        proto_labels = jnp.full(
            self.n_prototypes, self._target_label, dtype=jnp.int32
        )

        # Initialize thetas: sqrt of mean squared distance per prototype
        from prosemble.core.distance import squared_euclidean_distance_matrix
        dists = squared_euclidean_distance_matrix(X_target, prototypes)
        thetas = jnp.sqrt(jnp.mean(dists, axis=0) + 1e-10)

        # Lambda NG setup
        lambda_init = (
            self.lambda_init if self.lambda_init is not None
            else self.n_prototypes / 2.0
        )
        lambda_init = max(lambda_init, self.lambda_final + 1e-6)
        self._lambda_init_actual = lambda_init

        if self.lambda_decay is not None:
            self._lambda_decay = self.lambda_decay
        else:
            self._lambda_decay = (
                self.lambda_final / lambda_init
            ) ** (1.0 / self.max_iter)

        params = {
            'prototypes': prototypes,
            'thetas': thetas,
            'lambda_ng': jnp.array(lambda_init, dtype=jnp.float32),
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
        thetas = params['thetas']
        lambda_ng = params['lambda_ng']

        n_protos = prototypes.shape[0]

        # Squared Euclidean distances: (n, K)
        diff = X[:, None, :] - prototypes[None, :, :]
        sq_distances = jnp.sum(diff ** 2, axis=2)

        # Target / non-target masks
        target_mask = (y == self._target_label)

        # ===== Representation cost R (target data only) =====
        # Neural Gas ranking over all samples but weighted for target only
        order = jnp.argsort(sq_distances, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)

        h_ng = jnp.exp(-ranks / (lambda_ng + 1e-10))
        R_per_sample = jnp.sum(h_ng * sq_distances, axis=1)
        R_per_sample = jnp.where(target_mask, R_per_sample, 0.0)
        n_target = jnp.sum(target_mask) + 1e-10
        R = jnp.sum(R_per_sample) / n_target

        # ===== Classification cost C =====
        # Response probability p(w_k | x)
        if self.response_type == 'gaussian':
            logits = -self.gamma_resp * sq_distances
            p_k = jax.nn.softmax(logits, axis=1)
        elif self.response_type == 'student_t':
            p_unnorm = (1.0 + sq_distances / self.nu) ** (-(self.nu + 1) / 2)
            p_k = p_unnorm / (jnp.sum(p_unnorm, axis=1, keepdims=True) + 1e-10)
        else:  # uniform
            p_k = jnp.ones_like(sq_distances) / n_protos

        # Sigmoid approximation of Heaviside: sgd_σ(d, θ_k) = σ((θ_k - d) / σ)
        # Keep thetas positive
        thetas_pos = jnp.maximum(thetas, 1e-6)
        heaviside = jax.nn.sigmoid(
            (thetas_pos[None, :] - sq_distances) / (self.sigma + 1e-10)
        )

        # Local responsibility: r(x, w_k) = p(w_k|x) · H(θ_k - d(x, w_k))
        responsibility = p_k * heaviside

        # Summed responsibility per sample
        total_resp = jnp.sum(responsibility, axis=1)
        total_resp = jnp.clip(total_resp, 1e-10, 1.0 - 1e-10)

        # Binary labels: 1 for target, 0 for non-target
        y_binary = target_mask.astype(jnp.float32)

        if self.cost_function == 'contrastive':
            # Probabilistic confusion matrix
            TP = jnp.sum(y_binary * total_resp)
            FN = jnp.sum(y_binary) - TP
            FP = jnp.sum((1.0 - y_binary) * total_resp)
            TN = jnp.sum(1.0 - y_binary) - FP

            # CS_W = 1 - (TP·TN - FP·FN) / ((TP+FP)(TN+FN))
            numerator = TP * TN - FP * FN
            denominator = (TP + FP + 1e-10) * (TN + FN + 1e-10)
            C = 1.0 - numerator / denominator

        elif self.cost_function == 'brier':
            # Brier Score: mean (y - Σ_k r)²
            C = jnp.mean((y_binary - total_resp) ** 2)

        else:  # cross_entropy
            # Binary Cross Entropy
            C = -jnp.mean(
                y_binary * jnp.log(total_resp) +
                (1.0 - y_binary) * jnp.log(1.0 - total_resp)
            )

        return self.alpha * R + (1.0 - self.alpha) * C

    def _post_update(self, params):
        # Keep thetas positive
        thetas = jnp.maximum(params['thetas'], 1e-6)
        # Decay lambda_ng
        new_lambda = params['lambda_ng'] * self._lambda_decay
        new_lambda = jnp.maximum(new_lambda, self.lambda_final)
        return {**params, 'thetas': thetas, 'lambda_ng': new_lambda}

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.thetas_ = jnp.maximum(params['thetas'], 1e-6)
        self.lambda_ = float(params['lambda_ng'])

    def predict(self, X):
        """Predict target or non-target labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : array of shape (n_samples,)
            target_label for target, non_target_label for outliers.
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        scores = self.decision_function(X)
        return jnp.where(
            scores >= 0.5, self._target_label, self._non_target_label
        ).astype(jnp.int32)

    def predict_with_reject(self, X, upper=0.5, lower=None, reject_label=-1):
        """Predict with a reject option for uncertain samples.

        Samples with scores between lower and upper are rejected
        (labeled reject_label) instead of being forced into a class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        upper : float
            Scores >= upper are classified as target. Default: 0.5.
        lower : float, optional
            Scores < lower are classified as non-target. Scores in
            [lower, upper) are rejected. Default: same as upper
            (no rejection zone, equivalent to predict).
        reject_label : int
            Label for rejected samples. Default: -1.

        Returns
        -------
        labels : array of shape (n_samples,)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        if lower is None:
            lower = upper
        scores = self.decision_function(X)
        labels = jnp.full(scores.shape, reject_label, dtype=jnp.int32)
        labels = jnp.where(scores >= upper, self._target_label, labels)
        labels = jnp.where(scores < lower, self._non_target_label, labels)
        return labels

    def predict_proba(self, X):
        """Predict probability of being target class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : array of shape (n_samples,)
            Probability of each sample belonging to the target class.
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return self.decision_function(X)

    def decision_function(self, X):
        """Compute summed responsibility scores.

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

        diff = X[:, None, :] - self.prototypes_[None, :, :]
        sq_distances = jnp.sum(diff ** 2, axis=2)
        n_protos = self.prototypes_.shape[0]

        # Response probability
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
        else:  # uniform
            p_k = jnp.ones_like(sq_distances) / n_protos

        # Sigmoid Heaviside
        heaviside = jax.nn.sigmoid(
            (self.thetas_[None, :] - sq_distances) / (self.sigma + 1e-10)
        )

        # Summed responsibility
        responsibility = p_k * heaviside
        return jnp.clip(jnp.sum(responsibility, axis=1), 0.0, 1.0)

    @property
    def visibility_radii(self):
        """Return the learned visibility radii θ_k for each prototype."""
        if self.thetas_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.thetas_

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.thetas_ is not None:
            attrs.append('thetas_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.thetas_ is not None:
            arrays['thetas_'] = np.asarray(self.thetas_)
        if self.lambda_ is not None:
            arrays['lambda_'] = np.asarray(self.lambda_)
        if self._target_label is not None:
            arrays['_target_label'] = np.asarray(self._target_label)
        if self._non_target_label is not None:
            arrays['_non_target_label'] = np.asarray(self._non_target_label)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'thetas_' in arrays:
            self.thetas_ = jnp.asarray(arrays['thetas_'])
        if 'lambda_' in arrays:
            self.lambda_ = float(arrays['lambda_'])
        if '_target_label' in arrays:
            self._target_label = int(arrays['_target_label'])
        if '_non_target_label' in arrays:
            self._non_target_label = int(arrays['_non_target_label'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        # Remove n_prototypes_per_class — we use n_prototypes instead
        hp.pop('n_prototypes_per_class', None)
        hp['n_prototypes'] = self.n_prototypes
        hp['target_label'] = self.target_label
        hp['alpha'] = self.alpha
        hp['cost_function'] = self.cost_function
        hp['response_type'] = self.response_type
        hp['sigma'] = self.sigma
        hp['gamma_resp'] = self.gamma_resp
        hp['nu'] = self.nu
        hp['lambda_init'] = self.lambda_init
        hp['lambda_final'] = self.lambda_final
        hp['lambda_decay'] = self.lambda_decay
        return hp

