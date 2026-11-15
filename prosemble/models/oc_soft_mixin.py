"""
One-class learning for soft prototype models by contrast with a reference.

A soft prototype model scores a sample by its components,
:math:`f_k(x) = b_k - d_k(x) / 2\\sigma^2`. Here the components are not
normalised densities: the offsets :math:`b_k` are free parameters, and
the target class has the unnormalised density

.. math::

    p_T(x) = \\sum_{k \\in T} e^{f_k(x)}.

The model is trained to tell samples of the target class from samples of
a known reference distribution :math:`q`, drawn :math:`\\nu` times as
often. With labelled outliers, these get components of their own:

.. math::

    P(c \\mid x) = \\frac{\\sum_{k \\in c} e^{f_k(x)}}
                       {\\sum_k e^{f_k(x)} + \\nu\\, q(x)},
    \\qquad
    P(\\text{reference} \\mid x) = \\frac{\\nu\\, q(x)}
                       {\\sum_k e^{f_k(x)} + \\nu\\, q(x)},

and the cost is the negative log-likelihood of the labels target,
outlier and reference. This is noise-contrastive estimation: at the
optimum :math:`\\sum_{k \\in c} e^{f_k(x)}` is the density of class
:math:`c` times its share of the data, whatever :math:`q` is, so the
offsets take the place of the normalising constants and the cost has a
finite optimum when only target samples are given.

References
----------
.. [1] Gutmann, M., & Hyvarinen, A. (2012). Noise-contrastive estimation
       of unnormalized statistical models, with applications to natural
       image statistics. Journal of Machine Learning Research, 13,
       307-361.
.. [2] Seo, S., & Obermayer, K. (2003). Soft Learning Vector
       Quantization. Neural Computation, 15(7), 1589-1604.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from prosemble.core.manifolds import SO, SPD, Grassmannian, HyperbolicPoincare

#: Internal label codes of the training rows.
TARGET, OUTLIER, REFERENCE = 0, 1, 2


def kmeans_init(X, y, n_per_class, key, n_iter=20):
    """Prototypes from k-means run separately on each class.

    Seeds are chosen as in k-means++ and refined by ``n_iter`` Lloyd
    steps, so that the prototypes of a class cover its modes.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
        Class labels ``0..n_classes-1``.
    n_per_class : int, list or dict
        Number of prototypes per class.
    key : jax.random.PRNGKey

    Returns
    -------
    prototypes : array of shape (n_prototypes, n_features)
    prototype_labels : array of shape (n_prototypes,)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    seed = int(jax.random.randint(key, (), 0, np.iinfo(np.int32).max))
    rng = np.random.default_rng(seed)
    prototypes, labels = [], []
    for c in np.unique(y):
        n = n_per_class if isinstance(n_per_class, int) else n_per_class[int(c)]
        points = X[y == c]
        if points.shape[0] < n:
            raise ValueError(
                f"Class {int(c)} has {points.shape[0]} samples, fewer than "
                f"the {n} prototypes requested for it.")
        centres = [points[rng.integers(points.shape[0])]]
        for _ in range(n - 1):
            d = np.min(((points[:, None, :] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
            total = d.sum()
            pick = rng.choice(points.shape[0], p=d / total) if total > 0 else rng.integers(points.shape[0])
            centres.append(points[pick])
        centres = np.array(centres)
        for _ in range(n_iter):
            assign = np.argmin(((points[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
            for k in range(n):
                if np.any(assign == k):
                    centres[k] = points[assign == k].mean(axis=0)
        prototypes.append(centres)
        labels.extend([int(c)] * n)
    return (jnp.asarray(np.concatenate(prototypes), dtype=jnp.float32),
            jnp.asarray(labels, dtype=jnp.int32))


class OneClassSoftMixin:
    """One-class training and scoring for a soft prototype model.

    Place it before the soft model in the bases. The soft model supplies
    the distance through its ``_logits``; this mixin adds the offsets
    :math:`b_k`, the reference distribution, the contrastive cost and the
    one-class prediction interface.

    The reference of this class is for data in a vector space: a Gaussian
    with independent features, centred at the mean of the target samples
    and ``noise_scale`` times as wide as they are.
    :class:`RiemannianReferenceMixin` replaces it for manifold data.

    Parameters
    ----------
    n_prototypes : int
        Number of prototypes of the target class. Default: 3.
    n_outlier_prototypes : int
        Number of prototypes for labelled outliers, used only when the
        training labels contain non-target samples; never more than there
        are such samples. Default: 3.
    target_label : int, optional
        Label of the target class. Default: the most frequent label.
    noise_ratio : float
        Reference samples drawn per target sample, :math:`\\nu`.
        Default: 5.
    noise_scale : float
        Width of the reference relative to the spread of the target
        samples. Default: 2.5.
    nu : float
        Fraction of the training targets that ``predict`` rejects; sets
        the density level of the decision boundary. Default: 0.05.
    outlier_weight : float or 'balanced'
        Weight of a labelled outlier in the cost relative to a target
        sample. ``'balanced'`` (default) gives the outliers together the
        weight of the targets together.
    **kwargs
        Parameters of the soft model.
    """

    _encode_labels = False
    _hyperparams = ('n_prototypes', 'n_outlier_prototypes', 'target_label',
                    'noise_ratio', 'noise_scale', 'nu', 'outlier_weight')
    _fitted_array_names = ('bias_', 'threshold_', 'reference_loc_',
                           'reference_scale_', 'target_label_',
                           'non_target_label_')

    def __init__(self, n_prototypes=3, n_outlier_prototypes=3,
                 target_label=None, noise_ratio=5, noise_scale=2.5, nu=0.05,
                 outlier_weight='balanced', **kwargs):
        kwargs.pop('n_prototypes_per_class', None)
        super().__init__(n_prototypes_per_class=n_prototypes, **kwargs)
        if not (isinstance(n_prototypes, int) and n_prototypes > 0):
            raise ValueError(f"n_prototypes must be a positive integer, got {n_prototypes!r}")
        if not (isinstance(n_outlier_prototypes, int) and n_outlier_prototypes > 0):
            raise ValueError(
                f"n_outlier_prototypes must be a positive integer, got {n_outlier_prototypes!r}")
        if not noise_ratio > 0:
            raise ValueError(f"noise_ratio must be positive, got {noise_ratio!r}")
        if not noise_scale > 0:
            raise ValueError(f"noise_scale must be positive, got {noise_scale!r}")
        if not 0 <= nu < 1:
            raise ValueError(f"nu must be in [0, 1), got {nu!r}")
        if not (outlier_weight == 'balanced'
                or (isinstance(outlier_weight, (int, float)) and outlier_weight > 0)):
            raise ValueError(
                f"outlier_weight must be 'balanced' or a positive number, got {outlier_weight!r}")
        self.n_prototypes = n_prototypes
        self.n_outlier_prototypes = n_outlier_prototypes
        self.target_label = target_label
        self.noise_ratio = noise_ratio
        self.noise_scale = noise_scale
        self.nu = nu
        self.outlier_weight = outlier_weight
        self.bias_ = None
        self.threshold_ = None
        self.reference_loc_ = None
        self.reference_scale_ = None
        self.target_label_ = None
        self.non_target_label_ = None
        self._outlier_weight = 1.0

    # ------------------------------------------------------------------
    # Reference distribution
    # ------------------------------------------------------------------

    def _fit_reference(self, X_target):
        """Set the reference distribution from the target samples."""
        mean = jnp.mean(X_target, axis=0)
        std = jnp.std(X_target, axis=0)
        typical = jnp.mean(std)
        typical = jnp.where(typical > 0, typical, 1.0)
        # A feature that is constant in the target class still needs a
        # reference of finite width.
        std = jnp.where(std < 1e-3 * typical, typical, std)
        self.reference_loc_ = mean
        self.reference_scale_ = self.noise_scale * std

    def _sample_reference(self, key, n):
        """Draw ``n`` samples of the reference, shape (n, n_features)."""
        noise = jax.random.normal(key, (n, self.reference_loc_.shape[0]))
        return self.reference_loc_ + self.reference_scale_ * noise

    def _reference_log_density(self, X):
        """Log density of the reference at ``X``, up to a constant."""
        z = (X - self.reference_loc_) / self.reference_scale_
        return -0.5 * jnp.sum(z ** 2, axis=1) - jnp.sum(jnp.log(self.reference_scale_))

    def _default_initializer(self):
        return kmeans_init

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, y=None, **kwargs):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Labels. Samples with the target label are targets, all others
            labelled outliers. Default: every sample is a target.
        **kwargs
            Passed to the ``fit`` of the soft model.

        Returns
        -------
        self
        """
        X = jnp.asarray(X, dtype=jnp.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        codes = self._label_codes(X, y)
        X_target = X[np.asarray(codes) == TARGET]
        n_target = X_target.shape[0]
        n_outliers = X.shape[0] - n_target
        if self.outlier_weight == 'balanced':
            self._outlier_weight = n_target / n_outliers if n_outliers else 1.0
        else:
            self._outlier_weight = float(self.outlier_weight)

        if not kwargs.get('resume', False):
            self._fit_reference(X_target)
        n_reference = max(1, int(round(self.noise_ratio * n_target)))
        X_reference = self._sample_reference(
            jax.random.fold_in(self.key, 1), n_reference)

        if kwargs.get('sample_weight') is not None:
            kwargs['sample_weight'] = jnp.concatenate([
                jnp.asarray(kwargs['sample_weight'], dtype=jnp.float32),
                jnp.ones(n_reference)])
        if kwargs.get('validation_data') is not None:
            X_val, y_val = kwargs['validation_data']
            kwargs['validation_data'] = (X_val, jnp.where(
                jnp.asarray(y_val) == self._target_label, TARGET, OUTLIER))

        super().fit(
            jnp.concatenate([X, X_reference]),
            jnp.concatenate([codes, jnp.full(n_reference, REFERENCE, dtype=jnp.int32)]),
            **kwargs)
        self.threshold_ = jnp.quantile(self.score_samples(X_target), self.nu)
        return self

    def _label_codes(self, X, y):
        """Record the target label and return target / outlier codes."""
        if y is None:
            target = 0 if self.target_label is None else int(self.target_label)
            labels = np.full(X.shape[0], target)
        else:
            labels = np.asarray(y)
            if labels.shape[0] != X.shape[0]:
                raise ValueError("X and y have different numbers of samples")
            if self.target_label is not None:
                target = int(self.target_label)
            else:
                values, counts = np.unique(labels, return_counts=True)
                target = int(values[np.argmax(counts)])
        if not np.any(labels == target):
            raise ValueError(f"No training sample has the target label {target}")
        others = np.unique(labels[labels != target])
        non_target = int(others[0]) if others.size else 1 - target
        self.target_label_ = jnp.asarray(target, dtype=jnp.int32)
        self.non_target_label_ = jnp.asarray(non_target, dtype=jnp.int32)
        return jnp.asarray(np.where(labels == target, TARGET, OUTLIER), dtype=jnp.int32)

    @property
    def _target_label(self):
        return int(self.target_label_)

    @property
    def _non_target_label(self):
        return int(self.non_target_label_)

    def _offset(self, X):
        """:math:`\\log \\nu + \\log q(x)`."""
        return jnp.log(float(self.noise_ratio)) + self._reference_log_density(X)

    def _component_scores(self, params, X):
        """:math:`f_k(x) = b_k - d_k(x) / 2\\sigma^2`, shape (n, p)."""
        return self._logits(params, X) + params['bias'][None, :]

    def _init_state(self, X, y, key):
        labelled = np.asarray(y) != REFERENCE
        X_data, y_data = X[labelled], y[labelled]
        n_outliers = int(np.sum(np.asarray(y_data) == OUTLIER))
        counts = [self.n_prototypes]
        if n_outliers:
            counts.append(min(self.n_outlier_prototypes, n_outliers))

        saved = (self.n_prototypes_per_class, self.prototypes_initializer)
        self.n_prototypes_per_class = counts
        if self.prototypes_initializer is None:
            self.prototypes_initializer = self._default_initializer()
        try:
            state, params, proto_labels = super()._init_state(X_data, y_data, key)
        finally:
            self.n_prototypes_per_class, self.prototypes_initializer = saved

        # Start with the mean logit of the labelled samples at zero.
        scores = self._logits(params, X_data)
        same_class = y_data[:, None] == proto_labels[None, :]
        own = logsumexp(jnp.where(same_class, scores, -jnp.inf), axis=1)
        params['bias'] = jnp.full(
            proto_labels.shape[0], jnp.mean(self._offset(X_data) - own))
        state = state._replace(opt_state=self._optimizer.init(params))
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        scores = self._component_scores(params, X)
        offset = self._offset(X)
        total = jnp.logaddexp(logsumexp(scores, axis=1), offset)

        is_reference = y == REFERENCE
        same_class = y[:, None] == proto_labels[None, :]
        # Reference rows match no component; give them a finite value
        # here and replace it below, so that no gradient is undefined.
        own = logsumexp(
            jnp.where(same_class | is_reference[:, None], scores, -jnp.inf), axis=1)
        own = jnp.where(is_reference, offset, own)

        weights = jnp.where(y == OUTLIER, self._outlier_weight, 1.0)
        return jnp.sum(weights * (total - own)) / jnp.sum(weights)

    def _get_resume_params(self, params):
        params = super()._get_resume_params(params)
        params['bias'] = self.bias_
        return params

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.bias_ = params['bias']

    def _fitted_params(self):
        return {**super()._fitted_params(), 'bias': self.bias_}

    # ------------------------------------------------------------------
    # Scoring and prediction
    # ------------------------------------------------------------------

    def _class_scores(self, X):
        """Log unnormalised densities of the target and the known outliers."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        scores = self._component_scores(self._fitted_params(), X)
        is_target = (self.prototype_labels_ == TARGET)[None, :]
        target = logsumexp(jnp.where(is_target, scores, -jnp.inf), axis=1)
        outlier = logsumexp(jnp.where(is_target, -jnp.inf, scores), axis=1)
        return target, outlier

    def score_samples(self, X):
        """Log density of the target class, up to a constant.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : array of shape (n_samples,)
            :math:`\\log \\sum_{k \\in T} e^{f_k(x)}`; larger is more
            target-like.
        """
        return self._class_scores(X)[0]

    def decision_function(self, X):
        """Target-likeness in (0, 1), with the decision boundary at 0.5.

        The score is :math:`p_T(x) / (p_T(x) + p_O(x) + c)`, where
        :math:`p_O` is the density of the labelled outliers (zero without
        them) and :math:`c` the density level that the fraction ``nu`` of
        the training targets falls below. Without labelled outliers the
        accepted region is a level set of the target density; near
        labelled outliers the score is a likelihood ratio.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : array of shape (n_samples,)
        """
        target, outlier = self._class_scores(X)
        return jax.nn.sigmoid(target - jnp.logaddexp(outlier, self.threshold_))

    def predict(self, X):
        """Predict the target or the non-target label.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : array of shape (n_samples,)
        """
        return jnp.where(
            self.decision_function(X) >= 0.5,
            self._target_label, self._non_target_label,
        ).astype(jnp.int32)

    def predict_proba(self, X):
        """Target-likeness of each sample; same as :meth:`decision_function`."""
        return self.decision_function(X)

    def predict_with_reject(self, X, upper=0.5, lower=None, reject_label=-1):
        """Predict with a reject option for uncertain samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        upper : float
            Scores >= upper are classified as target. Default: 0.5.
        lower : float, optional
            Scores < lower are classified as non-target. Default: same
            as upper (no rejection zone).
        reject_label : int
            Label for rejected samples. Default: -1.

        Returns
        -------
        labels : array of shape (n_samples,)
        """
        if lower is None:
            lower = upper
        scores = self.decision_function(X)
        labels = jnp.full(scores.shape, reject_label, dtype=jnp.int32)
        labels = jnp.where(scores >= upper, self._target_label, labels)
        return jnp.where(scores < lower, self._non_target_label, labels)

    def predict_with_rejection(self, X, confidence=None):
        """Predict, labelling samples as -1 unless one side is confident.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        confidence : float, optional
            Minimum of the score, or of one minus the score, for a
            decision. Defaults to ``rejection_confidence``.
        """
        threshold = confidence if confidence is not None else self.rejection_confidence
        if threshold is None:
            return self.predict(X)
        return self.predict_with_reject(X, upper=threshold, lower=1.0 - threshold)

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp.pop('n_prototypes_per_class', None)
        return hp


class RiemannianReferenceMixin:
    """Reference distributions on the supported manifolds.

    Place it before :class:`OneClassSoftMixin`. Densities are taken with
    respect to the Riemannian volume, up to a constant.

    - SO(n) and the Grassmannian are compact, and the reference is the
      uniform distribution.
    - On SPD(n) and in the Poincare ball the reference is a Gaussian in
      the tangent space at a centre of the target samples, mapped to the
      manifold by the exponential map. Its density is the Gaussian
      density of the tangent coordinates :math:`u` divided by the volume
      distortion of the exponential map,
      :math:`\\prod_{i<j} \\operatorname{sinhc}((\\lambda_i - \\lambda_j) / 2)`
      on SPD(n), with :math:`\\lambda` the eigenvalues of :math:`u` and
      :math:`\\operatorname{sinhc}(t) = \\sinh(t) / t`, and
      :math:`\\operatorname{sinhc}(\\|u\\|)^{d - 1}` in the ball.
    """

    def _points(self, X):
        return X.reshape(X.shape[0], *self.manifold.point_shape)

    def _reference_coordinates(self, X):
        """Tangent coordinates of ``X`` at the centre of the reference."""
        from prosemble.models.riemannian_srng import _tangent_coordinates
        base = self.reference_loc_.reshape(self.manifold.point_shape)
        return jax.vmap(
            lambda x: _tangent_coordinates(self.manifold, base, x))(self._points(X))

    def _fit_reference(self, X_target):
        manifold = self.manifold
        points = self._points(X_target)
        if isinstance(manifold, (SO, Grassmannian)):
            self.reference_loc_ = jnp.zeros(X_target.shape[1])
            self.reference_scale_ = jnp.asarray(1.0)
            return
        if isinstance(manifold, SPD):
            # Log-Euclidean mean.
            values, vectors = jnp.linalg.eigh(points)
            logs = jnp.einsum('nij,nj,nkj->nik', vectors, jnp.log(values), vectors)
            values, vectors = jnp.linalg.eigh(jnp.mean(logs, axis=0))
            base = (vectors * jnp.exp(values)[None, :]) @ vectors.T
            dim = manifold.n * (manifold.n + 1) // 2
        elif isinstance(manifold, HyperbolicPoincare):
            origin = jnp.zeros(manifold.d)
            base = manifold.exp_map(origin, jnp.mean(
                jax.vmap(lambda x: manifold.log_map(origin, x))(points), axis=0))
            dim = manifold.d
        else:
            raise NotImplementedError(
                f"No reference distribution for {type(manifold).__name__}")
        self.reference_loc_ = base.reshape(-1)
        coords = self._reference_coordinates(X_target)
        spread = jnp.sqrt(jnp.mean(jnp.sum(coords.reshape(coords.shape[0], -1) ** 2, axis=1)) / dim)
        self.reference_scale_ = self.noise_scale * jnp.maximum(spread, 1e-3)

    def _sample_reference(self, key, n):
        manifold = self.manifold
        if isinstance(manifold, (SO, Grassmannian)):
            points = jax.vmap(manifold.random_point)(jax.random.split(key, n))
            return points.reshape(n, -1)
        base = self.reference_loc_.reshape(manifold.point_shape)
        noise = jax.random.normal(key, (n, *manifold.point_shape))
        if isinstance(manifold, SPD):
            noise = (noise + jnp.swapaxes(noise, 1, 2)) / 2.0
        coords = self.reference_scale_ * noise
        points = jax.vmap(
            lambda u: manifold.exp_map(base, manifold.frame_to_tangent(base, u)))(coords)
        return points.reshape(n, -1)

    def _reference_log_density(self, X):
        manifold = self.manifold
        if isinstance(manifold, (SO, Grassmannian)):
            return jnp.zeros(X.shape[0])
        coords = self._reference_coordinates(X)
        squared = jnp.sum(coords.reshape(coords.shape[0], -1) ** 2, axis=1)
        gaussian = -0.5 * squared / self.reference_scale_ ** 2
        if isinstance(manifold, SPD):
            values = jnp.linalg.eigvalsh(coords)
            gaps = jnp.abs(values[:, :, None] - values[:, None, :]) / 2.0
            upper = jnp.triu(jnp.ones((manifold.n, manifold.n), dtype=bool), k=1)
            distortion = jnp.sum(jnp.where(upper[None], _log_sinhc(gaps), 0.0), axis=(1, 2))
        else:
            distortion = (manifold.d - 1) * _log_sinhc(jnp.sqrt(squared))
        return gaussian - distortion

    def _default_initializer(self):
        # Prototypes have to be points of the manifold: select samples.
        return None


def _log_sinhc(t):
    """:math:`\\log(\\sinh(t) / t)` for :math:`t \\ge 0`, stable at 0 and for large t."""
    small = t < 1e-3
    safe = jnp.where(small, 1.0, t)
    # log sinh(t) = t + log(1 - exp(-2t)) - log 2
    value = safe + jnp.log1p(-jnp.exp(-2.0 * safe)) - jnp.log(2.0) - jnp.log(safe)
    return jnp.where(small, t ** 2 / 6.0, value)
