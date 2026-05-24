"""Curriculum learning (self-paced learning) for prototype-based models.

Implements the self-paced learning framework (Kumar et al. 2010) adapted
for LVQ models. Samples are presented in order of difficulty — easy
samples first, hard samples later.

For LVQ, sample difficulty is measured by the absolute value of the
mu-ratio: |mu(x)| close to 0 means the sample is on the decision
boundary (hard), while |mu(x)| >> 0 means it's far from the boundary
(easy).

Usage
-----
Curriculum learning is applied via sample masking in the loss function.
At each training step, a difficulty threshold lambda_t determines which
samples participate. The threshold increases over training to gradually
include harder samples.

References
----------
.. [1] Kumar, M. P., Packer, B., & Koller, D. (2010). Self-paced
       learning for latent variable models. NeurIPS.
.. [2] Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009).
       Curriculum learning. ICML.
"""

import jax.numpy as jnp


def curriculum_weights(per_sample_loss, threshold, mode='hard'):
    """Compute per-sample curriculum weights based on loss magnitude.

    Parameters
    ----------
    per_sample_loss : array of shape (n_samples,)
        Per-sample loss values (e.g., from GLVQ mu-ratio).
    threshold : float
        Difficulty threshold lambda_t. Samples with loss below this
        threshold are included. Increases over training.
    mode : str, {'hard', 'soft', 'linear'}
        Weight assignment mode:
        - 'hard': binary weights (0 or 1). Sample is either in or out.
        - 'soft': smooth exponential weighting.
          w_i = exp(-loss_i / threshold) if loss_i > threshold, else 1.
        - 'linear': linear ramp-down.
          w_i = max(0, 1 - (loss_i - threshold) / threshold).
        Default: 'hard'.

    Returns
    -------
    weights : array of shape (n_samples,)
        Per-sample weights in [0, 1]. Weights sum is used to normalize
        the loss.
    """
    if mode == 'hard':
        return jnp.where(per_sample_loss <= threshold, 1.0, 0.0)
    elif mode == 'soft':
        # Smooth transition: full weight below threshold, exponential decay above
        excess = jnp.maximum(per_sample_loss - threshold, 0.0)
        return jnp.exp(-excess / (threshold + 1e-10))
    elif mode == 'linear':
        # Linear ramp-down above threshold
        return jnp.clip(1.0 - (per_sample_loss - threshold) / (threshold + 1e-10), 0.0, 1.0)
    else:
        raise ValueError(f"Unknown curriculum mode '{mode}'. Use 'hard', 'soft', or 'linear'.")


def curriculum_threshold(iteration, max_iter, init_threshold=0.3,
                         final_threshold=None, schedule='linear'):
    """Compute the difficulty threshold at a given training iteration.

    The threshold increases over training: initially only easy samples
    are included, and harder samples are gradually added.

    Parameters
    ----------
    iteration : int or array
        Current training iteration.
    max_iter : int
        Maximum number of iterations.
    init_threshold : float
        Initial threshold (fraction of max loss to include). At the
        start, only samples with loss < init_threshold * max_loss are
        included. Default: 0.3.
    final_threshold : float, optional
        Final threshold. Default: None (uses a large value to include
        all samples by end of training).
    schedule : str, {'linear', 'exponential', 'cosine'}
        How the threshold grows over training:
        - 'linear': lambda_t = init + (final - init) * t / T
        - 'exponential': lambda_t = init * (final / init) ^ (t / T)
        - 'cosine': cosine annealing from init to final.
        Default: 'linear'.

    Returns
    -------
    threshold : float
        Difficulty threshold at the current iteration.
    """
    if final_threshold is None:
        final_threshold = 100.0  # Effectively include all samples

    progress = jnp.clip(iteration / jnp.maximum(max_iter - 1, 1), 0.0, 1.0)

    if schedule == 'linear':
        return init_threshold + (final_threshold - init_threshold) * progress
    elif schedule == 'exponential':
        # Avoid log(0): ensure init > 0
        log_ratio = jnp.log(final_threshold / (init_threshold + 1e-10))
        return init_threshold * jnp.exp(log_ratio * progress)
    elif schedule == 'cosine':
        # Cosine annealing: starts slow, accelerates, then slows
        cosine_factor = 0.5 * (1.0 - jnp.cos(jnp.pi * progress))
        return init_threshold + (final_threshold - init_threshold) * cosine_factor
    else:
        raise ValueError(
            f"Unknown schedule '{schedule}'. Use 'linear', 'exponential', or 'cosine'."
        )


def apply_curriculum_to_loss(per_sample_losses, iteration, max_iter,
                             init_percentile=0.3, schedule='linear',
                             mode='soft'):
    """Full curriculum pipeline: compute threshold and weighted loss.

    Convenience function that combines threshold scheduling and weight
    computation. Use inside ``_compute_loss`` to add curriculum learning
    to any model.

    Parameters
    ----------
    per_sample_losses : array of shape (n_samples,)
        Per-sample loss values.
    iteration : int or array
        Current training step.
    max_iter : int
        Total training steps.
    init_percentile : float
        Initial fraction of samples to include (by loss quantile).
        Default: 0.3 (include the easiest 30% of samples initially).
    schedule : str
        Threshold growth schedule. Default: 'linear'.
    mode : str
        Weighting mode. Default: 'soft'.

    Returns
    -------
    weighted_loss : scalar
        Curriculum-weighted mean loss.
    """
    # Estimate threshold from loss statistics
    # Use quantile of current losses as adaptive threshold
    sorted_losses = jnp.sort(per_sample_losses)
    n = per_sample_losses.shape[0]

    # Progress determines what fraction of samples to include
    progress = jnp.clip(iteration / jnp.maximum(max_iter - 1, 1), 0.0, 1.0)
    # Start with init_percentile fraction, end with all samples
    include_frac = init_percentile + (1.0 - init_percentile) * progress

    # Convert fraction to threshold via quantile
    idx = jnp.clip((include_frac * n).astype(jnp.int32), 0, n - 1)
    threshold = sorted_losses[idx]

    weights = curriculum_weights(per_sample_losses, threshold, mode=mode)

    # Weighted mean loss (avoid division by zero)
    total_weight = jnp.sum(weights)
    weighted_loss = jnp.sum(weights * per_sample_losses) / (total_weight + 1e-10)

    return weighted_loss
