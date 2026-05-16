"""
Large dataset training patterns for prosemble.

When datasets are too large to fit in memory, use external data loaders
(tf.data, grain, webdataset) and feed batches into prosemble via
partial_fit(). This example demonstrates three patterns:

  1. partial_fit() with simulated streaming data
  2. batched_iterator() for epoch-based training
  3. padded_batches() for lax.scan-compatible static batching

For datasets that DO fit in memory, prefer fit(batch_size=32) instead —
it handles mini-batch training, shuffling, and early stopping internally.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models import GLVQ
from prosemble.core.data import batched_iterator, padded_batches, shuffle_arrays

# ---------------------------------------------------------------------------
# Generate synthetic data (stand-in for your real dataset)
# ---------------------------------------------------------------------------

np.random.seed(42)
n_samples, n_features, n_classes = 501, 10, 3
X_all = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))
y_all = jnp.array(np.tile(np.arange(n_classes), n_samples // n_classes))

# Split into initial fit set and streaming set
# (y_all is interleaved [0,1,2,0,1,2,...] so every split has all classes)
X_init, y_init = X_all[:60], y_all[:60]
X_stream, y_stream = X_all[60:], y_all[60:]

# ===================================================================
# Pattern 1: partial_fit() with simulated streaming data
# ===================================================================
# Use this when data arrives in batches from an external source
# (database, file system, network stream, tf.data pipeline, etc.)

print("=" * 60)
print("Pattern 1: partial_fit() with streaming batches")
print("=" * 60)

model = GLVQ(
    n_prototypes_per_class=2,
    max_iter=10,
    lr=0.01,
    random_seed=42,
)

# Initial fit to initialize prototypes and optimizer
model.fit(X_init, y_init)
print(f"After initial fit: {model.n_iter_} iterations, loss={model.loss_:.4f}")

# Simulate streaming batches
# In practice, replace this loop with your data loader:
#   for batch in tf_dataset:
#   for batch in grain.DataLoader(...):
#   for batch in webdataset.WebLoader(...):
batch_size = 32
n_stream = len(X_stream)
losses = []

for start in range(0, n_stream, batch_size):
    end = min(start + batch_size, n_stream)
    X_batch = X_stream[start:end]
    y_batch = y_stream[start:end]

    model.partial_fit(X_batch, y_batch)
    losses.append(float(model.loss_))

print(f"After {len(losses)} streaming batches: loss={losses[-1]:.4f}")

preds = model.predict(X_all)
acc = float(jnp.mean(preds == y_all))
print(f"Accuracy: {acc:.2%}")

# ===================================================================
# Pattern 2: batched_iterator() for epoch-based training
# ===================================================================
# Use this when all data fits in memory but you want explicit
# epoch/batch control with partial_fit().

print()
print("=" * 60)
print("Pattern 2: batched_iterator() with epoch loop")
print("=" * 60)

model2 = GLVQ(
    n_prototypes_per_class=2,
    max_iter=5,
    lr=0.01,
    random_seed=42,
)

# Initial fit
model2.fit(X_init, y_init)

key = jax.random.PRNGKey(0)
n_epochs = 5

for epoch in range(n_epochs):
    key, subkey = jax.random.split(key)
    epoch_losses = []

    for X_batch, y_batch in batched_iterator(
        X_all, y_all, batch_size=64, key=subkey
    ):
        model2.partial_fit(X_batch, y_batch)
        epoch_losses.append(float(model2.loss_))

    mean_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Epoch {epoch + 1}/{n_epochs}: mean_loss={mean_loss:.4f}")

preds2 = model2.predict(X_all)
acc2 = float(jnp.mean(preds2 == y_all))
print(f"Accuracy: {acc2:.2%}")

# ===================================================================
# Pattern 3: padded_batches() for lax.scan compatibility
# ===================================================================
# Use this when building custom JIT-compiled training loops
# with jax.lax.scan. The output has static shapes required by XLA.

print()
print("=" * 60)
print("Pattern 3: padded_batches() for static-shaped batching")
print("=" * 60)

key = jax.random.PRNGKey(42)
X_batches, y_batches = padded_batches(X_all, y_all, batch_size=64, key=key)

print(f"Input:  X={X_all.shape}, y={y_all.shape}")
print(f"Output: X_batches={X_batches.shape}, y_batches={y_batches.shape}")
print(f"  {X_batches.shape[0]} batches of {X_batches.shape[1]} samples")
print(f"  (padded from {n_samples} to {X_batches.shape[0] * X_batches.shape[1]})")
print()
print("These static-shaped arrays can be passed directly to jax.lax.scan")
print("for fully JIT-compiled training loops.")

# ===================================================================
# Notes
# ===================================================================

print()
print("=" * 60)
print("When to use each pattern")
print("=" * 60)
print("""
  fit(batch_size=32)    Data fits in memory. Handles shuffling, early
                        stopping, and optimizer state internally.

  partial_fit()         Data too large for memory, or arrives as a
                        stream. Feed batches from any external loader:
                        tf.data, grain, webdataset, torch DataLoader.
                        Requires an initial fit() call to set up
                        prototypes and optimizer.

  batched_iterator()    Data fits in memory but you want explicit
                        epoch/batch control. Yields dynamic-sized
                        batches for Python loops.

  padded_batches()      Static-shaped output for jax.lax.scan.
                        Advanced: use when building custom JIT-compiled
                        training loops.
""")
