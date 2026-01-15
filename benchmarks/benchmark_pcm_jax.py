"""
Performance benchmarks for JAX PCM implementation.

Compares:
1. NumPy PCM vs JAX PCM (CPU)
2. JAX PCM (CPU) vs JAX PCM (GPU) if available
3. Effect of dataset size
4. Effect of number of clusters
5. Effect of dimensionality
6. Effect of k parameter
"""

import time
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import jax
    import jax.numpy as jnp
    from prosemble.models.jax import PCM_JAX
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not available. Install with: pip install jax jaxlib")
    sys.exit(1)

from prosemble.models.pcm import PCM


def benchmark_pcm_model(model, X, n_runs=5, warmup=2):
    """
    Benchmark a PCM model with warmup and multiple runs.

    Args:
        model: PCM model instance
        X: Data to fit
        n_runs: Number of benchmark runs
        warmup: Number of warmup runs

    Returns:
        dict: Statistics (mean, std, min, max time in seconds)
    """
    # Warmup
    for _ in range(warmup):
        if isinstance(model, PCM_JAX):
            model_copy = PCM_JAX(
                n_clusters=model.n_clusters,
                fuzzifier=model.fuzzifier,
                k=model.k,
                max_iter=model.max_iter,
                epsilon=model.epsilon,
                init_method=model.init_method,
                random_seed=42
            )
            model_copy.fit(X)
            if isinstance(X, jnp.ndarray):
                jax.block_until_ready(model_copy.centroids_)
        else:
            model_copy = PCM(
                data=X,
                c=model.num_clusters,
                m=model.fuzzifier,
                k=model.k,
                num_iter=model.num_iter,
                epsilon=model.epsilon,
                ord=model.ord,
                set_U_matrix='fcm'
            )
            model_copy.fit()

    # Benchmark
    times = []
    for _ in range(n_runs):
        if isinstance(model, PCM_JAX):
            model_copy = PCM_JAX(
                n_clusters=model.n_clusters,
                fuzzifier=model.fuzzifier,
                k=model.k,
                max_iter=model.max_iter,
                epsilon=model.epsilon,
                init_method=model.init_method,
                random_seed=42
            )
            start = time.perf_counter()
            model_copy.fit(X)
            if isinstance(X, jnp.ndarray):
                jax.block_until_ready(model_copy.centroids_)
            end = time.perf_counter()
        else:
            model_copy = PCM(
                data=X,
                c=model.num_clusters,
                m=model.fuzzifier,
                k=model.k,
                num_iter=model.num_iter,
                epsilon=model.epsilon,
                ord=model.ord,
                set_U_matrix='fcm'
            )
            start = time.perf_counter()
            model_copy.fit()
            end = time.perf_counter()

        times.append(end - start)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
    }


def benchmark_dataset_sizes(n_runs=5):
    """
    Benchmark PCM across different dataset sizes.

    Returns:
        DataFrame with benchmark results
    """
    print("\n" + "=" * 80)
    print("Benchmarking PCM: Dataset Size Scaling")
    print("=" * 80)

    sizes = [
        (100, 10, 3),      # Small
        (1000, 20, 5),     # Medium
        (5000, 50, 5),     # Large  (reduced from 10000 for NumPy)
    ]

    results = []

    for n_samples, n_features, n_clusters in sizes:
        print(f"\nDataset: n={n_samples}, d={n_features}, c={n_clusters}")

        # Generate data
        np.random.seed(42)
        X_np = np.random.randn(n_samples, n_features).astype(np.float32)
        X_jax = jnp.array(X_np)

        # NumPy version (skip for very large datasets as PCM is slow)
        if n_samples <= 1000:
            print("  NumPy PCM...", end=" ")
            try:
                pcm_np = PCM(
                    data=X_np,
                    c=n_clusters,
                    m=2.0,
                    k=1.0,
                    num_iter=30,  # Reduced iterations for NumPy
                    epsilon=1e-4,
                    ord='fro',
                    set_U_matrix='fcm'
                )
                stats_np = benchmark_pcm_model(pcm_np, X_np, n_runs=n_runs)
                print(f"{stats_np['mean']:.4f}s")
            except Exception as e:
                print(f"Failed: {e}")
                stats_np = {'mean': np.nan, 'std': np.nan}
        else:
            print("  NumPy PCM... skipped (too slow)")
            stats_np = {'mean': np.nan, 'std': np.nan}

        # JAX CPU version
        print("  JAX PCM (CPU)...", end=" ")
        with jax.default_device(jax.devices('cpu')[0]):
            pcm_jax_cpu = PCM_JAX(
                n_clusters=n_clusters,
                fuzzifier=2.0,
                k=1.0,
                max_iter=50,
                epsilon=1e-5,
                init_method='fcm',
                random_seed=42
            )
            stats_jax_cpu = benchmark_pcm_model(pcm_jax_cpu, X_jax, n_runs=n_runs)
        print(f"{stats_jax_cpu['mean']:.4f}s")

        # JAX GPU version (if available)
        if len(jax.devices('gpu')) > 0:
            print("  JAX PCM (GPU)...", end=" ")
            with jax.default_device(jax.devices('gpu')[0]):
                X_gpu = jax.device_put(X_jax, jax.devices('gpu')[0])
                pcm_jax_gpu = PCM_JAX(
                    n_clusters=n_clusters,
                    fuzzifier=2.0,
                    k=1.0,
                    max_iter=50,
                    epsilon=1e-5,
                    init_method='fcm',
                    random_seed=42
                )
                stats_jax_gpu = benchmark_pcm_model(pcm_jax_gpu, X_gpu, n_runs=n_runs)
            print(f"{stats_jax_gpu['mean']:.4f}s")
        else:
            stats_jax_gpu = {'mean': np.nan, 'std': np.nan}

        # Compute speedups
        speedup_cpu = stats_np['mean'] / stats_jax_cpu['mean'] if not np.isnan(stats_np['mean']) else np.nan
        speedup_gpu = stats_np['mean'] / stats_jax_gpu['mean'] if not np.isnan(stats_jax_gpu['mean']) else np.nan

        results.append({
            'n_samples': n_samples,
            'n_features': n_features,
            'n_clusters': n_clusters,
            'numpy_time': stats_np['mean'],
            'jax_cpu_time': stats_jax_cpu['mean'],
            'jax_gpu_time': stats_jax_gpu['mean'],
            'speedup_cpu': speedup_cpu,
            'speedup_gpu': speedup_gpu,
        })

    return pd.DataFrame(results)


def benchmark_num_clusters(n_runs=5):
    """
    Benchmark PCM with varying number of clusters.

    Returns:
        DataFrame with results
    """
    print("\n" + "=" * 80)
    print("Benchmarking PCM: Number of Clusters")
    print("=" * 80)

    n_samples = 5000
    n_features = 50
    cluster_counts = [3, 5, 10, 20]

    results = []

    np.random.seed(42)
    X_np = np.random.randn(n_samples, n_features).astype(np.float32)
    X_jax = jnp.array(X_np)

    for n_clusters in cluster_counts:
        print(f"\nClusters: {n_clusters}")

        # JAX CPU
        print("  JAX PCM (CPU)...", end=" ")
        with jax.default_device(jax.devices('cpu')[0]):
            pcm_jax = PCM_JAX(
                n_clusters=n_clusters,
                fuzzifier=2.0,
                k=1.0,
                max_iter=50,
                epsilon=1e-5,
                init_method='fcm',
                random_seed=42
            )
            stats = benchmark_pcm_model(pcm_jax, X_jax, n_runs=n_runs)
        print(f"{stats['mean']:.4f}s")

        results.append({
            'n_clusters': n_clusters,
            'time': stats['mean'],
            'std': stats['std'],
        })

    return pd.DataFrame(results)


def benchmark_dimensionality(n_runs=5):
    """
    Benchmark PCM with varying dimensionality.

    Returns:
        DataFrame with results
    """
    print("\n" + "=" * 80)
    print("Benchmarking PCM: Dimensionality")
    print("=" * 80)

    n_samples = 3000
    n_clusters = 5
    dimensions = [10, 50, 100, 200]

    results = []

    for n_features in dimensions:
        print(f"\nDimensions: {n_features}")

        np.random.seed(42)
        X_np = np.random.randn(n_samples, n_features).astype(np.float32)
        X_jax = jnp.array(X_np)

        # JAX CPU
        print("  JAX PCM (CPU)...", end=" ")
        with jax.default_device(jax.devices('cpu')[0]):
            pcm_jax = PCM_JAX(
                n_clusters=n_clusters,
                fuzzifier=2.0,
                k=1.0,
                max_iter=50,
                epsilon=1e-5,
                init_method='fcm',
                random_seed=42
            )
            stats = benchmark_pcm_model(pcm_jax, X_jax, n_runs=n_runs)
        print(f"{stats['mean']:.4f}s")

        results.append({
            'n_features': n_features,
            'time': stats['mean'],
            'std': stats['std'],
        })

    return pd.DataFrame(results)


def benchmark_k_parameter(n_runs=5):
    """
    Benchmark PCM with varying k parameter.

    Returns:
        DataFrame with results
    """
    print("\n" + "=" * 80)
    print("Benchmarking PCM: k Parameter")
    print("=" * 80)

    n_samples = 3000
    n_features = 50
    n_clusters = 5
    k_values = [0.01, 0.1, 1.0, 10.0]

    results = []

    np.random.seed(42)
    X_np = np.random.randn(n_samples, n_features).astype(np.float32)
    X_jax = jnp.array(X_np)

    for k in k_values:
        print(f"\nk parameter: {k}")

        # JAX CPU
        print("  JAX PCM (CPU)...", end=" ")
        with jax.default_device(jax.devices('cpu')[0]):
            pcm_jax = PCM_JAX(
                n_clusters=n_clusters,
                fuzzifier=2.0,
                k=k,
                max_iter=50,
                epsilon=1e-5,
                init_method='fcm',
                random_seed=42
            )
            stats = benchmark_pcm_model(pcm_jax, X_jax, n_runs=n_runs)
        print(f"{stats['mean']:.4f}s")

        results.append({
            'k': k,
            'time': stats['mean'],
            'std': stats['std'],
        })

    return pd.DataFrame(results)


def benchmark_convergence_analysis():
    """
    Analyze convergence behavior and iteration timing.

    Returns:
        DataFrame with convergence analysis
    """
    print("\n" + "=" * 80)
    print("PCM Convergence Analysis")
    print("=" * 80)

    n_samples = 3000
    n_features = 50
    n_clusters = 5

    np.random.seed(42)
    X = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))

    # Fit model and track objective
    model = PCM_JAX(
        n_clusters=n_clusters,
        fuzzifier=2.0,
        k=1.0,
        max_iter=100,
        epsilon=1e-5,
        init_method='fcm',
        random_seed=42
    )

    start = time.perf_counter()
    model.fit(X)
    jax.block_until_ready(model.centroids_)
    total_time = time.perf_counter() - start

    objectives = model.get_objective_history()

    # Find iteration where converged (objective change < threshold)
    diffs = np.abs(np.diff(objectives))
    converged_at = np.argmax(diffs < 1e-3) if np.any(diffs < 1e-3) else len(objectives)

    print(f"\nTotal iterations: {model.n_iter_}")
    print(f"Converged at iteration: {converged_at}")
    print(f"Total time: {total_time:.4f}s")
    print(f"Time per iteration: {total_time / model.n_iter_:.4f}s")
    print(f"Final objective: {objectives[-1]:.4f}")

    return pd.DataFrame({
        'iteration': np.arange(len(objectives)),
        'objective': objectives,
    })


def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("JAX PCM Benchmark Suite")
    print("=" * 80)

    # Check GPU availability
    if len(jax.devices('gpu')) > 0:
        print(f"GPU detected: {jax.devices('gpu')[0]}")
    else:
        print("No GPU detected. Running CPU benchmarks only.")

    # Benchmark 1: Dataset size scaling
    df_sizes = benchmark_dataset_sizes(n_runs=3)

    print("\n" + "=" * 80)
    print("Dataset Size Scaling Results")
    print("=" * 80)
    print(df_sizes.to_string(index=False))

    # Benchmark 2: Number of clusters
    df_clusters = benchmark_num_clusters(n_runs=3)

    print("\n" + "=" * 80)
    print("Number of Clusters Results")
    print("=" * 80)
    print(df_clusters.to_string(index=False))

    # Benchmark 3: Dimensionality
    df_dims = benchmark_dimensionality(n_runs=3)

    print("\n" + "=" * 80)
    print("Dimensionality Results")
    print("=" * 80)
    print(df_dims.to_string(index=False))

    # Benchmark 4: k parameter
    df_k = benchmark_k_parameter(n_runs=3)

    print("\n" + "=" * 80)
    print("k Parameter Results")
    print("=" * 80)
    print(df_k.to_string(index=False))

    # Benchmark 5: Convergence analysis
    df_convergence = benchmark_convergence_analysis()

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    df_sizes.to_csv(os.path.join(output_dir, 'benchmark_pcm_sizes.csv'), index=False)
    df_clusters.to_csv(os.path.join(output_dir, 'benchmark_pcm_clusters.csv'), index=False)
    df_dims.to_csv(os.path.join(output_dir, 'benchmark_pcm_dimensions.csv'), index=False)
    df_k.to_csv(os.path.join(output_dir, 'benchmark_pcm_k.csv'), index=False)
    df_convergence.to_csv(os.path.join(output_dir, 'benchmark_pcm_convergence.csv'), index=False)

    print(f"\nResults saved to {output_dir}/")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    if not df_sizes['speedup_cpu'].isna().all():
        avg_speedup_cpu = df_sizes['speedup_cpu'].mean()
        print(f"Average CPU speedup over NumPy: {avg_speedup_cpu:.2f}x")

    if not df_sizes['speedup_gpu'].isna().all():
        avg_speedup_gpu = df_sizes['speedup_gpu'].mean()
        print(f"Average GPU speedup over NumPy: {avg_speedup_gpu:.2f}x")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
