"""
Performance benchmarks for JAX distance functions.

Compares:
1. NumPy (original) vs JAX (CPU)
2. JAX (CPU) vs JAX (GPU) if available
3. Effect of dataset size
4. JIT compilation overhead
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Callable
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import jax
    import jax.numpy as jnp
    from prosemble.core.distance_jax import (
        euclidean_distance_matrix,
        squared_euclidean_distance_matrix,
        manhattan_distance_matrix,
        gaussian_kernel_matrix,
    )
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not available. Install with: pip install jax jaxlib")
    sys.exit(1)

from prosemble.core.distance import (
    euclidean_distance as np_euclidean,
    squared_euclidean_distance as np_squared_euclidean,
    manhattan_distance as np_manhattan,
)


def benchmark_function(func: Callable, *args, n_runs: int = 10, warmup: int = 2):
    """
    Benchmark a function with warmup and multiple runs.

    Args:
        func: Function to benchmark
        *args: Arguments to pass to function
        n_runs: Number of benchmark runs
        warmup: Number of warmup runs (for JIT compilation)

    Returns:
        dict: Statistics (mean, std, min, max time in seconds)
    """
    # Warmup (important for JIT compilation)
    for _ in range(warmup):
        _ = func(*args)

    # If JAX, block until computation completes
    if JAX_AVAILABLE and isinstance(args[0], jnp.ndarray):
        jax.block_until_ready(func(*args))

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)

        # Block until ready (for async GPU operations)
        if JAX_AVAILABLE and isinstance(result, jnp.ndarray):
            jax.block_until_ready(result)

        end = time.perf_counter()
        times.append(end - start)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
    }


def numpy_euclidean_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """NumPy implementation using loops."""
    return np.array([
        [np_euclidean(X[i], Y[j]) for j in range(len(Y))]
        for i in range(len(X))
    ])


def numpy_squared_euclidean_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """NumPy implementation using loops."""
    return np.array([
        [np_squared_euclidean(X[i], Y[j]) for j in range(len(Y))]
        for i in range(len(X))
    ])


def numpy_manhattan_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """NumPy implementation using loops."""
    return np.array([
        [np_manhattan(X[i], Y[j]) for j in range(len(Y))]
        for i in range(len(X))
    ])


def numpy_euclidean_vectorized(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Vectorized NumPy implementation for comparison."""
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    Y_sq = np.sum(Y**2, axis=1, keepdims=True).T
    XY = X @ Y.T
    D_sq = X_sq + Y_sq - 2 * XY
    return np.sqrt(np.maximum(D_sq, 0))


def benchmark_euclidean_distance(sizes: List[tuple], n_runs: int = 10) -> pd.DataFrame:
    """
    Benchmark Euclidean distance across different implementations and sizes.

    Args:
        sizes: List of (n_samples_X, n_samples_Y, n_features) tuples
        n_runs: Number of benchmark runs per configuration

    Returns:
        DataFrame with benchmark results
    """
    results = []

    for n_x, n_y, d in sizes:
        print(f"\nBenchmarking Euclidean distance: X={n_x}x{d}, Y={n_y}x{d}")

        # Generate data
        np.random.seed(42)
        X_np = np.random.randn(n_x, d).astype(np.float32)
        Y_np = np.random.randn(n_y, d).astype(np.float32)

        X_jax = jnp.array(X_np)
        Y_jax = jnp.array(Y_np)

        # Benchmark NumPy (original with loops) - skip for large sizes
        if n_x * n_y < 10000:
            print("  NumPy (loops)...", end=" ")
            stats_np_loops = benchmark_function(numpy_euclidean_matrix, X_np, Y_np, n_runs=n_runs)
            print(f"{stats_np_loops['mean']:.4f}s")
        else:
            print("  NumPy (loops)... skipped (too slow)")
            stats_np_loops = {'mean': np.nan, 'std': np.nan}

        # Benchmark NumPy (vectorized)
        print("  NumPy (vectorized)...", end=" ")
        stats_np_vec = benchmark_function(numpy_euclidean_vectorized, X_np, Y_np, n_runs=n_runs)
        print(f"{stats_np_vec['mean']:.4f}s")

        # Benchmark JAX (CPU)
        print("  JAX (CPU)...", end=" ")
        with jax.default_device(jax.devices('cpu')[0]):
            stats_jax_cpu = benchmark_function(euclidean_distance_matrix, X_jax, Y_jax, n_runs=n_runs)
        print(f"{stats_jax_cpu['mean']:.4f}s")

        # Benchmark JAX (GPU) if available
        if len(jax.devices('gpu')) > 0:
            print("  JAX (GPU)...", end=" ")
            with jax.default_device(jax.devices('gpu')[0]):
                X_gpu = jax.device_put(X_jax, jax.devices('gpu')[0])
                Y_gpu = jax.device_put(Y_jax, jax.devices('gpu')[0])
                stats_jax_gpu = benchmark_function(euclidean_distance_matrix, X_gpu, Y_gpu, n_runs=n_runs)
            print(f"{stats_jax_gpu['mean']:.4f}s")
        else:
            stats_jax_gpu = {'mean': np.nan, 'std': np.nan}
            print("  JAX (GPU)... not available")

        # Compute speedups
        speedup_cpu = stats_np_vec['mean'] / stats_jax_cpu['mean']
        speedup_gpu = stats_np_vec['mean'] / stats_jax_gpu['mean'] if not np.isnan(stats_jax_gpu['mean']) else np.nan

        results.append({
            'n_samples_X': n_x,
            'n_samples_Y': n_y,
            'n_features': d,
            'numpy_loops_time': stats_np_loops['mean'],
            'numpy_vec_time': stats_np_vec['mean'],
            'jax_cpu_time': stats_jax_cpu['mean'],
            'jax_gpu_time': stats_jax_gpu['mean'],
            'speedup_cpu_vs_numpy': speedup_cpu,
            'speedup_gpu_vs_numpy': speedup_gpu,
        })

    return pd.DataFrame(results)


def benchmark_all_distances(n_x: int = 1000, n_y: int = 500, d: int = 50, n_runs: int = 10) -> pd.DataFrame:
    """
    Benchmark all distance functions on same data size.

    Args:
        n_x: Number of samples in X
        n_y: Number of samples in Y
        d: Number of features
        n_runs: Number of runs

    Returns:
        DataFrame with results for all distance functions
    """
    print(f"\nBenchmarking all distance functions: X={n_x}x{d}, Y={n_y}x{d}")

    np.random.seed(42)
    X_np = np.random.randn(n_x, d).astype(np.float32)
    Y_np = np.random.randn(n_y, d).astype(np.float32)

    X_jax = jnp.array(X_np)
    Y_jax = jnp.array(Y_np)

    results = []

    # Test different distance functions
    tests = [
        ('Euclidean', numpy_euclidean_vectorized, euclidean_distance_matrix),
        ('Squared Euclidean', lambda X, Y: numpy_euclidean_vectorized(X, Y)**2, squared_euclidean_distance_matrix),
        ('Manhattan', None, manhattan_distance_matrix),
    ]

    for name, np_func, jax_func in tests:
        print(f"  {name}...")

        # NumPy
        if np_func is not None:
            stats_np = benchmark_function(np_func, X_np, Y_np, n_runs=n_runs)
            np_time = stats_np['mean']
        else:
            np_time = np.nan

        # JAX CPU
        with jax.default_device(jax.devices('cpu')[0]):
            stats_jax_cpu = benchmark_function(jax_func, X_jax, Y_jax, n_runs=n_runs)
        jax_cpu_time = stats_jax_cpu['mean']

        # JAX GPU
        if len(jax.devices('gpu')) > 0:
            with jax.default_device(jax.devices('gpu')[0]):
                X_gpu = jax.device_put(X_jax, jax.devices('gpu')[0])
                Y_gpu = jax.device_put(Y_jax, jax.devices('gpu')[0])
                stats_jax_gpu = benchmark_function(jax_func, X_gpu, Y_gpu, n_runs=n_runs)
            jax_gpu_time = stats_jax_gpu['mean']
        else:
            jax_gpu_time = np.nan

        speedup_cpu = np_time / jax_cpu_time if not np.isnan(np_time) else np.nan
        speedup_gpu = np_time / jax_gpu_time if not np.isnan(np_time) and not np.isnan(jax_gpu_time) else np.nan

        results.append({
            'distance_function': name,
            'numpy_time': np_time,
            'jax_cpu_time': jax_cpu_time,
            'jax_gpu_time': jax_gpu_time,
            'speedup_cpu': speedup_cpu,
            'speedup_gpu': speedup_gpu,
        })

    return pd.DataFrame(results)


def benchmark_jit_overhead() -> pd.DataFrame:
    """
    Benchmark JIT compilation overhead.

    Returns:
        DataFrame showing first-call vs subsequent-call times
    """
    print("\nBenchmarking JIT compilation overhead...")

    sizes = [
        (100, 50, 20),
        (1000, 500, 50),
        (10000, 1000, 100),
    ]

    results = []

    for n_x, n_y, d in sizes:
        print(f"  Size: X={n_x}x{d}, Y={n_y}x{d}")

        np.random.seed(42)
        X = jnp.array(np.random.randn(n_x, d).astype(np.float32))
        Y = jnp.array(np.random.randn(n_y, d).astype(np.float32))

        # First call (includes compilation)
        start = time.perf_counter()
        result = euclidean_distance_matrix(X, Y)
        jax.block_until_ready(result)
        first_call_time = time.perf_counter() - start

        # Second call (cached)
        start = time.perf_counter()
        result = euclidean_distance_matrix(X, Y)
        jax.block_until_ready(result)
        second_call_time = time.perf_counter() - start

        # Average of next 10 calls
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = euclidean_distance_matrix(X, Y)
            jax.block_until_ready(result)
            times.append(time.perf_counter() - start)
        avg_time = np.mean(times)

        results.append({
            'n_samples_X': n_x,
            'n_samples_Y': n_y,
            'n_features': d,
            'first_call_time': first_call_time,
            'second_call_time': second_call_time,
            'avg_subsequent_time': avg_time,
            'compilation_overhead': first_call_time - second_call_time,
        })

    return pd.DataFrame(results)


def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("JAX Distance Functions Benchmark Suite")
    print("=" * 80)

    # Check GPU availability
    if len(jax.devices('gpu')) > 0:
        print(f"GPU detected: {jax.devices('gpu')[0]}")
    else:
        print("No GPU detected. Running CPU benchmarks only.")

    # Benchmark 1: Euclidean distance across sizes
    sizes_euclidean = [
        (100, 50, 20),      # Small
        (1000, 500, 50),    # Medium
        (10000, 1000, 100), # Large
        (100000, 500, 50),  # Very large (X only)
    ]

    df_euclidean = benchmark_euclidean_distance(sizes_euclidean, n_runs=10)

    print("\n" + "=" * 80)
    print("Euclidean Distance Benchmark Results")
    print("=" * 80)
    print(df_euclidean.to_string(index=False))

    # Benchmark 2: All distance functions
    df_all = benchmark_all_distances(n_x=1000, n_y=500, d=50, n_runs=10)

    print("\n" + "=" * 80)
    print("All Distance Functions Comparison")
    print("=" * 80)
    print(df_all.to_string(index=False))

    # Benchmark 3: JIT overhead
    df_jit = benchmark_jit_overhead()

    print("\n" + "=" * 80)
    print("JIT Compilation Overhead")
    print("=" * 80)
    print(df_jit.to_string(index=False))

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    df_euclidean.to_csv(os.path.join(output_dir, 'benchmark_euclidean.csv'), index=False)
    df_all.to_csv(os.path.join(output_dir, 'benchmark_all_distances.csv'), index=False)
    df_jit.to_csv(os.path.join(output_dir, 'benchmark_jit_overhead.csv'), index=False)

    print(f"\nResults saved to {output_dir}/")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    avg_speedup_cpu = df_euclidean['speedup_cpu_vs_numpy'].mean()
    print(f"Average CPU speedup over NumPy (vectorized): {avg_speedup_cpu:.2f}x")

    if not df_euclidean['speedup_gpu_vs_numpy'].isna().all():
        avg_speedup_gpu = df_euclidean['speedup_gpu_vs_numpy'].mean()
        print(f"Average GPU speedup over NumPy (vectorized): {avg_speedup_gpu:.2f}x")

    print(f"\nAverage JIT compilation overhead: {df_jit['compilation_overhead'].mean():.4f}s")
    print(f"Average subsequent call time (after JIT): {df_jit['avg_subsequent_time'].mean():.4f}s")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
