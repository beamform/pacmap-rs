# pacmap

[![Crates.io](https://img.shields.io/crates/v/pacmap)](https://crates.io/crates/pacmap)
[![Documentation](https://docs.rs/pacmap/badge.svg)](https://docs.rs/pacmap)
[![CI](https://github.com/beamform/pacmap-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/beamform/pacmap-rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A Rust implementation of PaCMAP (Pairwise Controlled Manifold Approximation) for dimensionality reduction based on the
original [Python implementation](https://github.com/YingfanWang/PaCMAP).

## Overview

Dimensionality reduction transforms high-dimensional data into a lower-dimensional representation while preserving
important relationships between points. This is useful for visualization, analysis, and as preprocessing for other
algorithms.

PaCMAP is a relatively recent dimensionality reduction technique that preserves both local and global structure through
three types of point relationships:

- Nearest neighbor pairs preserve local structure
- Mid-near pairs preserve intermediate structure
- Far pairs prevent collapse and maintain separation

For details on the algorithm, see the [original paper](https://jmlr.org/papers/v22/20-1061.html).

## Features

- Fast approximate nearest neighbors for large datasets using USearch
- SIMD-optimized distance calculations
- Parallel processing with Rayon
- Optional PCA initialization using various BLAS backends
- Reproducible results with optional seeding

## Usage

Basic usage with default parameters:

```rust
use anyhow::Result;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use pacmap::{Configuration, fit_transform};

fn main() -> Result<()> {
    // Your high-dimensional data as an n × d array
    let n_samples = 1000;
    let n_features = 1000;
    let mut data = Array2::random((n_samples, n_features), Uniform::new(-1.0, 1.0));

    let config = Configuration::default();
    let (embedding, _) = fit_transform(data.view(), config)?;

    // embedding is now an n × 2 array
    Ok(())
}
```

Customized embedding:

```rust
use anyhow::Result;
use pacmap::{Configuration, Initialization};

fn main() -> Result<()> {
    let config = Configuration::builder()
        .embedding_dimensions(3)
        .initialization(Initialization::Random(Some(42)))
        .learning_rate(0.8)
        .num_iters((50, 50, 100))
        .mid_near_ratio(0.3)
        .far_pair_ratio(2.0)
        .approx_threshold(8_000)  // Use approximate neighbors above this size
        .build();

    let (embedding, _) = fit_transform(data.view(), config)?;
    Ok(())
}
```

Capturing intermediate states:

```rust
use anyhow::Result;
use pacmap::Configuration;

fn main() -> Result<()> {
    let config = Configuration::builder()
        .snapshots(vec![100, 200, 300])
        .build();

    let (embedding, Some(states)) = fit_transform(data.view(), config)?;

    // states is now an s × n × d array where s is the number of snapshots
    Ok(())
}
```

For a standalone example, see the [pacmap-rs-example repository](https://github.com/beamform/pacmap-rs-example).

## Configuration

### Core Parameters

- `embedding_dimensions`: Output dimensionality (default: 2)
- `initialization`: How to initialize coordinates:
    - `Pca` - Project data using PCA (default)
    - `Value(array)` - Use provided coordinates
    - `Random(seed)` - Random initialization with optional seed
- `learning_rate`: Learning rate for Adam optimizer (default: 1.0)
- `num_iters`: Iteration counts for three optimization phases (default: (100, 100, 250)):
    1. Mid-near weight reduction phase
    2. Balanced weight phase
    3. Local structure focus phase
- `snapshots`: Optional vector of iterations at which to save embedding states
- `approx_threshold`: Number of samples above which to use approximate nearest neighbors (default: 8,000)

### Pair Sampling Parameters

- `mid_near_ratio`: Ratio of mid-near to nearest neighbor pairs (default: 0.5)
- `far_pair_ratio`: Ratio of far to nearest neighbor pairs (default: 2.0)
- `override_neighbors`: Optional fixed neighbor count override (default: None, auto-scaled with dataset size)
- `seed`: Optional random seed for reproducible sampling and initialization

### Pair Configuration

- `PairConfiguration::Generate` - Generate all pairs from scratch (default)
- `PairConfiguration::NeighborsProvided { pair_neighbors }` - Use provided nearest neighbors, generate remaining pairs
- `PairConfiguration::AllProvided { pair_neighbors, pair_mn, pair_fp }` - Use all provided pairs

## Feature Flags

### BLAS/LAPACK Backends

Only one BLAS/LAPACK backend feature should be enabled at a time.
These are required for PCA operations except on macOS which uses Accelerate by default.

- `intel-mkl-static` - Static linking with Intel MKL
- `intel-mkl-system` - Dynamic linking with system Intel MKL
- `openblas-static` - Static linking with OpenBLAS
- `openblas-system` - Dynamic linking with system OpenBLAS
- `netlib-static` - Static linking with Netlib
- `netlib-system` - Dynamic linking with system Netlib

For example:

```toml
[dependencies]
pacmap = { version = "0.2", features = ["openblas-static"] }
```

See [ndarray-linalg's documentation](https://github.com/rust-ndarray/ndarray-linalg#backend-features) for detailed
information about BLAS/LAPACK configuration and performance considerations.

### Performance Features

- `simsimd` - Enable SIMD optimizations in USearch for faster approximate nearest neighbor search.
  Requires GCC 13+ for compilation and a recent glibc at runtime.

## Limitations

This implementation currently:

- Only supports Euclidean distances
- Does not support incremental transform

## References

[Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization](https://jmlr.org/papers/v22/20-1061.html).
Wang, Y., Huang, H., Rudin, C., & Shaposhnik, Y. (2021). Journal of Machine Learning Research, 22(201), 1-73.

## License

Apache License, Version 2.0
