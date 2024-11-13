//! Benchmarks for k-nearest neighbors implementations in PaCMAP.
//!
//! Compares performance of exact and approximate KNN algorithms across
//! different dataset sizes and dimensions. Uses criterion for structured
//! benchmarking and MiMalloc for optimized memory allocation.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mimalloc::MiMalloc;
use ndarray::Array2;
use pacmap::knn::{find_k_nearest_neighbors, find_k_nearest_neighbors_approx};
use rand::prelude::*;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// Use MiMalloc as the global allocator for optimal memory performance
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Benchmarks exact vs approximate k-nearest neighbors algorithms.
///
/// Measures performance across multiple dataset sizes, keeping dimension and k
/// fixed. Uses reproducible random data generated with a fixed seed.
///
/// # Arguments
/// * `c` - Criterion benchmark configuration
fn knn_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("KNN Benchmark");
    group.sample_size(10);

    let sizes = vec![500, 1000, 5000, 10000, 20000, 50000, 70000];
    let k = 60;
    let dim = 784;

    let seed = [0u8; 16];
    let mut rng = Pcg64Mcg::from_seed(seed);

    // Generate all datasets upfront to avoid including data generation in timing
    let datasets: Vec<(usize, Array2<f32>)> = sizes
        .iter()
        .map(|&size| (size, generate_random_data(size, dim, &mut rng)))
        .collect();

    for (size, data) in datasets {
        // Benchmark exact KNN implementation
        group.bench_with_input(BenchmarkId::new("Exact", size), &data, |b, data| {
            b.iter(|| {
                let (neighbors, distances) = find_k_nearest_neighbors(data.view(), k);
                black_box((neighbors, distances));
            });
        });

        // Benchmark approximate KNN implementation
        group.bench_with_input(BenchmarkId::new("Approximate", size), &data, |b, data| {
            b.iter(|| match find_k_nearest_neighbors_approx(data.view(), k) {
                Ok((neighbors, distances)) => {
                    black_box((neighbors, distances));
                }
                Err(err) => {
                    panic!("Error during approximate KNN: {:?}", err);
                }
            });
        });
    }

    group.finish();
}

/// Generates random test data for KNN benchmarking.
///
/// Creates a matrix of uniformly distributed random floats.
///
/// # Arguments
/// * `n` - Number of samples/rows
/// * `dim` - Number of features/columns
/// * `rng` - Random number generator
///
/// # Returns
/// An n × dim matrix of random floats
fn generate_random_data(n: usize, dim: usize, rng: &mut impl Rng) -> Array2<f32> {
    Array2::from_shape_fn((n, dim), |_| rng.gen())
}

criterion_group!(benches, knn_benchmark);
criterion_main!(benches);
