use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mimalloc::MiMalloc;
use ndarray::Array2;
use pacmap::knn::{find_k_nearest_neighbors, find_k_nearest_neighbors_approx};
use rand::prelude::*;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn knn_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("KNN Benchmark");
    group.sample_size(10);

    // Data sizes to test
    let sizes = vec![500, 1000, 5000, 10000, 20000, 50000, 70000];
    let k = 60;
    let dim = 784;

    // Fixed seed for reproducibility
    let seed = [0u8; 16];
    let mut rng = Pcg64Mcg::from_seed(seed);

    // Generate random data outside the benchmark loops
    let datasets: Vec<(usize, Array2<f32>)> = sizes
        .iter()
        .map(|&size| (size, generate_random_data(size, dim, &mut rng)))
        .collect();

    for (size, data) in datasets {
        // Benchmark the exact method
        group.bench_with_input(BenchmarkId::new("Exact", size), &data, |b, data| {
            b.iter(|| {
                let (neighbors, distances) = find_k_nearest_neighbors(data.view(), k);
                black_box((neighbors, distances));
            });
        });

        // Benchmark the approximate method
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

fn generate_random_data(n: usize, dim: usize, rng: &mut impl Rng) -> Array2<f32> {
    // Generate an Array2 of shape (n, dim) filled with random floats
    Array2::from_shape_fn((n, dim), |_| rng.gen())
}

criterion_group!(benches, knn_benchmark);
criterion_main!(benches);
