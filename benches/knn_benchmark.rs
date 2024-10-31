use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use pacmap_rs::knn::find_k_nearest_neighbors;
use rand::prelude::*;

fn generate_random_data(n: usize, d: usize, seed: u64) -> Array2<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    Array2::from_shape_fn((n, d), |_| rng.gen())
}

fn bench_knn_functions(c: &mut Criterion) {
    let k_values = [15];
    let embedding_lengths = [1024, 4096];
    let n_values = [10, 100, 500, 1000, 5000, 10000];
    let seed = 42;

    let mut group = c.benchmark_group("k_nearest_neighbors");

    for &n in &n_values {
        for &d in &embedding_lengths {
            let data = generate_random_data(n, d, seed);
            let data_view = data.view();

            for &k in &k_values {
                group.bench_with_input(
                    BenchmarkId::new("exact", format!("n={},d={},k={}", n, d, k)),
                    &(data_view, k),
                    |b, &(data, k)| b.iter(|| find_k_nearest_neighbors(data, k)),
                );
            }
        }
    }

    group.finish();
}

criterion_group!(benches, bench_knn_functions);
criterion_main!(benches);
