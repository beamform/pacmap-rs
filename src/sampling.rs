//! `PaCMAP` pair sampling implementations.
//!
//! This module provides functions for sampling three types of point pairs used
//! in `PaCMAP` dimensionality reduction:
//!
//! - Far pairs (FP): Random distant points sampled from outside each point's
//!   nearest neighbors
//! - Mid-near pairs (MN): Points sampled to preserve mid-range distances and
//!   global structure
//! - Nearest neighbors (NN): Close points based on distance metrics that
//!   preserve local structure
//!
//! Both deterministic (seeded) and non-deterministic sampling strategies are
//! supported.

use crate::distance::array_euclidean_distance;
use ndarray::parallel::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use ndarray::{azip, s, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis, Zip};
use rand::rngs::SmallRng;
use rand::{thread_rng, Rng, SeedableRng};
use rayon::slice::ParallelSliceMut;
use std::cmp::min;

/// Samples random indices while avoiding excluded values.
///
/// # Arguments
/// * `n_samples` - Number of unique indices to sample
/// * `maximum` - Maximum index value (exclusive)
/// * `reject_ind` - Array of indices that cannot be sampled
/// * `self_ind` - Index to exclude from sampling (typically the source point
///   index)
/// * `rng` - Random number generator to use
///
/// # Returns
/// A vector of `n_samples` unique indices, each < `maximum` and not in
/// `reject_ind`
fn sample_fp<R>(
    n_samples: usize,
    maximum: u32,
    reject_ind: ArrayView1<u32>,
    self_ind: u32,
    rng: &mut R,
) -> Vec<u32>
where
    R: Rng,
{
    let available_indices = (maximum as usize)
        .saturating_sub(reject_ind.len())
        .saturating_sub(reject_ind.iter().all(|&i| i != self_ind) as usize);

    let n_samples = min(n_samples, available_indices);
    let mut result = Vec::with_capacity(n_samples);

    while result.len() < n_samples {
        let j = rng.gen_range(0..maximum);
        if j != self_ind && !result.contains(&j) && reject_ind.iter().all(|&k| k != j) {
            result.push(j);
        }
    }
    result
}

/// Samples far pairs deterministically using a fixed random seed.
///
/// Generates pairs of points by selecting random indices far from each point's
/// nearest neighbors. The sampling is reproducible when using the same seed.
///
/// # Arguments
/// * `x` - Input data matrix where each row is a point
/// * `pair_neighbors` - Matrix of nearest neighbor indices for each point
/// * `n_neighbors` - Number of nearest neighbors per point
/// * `n_fp` - Number of far pairs to sample per point
/// * `random_state` - Seed for random number generation
///
/// # Returns
/// Matrix of shape `(n * n_fp, 2)` containing sampled far point pairs
pub fn sample_fp_pair_deterministic(
    x: ArrayView2<f32>,
    pair_neighbors: ArrayView2<u32>,
    n_neighbors: usize,
    n_fp: usize,
    random_state: u64,
) -> Array2<u32> {
    let n = x.nrows();
    let mut pair_fp = Array2::zeros((n * n_fp, 2));
    let n = n as u32;

    // Sample n_fp far pairs for each point in parallel
    pair_fp
        .axis_chunks_iter_mut(Axis(0), n_fp)
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut pairs)| {
            let reject_ind =
                pair_neighbors.slice(s![i * n_neighbors..(i + 1) * n_neighbors, 1_usize]);

            let mut rng = SmallRng::seed_from_u64(random_state + i as u64);
            let fp_index = sample_fp(n_fp, n, reject_ind, i as u32, &mut rng);

            if fp_index.is_empty() {
                return;
            }

            azip!((mut pair in pairs.rows_mut(), &index in &fp_index) {
                pair[0] = i as u32;
                pair[1] = index;
            });
        });

    pair_fp
}

/// Samples mid-near pairs deterministically using a fixed random seed.
///
/// Generates pairs of points with intermediate distances to help preserve
/// global structure. Each pair is selected by sampling 6 random points and
/// picking the second closest.
///
/// # Arguments
/// * `x` - Input data matrix where each row is a point
/// * `n_mn` - Number of mid-near pairs to sample per point
/// * `random_state` - Seed for random number generation
///
/// # Returns
/// Matrix of shape `(n * n_mn, 2)` containing sampled mid-near pairs
pub fn sample_mn_pair_deterministic(
    x: ArrayView2<f32>,
    n_mn: usize,
    random_state: u64,
) -> Array2<u32> {
    let n = x.nrows();
    let mut pair_mn = Array2::<u32>::zeros((n * n_mn, 2));
    let n = n as u32;

    // Sample n_mn mid-near pairs for each point in parallel
    pair_mn
        .axis_chunks_iter_mut(Axis(0), n_mn)
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut pairs)| {
            let mut rng = SmallRng::seed_from_u64(random_state + i as u64);
            for j in 0..n_mn {
                let reject_ind = pairs.slice(s![0..j, 1_usize]);
                let sampled = sample_fp(6, n, reject_ind, i as u32, &mut rng);
                sample_mn_pair_impl(x, pairs.row_mut(j), i, &sampled);
            }
        });

    pair_mn
}

/// Samples far pairs using the global thread RNG.
///
/// Non-deterministic version of `sample_fp_pair_deterministic` that uses the
/// global thread-local random number generator instead of a seeded RNG.
///
/// # Arguments
/// * `x` - Input data matrix where each row is a point
/// * `pair_neighbors` - Matrix of nearest neighbor indices for each point
/// * `n_neighbors` - Number of nearest neighbors per point
/// * `n_fp` - Number of far pairs to sample per point
///
/// # Returns
/// Matrix of shape `(n * n_fp, 2)` containing sampled far point pairs
pub fn sample_fp_pair(
    x: ArrayView2<f32>,
    pair_neighbors: ArrayView2<u32>,
    n_neighbors: usize,
    n_fp: usize,
) -> Array2<u32> {
    let n = x.nrows();
    let mut pair_fp = Array2::zeros((n * n_fp, 2));
    let n = n as u32;

    // Sample n_fp far pairs for each point in parallel
    pair_fp
        .axis_chunks_iter_mut(Axis(0), n_fp)
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut pairs)| {
            let reject_ind =
                pair_neighbors.slice(s![i * n_neighbors..(i + 1) * n_neighbors, 1_usize]);

            let mut rng = thread_rng();
            let fp_index = sample_fp(n_fp, n, reject_ind, i as u32, &mut rng);

            if fp_index.is_empty() {
                return;
            }

            azip!((mut pair in pairs.rows_mut(), &index in &fp_index) {
                pair[0] = i as u32;
                pair[1] = index;
            });
        });

    pair_fp
}

/// Samples mid-near pairs using the global thread RNG.
///
/// Non-deterministic version of `sample_mn_pair_deterministic` that uses the
/// global thread-local random number generator instead of a seeded RNG.
///
/// # Arguments
/// * `x` - Input data matrix where each row is a point
/// * `n_mn` - Number of mid-near pairs to sample per point
///
/// # Returns
/// Matrix of shape `(n * n_mn, 2)` containing sampled mid-near pairs
pub fn sample_mn_pair(x: ArrayView2<f32>, n_mn: usize) -> Array2<u32> {
    let n = x.nrows();
    let mut pair_mn = Array2::<u32>::zeros((n * n_mn, 2));
    let n = n as u32;

    pair_mn
        .axis_chunks_iter_mut(Axis(0), n_mn)
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut pairs)| {
            let mut rng = thread_rng();
            for j in 0..n_mn {
                let reject_ind = pairs.slice(s![0..j, 1_usize]);
                let sampled = sample_fp(6, n, reject_ind, i as u32, &mut rng);
                sample_mn_pair_impl(x, pairs.row_mut(j), i, &sampled);
            }
        });

    pair_mn
}

/// Samples nearest neighbor pairs based on scaled distances.
///
/// Generates pairs by connecting each point with its nearest neighbors
/// according to a distance matrix. Neighbors are sorted by their scaled
/// distances.
///
/// # Arguments
/// * `x` - Input data matrix where each row is a point
/// * `scaled_dist` - Matrix of scaled distances between points
/// * `neighbors` - Matrix of nearest neighbor indices for each point
/// * `n_neighbors` - Number of neighbors to sample per point
///
/// # Returns
/// Matrix of shape `(n * n_neighbors, 2)` containing sampled neighbor pairs
pub fn sample_neighbors_pair(
    x: ArrayView2<f32>,
    scaled_dist: ArrayView2<f32>,
    neighbors: ArrayView2<u32>,
    n_neighbors: usize,
) -> Array2<u32> {
    let n = x.nrows();
    let mut sorted_dist_indices = Array2::<u32>::zeros(scaled_dist.dim());

    // Sort scaled distances for each point and store sorted indices
    Zip::from(scaled_dist.axis_iter(Axis(0)))
        .and(sorted_dist_indices.axis_iter_mut(Axis(0)))
        .par_for_each(|distances, mut indices| {
            let mut distance_indices = distances.into_iter().enumerate().collect::<Vec<_>>();
            distance_indices.par_sort_unstable_by(|a, b| f32::total_cmp(a.1, b.1));
            for (i, (index, _)) in distance_indices.iter().enumerate() {
                indices[i] = *index as u32;
            }
        });

    let mut pair_neighbors = Array2::zeros((n * n_neighbors, 2));
    pair_neighbors
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(pair_index, mut pair)| {
            let i = pair_index / n_neighbors;
            let j = pair_index % n_neighbors;
            let scaled_sort = sorted_dist_indices.row(i);

            pair[0] = i as u32;
            pair[1] = neighbors[[i, scaled_sort[j] as usize]];
        });

    pair_neighbors
}

/// Creates a mid-near pair by finding the second closest point from sampled
/// candidates.
///
/// # Arguments
/// * `x` - Input data matrix where each row is a point
/// * `pair` - Output array to store the sampled pair indices
/// * `i` - Index of source point
/// * `sampled` - Array of randomly sampled candidate indices
fn sample_mn_pair_impl(
    x: ArrayView2<f32>,
    mut pair: ArrayViewMut1<u32>,
    i: usize,
    sampled: &[u32],
) {
    let mut distance_indices = [(0.0, 0); 6];
    for (&s, entry) in sampled.iter().zip(distance_indices.iter_mut()) {
        let distance = array_euclidean_distance(x.row(i), x.row(s as usize));
        *entry = (distance, s);
    }

    distance_indices.sort_unstable_by(|a, b| f32::total_cmp(&a.0, &b.0));
    let picked = distance_indices[1].1;

    pair[0] = i as u32;
    pair[1] = picked;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use ndarray_rand::RandomExt;
    use rand::distributions::Uniform;
    use rand::SeedableRng;

    #[test]
    fn test_sample_fp() {
        let mut rng = SmallRng::from_seed([0; 32]);
        let n_samples = 5;
        let maximum = 10;
        let reject_ind = array![2, 4, 6];

        let result = sample_fp(n_samples, maximum, reject_ind.view(), 0, &mut rng);

        assert_eq!(result.len(), n_samples);
        for &x in result.iter() {
            assert!(x < maximum);
            assert!(!reject_ind.iter().any(|&k| k == x));
        }

        // Check for uniqueness
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                assert_ne!(result[i], result[j]);
            }
        }
    }

    #[test]
    fn test_sample_fp_pair_deterministic() {
        let x = Array2::ones((100, 10));
        let pair_neighbors = Array2::from_shape_fn((1000, 2), |(i, j)| {
            if j == 0 {
                (i / 10) as u32
            } else {
                (i % 10) as u32
            }
        });
        let n_neighbors = 10;
        let n_fp = 5;
        let random_state = 42;

        let result = sample_fp_pair_deterministic(
            x.view(),
            pair_neighbors.view(),
            n_neighbors,
            n_fp,
            random_state,
        );

        // Check shape
        assert_eq!(result.shape(), &[500, 2]);

        // Check that each point has exactly n_fp samples
        for i in 0..100 {
            assert_eq!(
                result
                    .slice(s![i * n_fp..(i + 1) * n_fp, 0])
                    .iter()
                    .all(|&x| x == i as u32),
                true
            );
        }

        // Check that samples are not in the nearest neighbors
        for i in 0..100 {
            let neighbors = pair_neighbors.slice(s![i * n_neighbors..(i + 1) * n_neighbors, 1]);
            for j in 0..n_fp {
                assert!(!neighbors.iter().any(|x| x == &result[[i * n_fp + j, 1]]));
            }
        }

        // Check determinism
        let result2 = sample_fp_pair_deterministic(
            x.view(),
            pair_neighbors.view(),
            n_neighbors,
            n_fp,
            random_state,
        );

        assert_eq!(result, result2);
    }

    #[test]
    fn test_sample_mn_pair_deterministic() {
        let x = Array2::random((1000, 20), Uniform::new(-1.0, 1.0));
        let n_mn = 5;
        let random_state = 42;

        let result = sample_mn_pair_deterministic(x.view(), n_mn, random_state);

        // Check shape
        assert_eq!(result.shape(), &[1000 * n_mn, 2]);

        // Check if all pairs are valid
        for pair in result.rows() {
            assert!(pair[0] < 1000);
            assert!(pair[1] < 1000);
            assert_ne!(pair[0], pair[1]);
        }

        // Check determinism
        let result2 = sample_mn_pair_deterministic(x.view(), n_mn, random_state);
        assert_eq!(result, result2);
    }

    #[test]
    fn test_sample_fp_pair() {
        let x = Array2::random((100, 10), Uniform::new(0., 1.));
        let pair_neighbors = Array2::from_shape_fn((1000, 2), |(i, j)| {
            if j == 0 {
                (i / 10) as u32
            } else {
                (i % 10) as u32
            }
        });
        let n_neighbors = 10;
        let n_fp = 5;

        let result = sample_fp_pair(x.view(), pair_neighbors.view(), n_neighbors, n_fp);

        // Check shape
        assert_eq!(result.shape(), &[500, 2]);

        // Check that each point has exactly n_fp samples
        for i in 0..100 {
            assert_eq!(
                result
                    .slice(s![i * n_fp..(i + 1) * n_fp, 0])
                    .iter()
                    .all(|&x| x == i as u32),
                true
            );
        }

        // Check that samples are not in the nearest neighbors
        for i in 0..100 {
            let neighbors = pair_neighbors.slice(s![i * n_neighbors..(i + 1) * n_neighbors, 1]);
            for j in 0..n_fp {
                assert!(!neighbors.iter().any(|x| x == &result[[i * n_fp + j, 1]]));
            }
        }
    }

    #[test]
    fn test_sample_mn_pair() {
        let x = Array2::random((1000, 20), Uniform::new(-1.0, 1.0));
        let n_mn = 5;

        let result = sample_mn_pair(x.view(), n_mn);

        // Check shape
        assert_eq!(result.shape(), &[1000 * n_mn, 2]);

        // Check if all pairs are valid
        for pair in result.rows() {
            assert!(pair[0] < 1000);
            assert!(pair[1] < 1000);
        }

        // Check if each point has exactly n_mn pairs
        for i in 0..1000 {
            let count = result
                .rows()
                .into_iter()
                .filter(|row| row[0] == i as u32)
                .count();

            assert_eq!(count, n_mn);
        }
    }

    #[test]
    fn test_sample_neighbors_pair() {
        let x = Array2::random((10, 5), Uniform::new(0., 1.));
        let scaled_dist = Array2::random((10, 10), Uniform::new(0., 1.));
        let neighbors = Array2::from_shape_fn((10, 10), |(_i, j)| j as u32);
        let n_neighbors = 5;

        let result =
            sample_neighbors_pair(x.view(), scaled_dist.view(), neighbors.view(), n_neighbors);

        // Check shape
        assert_eq!(result.shape(), &[50, 2]);

        // Check if all pairs are valid
        for pair in result.rows() {
            assert!(pair[0] < 10);
            assert!(pair[1] < 10);
        }

        // Check if each point has exactly n_neighbors pairs
        for i in 0..10 {
            let count = result
                .rows()
                .into_iter()
                .filter(|row| row[0] == i as u32)
                .count();

            assert_eq!(count, n_neighbors);
        }

        // Check if neighbors are sorted by scaled distance
        for i in 0..10 {
            let neighbors = result
                .rows()
                .into_iter()
                .filter(|row| row[0] == i as u32)
                .map(|row| row[1])
                .collect::<Vec<_>>();

            let row = scaled_dist.row(i as usize);
            let mut sorted_distances = row.iter().enumerate().collect::<Vec<_>>();
            sorted_distances.sort_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap());

            for (j, &neighbor) in neighbors.iter().enumerate() {
                assert_eq!(neighbor, sorted_distances[j].0 as u32);
            }
        }
    }
}
