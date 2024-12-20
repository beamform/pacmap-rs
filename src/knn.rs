//! K-nearest neighbor computation for `PaCMAP` dimensionality reduction.
//!
//! Efficiently computes k-nearest neighbors for high-dimensional data points
//! using SIMD-accelerated Euclidean distance calculations and parallel
//! processing. Provides both exact and approximate neighbor finding algorithms.
//!
//! The neighbors and distances are used by `PaCMAP` to preserve local structure
//! during the dimensionality reduction process.
//!
//! The approximate algorithm uses a spatial index for fast lookups on large
//! datasets, while the exact algorithm computes all pairwise distances.

use crate::distance::simd_euclidean_distance;
use ndarray::{Array2, ArrayView2, Axis, Zip};
use rayon::prelude::*;
use std::cmp::min;
use thiserror::Error;
use tracing::error;
use usearch::ffi::Matches;
use usearch::{IndexOptions, MetricKind};

/// Finds k-nearest neighbors approximately using a spatial index structure.
///
/// Builds a spatial index optimized for L2 distance queries and searches it in
/// parallel. Appropriate for large datasets where exact neighbor finding is too
/// expensive.
///
/// # Arguments
/// * `data` - Input data matrix where each row is a point
/// * `k` - Number of nearest neighbors to find per point
///
/// # Returns
/// A tuple containing:
/// * `neighbor_array` - Matrix of shape `(n, k)` containing indices of nearest
///   neighbors
/// * `distance_array` - Matrix of shape `(n, k)` containing distances to
///   nearest neighbors
///
/// # Errors
/// * `KnnError::Construction` if the index cannot be created
/// * `KnnError::Reservation` if memory cannot be allocated for the index
pub fn find_k_nearest_neighbors_approx(
    data: ArrayView2<f32>,
    k: usize,
) -> Result<(Array2<u32>, Array2<f32>), KnnError> {
    let n = data.nrows();

    // Configure index for parallel L2 distance search
    let options = IndexOptions {
        dimensions: data.ncols(),
        metric: MetricKind::L2sq,
        multi: true,
        ..Default::default()
    };

    let index = usearch::new_index(&options).map_err(|e| KnnError::Construction(e.to_string()))?;

    index
        .reserve(n)
        .map_err(|e| KnnError::Reservation(e.to_string()))?;

    // Add data points to index in parallel
    Zip::indexed(data.axis_iter(Axis(0))).par_for_each(|i, vector| {
        let result = match vector.as_slice() {
            None => {
                let vector = vector.to_vec();
                index.add(i as u64, &vector)
            }
            Some(slice) => index.add(i as u64, slice),
        };

        if let Err(error) = result {
            error!("failed to add vector to index: {error}; skipping");
        }
    });

    let mut neighbors = Array2::zeros((n, k));
    let mut distances = Array2::zeros((n, k));

    // Search for neighbors of each point in parallel, excluding self matches
    Zip::indexed(data.axis_iter(Axis(0)))
        .and(neighbors.axis_iter_mut(Axis(0)))
        .and(distances.axis_iter_mut(Axis(0)))
        .par_for_each(|i, vector, mut nbrs, mut dists| {
            let identity_check = |key| key != i as u64;
            let result = match vector.as_slice() {
                None => {
                    let vector = vector.to_vec();
                    index.filtered_search(&vector, k, identity_check)
                }
                Some(slice) => index.filtered_search(slice, k, identity_check),
            };

            let (keys, distances) = match result {
                Ok(Matches { keys, distances }) => (keys, distances),
                Err(error) => {
                    error!("index search failed: {error}; skipping");
                    return;
                }
            };

            // Store neighbor indices and distances
            for (((nbr, dist), key), distance) in nbrs
                .iter_mut()
                .zip(dists.iter_mut())
                .zip(keys.into_iter())
                .zip(distances.into_iter())
            {
                *nbr = key as u32;
                *dist = distance.sqrt();
            }
        });

    Ok((neighbors, distances))
}

/// Finds k-nearest neighbors exactly by computing all pairwise distances.
///
/// Uses parallel processing and SIMD to accelerate the computation of pairwise
/// distances. Appropriate for small to medium datasets where exact neighbors
/// are desired.
///
/// # Arguments
/// * `data` - Input data matrix where each row is a point
/// * `k` - Number of nearest neighbors to find per point
///
/// # Returns
/// A tuple containing:
/// * `neighbor_array` - Matrix of shape `(n, min(k, n-1))` containing indices
///   of nearest neighbors
/// * `distance_array` - Matrix of shape `(n, min(k, n-1))` containing distances
///   to nearest neighbors
///
/// Returns empty arrays if input is empty. For a single input point, arrays
/// will have 0 columns.
#[must_use]
pub fn find_k_nearest_neighbors(data: ArrayView2<f32>, k: usize) -> (Array2<u32>, Array2<f32>) {
    let n = data.nrows();

    // Handle empty input case
    if n == 0 {
        return (Array2::<u32>::zeros((0, 0)), Array2::<f32>::zeros((0, 0)));
    }

    // Limit k to available neighbors
    let k = min(k, n - 1);

    // Compute upper triangular pairwise distances in parallel using SIMD
    let distances: Vec<_> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let row_i = data.row(i);
            let a_slice = row_i.as_slice().unwrap_or(&[]);
            (i + 1..n)
                .map(move |j| {
                    let row_j = data.row(j);
                    let b_slice = row_j.as_slice().unwrap_or(&[]);
                    let dist = simd_euclidean_distance(a_slice, b_slice);
                    (i as u32, j as u32, dist)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Sort distances to find k nearest neighbors
    let mut distances_sorted = distances;
    distances_sorted.par_sort_unstable_by(|a, b| f32::total_cmp(&a.2, &b.2));

    // Initialize output arrays
    let mut neighbor_array = Array2::<u32>::zeros((n, k));
    let mut distance_array = Array2::<f32>::from_elem((n, k), f32::MAX);
    let mut counts = vec![0; n];

    // Fill arrays with k nearest neighbors for each point
    // Each distance pair (i,j) counts as a neighbor for both i and j
    for &(i, j, distance) in &distances_sorted {
        let ix = i as usize;
        let jx = j as usize;

        if counts[ix] < k {
            neighbor_array[(ix, counts[ix])] = j;
            distance_array[(ix, counts[ix])] = distance;
            counts[ix] += 1;
        }

        if counts[jx] < k {
            neighbor_array[(jx, counts[jx])] = i;
            distance_array[(jx, counts[jx])] = distance;
            counts[jx] += 1;
        }
    }

    (neighbor_array, distance_array)
}

/// Errors that can occur during k-nearest neighbor computation.
#[derive(Debug, Error)]
pub enum KnnError {
    /// Failed to construct the spatial index
    #[error("failed to construct index: {0}")]
    Construction(String),

    /// Failed to allocate memory for the index
    #[error("failed to reserve space for vectors: {0}")]
    Reservation(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{array, Array2};
    use std::f32::consts::FRAC_1_SQRT_2;

    /// Tests finding k-nearest neighbors with empty input
    #[test]
    fn test_empty_embeddings() {
        let embeddings = Array2::<f32>::zeros((0, 128));
        let k = 5;
        let (neighbor_indices, distances) = find_k_nearest_neighbors(embeddings.view(), k);
        assert_eq!(neighbor_indices.shape(), &[0, 0]);
        assert_eq!(distances.shape(), &[0, 0]);
    }

    /// Tests finding k-nearest neighbors with single point input
    #[test]
    fn test_single_embedding() {
        let embeddings = array![[1.0, 0.0, 0.0]];
        let k = 1;
        let (neighbor_indices, distances) = find_k_nearest_neighbors(embeddings.view(), k);
        assert_eq!(neighbor_indices.shape(), &[1, 0]);
        assert_eq!(distances.shape(), &[1, 0]);
    }

    /// Tests finding k-nearest neighbors with k=0
    #[test]
    fn test_k_zero() {
        let embeddings = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let k = 0;
        let (neighbor_indices, distances) = find_k_nearest_neighbors(embeddings.view(), k);
        assert_eq!(neighbor_indices.shape(), &[3, 0]);
        assert_eq!(distances.shape(), &[3, 0]);
    }

    /// Tests finding k-nearest neighbors when k exceeds available neighbors
    #[test]
    fn test_k_exceeds() {
        let embeddings: Array2<f32> = array![[1.0], [2.0], [3.0]];
        let k = 5;
        let (neighbor_indices, distances) = find_k_nearest_neighbors(embeddings.view(), k);
        assert_eq!(neighbor_indices.shape(), &[3, 2]);
        assert_eq!(distances.shape(), &[3, 2]);
        assert!(neighbor_indices.iter().all(|&idx| idx < 3));
        assert!(distances.iter().all(|&d| d >= 0.0));
    }

    /// Tests finding k-nearest neighbors for points with unique locations
    #[test]
    fn test_normal_case() {
        let embeddings: Array2<f32> = array![
            [1.0, 0.0], // E0
            [0.0, 1.0], // E1
            [0.5, 0.5]  // E2
        ];

        let k = 2;
        let (neighbor_indices, distances) = find_k_nearest_neighbors(embeddings.view(), k);
        assert_eq!(neighbor_indices.shape(), &[3, 2]);
        assert_eq!(distances.shape(), &[3, 2]);

        let expected = vec![
            (0, vec![(2, FRAC_1_SQRT_2), (1, 1.4142)]),
            (1, vec![(2, FRAC_1_SQRT_2), (0, 1.4142)]),
            (2, vec![(0, FRAC_1_SQRT_2), (1, FRAC_1_SQRT_2)]),
        ];

        check_neighbors_and_distances(&neighbor_indices, &distances, &expected);
    }

    /// Tests finding k-nearest neighbors with duplicate points
    #[test]
    fn test_duplicate_embeddings() {
        let embeddings: Array2<f32> = array![
            [1.0, 0.0], // E0
            [1.0, 0.0], // E1 (duplicate of E0)
            [0.0, 1.0]  // E2
        ];

        let k = 2;
        let (neighbor_indices, distances) = find_k_nearest_neighbors(embeddings.view(), k);
        assert_eq!(neighbor_indices.shape(), &[3, 2]);
        assert_eq!(distances.shape(), &[3, 2]);

        let expected = vec![
            (0, vec![(1, 0.0), (2, 1.4142)]),
            (1, vec![(0, 0.0), (2, 1.4142)]),
            (2, vec![(0, 1.4142), (1, 1.4142)]),
        ];

        check_neighbors_and_distances(&neighbor_indices, &distances, &expected);
    }

    /// Tests finding k-nearest neighbors with negative coordinates
    #[test]
    fn test_negative_components() {
        let embeddings: Array2<f32> = array![
            [1.0, 0.0],  // E0
            [-1.0, 0.0], // E1
            [0.0, 1.0]   // E2
        ];

        let k = 2;
        let (neighbor_indices, distances) = find_k_nearest_neighbors(embeddings.view(), k);
        assert_eq!(neighbor_indices.shape(), &[3, 2]);
        assert_eq!(distances.shape(), &[3, 2]);

        let expected = vec![
            (0, vec![(2, 1.4142), (1, 2.0)]),
            (1, vec![(2, 1.4142), (0, 2.0)]),
            (2, vec![(0, 1.4142), (1, 1.4142)]),
        ];

        check_neighbors_and_distances(&neighbor_indices, &distances, &expected);
    }

    /// Verifies nearest neighbor computation results match expected values
    ///
    /// # Arguments
    /// * `neighbor_indices` - Computed matrix of nearest neighbor indices
    /// * `distances` - Computed matrix of distances to nearest neighbors
    /// * `expected` - Vector of expected (point_idx, vec![(neighbor_idx,
    ///   distance)]) tuples
    fn check_neighbors_and_distances(
        neighbor_indices: &Array2<u32>,
        distances: &Array2<f32>,
        expected: &Vec<(usize, Vec<(usize, f32)>)>,
    ) {
        for &(point_idx, ref expected_neighbors) in expected {
            let neighbors = neighbor_indices.row(point_idx);
            let neighbor_distances = distances.row(point_idx);

            assert_eq!(
                neighbors.len(),
                expected_neighbors.len(),
                "Mismatch in number of neighbors for point {}",
                point_idx
            );

            let mut neighbor_info: Vec<(usize, f32)> = neighbors
                .iter()
                .zip(neighbor_distances.iter())
                .map(|(&idx, &dist)| (idx as usize, dist))
                .collect();

            let mut expected_sorted = expected_neighbors.clone();
            neighbor_info.sort_by_key(|&(idx, _)| idx);
            expected_sorted.sort_by_key(|&(idx, _)| idx);

            for (&(neighbor_idx, distance), &(exp_neighbor_idx, exp_distance)) in
                neighbor_info.iter().zip(expected_sorted.iter())
            {
                assert_eq!(
                    neighbor_idx, exp_neighbor_idx,
                    "Mismatch in neighbor index for point {}",
                    point_idx
                );
                assert_relative_eq!(distance, exp_distance, epsilon = 1e-4, max_relative = 1e-4);
            }
        }
    }
}
