//! Optimized distance calculations for `PaCMAP` dimensionality reduction.
//!
//! This module provides efficient implementations of distance metrics using
//! SIMD instructions where possible. It includes functions for:
//!
//! - Computing Euclidean distances between vectors using SIMD
//! - Scaling distances based on per-point sigma values
//! - Handling both contiguous and non-contiguous array views

use ndarray::{Array2, ArrayView1, ArrayView2, Zip};
use tracing::warn;
use wide::f32x8;

/// Scales distances between points using per-point sigma values for adaptive
/// scaling.
///
/// The scaling formula is: `scaled_dist` = dist^2 / (`sigma_i` * `sigma_j`)
/// where `sigma_i` and `sigma_j` are the scaling factors for points i and j.
///
/// # Arguments
/// * `knn_distances` - Matrix of distances to k-nearest neighbors
/// * `sig` - Scaling factor for each point
/// * `neighbors` - Indices of k-nearest neighbors for each point
///
/// # Returns
/// A matrix of scaled distances with same dimensions as `knn_distances`
pub fn scale_dist(
    knn_distances: ArrayView2<f32>,
    sig: ArrayView1<f32>,
    neighbors: ArrayView2<u32>,
) -> Array2<f32> {
    Zip::indexed(knn_distances)
        .and(neighbors)
        .par_map_collect(|(i, _), knn_dist, neighbor| {
            knn_dist * knn_dist / (sig[i] * sig[*neighbor as usize])
        })
}

/// Computes Euclidean distance between vectors using SIMD operations.
///
/// Processes vectors in chunks of 8 elements using SIMD instructions for
/// improved performance. Handles remaining elements sequentially.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Panics
/// * If vectors have different lengths
pub fn simd_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let a_chunks = a.chunks_exact(8);
    let a_remainder = a_chunks.remainder();

    let b_chunks = b.chunks_exact(8);
    let b_remainder = b_chunks.remainder();

    // Process 8 elements at a time using SIMD
    let mut sum_sq = f32x8::splat(0.0);
    for (a_chunk, b_chunk) in a_chunks.zip(b_chunks) {
        let diff = f32x8::from(a_chunk) - f32x8::from(b_chunk);
        sum_sq += diff * diff;
    }

    let mut total_sum_sq: f32 = sum_sq.as_array_ref().iter().sum();

    // Handle remaining elements sequentially
    for (a, b) in a_remainder.iter().zip(b_remainder) {
        let diff = a - b;
        total_sum_sq += diff * diff;
    }

    total_sum_sq.sqrt()
}

/// Computes Euclidean distance between array views with optimized path for
/// contiguous data.
///
/// Attempts to use SIMD operations on contiguous memory first, falling back to
/// slower methods for non-contiguous data with appropriate warnings.
///
/// # Arguments
/// * `a` - First vector as array view
/// * `b` - Second vector as array view
///
/// # Returns
/// Euclidean distance between the vectors
pub fn array_euclidean_distance(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    let a_slice = a.as_slice();
    let b_slice = b.as_slice();

    match (a_slice, b_slice) {
        (Some(a), Some(b)) => simd_euclidean_distance(a, b),
        (Some(a), None) => {
            warn!("b is non-contiguous, requiring allocation to compute distance");
            simd_euclidean_distance(a, &b.to_vec())
        }
        (None, Some(b)) => {
            warn!("a is non-contiguous, requiring allocation to compute distance");
            simd_euclidean_distance(&a.to_vec(), b)
        }
        (None, None) => {
            warn!("both a and b are non-contiguous, requiring allocation to compute distance");
            simd_euclidean_distance(&a.to_vec(), &b.to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    /// Tests for Euclidean distance calculations
    mod euclidean {
        use crate::distance::{scale_dist, simd_euclidean_distance};
        use approx::abs_diff_eq;
        use ndarray::Zip;
        use quickcheck::{Arbitrary, Gen, TestResult};
        use quickcheck_macros::quickcheck;

        #[test]
        fn test_scale_dist() {
            use ndarray::array;
            use ndarray::Array2;

            let knn_distances = array![[1.0, 2.0], [3.0, 4.0]];
            let sig = array![1.0, 2.0];
            let neighbors = array![[0, 1], [1, 0]];

            let expected: Array2<f32> = array![[1.0, 2.0], [2.25, 8.0]];
            let result = scale_dist(knn_distances.view(), sig.view(), neighbors.view());

            assert_eq!(result, expected);
        }

        #[test]
        fn test_scale_dist_large() {
            use ndarray::array;
            use ndarray::Array2;

            let knn_distances = array![
                [0.495694, 0.164182, 0.38736224, 0.7196466, 0.38224003],
                [0.36155823, 0.50222975, 0.9781539, 0.98941207, 0.8973262],
                [0.1860112, 0.42378604, 0.2788846, 0.669746, 0.55980533],
                [0.11129272, 0.6330095, 0.13824642, 0.79582757, 0.85618436],
                [0.35381937, 0.14951079, 0.5282848, 0.15796484, 0.4141447],
                [0.88824165, 0.08663641, 0.64704186, 0.010489, 0.57278913],
                [0.85442775, 0.35388637, 0.73989093, 0.27819583, 0.1791155],
                [0.38080302, 0.4854166, 0.77916056, 0.04510982, 0.73329055],
                [0.8139709, 0.74042857, 0.6179393, 0.46581566, 0.36481187],
                [0.37006631, 0.73663354, 0.425183, 0.03042419, 0.324348]
            ];

            let sig = array![
                1.3664602, 1.5004125, 1.841473, 1.2978389, 1.4815831, 1.0259365, 1.391211,
                1.2620935, 1.3638133, 1.1639808
            ];

            let neighbors = array![
                [1, 5, 3, 4, 7],
                [6, 6, 8, 2, 3],
                [7, 3, 6, 6, 7],
                [7, 6, 3, 0, 4],
                [6, 8, 1, 4, 0],
                [1, 4, 1, 0, 5],
                [0, 2, 5, 2, 2],
                [1, 5, 3, 6, 8],
                [9, 5, 5, 6, 3],
                [7, 3, 7, 8, 4]
            ];

            let expected: Array2<f32> = array![
                [0.11984493, 0.019228, 0.08460905, 0.2558088, 0.08471958],
                [0.06262574, 0.12083735, 0.4675723, 0.35430592, 0.41349402],
                [0.01488748, 0.07514622, 0.03035922, 0.17509021, 0.1348395],
                [0.00756173, 0.2219252, 0.01134661, 0.35712493, 0.3812306],
                [0.06073572, 0.01106278, 0.12554504, 0.0113676, 0.08471907],
                [0.51254398, 0.00493804, 0.2719779, 0.00007848, 0.31170845],
                [0.38402605, 0.04888429, 0.38355, 0.03020945, 0.01252299],
                [0.07657704, 0.18197729, 0.37063086, 0.00115893, 0.31239617],
                [0.41736597, 0.39182392, 0.27290797, 0.11436161, 0.07519037],
                [0.09322271, 0.3592, 0.12305929, 0.00058309, 0.06100292]
            ];

            let result = scale_dist(knn_distances.view(), sig.view(), neighbors.view());
            let tol = 1e-6;

            Zip::from(&result).and(&expected).for_each(|&a, &b| {
                assert!(abs_diff_eq!(a, b, epsilon = tol), "a: {}, b: {}", a, b);
            });
        }

        /// Test type for QuickCheck property testing of vector pairs.
        ///
        /// Generates pairs of vectors with same length containing only finite
        /// values.
        #[derive(Clone, Debug)]
        struct VecPair(Vec<f32>, Vec<f32>);

        impl Arbitrary for VecPair {
            fn arbitrary(g: &mut Gen) -> VecPair {
                loop {
                    let len = u8::arbitrary(g) as usize;
                    let a: Vec<_> = (0..len).map(|_| f32::arbitrary(g)).collect();
                    let b: Vec<_> = (0..len).map(|_| f32::arbitrary(g)).collect();

                    if !a
                        .iter()
                        .chain(b.iter())
                        .any(|v| v.is_nan() || v.is_infinite())
                    {
                        break VecPair(a, b);
                    }
                }
            }
        }

        /// Reference implementation of Euclidean distance for testing.
        pub fn standard_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
            debug_assert_eq!(a.len(), b.len(), "Vectors must have the same length");

            let sum_sq: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(a_i, b_i)| (a_i - b_i).powi(2))
                .sum();

            sum_sq.sqrt()
        }

        #[quickcheck]
        fn non_negative(pair: VecPair) -> bool {
            let VecPair(a, b) = pair;
            simd_euclidean_distance(&a, &b) >= 0.0
        }

        #[quickcheck]
        fn zero_when_equal(pair: VecPair) -> bool {
            let VecPair(a, _) = pair;
            simd_euclidean_distance(&a, &a) == 0.0
        }

        #[quickcheck]
        fn distance_symmetry(pair: VecPair) -> TestResult {
            let VecPair(a, b) = pair;
            let d1 = simd_euclidean_distance(&a, &b);
            let d2 = simd_euclidean_distance(&b, &a);
            let difference = (d1 - d2).abs();

            if difference > f32::EPSILON {
                TestResult::error(format!("difference is {difference}"))
            } else {
                TestResult::passed()
            }
        }

        #[quickcheck]
        fn correctness(pair: VecPair) -> TestResult {
            let VecPair(a, b) = pair;
            let simd_result = simd_euclidean_distance(&a, &b);
            let standard_result = standard_euclidean_distance(&a, &b);
            let difference = (simd_result - standard_result).abs();

            if difference > f32::EPSILON {
                TestResult::error(format!("difference is {difference}"))
            } else {
                TestResult::passed()
            }
        }
    }
}
