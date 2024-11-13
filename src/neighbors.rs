//! Manages pair generation for `PaCMAP` dimensionality reduction.
//!
//! This module generates three types of point pairs used in `PaCMAP`:
//! - Nearest neighbor pairs: Preserve local structure
//! - Mid-near pairs: Preserve intermediate structure
//! - Far pairs: Preserve global structure and prevent collapse
//!
//! The module supports both deterministic and non-deterministic pair sampling.

use crate::distance::scale_dist;
use crate::knn::{find_k_nearest_neighbors, find_k_nearest_neighbors_approx, KnnError};
use crate::sampling::{
    sample_fp_pair, sample_fp_pair_deterministic, sample_mn_pair, sample_mn_pair_deterministic,
    sample_neighbors_pair,
};
use crate::Pairs;
use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use std::cmp::min;

/// Generates all three pair types needed for `PaCMAP`: nearest neighbors,
/// mid-near, and far pairs.
///
/// This function handles the complete pair sampling pipeline:
/// 1. Finds k-nearest neighbors with extra padding for robustness
/// 2. Computes distance scaling factors from moderately distant neighbors
/// 3. Scales raw distances using these factors
/// 4. Samples pairs of each type either deterministically or randomly
///
/// # Arguments
/// * `x` - Input data matrix with dimensions (`n_samples`, `n_features`)
/// * `n_neighbors` - Number of nearest neighbors per point
/// * `n_mn` - Number of mid-near pairs per point
/// * `n_fp` - Number of far pairs per point
/// * `random_state` - Optional seed for deterministic sampling
///
/// # Returns
/// A `Pairs` struct containing arrays of indices for each pair type
pub fn generate_pairs(
    x: ArrayView2<f32>,
    n_neighbors: usize,
    n_mn: usize,
    n_fp: usize,
    random_state: Option<u64>,
) -> Result<Pairs, KnnError> {
    let n = x.nrows();

    // Add padding neighbors for robustness in selecting final pairs
    let n_neighbors_extra = (n_neighbors + 50).min(n - 1);
    let n_neighbors = n_neighbors.min(n - 1);

    // Find k-nearest neighbors and distances
    let (neighbors, knn_distances) = if n < 8_000 {
        find_k_nearest_neighbors(x, n_neighbors_extra)
    } else {
        find_k_nearest_neighbors_approx(x, n_neighbors_extra)?
    };

    // Calculate scaling using mean distance of moderately distant neighbors
    // (indices 3-5) This provides robustness compared to using nearest or
    // farthest neighbors
    let start = min(3, knn_distances.ncols().saturating_sub(1));
    let end = min(6, knn_distances.ncols());
    let sig = knn_distances
        .slice(s![.., start..end])
        .mean_axis(Axis(1))
        .map_or_else(|| Array1::from_elem(n, 1e-10), |d| d.mapv(|x| x.max(1e-10)));

    // Scale distances and sample neighbor pairs
    let neighbors_view = neighbors.view();
    let scaled_dist = scale_dist(knn_distances.view(), sig.view(), neighbors_view);
    let pair_neighbors =
        sample_neighbors_pair(x.view(), scaled_dist.view(), neighbors_view, n_neighbors);

    // Generate remaining pairs either deterministically or randomly
    let (pair_mn, pair_fp) = match random_state {
        Some(seed) => (
            sample_mn_pair_deterministic(x.view(), n_mn, seed),
            sample_fp_pair_deterministic(x.view(), pair_neighbors.view(), n_neighbors, n_fp, seed),
        ),
        None => (
            sample_mn_pair(x.view(), n_mn),
            sample_fp_pair(x.view(), pair_neighbors.view(), n_neighbors, n_fp),
        ),
    };

    Ok(Pairs {
        pair_neighbors,
        pair_mn,
        pair_fp,
    })
}

/// Generates only mid-near and far pairs using pre-computed nearest neighbor
/// pairs.
///
/// This function is used when nearest neighbor pairs are already available,
/// avoiding redundant neighbor computation while still generating the other
/// required pair types.
///
/// # Arguments
/// * `x` - Input data matrix with dimensions (`n_samples`, `n_features`)
/// * `n_neighbors` - Number of nearest neighbors used to generate input pairs
/// * `n_mn` - Number of mid-near pairs to generate per point
/// * `n_fp` - Number of far pairs to generate per point
/// * `pair_neighbors` - Pre-computed nearest neighbor pair indices
/// * `random_seed` - Optional seed for deterministic sampling
///
/// # Returns
/// Tuple of arrays containing:
/// - Mid-near pair indices with shape (`n_samples` * `n_mn`, 2)
/// - Far pair indices with shape (`n_samples` * `n_fp`, 2)
pub fn generate_pair_no_neighbors(
    x: ArrayView2<f32>,
    n_neighbors: usize,
    n_mn: usize,
    n_fp: usize,
    pair_neighbors: ArrayView2<u32>,
    random_seed: Option<u64>,
) -> (Array2<u32>, Array2<u32>) {
    match random_seed {
        Some(seed) => (
            sample_mn_pair_deterministic(x, n_mn, seed),
            sample_fp_pair_deterministic(x, pair_neighbors, n_neighbors, n_fp, seed),
        ),
        None => (
            sample_mn_pair(x, n_mn),
            sample_fp_pair(x, pair_neighbors, n_neighbors, n_fp),
        ),
    }
}
