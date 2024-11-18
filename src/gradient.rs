//! `PaCMAP` gradient calculation implementation.
//!
//! This module provides the core gradient computation functionality for
//! `PaCMAP`'s loss function, which balances attractive forces between nearby
//! points and repulsive forces between distant points. The gradient is used to
//! iteratively optimize the low-dimensional embedding coordinates.

use crate::weights::Weights;
use ndarray::{Array2, ArrayView2, Axis};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// Calculates the gradient of the `PaCMAP` loss function for the current
/// embedding.
///
/// Computes contributions to the gradient from three types of point pairs:
/// - Nearest neighbor pairs that preserve local structure through attraction
/// - Mid-near pairs that preserve intermediate structure through attraction
/// - Far pairs that prevent collapse through repulsion
///
/// The gradient contributions are calculated in parallel using chunked
/// processing for memory efficiency.
///
/// # Arguments
/// * `y` - Current embedding coordinates as an n × d matrix
/// * `pair_neighbors` - Matrix of nearest neighbor pair indices
/// * `pair_mn` - Matrix of mid-near pair indices
/// * `pair_fp` - Matrix of far pair indices
/// * `weights` - Weight parameters controlling the relative strength of each
///   pair type
///
/// # Returns
/// An (n+1) × d matrix containing:
/// - Gradient values for each point in the first n rows
/// - Total loss value in the first column of the last row
pub fn pacmap_grad<'a>(
    y: ArrayView2<f32>,
    pair_neighbors: ArrayView2<'a, u32>,
    pair_mn: ArrayView2<'a, u32>,
    pair_fp: ArrayView2<'a, u32>,
    weights: &Weights,
) -> Array2<f32> {
    let (n, dim) = y.dim();

    // Define parameters for each pair type:
    // (pairs, weight, denominator constant, weight constant, is_far_pair)
    let pair_params = [
        (pair_neighbors, weights.w_neighbors, 10.0, 20.0, false),
        (pair_mn, weights.w_mn, 10000.0, 20000.0, false),
        (pair_fp, weights.w_fp, 1.0, 2.0, true),
    ];

    // Process chunks of pairs in parallel and sum their gradient contributions
    let (mut grad, total_loss) = pair_params
        .iter()
        .flat_map(|(pairs, w, denom_const, w_const, is_fp)| {
            pairs
                .axis_chunks_iter(Axis(0), 1024)
                .map(move |chunk| (chunk, *w, *denom_const, *w_const, *is_fp))
        })
        .collect::<Vec<_>>()
        .par_iter()
        .map(|&(pairs, w, denom_const, w_const, is_fp)| {
            process_pairs(y, pairs, w, denom_const, w_const, is_fp, n, dim)
        })
        .reduce(
            || (Array2::zeros((n + 1, dim)), 0.0),
            |(mut grad1, loss1), (grad2, loss2)| {
                grad1 += &grad2;
                (grad1, loss1 + loss2)
            },
        );

    grad[[n, 0]] = total_loss;
    grad
}

/// Processes a chunk of point pairs to compute their gradient contributions.
///
/// For each pair, calculates:
/// - The squared distance between points in the current embedding
/// - The contribution to the loss based on pair type and distance
/// - Gradient updates that either attract or repel the points
///
/// # Arguments
/// * `y` - Current embedding coordinates
/// * `pairs` - Chunk of point pair indices to process
/// * `w` - Weight factor for this pair type
/// * `denom_const` - Denominator constant in weight formula
/// * `w_const` - Weight constant in gradient formula
/// * `is_fp` - True if these are far pairs with repulsive forces
/// * `n` - Number of points
/// * `dim` - Embedding dimension
///
/// # Returns
/// A tuple containing:
/// - Gradient matrix for this chunk of pairs
/// - Loss value for this chunk
#[allow(clippy::too_many_arguments)]
fn process_pairs(
    y: ArrayView2<f32>,
    pairs: ArrayView2<u32>,
    w: f32,
    denom_const: f32,
    w_const: f32,
    is_fp: bool,
    n: usize,
    dim: usize,
) -> (Array2<f32>, f32) {
    let mut grad = Array2::zeros((n + 1, dim));
    let mut loss = 0.0;
    let mut y_ij = vec![0.0; dim];

    for pair_row in pairs.rows() {
        let i = pair_row[0] as usize;
        let j = pair_row[1] as usize;

        if i == j {
            continue;
        }

        // Calculate squared distance between points
        let mut d_ij = 1.0f32;
        for d in 0..dim {
            y_ij[d] = y[[i, d]] - y[[j, d]];
            d_ij += y_ij[d].powi(2);
        }

        if is_fp {
            // Repulsive updates for far pairs
            loss += w * (1.0 / (1.0 + d_ij));
            let w1 = w * (2.0 / (1.0 + d_ij).powi(2));

            for d in 0..dim {
                let grad_update = w1 * y_ij[d];
                grad[[i, d]] -= grad_update;
                grad[[j, d]] += grad_update;
            }
        } else {
            // Attractive updates for neighbor/mid-near pairs
            loss += w * (d_ij / (denom_const + d_ij));
            let w1 = w * (w_const / (denom_const + d_ij).powi(2));

            for d in 0..dim {
                let grad_update = w1 * y_ij[d];
                grad[[i, d]] += grad_update;
                grad[[j, d]] -= grad_update;
            }
        }
    }

    (grad, loss)
}

#[cfg(test)]
mod test {
    use super::pacmap_grad;
    use crate::weights::Weights;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Zip};

    #[test]
    fn test_pacmap_grad() {
        let y_test = array![
            [-0.70575494, 0.4136191],
            [-0.5127779, 1.060248],
            [-1.0165913, -1.1657093],
            [-0.8206925, 0.9737984],
            [-1.0650787, -1.5299057],
            [-0.02214996, -1.4788837],
            [0.37072298, 1.6783544],
            [-1.0666362, 1.1047112],
            [-0.2004564, -0.08376265],
            [-1.1240833, 0.10645787],
        ];

        let pair_neighbors = array![[0, 1], [2, 3], [4, 5]];
        let pair_mn = array![[6, 7], [8, 9]];
        let pair_fp = array![[0, 2], [3, 5]];

        let w_neighbors = 0.5;
        let w_mn = 0.3;
        let w_fp = 0.2;

        let grad_python = array![
            [-0.020605005, -0.07924966],
            [0.014705758, 0.04927617],
            [-0.0021341527, -0.057763252],
            [0.012299123, 0.0746348],
            [-0.071347743, -0.0034904587],
            [0.067082018, 0.016592404],
            [0.00008618303, 0.000034395234],
            [-0.00008618303, -0.000034395234],
            [0.000055396682, -0.000011408921],
            [-0.000055396682, 0.000011408921],
            [0.39661729, 0.0],
        ];

        let weights = Weights {
            w_mn,
            w_neighbors,
            w_fp,
        };

        let grad_rust = pacmap_grad(
            y_test.view(),
            pair_neighbors.view(),
            pair_mn.view(),
            pair_fp.view(),
            &weights,
        );

        Zip::from(grad_python.view())
            .and(grad_rust.view())
            .for_each(|&a, &b| {
                assert_abs_diff_eq!(a, b, epsilon = 1e-6);
            });
    }
}
