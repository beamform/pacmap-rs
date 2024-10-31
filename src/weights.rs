//! Manages weights used during the `PaCMAP` optimization process.
//!
//! The weights control how different types of point pairs influence the embedding:
//! - Nearest neighbor pairs preserve local structure
//! - Mid-near pairs preserve medium-range structure
//! - Far pairs prevent collapse by maintaining separation
//!
//! The weights evolve through three phases during optimization:
//! 1. Gradually reduce mid-near weight to allow initial structure formation
//! 2. Balance local and global structure with equal mid-near and neighbor weights
//! 3. Focus on local structure by zeroing mid-near weight

/// Weight parameters applied to each type of point pair during optimization.
#[derive(Debug, Clone, Copy)]
pub struct Weights {
    /// Weight for mid-near pairs, controlling medium-range structure preservation.
    /// Gradually decreases from initial value to 0.0 across optimization phases.
    pub w_mn: f32,

    /// Weight for nearest neighbor pairs, controlling local structure preservation.
    /// Varies between 1.0-3.0 across optimization phases.
    pub w_neighbors: f32,

    /// Weight for far pairs, controlling global structure by preventing collapse.
    /// Remains constant at 1.0 throughout optimization.
    pub w_fp: f32,
}

/// Calculate optimization weights based on the current iteration and phase durations.
///
/// The weights smoothly transition through three phases:
/// 1. Reduce mid-near weight linearly from initial value to 3.0, with fixed neighbor and far weights
/// 2. Fix all weights to balance local and global structure (mid-near=3.0, neighbor=3.0, far=1.0)
/// 3. Zero mid-near weight and reduce neighbor weight to focus on local structure
///
/// # Arguments
/// * `w_mn_init` - Initial weight for mid-near pairs
/// * `itr` - Current iteration number
/// * `phase_1_iters` - Number of iterations in phase 1 (mid-near reduction)
/// * `phase_2_iters` - Number of iterations in phase 2 (balanced weights)
///
/// # Returns
/// A `Weights` struct containing the calculated weights for this iteration
#[allow(clippy::cast_precision_loss)]
pub fn find_weights(
    w_mn_init: f32,
    itr: usize,
    phase_1_iters: usize,
    phase_2_iters: usize,
) -> Weights {
    if itr < phase_1_iters {
        // Phase 1: Linear interpolation of mid-near weight
        let progress = itr as f32 / phase_1_iters as f32;
        Weights {
            w_mn: (1.0 - progress) * w_mn_init + progress * 3.0,
            w_neighbors: 2.0,
            w_fp: 1.0,
        }
    } else if itr < phase_1_iters + phase_2_iters {
        // Phase 2: Fixed balanced weights
        Weights {
            w_mn: 3.0,
            w_neighbors: 3.0,
            w_fp: 1.0,
        }
    } else {
        // Phase 3: Local structure focus
        Weights {
            w_mn: 0.0,
            w_neighbors: 1.0,
            w_fp: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_find_weight() {
        let w_mn_init = 1000.0;

        // Test Phase 1
        let w0 = find_weights(w_mn_init, 0, 100, 100);
        assert_abs_diff_eq!(w0.w_mn, 1000.0);
        assert_abs_diff_eq!(w0.w_neighbors, 2.0);
        assert_abs_diff_eq!(w0.w_fp, 1.0);

        let w50 = find_weights(w_mn_init, 50, 100, 100);
        assert_abs_diff_eq!(w50.w_mn, 501.5);
        assert_abs_diff_eq!(w50.w_neighbors, 2.0);
        assert_abs_diff_eq!(w50.w_fp, 1.0);

        // Test Phase 2
        let w150 = find_weights(w_mn_init, 150, 100, 100);
        assert_abs_diff_eq!(w150.w_mn, 3.0);
        assert_abs_diff_eq!(w150.w_neighbors, 3.0);
        assert_abs_diff_eq!(w150.w_fp, 1.0);

        // Test Phase 3
        let w300 = find_weights(w_mn_init, 300, 100, 100);
        assert_abs_diff_eq!(w300.w_mn, 0.0);
        assert_abs_diff_eq!(w300.w_neighbors, 1.0);
        assert_abs_diff_eq!(w300.w_fp, 1.0);
    }
}
