//! Adam optimization updates for `PaCMAP` embeddings
//!
//! This module implements gradient updates using the Adam optimizer during
//! `PaCMAP` dimensionality reduction. The Adam optimizer adapts learning rates
//! per parameter using moment estimates.

use ndarray::{s, ArrayView2, ArrayViewMut2, Zip};

/// Updates embedding coordinates using adaptive moment estimation (Adam).
///
/// Performs a single Adam optimization step to update the low-dimensional
/// embedding based on computed gradients. Uses exponential moving averages of
/// gradients and squared gradients to adapt learning rates per parameter.
///
/// # Arguments
/// * `y` - Current embedding coordinates to update
/// * `grad` - Gradient for this iteration
/// * `m` - First moment estimate (gradient moving average)
/// * `v` - Second moment estimate (squared gradient moving average)
/// * `beta1` - First moment decay rate (typically 0.9)
/// * `beta2` - Second moment decay rate (typically 0.999)
/// * `lr` - Base learning rate
/// * `itr` - Current iteration number (0-based)
///
/// # Implementation Notes
/// - Applies bias correction to moment estimates based on iteration number
/// - Updates moment estimates using exponential moving averages
/// - Updates parameters with Adam rule: y -= lr * m / (sqrt(v) + eps)
/// - Uses parallel iteration for efficiency
///
/// # Panics
/// * If moment tensors m and v have different shapes than y
/// * If grad slice has insufficient rows for y
#[allow(clippy::too_many_arguments)]
pub fn update_embedding_adam(
    y: ArrayViewMut2<f32>,
    grad: ArrayView2<f32>,
    m: ArrayViewMut2<f32>,
    v: ArrayViewMut2<f32>,
    beta1: f32,
    beta2: f32,
    lr: f32,
    itr: usize,
) {
    // Compute bias-corrected learning rate
    let itr = (itr + 1) as i32;
    let lr_t = lr * (1.0 - beta2.powi(itr)).sqrt() / (1.0 - beta1.powi(itr));
    let grad = grad.slice(s![..y.nrows(), ..]);

    // Update moment estimates and parameters in parallel
    Zip::from(y)
        .and(grad)
        .and(m)
        .and(v)
        .par_for_each(|y, &grad, m, v| {
            *m += (1.0 - beta1) * (grad - *m);
            *v += (1.0 - beta2) * (grad.powi(2) - *v);
            *y -= lr_t * *m / (v.sqrt() + 1e-7);
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2};

    #[test]
    fn test_update_embedding_adam() {
        // Define test inputs
        let mut y = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let grad = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
        let mut m = Array2::zeros((3, 2));
        let mut v = Array2::zeros((3, 2));
        let beta1 = 0.9;
        let beta2 = 0.999;
        let lr = 0.001;
        let itr = 0;

        // Run update step
        update_embedding_adam(
            y.view_mut(),
            grad.view(),
            m.view_mut(),
            v.view_mut(),
            beta1,
            beta2,
            lr,
            itr,
        );

        // Define expected outputs
        let y_expected = array![[0.999, 1.9990001], [2.999, 3.999], [4.999, 5.999]];
        let m_expected = array![
            [0.01, 0.02000001],
            [0.03000001, 0.04000001],
            [0.05000001, 0.06000002]
        ];
        let v_expected = array![
            [9.9998715e-06, 3.9999486e-05],
            [8.9998844e-05, 1.5999794e-04],
            [2.4999678e-04, 3.5999538e-04]
        ];

        // Verify outputs match expected values
        Zip::from(&y).and(&y_expected).for_each(|&y_val, &y_exp| {
            assert_abs_diff_eq!(y_val, y_exp, epsilon = 1e-6);
        });

        Zip::from(&m).and(&m_expected).for_each(|&m_val, &m_exp| {
            assert_abs_diff_eq!(m_val, m_exp, epsilon = 1e-6);
        });

        Zip::from(&v).and(&v_expected).for_each(|&v_val, &v_exp| {
            assert_abs_diff_eq!(v_val, v_exp, epsilon = 1e-6);
        });
    }
}
