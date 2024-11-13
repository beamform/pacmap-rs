#![allow(clippy::multiple_crate_versions)]

//! # `PaCMAP`: Pairwise Controlled Manifold Approximation
//!
//! This crate provides a Rust implementation of `PaCMAP` (Pairwise Controlled
//! Manifold Approximation), a dimensionality reduction technique that preserves
//! both local and global structure of high-dimensional data.
//!
//! `PaCMAP` transforms high-dimensional data into a lower-dimensional
//! representation while preserving important relationships between points. This
//! is useful for visualization, analysis, and as preprocessing for other
//! algorithms.
//!
//! ## Key Features
//!
//! `PaCMAP` preserves both local and global structure through three types of
//! point relationships:
//! - Nearest neighbor pairs preserve local structure
//! - Mid-near pairs preserve intermediate structure
//! - Far pairs prevent collapse and maintain separation
//!
//! The implementation provides:
//! - Configurable optimization with adaptive learning rates via Adam
//!   optimization
//! - Phase-based weight schedules to balance local and global preservation
//! - Multiple initialization options including PCA and random seeding
//! - Optional snapshot capture of intermediate states
//!
//! ## Examples
//!
//! Basic usage with default parameters:
//! ```rust,no_run
//! use ndarray::Array2;
//! use pacmap::{Configuration, fit_transform};
//!
//! let data: Array2<f32> = // ... load your high-dimensional data
//! # Array2::zeros((100, 50));
//! let config = Configuration::default();
//! let (embedding, _) = fit_transform(data.view(), config).unwrap();
//! ```
//!
//! Customized embedding:
//! ```rust,no_run
//! use pacmap::{Configuration, Initialization};
//!
//! let config = Configuration::builder()
//!     .embedding_dimensions(3)
//!     .initialization(Initialization::Random(Some(42)))
//!     .learning_rate(0.8)
//!     .num_iters((50, 50, 100))
//!     .mid_near_ratio(0.3)
//!     .far_pair_ratio(2.0)
//!     .build();
//! ```
//!
//! Capturing intermediate states:
//! ```rust,no_run
//! use pacmap::Configuration;
//!
//! let config = Configuration::builder()
//!     .snapshots(vec![100, 200, 300])
//!     .build();
//! ```
//!
//! ## Configuration
//!
//! Core parameters:
//! - `embedding_dimensions`: Output dimensionality (default: 2)
//! - `initialization`: How to initialize coordinates:
//!   - `Pca` - Project data using PCA (default)
//!   - `Value(array)` - Use provided coordinates
//!   - `Random(seed)` - Random initialization with optional seed
//! - `learning_rate`: Learning rate for Adam optimizer (default: 1.0)
//! - `num_iters`: Iteration counts for three optimization phases (default:
//!   (100, 100, 250))
//! - `snapshots`: Optional vector of iterations at which to save embedding
//!   states
//! - `approx_threshold`: Number of points above which approximate neighbor
//!   search is used
//!
//! Pair sampling parameters:
//! - `mid_near_ratio`: Ratio of mid-near to nearest neighbor pairs (default:
//!   0.5)
//! - `far_pair_ratio`: Ratio of far to nearest neighbor pairs (default: 2.0)
//! - `override_neighbors`: Optional fixed neighbor count override
//! - `seed`: Optional random seed for reproducible sampling
//!
//! ## Implementation Notes
//!
//! - Supports both exact and approximate nearest neighbor search
//! - Uses Euclidean distances for pair relationships
//! - Leverages ndarray for efficient matrix operations
//! - Employs parallel iterators via rayon for performance
//! - Provides detailed error handling with custom error types
//!
//! ## References
//!
//! [Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization](https://jmlr.org/papers/v22/20-1061.html).
//! Wang, Y., Huang, H., Rudin, C., & Shaposhnik, Y. (2021).
//! Journal of Machine Learning Research, 22(201), 1-73.
//!
//! Original Python implementation: <https://github.com/YingfanWang/PaCMAP>

// Submodule imports
mod adam;
mod distance;
mod gradient;
pub mod knn;
mod neighbors;
mod sampling;
mod weights;

#[cfg(test)]
mod tests;

use bon::Builder;
use ndarray::{s, Array1, Array2, Array3, ArrayView2, Axis, Zip};
use ndarray_rand::rand_distr::{Normal, NormalError};
use ndarray_rand::RandomExt;
use petal_decomposition::{DecompositionError, Pca, RandomizedPca, RandomizedPcaBuilder};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_pcg::Mcg128Xsl64;
use std::cmp::min;
use std::time::Instant;
use thiserror::Error;
use tracing::{debug, warn};

use crate::adam::update_embedding_adam;
use crate::gradient::pacmap_grad;
use crate::knn::KnnError;
use crate::neighbors::{generate_pair_no_neighbors, generate_pairs};
use crate::weights::find_weights;

/// Configuration options for the `PaCMAP` embedding process.
///
/// Controls initialization, sampling ratios, optimization parameters, and
/// snapshot capture.
#[derive(Builder, Clone, Debug)]
pub struct Configuration {
    /// Number of dimensions in the output embedding space, typically 2 or 3
    #[builder(default = 2)]
    pub embedding_dimensions: usize,

    /// Method for initializing the embedding coordinates
    #[builder(default)]
    pub initialization: Initialization,

    /// Ratio of mid-near pairs to nearest neighbor pairs
    #[builder(default = 0.5)]
    pub mid_near_ratio: f32,

    /// Ratio of far pairs to nearest neighbor pairs
    #[builder(default = 2.0)]
    pub far_pair_ratio: f32,

    /// Optional fixed neighbor count override
    pub override_neighbors: Option<usize>,

    /// Optional random seed for reproducibility
    pub seed: Option<u64>,

    /// Controls how point pairs are sampled or provided
    #[builder(default)]
    pub pair_configuration: PairConfiguration,

    /// Learning rate for the Adam optimizer
    #[builder(default = 1.0)]
    pub learning_rate: f32,

    /// Number of iterations for attraction, local structure, and global
    /// structure phases
    #[builder(default = (100, 100, 250))]
    pub num_iters: (usize, usize, usize),

    /// Optional iteration indices at which to save embedding states
    pub snapshots: Option<Vec<usize>>,

    /// Number of points above which approximate neighbor search is used
    #[builder(default = 8_000)]
    pub approx_threshold: usize,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            embedding_dimensions: 2,
            initialization: Initialization::default(),
            mid_near_ratio: 0.5,
            far_pair_ratio: 2.0,
            override_neighbors: None,
            seed: None,
            pair_configuration: PairConfiguration::default(),
            learning_rate: 1.0,
            num_iters: (100, 100, 250),
            snapshots: None,
            approx_threshold: 8_000,
        }
    }
}

/// Methods for initializing the embedding coordinates.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub enum Initialization {
    /// Project data using PCA
    #[default]
    Pca,

    /// Use provided coordinate values
    Value(Array2<f32>),

    /// Initialize randomly with optional seed
    Random(Option<u64>),
}

/// Strategy for sampling pairs during optimization.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub enum PairConfiguration {
    /// Sample all pairs from scratch based on distances.
    /// Most computationally intensive but requires no prior information.
    #[default]
    Generate,

    /// Use provided nearest neighbors and generate mid-near and far pairs.
    /// Useful when nearest neighbors are pre-computed.
    NeighborsProvided {
        /// Matrix of shape (n * k, 2) containing nearest neighbor pair indices
        pair_neighbors: Array2<u32>,
    },

    /// Use all provided pair indices without additional sampling.
    /// Most efficient when all required pairs are pre-computed.
    AllProvided {
        /// Nearest neighbor pair indices
        pair_neighbors: Array2<u32>,
        /// Mid-near pair indices
        pair_mn: Array2<u32>,
        /// Far pair indices
        pair_fp: Array2<u32>,
    },
}

/// Reduces dimensionality of input data using `PaCMAP`.
///
/// # Arguments
/// * `x` - Input data matrix where each row is a sample
/// * `config` - Configuration options controlling the embedding process
///
/// # Returns
/// A tuple containing:
/// * Final embedding coordinates as a matrix
/// * Optional array of intermediate embedding states if snapshots were
///   requested
///
/// # Errors
/// * `PaCMapError::SampleSize` - Input has <= 1 samples
/// * `PaCMapError::InvalidNeighborCount` - Calculated neighbor count is invalid
/// * `PaCMapError::InvalidFarPointCount` - Calculated far point count is
///   invalid
/// * `PaCMapError::InvalidNearestNeighborShape` - Provided neighbor pairs have
///   wrong shape
/// * `PaCMapError::EmptyArrayMean` - Mean cannot be calculated for
///   preprocessing
/// * `PaCMapError::EmptyArrayMinMax` - Min/max cannot be found during
///   preprocessing
/// * `PaCMapError::Pca` - PCA decomposition fails
/// * `PaCMapError::Normal` - Random initialization fails
pub fn fit_transform(
    x: ArrayView2<f32>,
    config: Configuration,
) -> Result<(Array2<f32>, Option<Array3<f32>>), PaCMapError> {
    // Input validation
    let (n, dim) = x.dim();
    if n <= 1 {
        return Err(PaCMapError::SampleSize);
    }

    // Preprocess input data with optional dimensionality reduction
    let PreprocessingResult {
        x,
        pca_solution,
        transform,
        ..
    } = preprocess_x(
        x,
        matches!(config.initialization, Initialization::Pca),
        dim,
        config.embedding_dimensions,
        config.seed,
    )?;

    // Initialize embedding coordinates
    let embedding_init = if pca_solution {
        YInit::Preprocessed
    } else {
        match config.initialization {
            Initialization::Pca => YInit::DimensionalReduction(transform),
            Initialization::Value(value) => YInit::Value(value),
            Initialization::Random(maybe_seed) => YInit::Random(maybe_seed),
        }
    };

    // Calculate pair sampling parameters
    let pair_decision = decide_num_pairs(
        n,
        config.override_neighbors,
        config.mid_near_ratio,
        config.far_pair_ratio,
    )?;

    if n - 1 < pair_decision.n_neighbors {
        warn!("Sample size is smaller than n_neighbors. n_neighbors will be reduced.");
    }

    // Sample point pairs for optimization
    let pairs = sample_pairs(
        x.view(),
        pair_decision.n_neighbors,
        pair_decision.n_mn,
        pair_decision.n_fp,
        config.pair_configuration,
        config.seed,
        config.approx_threshold,
    )?;

    // Run optimization to compute embedding
    pacmap(
        x.view(),
        config.embedding_dimensions,
        pairs.pair_neighbors.view(),
        pairs.pair_mn.view(),
        pairs.pair_fp.view(),
        config.learning_rate,
        config.num_iters,
        embedding_init,
        config.snapshots.as_deref(),
    )
}

/// Results from preprocessing input data.
#[allow(dead_code)]
struct PreprocessingResult {
    /// Preprocessed data matrix
    x: Array2<f32>,

    /// Whether PCA dimensionality reduction was applied
    pca_solution: bool,

    /// Fitted dimensionality reduction transform
    transform: Transform,

    /// Minimum x value
    x_min: f32,

    /// Maximum x value
    x_max: f32,

    /// Mean of x along axis 0
    x_mean: Array1<f32>,
}

/// Types of dimensionality reduction transforms used for initialization.
#[non_exhaustive]
enum Transform {
    /// Standard PCA
    Pca(Pca<f32>),

    /// Randomized PCA without fixed seed for efficiency on large datasets
    RandomizedPca(RandomizedPca<f32, Mcg128Xsl64>),

    /// Randomized PCA with fixed seed for reproducibility
    SeededPca(RandomizedPca<f32, SmallRng>),
}

impl Transform {
    /// Applies the transform to new data.
    ///
    /// # Arguments
    /// * `x` - Input data to transform
    ///
    /// # Errors
    /// * `DecompositionError` if transform fails
    pub fn transform(&self, x: ArrayView2<f32>) -> Result<Array2<f32>, DecompositionError> {
        match self {
            Transform::Pca(pca) => pca.transform(&x),
            Transform::RandomizedPca(pca) => pca.transform(&x),
            Transform::SeededPca(pca) => pca.transform(&x),
        }
    }
}

/// Preprocesses input data through normalization and optional dimensionality
/// reduction.
///
/// For high dimensional data (>100 dimensions), optionally applies PCA to
/// reduce to 100 dimensions. Otherwise normalizes the data by centering and
/// scaling.
///
/// # Arguments
/// * `x` - Input data matrix
/// * `apply_pca` - Whether to apply PCA dimensionality reduction
/// * `high_dim` - Original data dimensionality
/// * `low_dim` - Target dimensionality after reduction
/// * `maybe_seed` - Optional random seed for reproducibility
///
/// # Returns
/// A `PreprocessingResult` containing the processed data and transform
///
/// # Errors
/// * `PaCMapError::EmptyArrayMean` if mean cannot be calculated
/// * `PaCMapError::EmptyArrayMinMax` if min/max cannot be found
/// * `PaCMapError::Pca` if PCA decomposition fails
fn preprocess_x(
    x: ArrayView2<f32>,
    apply_pca: bool,
    high_dim: usize,
    low_dim: usize,
    maybe_seed: Option<u64>,
) -> Result<PreprocessingResult, PaCMapError> {
    let mut pca_solution = false;
    let mut x_out: Array2<f32>;
    let x_mean: Array1<f32>;
    let x_min: f32;
    let x_max: f32;
    let transform: Transform;

    if high_dim > 100 && apply_pca {
        let n_components = min(100, x.nrows());
        // Compute the mean of x along axis 0
        x_mean = x.mean_axis(Axis(0)).ok_or(PaCMapError::EmptyArrayMean)?;

        // Initialize PCA and transform
        match maybe_seed {
            None => {
                let mut pca = RandomizedPca::new(n_components);
                x_out = pca.fit_transform(&x)?;
                transform = Transform::RandomizedPca(pca);
            }
            Some(seed) => {
                let mut pca =
                    RandomizedPcaBuilder::with_rng(SmallRng::seed_from_u64(seed), n_components)
                        .build();

                x_out = pca.fit_transform(&x)?;
                transform = Transform::SeededPca(pca);
            }
        };

        pca_solution = true;

        // Set x_min and x_max to zero
        x_min = 0.0;
        x_max = 0.0;

        debug!("Applied PCA, the dimensionality becomes {n_components}");
    } else {
        x_out = x.to_owned();

        // Compute x_min and x_max
        x_min = *x_out
            .iter()
            .min_by(|&a, &b| f32::total_cmp(a, b))
            .ok_or(PaCMapError::EmptyArrayMinMax)?;

        x_max = *x_out
            .iter()
            .max_by(|&a, &b| f32::total_cmp(a, b))
            .ok_or(PaCMapError::EmptyArrayMinMax)?;

        // Subtract x_min from x
        x_out.mapv_inplace(|val| val - x_min);

        // Divide by x_max (not the range) to replicate the Python function
        x_out.mapv_inplace(|val| val / x_max);

        // Compute x_mean
        x_mean = x_out
            .mean_axis(Axis(0))
            .ok_or(PaCMapError::EmptyArrayMean)?;

        // Subtract x_mean from x
        x_out -= &x_mean;

        // Proceed with PCA
        let n_components = min(x_out.nrows(), low_dim);
        let mut pca = Pca::new(n_components);
        pca.fit(&x_out)?;
        transform = Transform::Pca(pca);

        debug!("x is normalized");
    };

    Ok(PreprocessingResult {
        x: x_out,
        pca_solution,
        x_min,
        x_max,
        x_mean,
        transform,
    })
}

/// Parameters controlling pair sampling based on dataset size.
struct PairDecision {
    /// Number of nearest neighbors per point
    n_neighbors: usize,

    /// Number of mid-near pairs per point
    n_mn: usize,

    /// Number of far pairs per point
    n_fp: usize,
}

/// Calculates number of pairs to use based on dataset size and configuration.
///
/// Automatically scales neighbor counts with dataset size unless overridden.
///
/// # Arguments
/// * `n` - Number of samples in dataset
/// * `n_neighbors` - Optional fixed neighbor count override
/// * `mn_ratio` - Ratio of mid-near pairs to nearest neighbors
/// * `fp_ratio` - Ratio of far pairs to nearest neighbors
///
/// # Returns
/// A `PairDecision` containing the calculated pair counts
///
/// # Errors
/// * `PaCMapError::InvalidNeighborCount` - Calculated neighbor count is less
///   than 1
/// * `PaCMapError::InvalidFarPointCount` - Calculated far pair count is less
///   than 1
#[allow(clippy::cast_precision_loss)]
fn decide_num_pairs(
    n: usize,
    n_neighbors: Option<usize>,
    mn_ratio: f32,
    fp_ratio: f32,
) -> Result<PairDecision, PaCMapError> {
    // Scale neighbors with data size or use override
    let n_neighbors = n_neighbors.unwrap_or_else(|| {
        if n <= 10000 {
            10
        } else {
            (10.0 + 15.0 * ((n as f32).log10() - 4.0)).round() as usize
        }
    });

    let n_mn = (n_neighbors as f32 * mn_ratio).round() as usize;
    let n_fp = (n_neighbors as f32 * fp_ratio).round() as usize;

    // Validate calculated pair counts
    if n_neighbors < 1 {
        return Err(PaCMapError::InvalidNeighborCount);
    }

    if n_fp < 1 {
        return Err(PaCMapError::InvalidFarPointCount);
    }

    Ok(PairDecision {
        n_neighbors,
        n_mn,
        n_fp,
    })
}

/// Collection of sampled point pairs used during optimization.
struct Pairs {
    /// Nearest neighbor pairs preserving local structure
    pair_neighbors: Array2<u32>,

    /// Mid-near pairs preserving medium-range structure
    pair_mn: Array2<u32>,

    /// Far pairs preventing collapse
    pair_fp: Array2<u32>,
}

/// Samples point pairs according to the `PaCMAP` strategy.
///
/// # Arguments
/// * `x` - Input data matrix
/// * `n_neighbors` - Number of nearest neighbors per point
/// * `n_mn` - Number of mid-near pairs per point
/// * `n_fp` - Number of far pairs per point
/// * `pair_config` - Configuration for pair sampling
/// * `random_state` - Optional random seed
/// * `approx_threshold` - Number of points above which approximate search is
///   used
///
/// # Returns
/// A `Pairs` struct containing the sampled pair indices
///
/// # Errors
/// * `PaCMapError::InvalidNearestNeighborShape` if provided pairs have invalid
///   shape
fn sample_pairs(
    x: ArrayView2<f32>,
    n_neighbors: usize,
    n_mn: usize,
    n_fp: usize,
    pair_config: PairConfiguration,
    random_state: Option<u64>,
    approx_threshold: usize,
) -> Result<Pairs, PaCMapError> {
    debug!("Finding pairs");
    match pair_config {
        // Generate all pairs from scratch
        PairConfiguration::Generate => Ok(generate_pairs(
            x,
            n_neighbors,
            n_mn,
            n_fp,
            random_state,
            approx_threshold,
        )?),

        // Use provided nearest neighbors, generate remaining pairs
        PairConfiguration::NeighborsProvided { pair_neighbors } => {
            let expected_shape = [x.nrows() * n_neighbors, 2];
            if pair_neighbors.shape() != expected_shape {
                return Err(PaCMapError::InvalidNearestNeighborShape {
                    expected: expected_shape,
                    actual: pair_neighbors.shape().to_vec(),
                });
            }

            debug!("Using provided nearest neighbor pairs.");
            let (pair_mn, pair_fp) = generate_pair_no_neighbors(
                x,
                n_neighbors,
                n_mn,
                n_fp,
                pair_neighbors.view(),
                random_state,
            );

            debug!("Additional pairs sampled successfully.");
            Ok(Pairs {
                pair_neighbors,
                pair_mn,
                pair_fp,
            })
        }

        // Use all provided pairs without additional sampling
        PairConfiguration::AllProvided {
            pair_neighbors,
            pair_mn,
            pair_fp,
        } => {
            debug!("Using all provided pairs.");
            Ok(Pairs {
                pair_neighbors,
                pair_mn,
                pair_fp,
            })
        }
    }
}

/// Methods for initializing embedding coordinates.
enum YInit {
    /// Use provided coordinate values
    Value(Array2<f32>),

    /// Use preprocessed data directly
    Preprocessed,

    /// Apply dimensionality reduction transform
    DimensionalReduction(Transform),

    /// Initialize randomly with optional seed
    Random(Option<u64>),
}

/// Core `PaCMAP` optimization function.
///
/// Iteratively updates embedding coordinates through gradient descent to
/// preserve data structure. Uses phase-based weight schedules to balance local
/// and global structure preservation.
///
/// # Arguments
/// * `x` - Input data matrix
/// * `n_dims` - Desired output dimensionality
/// * `pair_neighbors` - Nearest neighbor pairs
/// * `pair_mn` - Mid-near pairs
/// * `pair_fp` - Far pairs
/// * `lr` - Learning rate
/// * `num_iters` - Number of iterations for each optimization phase
/// * `y_initialization` - Method for initializing coordinates
/// * `inter_snapshots` - Optional indices for saving intermediate states
///
/// # Returns
/// A tuple containing:
/// * Final embedding coordinates
/// * Optional intermediate states if snapshots were requested
///
/// # Errors
/// * `PaCMapError::EmptyArrayMean` if mean cannot be calculated
/// * `PaCMapError::Normal` if random initialization fails
/// * `PaCMapError::Pca` if PCA transform fails
#[allow(clippy::too_many_arguments)]
fn pacmap<'a>(
    x: ArrayView2<f32>,
    n_dims: usize,
    pair_neighbors: ArrayView2<'a, u32>,
    pair_mn: ArrayView2<'a, u32>,
    pair_fp: ArrayView2<'a, u32>,
    lr: f32,
    num_iters: (usize, usize, usize),
    y_initialization: YInit,
    inter_snapshots: Option<&[usize]>,
) -> Result<(Array2<f32>, Option<Array3<f32>>), PaCMapError> {
    let start_time = Instant::now();
    let n = x.nrows();
    let mut inter_snapshots = Snapshots::from(n_dims, n, inter_snapshots);

    // Initialize embedding coordinates based on specified method
    let mut y: Array2<f32> = match y_initialization {
        YInit::Value(mut y) => {
            let mean = y.mean_axis(Axis(0)).ok_or(PaCMapError::EmptyArrayMean)?;
            let std = y.std_axis(Axis(0), 0.0);

            // Center and scale provided coordinates
            Zip::from(&mut y)
                .and_broadcast(&mean)
                .and_broadcast(&std)
                .par_for_each(|y_elem, &mean_elem, &std| {
                    *y_elem = (*y_elem - mean_elem) * 0.0001 / std;
                });

            y
        }
        YInit::Preprocessed => x.slice(s![.., ..n_dims]).to_owned() * 0.01,
        YInit::DimensionalReduction(transform) => transform.transform(x)? * 0.01,
        YInit::Random(maybe_seed) => {
            let normal = Normal::new(0.0, 1.0)?;

            match maybe_seed {
                None => Array2::random((n, n_dims), normal),
                Some(seed) => {
                    Array2::random_using((n, n_dims), normal, &mut SmallRng::seed_from_u64(seed))
                }
            }
        }
    };

    // Initialize optimizer parameters
    let w_mn_init = 1000.0;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let mut m = Array2::zeros(y.dim());
    let mut v = Array2::zeros(y.dim());

    // Store initial state if snapshots requested
    if let Some(ref mut snapshots) = inter_snapshots {
        snapshots.states.slice_mut(s![0_usize, .., ..]).assign(&y);
    }

    debug!(
        "Pair shapes: neighbors {:?}, MN {:?}, FP {:?}",
        pair_neighbors.dim(),
        pair_mn.dim(),
        pair_fp.dim()
    );

    let num_iters_total = num_iters.0 + num_iters.1 + num_iters.2;

    // Main optimization loop
    for itr in 0..num_iters_total {
        // Update weights based on phase
        let weights = find_weights(w_mn_init, itr, num_iters.0, num_iters.1);
        let grad = pacmap_grad(y.view(), pair_neighbors, pair_mn, pair_fp, &weights);

        let c = grad[(n, 0)];
        if itr == 0 {
            debug!("Initial Loss: {}", c);
        }

        // Update embedding with Adam optimizer
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

        if (itr + 1) % 10 == 0 {
            debug!("Iteration: {:4}, Loss: {}", itr + 1, c);
        }

        // Store intermediate state if requested
        let Some(ref mut snapshots) = &mut inter_snapshots else {
            continue;
        };

        if let Some(index) = snapshots.indices.iter().position(|&x| x == itr + 1) {
            snapshots.states.slice_mut(s![index, .., ..]).assign(&y);
        }
    }

    let elapsed = start_time.elapsed();
    debug!("Elapsed time: {:.2?}", elapsed);

    Ok((y, inter_snapshots.map(|s| s.states)))
}

/// Manages intermediate embedding states during optimization.
struct Snapshots<'a> {
    /// Stored embedding states
    states: Array3<f32>,

    /// Indices at which to take snapshots
    indices: &'a [usize],
}

impl<'a> Snapshots<'a> {
    /// Creates new snapshot manager if indices are provided.
    ///
    /// # Arguments
    /// * `n_dims` - Dimensionality of embedding
    /// * `n` - Number of samples
    /// * `maybe_snapshots` - Optional snapshot indices
    fn from(n_dims: usize, n: usize, maybe_snapshots: Option<&'a [usize]>) -> Option<Self> {
        let snapshots = maybe_snapshots?;
        Some(Self {
            states: Array3::zeros((snapshots.len(), n, n_dims)),
            indices: snapshots,
        })
    }
}

/// Errors that can occur during `PaCMAP` embedding.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum PaCMapError {
    /// Input data has 1 or fewer samples
    #[error("Sample size must be larger than one")]
    SampleSize,

    /// Provided nearest neighbor pairs have incorrect dimensions
    #[error("Invalid shape for nearest neighbor pairs. Expected {expected:?}, got {actual:?}")]
    InvalidNearestNeighborShape {
        /// Expected shape: [`n_samples` * `n_neighbors`, 2]
        expected: [usize; 2],
        /// Actual shape of provided matrix
        actual: Vec<usize>,
    },

    /// Mean calculation failed due to empty array
    #[error("Failed to calculate mean axis: the array is empty")]
    EmptyArrayMean,

    /// Normal distribution creation failed
    #[error(transparent)]
    Normal(#[from] NormalError),

    /// Calculated number of nearest neighbors is less than 1
    #[error("The number of nearest neighbors can't be less than 1")]
    InvalidNeighborCount,

    /// Calculated number of far points is less than 1
    #[error("The number of far points can't be less than 1")]
    InvalidFarPointCount,

    /// Min/max computation failed due to empty array
    #[error("Failed to compute min or max of X: the array is empty")]
    EmptyArrayMinMax,

    /// Data cannot be normalized due to zero range
    #[error("The range of X is zero (max - min = 0), cannot normalize")]
    ZeroRange,

    /// PCA decomposition failed
    #[error(transparent)]
    Pca(#[from] DecompositionError),

    /// K-nearest neighbors error
    #[error(transparent)]
    Neighbors(#[from] KnnError),
}
