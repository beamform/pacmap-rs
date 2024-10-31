use crate::distance::array_euclidean_distance;
use crate::{fit_transform, Configuration, Initialization, PairConfiguration};
use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

#[test]
fn test_configuration_builder() {
    let config = Configuration::builder()
        // .embedding_dimensions(3)
        .initialization(Initialization::Random(Some(42)))
        .learning_rate(0.8)
        .num_iters((50, 50, 100))
        .build();

    assert_eq!(config.embedding_dimensions, 2);
    assert_eq!(config.learning_rate, 0.8);
    assert_eq!(config.num_iters, (50, 50, 100));
}

#[test]
fn test_fit_transform() {
    // Create a synthetic dataset with clear structure
    let n_samples = 1000;
    let n_features = 50;
    let embedding_dims = 2;

    // Generate random data with some structure
    let mut x = Array2::random((n_samples, n_features), Uniform::new(-1.0, 1.0));

    // Add some cluster structure
    for i in 0..n_samples {
        let cluster = i / 200; // 5 clusters
        for j in 0..n_features {
            x[[i, j]] += cluster as f32;
        }
    }

    // Test different configurations
    let configs = vec![
        // Default configuration
        Configuration::default(),
        // Custom configuration with PCA initialization
        Configuration {
            embedding_dimensions: embedding_dims,
            initialization: Initialization::Pca,
            mid_near_ratio: 0.5,
            far_pair_ratio: 2.0,
            override_neighbors: Some(15),
            seed: Some(42),
            pair_configuration: PairConfiguration::Generate,
            learning_rate: 1.0,
            num_iters: (100, 100, 250),
            snapshots: Some(vec![100, 200, 300]),
        },
        // Random initialization with seed
        Configuration {
            embedding_dimensions: embedding_dims,
            initialization: Initialization::Random(Some(42)),
            mid_near_ratio: 0.3,
            far_pair_ratio: 1.5,
            override_neighbors: Some(10),
            seed: Some(42),
            pair_configuration: PairConfiguration::Generate,
            learning_rate: 0.8,
            num_iters: (50, 50, 100),
            snapshots: None,
        },
    ];

    for (i, config) in configs.into_iter().enumerate() {
        let result = fit_transform(x.view(), config.clone());
        assert!(result.is_ok(), "Configuration {} failed", i);

        let (embedding, snapshots) = result.unwrap();

        // Check dimensions
        assert_eq!(
            embedding.shape(),
            &[n_samples, embedding_dims],
            "Wrong embedding shape for configuration {}",
            i
        );

        // Check that the embedding is not degenerate
        let mean = embedding.mean_axis(Axis(0)).unwrap();
        let std = embedding.std_axis(Axis(0), 0.0);

        // Embedding should be centered
        assert!(
            mean.iter().all(|&x| x.abs() < 1.0),
            "Embedding not centered for configuration {}",
            i
        );

        // Embedding should have non-zero variance
        assert!(
            std.iter().all(|&x| x > 1e-6),
            "Degenerate embedding for configuration {}",
            i
        );

        // Check cluster preservation (basic test)
        // Points from the same cluster should be closer to each other
        let mut intra_cluster_dist = 0.0;
        let mut inter_cluster_dist = 0.0;
        let mut intra_count = 0;
        let mut inter_count = 0;

        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let dist = array_euclidean_distance(embedding.row(i), embedding.row(j));

                if (i / 200) == (j / 200) {
                    intra_cluster_dist += dist;
                    intra_count += 1;
                } else {
                    inter_cluster_dist += dist;
                    inter_count += 1;
                }
            }
        }

        intra_cluster_dist /= intra_count as f32;
        inter_cluster_dist /= inter_count as f32;

        assert!(
            intra_cluster_dist < inter_cluster_dist,
            "Cluster structure not preserved for configuration {}, intra: {}, inter: {}",
            i,
            intra_cluster_dist,
            inter_cluster_dist
        );

        // Check snapshots if they were requested
        if let Some(snapshot_array) = snapshots {
            if let Some(expected_snapshots) = config.snapshots {
                assert_eq!(
                    snapshot_array.shape(),
                    &[expected_snapshots.len(), n_samples, embedding_dims],
                    "Wrong snapshot shape for configuration {}",
                    i
                );
            }
        }
    }
}
