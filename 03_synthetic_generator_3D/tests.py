"""
Unit tests for the 3D Synthetic Dataset Generator.

Run with: python tests.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PriorConfig3D, DatasetConfig3D
from temporal_dag_builder import TemporalDAGBuilder, TemporalDAG
from row_generator_3d import RowGenerator3D, create_transformations_3d
from feature_selector_3d import FeatureSelector3D, TableBuilder3D
from generator import SyntheticDatasetGenerator3D, SyntheticDataset3D


class TestConfig3D:
    """Tests for 3D configuration classes."""
    
    def test_prior_config_defaults(self):
        """Test that PriorConfig3D has valid defaults."""
        prior = PriorConfig3D()
        
        assert prior.n_samples_range[0] < prior.n_samples_range[1]
        assert prior.n_features_range[0] <= prior.n_features_range[1]
        assert prior.n_timesteps_range[0] < prior.n_timesteps_range[1]
        assert 0 <= prior.prob_univariate <= 1
        assert prior.max_classes >= 2
    
    def test_sample_hyperparams(self):
        """Test sampling dataset config from prior."""
        prior = PriorConfig3D()
        rng = np.random.default_rng(42)
        
        config = prior.sample_hyperparams(rng)
        
        assert isinstance(config, DatasetConfig3D)
        assert config.n_samples >= prior.n_samples_range[0]
        assert config.n_features >= 1
        assert config.n_timesteps >= prior.n_timesteps_range[0]
        assert len(config.temporal_connections) >= 0
    
    def test_temporal_connections_sampled(self):
        """Test that temporal connections are properly sampled."""
        prior = PriorConfig3D()
        rng = np.random.default_rng(42)
        
        config = prior.sample_hyperparams(rng)
        
        valid_types = [
            'self', 'cross', 'many_to_one', 'one_to_many',
            'broadcast_multiskip', 'conditional_lag', 'conditional_dest'
        ]
        for conn in config.temporal_connections:
            assert conn.skip >= 1
            assert len(conn.source_nodes) >= 1
            assert len(conn.target_nodes) >= 1
            assert conn.connection_type in valid_types, f"Invalid type: {conn.connection_type}"
    
    def test_config_to_dict(self):
        """Test serialization of DatasetConfig3D."""
        prior = PriorConfig3D()
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        d = config.to_dict()
        
        assert 'n_samples' in d
        assert 'n_features' in d
        assert 'n_timesteps' in d
        assert 'feature_window_start' in d
        assert 'target_timestep' in d


class TestTemporalDAGBuilder:
    """Tests for temporal DAG construction."""
    
    def test_basic_temporal_dag(self):
        """Test basic temporal DAG construction."""
        prior = PriorConfig3D(
            n_nodes_range=(5, 10),
            n_timesteps_range=(10, 20)
        )
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        builder = TemporalDAGBuilder(config, rng)
        dag = builder.build()
        
        assert isinstance(dag, TemporalDAG)
        assert dag.n_timesteps == config.n_timesteps
        assert dag.n_base_nodes > 0
        assert len(dag.nodes) == dag.n_timesteps * dag.n_base_nodes
    
    def test_temporal_dag_acyclicity(self):
        """Test that temporal DAG has no cycles."""
        prior = PriorConfig3D(n_nodes_range=(5, 8), n_timesteps_range=(5, 10))
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        builder = TemporalDAGBuilder(config, rng)
        dag = builder.build()
        
        # Topological order should exist
        assert len(dag.topological_order) == len(dag.nodes)
        
        # All temporal edges should go forward in time
        for from_t, to_t, _, _ in dag.temporal_edges:
            assert from_t < to_t
    
    def test_global_id_conversion(self):
        """Test global ID conversion functions."""
        prior = PriorConfig3D(n_nodes_range=(5, 5), n_timesteps_range=(10, 10))
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        builder = TemporalDAGBuilder(config, rng)
        dag = builder.build()
        
        # Test round-trip conversion
        for t in range(dag.n_timesteps):
            for base_id in range(dag.n_base_nodes):
                global_id = dag.get_global_id(t, base_id)
                t2, base2 = dag.get_timestep_and_base(global_id)
                assert t == t2
                assert base_id == base2


class TestRowGenerator3D:
    """Tests for 3D row generation."""
    
    def test_basic_generation(self):
        """Test basic row generation."""
        prior = PriorConfig3D(
            n_samples_range=(50, 50),
            n_nodes_range=(5, 8),
            n_timesteps_range=(10, 15)
        )
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        builder = TemporalDAGBuilder(config, rng)
        dag = builder.build()
        
        spatial_t, temporal_t = create_transformations_3d(dag, config, rng)
        generator = RowGenerator3D(config, dag, spatial_t, temporal_t, rng)
        
        propagated = generator.generate(n_samples=50)
        
        assert propagated.n_samples == 50
        assert propagated.n_timesteps == config.n_timesteps
        assert len(propagated.values) == len(dag.nodes)
    
    def test_time_series_extraction(self):
        """Test extracting time series from propagated values."""
        prior = PriorConfig3D(
            n_samples_range=(30, 30),
            n_nodes_range=(5, 5),
            n_timesteps_range=(20, 20)
        )
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        builder = TemporalDAGBuilder(config, rng)
        dag = builder.build()
        
        spatial_t, temporal_t = create_transformations_3d(dag, config, rng)
        generator = RowGenerator3D(config, dag, spatial_t, temporal_t, rng)
        
        propagated = generator.generate(n_samples=30)
        
        # Extract time series for first node
        series = propagated.get_time_series(0, 0, 10)
        assert series.shape == (30, 10)
        
        all_series = propagated.get_all_timesteps(0)
        assert all_series.shape == (30, 20)


class TestFeatureSelector3D:
    """Tests for 3D feature selection."""
    
    def test_basic_selection(self):
        """Test basic feature selection."""
        prior = PriorConfig3D(
            n_features_range=(3, 5),
            n_nodes_range=(10, 15),
            n_timesteps_range=(30, 50)
        )
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        builder = TemporalDAGBuilder(config, rng)
        dag = builder.build()
        
        selector = FeatureSelector3D(config, dag, rng)
        selection = selector.select()
        
        assert len(selection.feature_base_nodes) <= config.n_features
        assert selection.target_base_node not in selection.feature_base_nodes
        assert selection.feature_window_start < selection.feature_window_end
        assert selection.target_position in ['before', 'within', 'after']
    
    def test_target_position_valid(self):
        """Test that target position is valid relative to feature window."""
        prior = PriorConfig3D()
        rng = np.random.default_rng(42)
        
        for _ in range(10):  # Test multiple configs
            config = prior.sample_hyperparams(rng)
            builder = TemporalDAGBuilder(config, rng)
            dag = builder.build()
            
            selector = FeatureSelector3D(config, dag, rng)
            selection = selector.select()
            
            if selection.target_position == 'before':
                assert selection.target_timestep <= selection.feature_window_start
            elif selection.target_position == 'within':
                assert selection.feature_window_start <= selection.target_timestep < selection.feature_window_end
            else:  # after
                # Target should be at or after feature_window_end, but capped at n_timesteps-1
                # In edge cases where feature_window_end == n_timesteps, target may equal it
                assert selection.target_timestep >= min(selection.feature_window_end, config.n_timesteps - 1)


class TestTableBuilder3D:
    """Tests for building 3D tables."""
    
    def test_build_table(self):
        """Test building X and y from propagated values."""
        prior = PriorConfig3D(
            n_samples_range=(50, 50),
            n_features_range=(3, 3),
            n_timesteps_range=(30, 30),
            n_nodes_range=(10, 10)
        )
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        builder = TemporalDAGBuilder(config, rng)
        dag = builder.build()
        
        spatial_t, temporal_t = create_transformations_3d(dag, config, rng)
        row_gen = RowGenerator3D(config, dag, spatial_t, temporal_t, rng)
        propagated = row_gen.generate()
        
        selector = FeatureSelector3D(config, dag, rng)
        selection = selector.select()
        
        table_builder = TableBuilder3D(selection, config, rng)
        X, y = table_builder.build(propagated)
        
        expected_timesteps = selection.feature_window_end - selection.feature_window_start
        assert X.shape == (50, len(selection.feature_base_nodes), expected_timesteps)
        assert y.shape == (50,)
        assert len(np.unique(y)) <= config.n_classes


class TestSyntheticDatasetGenerator3D:
    """Tests for the main 3D generator."""
    
    def test_basic_generation(self):
        """Test basic dataset generation."""
        generator = SyntheticDatasetGenerator3D(seed=42)
        dataset = generator.generate()
        
        assert isinstance(dataset, SyntheticDataset3D)
        assert dataset.X.ndim == 3
        assert dataset.y.ndim == 1
        assert dataset.X.shape[0] == dataset.y.shape[0]
    
    def test_shape_properties(self):
        """Test shape properties."""
        generator = SyntheticDatasetGenerator3D(seed=42)
        dataset = generator.generate()
        
        assert dataset.shape == dataset.X.shape
        assert dataset.n_samples == dataset.X.shape[0]
        assert dataset.n_features == dataset.X.shape[1]
        assert dataset.n_timesteps == dataset.X.shape[2]
    
    def test_generate_many(self):
        """Test generating multiple datasets."""
        generator = SyntheticDatasetGenerator3D(seed=42)
        datasets = list(generator.generate_many(5))
        
        assert len(datasets) == 5
        for ds in datasets:
            assert ds.X.ndim == 3
            assert ds.n_classes >= 2
    
    def test_univariate_datasets(self):
        """Test that univariate datasets are generated."""
        prior = PriorConfig3D(prob_univariate=1.0)  # Force univariate
        generator = SyntheticDatasetGenerator3D(prior=prior, seed=42)
        
        for _ in range(5):
            dataset = generator.generate()
            assert dataset.n_features == 1
    
    def test_to_2d_conversion(self):
        """Test conversion to 2D format."""
        generator = SyntheticDatasetGenerator3D(seed=42)
        dataset = generator.generate()
        
        X_2d, y = dataset.to_2d()
        
        assert X_2d.ndim == 2
        assert X_2d.shape[0] == dataset.n_samples
        assert X_2d.shape[1] == dataset.n_features * dataset.n_timesteps
        assert np.array_equal(y, dataset.y)
    
    def test_metadata(self):
        """Test that metadata is properly populated."""
        generator = SyntheticDatasetGenerator3D(seed=42)
        dataset = generator.generate()
        
        assert 'n_base_nodes' in dataset.metadata
        assert 'n_temporal_edges' in dataset.metadata
        assert 'feature_selection' in dataset.metadata
        assert 'post_processing' in dataset.metadata
    
    def test_reproducibility(self):
        """Test that same seed produces consistent results."""
        gen1 = SyntheticDatasetGenerator3D(seed=42)
        gen2 = SyntheticDatasetGenerator3D(seed=42)
        
        ds1 = gen1.generate()
        ds2 = gen2.generate()
        
        assert ds1.shape == ds2.shape
        assert ds1.n_classes == ds2.n_classes


def run_tests():
    """Run all tests."""
    import traceback
    
    test_classes = [
        TestConfig3D,
        TestTemporalDAGBuilder,
        TestRowGenerator3D,
        TestFeatureSelector3D,
        TestTableBuilder3D,
        TestSyntheticDatasetGenerator3D,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)
        
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    getattr(instance, method_name)()
                    print(f"  [PASS] {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  [FAIL] {method_name}")
                    print(f"    Error: {e}")
                    traceback.print_exc()
                    failed_tests.append((test_class.__name__, method_name, str(e)))
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

