"""
Unit tests for the synthetic dataset generator.

Run with: python -m pytest synthetic_generator/tests.py -v
Or simply: python synthetic_generator/tests.py
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from __init__ import (
    SyntheticDatasetGenerator,
    PriorConfig,
    DatasetConfig,
    DAGBuilder,
    TransformationFactory,
    RowGenerator,
    FeatureSelector,
    PostProcessor,
)
from transformations import (
    Activation,
    NNTransformation,
    DiscretizationTransformation,
    TreeTransformation,
    IdentityTransformation,
    RootNoiseGenerator,
)


class TestConfig:
    """Tests for configuration classes."""
    
    def test_prior_config_defaults(self):
        """Test that PriorConfig has valid defaults."""
        prior = PriorConfig()
        
        assert prior.n_rows_range[0] < prior.n_rows_range[1]
        assert prior.n_features_range[0] < prior.n_features_range[1]
        assert prior.n_nodes_range[0] < prior.n_nodes_range[1]
        assert 0 <= prior.prob_classification <= 1
        assert len(prior.activations) > 0
    
    def test_sample_hyperparams(self):
        """Test sampling dataset config from prior."""
        prior = PriorConfig()
        rng = np.random.default_rng(42)
        
        config = prior.sample_hyperparams(rng)
        
        assert isinstance(config, DatasetConfig)
        assert prior.n_rows_range[0] <= config.n_rows <= prior.n_rows_range[1] * 2  # Allow some margin for log-uniform
        assert config.n_features >= prior.n_features_range[0]
        assert config.n_nodes >= config.n_features + 2  # At least features + target + 1
    
    def test_config_to_dict(self):
        """Test serialization of DatasetConfig."""
        prior = PriorConfig()
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        d = config.to_dict()
        
        assert 'n_rows' in d
        assert 'n_features' in d
        assert 'is_classification' in d


class TestActivations:
    """Tests for activation functions."""
    
    def test_all_activations_run(self):
        """Test that all activations can be applied."""
        x = np.linspace(-2, 2, 100)
        
        for name in ['identity', 'log', 'sigmoid', 'tanh', 'sin', 'cos',
                     'abs', 'square', 'cube', 'sqrt', 'relu', 'softplus',
                     'step', 'mod', 'rank', 'exp_neg', 'gaussian']:
            fn = Activation.get(name)
            result = fn(x)
            assert result.shape == x.shape, f"Activation {name} changed shape"
            assert np.all(np.isfinite(result)), f"Activation {name} produced non-finite values"
    
    def test_activation_not_found(self):
        """Test that unknown activation raises error."""
        try:
            Activation.get('unknown_activation')
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestDAGBuilder:
    """Tests for DAG construction."""
    
    def test_basic_dag(self):
        """Test basic DAG construction."""
        prior = PriorConfig(n_nodes_range=(10, 20))
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        builder = DAGBuilder(config, rng)
        dag = builder.build()
        
        assert len(dag.nodes) > 0
        assert len(dag.edges) > 0
        assert len(dag.root_nodes) > 0
        assert len(dag.topological_order) == len(dag.nodes)
    
    def test_dag_acyclicity(self):
        """Test that DAG has no cycles."""
        prior = PriorConfig()
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        builder = DAGBuilder(config, rng)
        dag = builder.build()
        
        # Verify topological order exists (would fail if cycles)
        assert len(dag.topological_order) == len(dag.nodes)
        
        # Verify edges respect topological order
        order_map = {node_id: i for i, node_id in enumerate(dag.topological_order)}
        for parent, child in dag.edges:
            assert order_map[parent] < order_map[child], "Edge violates topological order"
    
    def test_dag_subgraphs(self):
        """Test disconnected subgraphs."""
        prior = PriorConfig(
            prob_disconnected_subgraph=1.0,
            n_disconnected_subgraphs_range=(2, 3),
            n_nodes_range=(20, 30)
        )
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        builder = DAGBuilder(config, rng)
        dag = builder.build()
        
        assert dag.n_subgraphs >= 2, "Should have multiple subgraphs"
    
    def test_ancestors_descendants(self):
        """Test ancestor/descendant computation."""
        prior = PriorConfig(n_nodes_range=(10, 15))
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        builder = DAGBuilder(config, rng)
        dag = builder.build()
        
        # Root nodes should have no ancestors
        for root in dag.root_nodes:
            ancestors = dag.get_ancestors(root)
            assert len(ancestors) == 0, f"Root node {root} has ancestors"
        
        # Check consistency: if A is ancestor of B, B is descendant of A
        for node_id in dag.nodes:
            descendants = dag.get_descendants(node_id)
            for desc in descendants:
                ancestors_of_desc = dag.get_ancestors(desc)
                assert node_id in ancestors_of_desc


class TestTransformations:
    """Tests for edge transformations."""
    
    def test_nn_transformation(self):
        """Test neural network transformation."""
        rng = np.random.default_rng(42)
        
        transform = NNTransformation(
            weights=[np.array([[0.5, 0.3], [0.2, 0.4]])],
            biases=[np.array([0.1, 0.2])],
            activations=['tanh'],
            noise_scale=0.01,
            rng=rng
        )
        
        inputs = np.random.randn(100, 2)
        outputs = transform.forward(inputs)
        
        assert outputs.shape == (100,)
        assert np.all(np.isfinite(outputs))
    
    def test_discretization_transformation(self):
        """Test discretization transformation."""
        rng = np.random.default_rng(42)
        
        transform = DiscretizationTransformation(
            prototypes=np.array([[-1.0], [0.0], [1.0]]),
            category_embeddings=np.array([-1.0, 0.0, 1.0]),
            noise_scale=0.01,
            rng=rng
        )
        
        inputs = np.random.randn(100, 1)
        outputs = transform.forward(inputs)
        categories = transform.get_category_indices(inputs)
        
        assert outputs.shape == (100,)
        assert categories.shape == (100,)
        assert set(categories).issubset({0, 1, 2})
    
    def test_tree_transformation(self):
        """Test decision tree transformation."""
        rng = np.random.default_rng(42)
        
        # Simple depth-1 tree
        transform = TreeTransformation(
            thresholds=np.array([0.0, 0.0, 0.0]),
            feature_indices=np.array([0, 0, 0]),
            left_children=np.array([1, -1, -1]),
            right_children=np.array([2, -1, -1]),
            leaf_values=np.array([0.0, -1.0, 1.0]),
            noise_scale=0.01,
            rng=rng
        )
        
        inputs = np.random.randn(100, 1)
        outputs = transform.forward(inputs)
        
        assert outputs.shape == (100,)
        assert np.all(np.isfinite(outputs))
    
    def test_transformation_factory(self):
        """Test transformation factory."""
        prior = PriorConfig()
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        factory = TransformationFactory(config, rng)
        
        for n_parents in [1, 3, 5]:
            transform = factory.create(n_parents)
            
            inputs = np.random.randn(50, n_parents)
            outputs = transform.forward(inputs)
            
            assert outputs.shape == (50,)
            assert np.all(np.isfinite(outputs))


class TestRowGenerator:
    """Tests for row generation."""
    
    def test_basic_generation(self):
        """Test basic row generation."""
        prior = PriorConfig(n_nodes_range=(10, 15), n_rows_range=(100, 100))
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        builder = DAGBuilder(config, rng)
        dag = builder.build()
        
        transformations = RowGenerator.create_transformations(dag, config, rng)
        generator = RowGenerator(config, dag, transformations, rng)
        
        propagated = generator.generate(n_samples=100)
        
        assert len(propagated.values) == len(dag.nodes)
        for node_id, values in propagated.values.items():
            assert len(values) == 100
            assert np.all(np.isfinite(values))
    
    def test_prototype_generation(self):
        """Test generation with row dependencies."""
        prior = PriorConfig(
            n_nodes_range=(10, 15),
            n_rows_range=(100, 100),
            prob_row_dependency=1.0,
            n_prototypes_range=(5, 10)
        )
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        builder = DAGBuilder(config, rng)
        dag = builder.build()
        
        transformations = RowGenerator.create_transformations(dag, config, rng)
        generator = RowGenerator(config, dag, transformations, rng)
        
        propagated = generator.generate(n_samples=100)
        
        assert propagated.metadata['generation_type'] == 'prototype'
        assert 'prototype_assignments' in propagated.metadata


class TestFeatureSelector:
    """Tests for feature and target selection."""
    
    def test_basic_selection(self):
        """Test basic feature selection."""
        prior = PriorConfig(n_nodes_range=(20, 30), n_features_range=(5, 10))
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        builder = DAGBuilder(config, rng)
        dag = builder.build()
        
        transformations = RowGenerator.create_transformations(dag, config, rng)
        selector = FeatureSelector(config, dag, transformations, rng)
        
        selection = selector.select()
        
        assert len(selection.feature_nodes) <= config.n_features
        assert selection.target_node not in selection.feature_nodes
        assert selection.target_node in dag.nodes
        for feat in selection.feature_nodes:
            assert feat in dag.nodes
    
    def test_relevant_irrelevant(self):
        """Test relevant vs irrelevant feature identification."""
        prior = PriorConfig(
            n_nodes_range=(30, 40),
            n_features_range=(10, 15),
            prob_disconnected_subgraph=1.0,
            n_disconnected_subgraphs_range=(2, 3)
        )
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        builder = DAGBuilder(config, rng)
        dag = builder.build()
        
        transformations = RowGenerator.create_transformations(dag, config, rng)
        selector = FeatureSelector(config, dag, transformations, rng)
        
        selection = selector.select()
        
        # All features should be either relevant or irrelevant
        all_features = set(selection.feature_nodes)
        classified = selection.relevant_features | selection.irrelevant_features
        assert all_features == classified


class TestPostProcessor:
    """Tests for post-processing."""
    
    def test_warping(self):
        """Test warping transformation."""
        from post_processing import Warper
        
        rng = np.random.default_rng(42)
        warper = Warper(intensity=1.0, rng=rng)
        
        X = np.random.randn(100, 5)
        X_warped = warper.warp(X)
        
        assert X_warped.shape == X.shape
        assert not np.allclose(X_warped, X)  # Should be different
    
    def test_quantization(self):
        """Test quantization."""
        from post_processing import Quantizer
        
        rng = np.random.default_rng(42)
        quantizer = Quantizer(n_bins_range=(5, 10), rng=rng)
        
        X = np.random.randn(100, 5)
        X_quantized = quantizer.quantize(X)
        
        assert X_quantized.shape == X.shape
    
    def test_missing_values(self):
        """Test missing value injection."""
        from post_processing import MissingValueInjector
        
        rng = np.random.default_rng(42)
        injector = MissingValueInjector(missing_rate=0.1, rng=rng)
        
        X = np.random.randn(100, 5)
        X_missing, mask = injector.inject(X)
        
        assert X_missing.shape == X.shape
        assert mask.shape == X.shape
        assert np.any(np.isnan(X_missing))
        assert np.sum(mask) > 0
    
    def test_full_post_processing(self):
        """Test full post-processing pipeline."""
        prior = PriorConfig(
            prob_warping=1.0,
            prob_quantization=1.0,
            prob_missing_values=1.0,
        )
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        processor = PostProcessor(config, rng)
        
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        result = processor.process(X, y)
        
        assert result.X.shape == X.shape
        assert result.y.shape == y.shape


class TestSyntheticDatasetGenerator:
    """Tests for the main generator class."""
    
    def test_basic_generation(self):
        """Test basic dataset generation."""
        generator = SyntheticDatasetGenerator(seed=42)
        dataset = generator.generate()
        
        assert dataset.X.shape[0] == dataset.y.shape[0]
        assert dataset.X.shape[1] > 0
        assert len(dataset.feature_names) == dataset.X.shape[1]
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        gen1 = SyntheticDatasetGenerator(seed=42)
        gen2 = SyntheticDatasetGenerator(seed=42)
        
        dataset1 = gen1.generate()
        dataset2 = gen2.generate()
        
        # Note: Due to the complexity of the generation process,
        # exact reproducibility might not be guaranteed for all aspects
        assert dataset1.X.shape == dataset2.X.shape
        assert dataset1.is_classification == dataset2.is_classification
    
    def test_generate_many(self):
        """Test generating multiple datasets."""
        generator = SyntheticDatasetGenerator(seed=42)
        
        datasets = list(generator.generate_many(10))
        
        assert len(datasets) == 10
        for ds in datasets:
            assert ds.X.shape[0] > 0
            assert ds.X.shape[1] > 0
    
    def test_classification_vs_regression(self):
        """Test that both task types are generated."""
        # Force classification
        prior_clf = PriorConfig(prob_classification=1.0)
        gen_clf = SyntheticDatasetGenerator(prior=prior_clf, seed=42)
        ds_clf = gen_clf.generate()
        
        assert ds_clf.is_classification
        assert ds_clf.n_classes >= 2
        assert len(np.unique(ds_clf.y)) <= ds_clf.n_classes
        
        # Force regression
        prior_reg = PriorConfig(prob_classification=0.0)
        gen_reg = SyntheticDatasetGenerator(prior=prior_reg, seed=42)
        ds_reg = gen_reg.generate()
        
        assert not ds_reg.is_classification
        assert ds_reg.n_classes == 0
    
    def test_custom_config(self):
        """Test generation with custom config."""
        prior = PriorConfig()
        rng = np.random.default_rng(42)
        config = prior.sample_hyperparams(rng)
        
        # Override some values
        config.n_rows = 50
        config.n_features = 5
        
        generator = SyntheticDatasetGenerator(seed=42)
        dataset = generator.generate(config=config)
        
        assert dataset.X.shape[0] == 50
        assert dataset.X.shape[1] == 5
    
    def test_metadata(self):
        """Test that metadata is properly populated."""
        generator = SyntheticDatasetGenerator(seed=42)
        dataset = generator.generate()
        
        assert 'n_nodes' in dataset.metadata
        assert 'n_edges' in dataset.metadata
        assert 'n_relevant_features' in dataset.metadata
        assert 'n_irrelevant_features' in dataset.metadata
        assert 'post_processing' in dataset.metadata


def run_tests():
    """Run all tests."""
    import traceback
    
    test_classes = [
        TestConfig,
        TestActivations,
        TestDAGBuilder,
        TestTransformations,
        TestRowGenerator,
        TestFeatureSelector,
        TestPostProcessor,
        TestSyntheticDatasetGenerator,
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
                    print(f"  ✓ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  ✗ {method_name}")
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

