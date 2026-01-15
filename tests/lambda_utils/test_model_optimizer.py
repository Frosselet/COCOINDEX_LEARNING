"""
Tests for Lambda model optimization utilities.

Tests COLPALI-901: Optimize Lambda container for 3B model deployment.

Environment Requirements:
- PyTorch 2.0+
- FBGEMM backend for INT8 quantization tests (Linux with specific CPU features)
- psutil for memory monitoring

Skip Conditions:
- Quantization tests skip if FBGEMM backend unavailable
- torch.compile tests skip if inductor backend unavailable
"""

import sys
import tempfile
from pathlib import Path
import importlib.util

import pytest
import torch
from torch import nn

# Load module directly to avoid tatforge package dependencies
def load_module_directly(module_name: str, file_path: Path):
    """Load a module directly from file."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Get paths
project_root = Path(__file__).parent.parent.parent
lambda_utils_path = project_root / "tatforge" / "lambda_utils"

# Load module
model_optimizer = load_module_directly(
    "model_optimizer_test",
    lambda_utils_path / "model_optimizer.py"
)

# Get classes
OptimizationConfig = model_optimizer.OptimizationConfig
OptimizationMetrics = model_optimizer.OptimizationMetrics
LambdaModelOptimizer = model_optimizer.LambdaModelOptimizer


# Environment detection for conditional tests
def check_fbgemm_available():
    """Check if FBGEMM quantization backend is available.

    FBGEMM is only available on Linux with specific CPU features.
    The error occurs when actually running inference with a model that has
    Linear layers as submodules, not when quantizing a standalone Linear layer.
    """
    try:
        # Must use a model with Linear as submodule, not standalone
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        test_model = TestModel()
        quantized = torch.quantization.quantize_dynamic(test_model, {nn.Linear}, dtype=torch.qint8)
        # Must actually run inference to trigger the FBGEMM check
        test_input = torch.randn(1, 10)
        _ = quantized(test_input)
        return True
    except RuntimeError:
        return False

FBGEMM_AVAILABLE = check_fbgemm_available()
TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile')


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size=100, hidden_size=50, output_size=10):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class TestOptimizationConfig:
    """Test OptimizationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OptimizationConfig()

        assert config.enable_int8_quantization is True
        assert config.quantization_dtype == torch.qint8
        assert config.max_memory_gb == 8.0
        assert config.enable_pruning is False
        assert config.num_threads == 4

    def test_custom_values(self):
        """Test custom configuration values."""
        config = OptimizationConfig(
            enable_int8_quantization=False,
            max_memory_gb=16.0,
            enable_pruning=True,
            pruning_ratio=0.2
        )

        assert config.enable_int8_quantization is False
        assert config.max_memory_gb == 16.0
        assert config.enable_pruning is True
        assert config.pruning_ratio == 0.2


class TestOptimizationMetrics:
    """Test OptimizationMetrics dataclass."""

    def test_default_values(self):
        """Test default metrics values."""
        metrics = OptimizationMetrics()

        assert metrics.original_size_mb == 0.0
        assert metrics.optimized_size_mb == 0.0
        assert metrics.compression_ratio == 0.0
        assert metrics.optimizations_applied == []

    def test_custom_values(self):
        """Test custom metrics values."""
        metrics = OptimizationMetrics(
            original_size_mb=100.0,
            optimized_size_mb=50.0,
            compression_ratio=2.0,
            optimizations_applied=["int8_quantization"]
        )

        assert metrics.original_size_mb == 100.0
        assert metrics.compression_ratio == 2.0


class TestLambdaModelOptimizer:
    """Test LambdaModelOptimizer class."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = LambdaModelOptimizer()

        assert optimizer.config is not None
        assert optimizer.metrics is not None

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = OptimizationConfig(num_threads=8)
        optimizer = LambdaModelOptimizer(config=config)

        assert optimizer.config.num_threads == 8

    @pytest.mark.skipif(not FBGEMM_AVAILABLE, reason="Requires FBGEMM quantization engine (Linux)")
    def test_quantize_model(self):
        """Test model quantization with FBGEMM backend."""
        model = SimpleModel()
        optimizer = LambdaModelOptimizer()

        quantized_model, info = optimizer.quantize_model(model)

        assert info["quantized"] is True
        assert "original_size_mb" in info
        assert "quantized_size_mb" in info
        assert "compression_ratio" in info
        assert info["compression_ratio"] > 1.0  # Should compress

    def test_quantize_model_disabled(self):
        """Test quantization when disabled."""
        model = SimpleModel()
        config = OptimizationConfig(enable_int8_quantization=False)
        optimizer = LambdaModelOptimizer(config=config)

        result_model, info = optimizer.quantize_model(model)

        assert info["quantized"] is False

    def test_optimize_for_inference(self):
        """Test inference optimization (without torch.compile)."""
        model = SimpleModel()
        config = OptimizationConfig(enable_cpu_optimization=False)
        optimizer = LambdaModelOptimizer(config=config)

        optimized = optimizer.optimize_for_inference(model)

        # Model should be in eval mode
        assert not optimized.training

        # Gradients should be disabled
        for param in optimized.parameters():
            assert not param.requires_grad

        # Check optimizations were recorded
        assert "eval_mode" in optimizer.metrics.optimizations_applied
        assert "gradients_disabled" in optimizer.metrics.optimizations_applied

    def test_prune_model(self):
        """Test model pruning."""
        model = SimpleModel()
        config = OptimizationConfig(enable_pruning=True, pruning_ratio=0.1)
        optimizer = LambdaModelOptimizer(config=config)

        pruned_model, info = optimizer.prune_model(model)

        assert info["pruned"] is True
        assert info["pruning_ratio"] == 0.1
        assert info["layers_pruned"] > 0

    def test_prune_model_disabled(self):
        """Test pruning when disabled."""
        model = SimpleModel()
        optimizer = LambdaModelOptimizer()

        result_model, info = optimizer.prune_model(model)

        assert info["pruned"] is False

    def test_estimate_memory_requirements(self):
        """Test memory estimation."""
        model = SimpleModel()
        optimizer = LambdaModelOptimizer()

        estimates = optimizer.estimate_memory_requirements(model)

        assert "model_size_mb" in estimates
        assert "estimated_inference_mb" in estimates
        assert "available_memory_gb" in estimates
        assert "fits_in_memory" in estimates
        assert "recommended_batch_size" in estimates

    def test_save_and_load_model(self):
        """Test model save."""
        model = SimpleModel()
        optimizer = LambdaModelOptimizer()

        class MockProcessor:
            def save_pretrained(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "config.json").write_text("{}")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_info = optimizer.save_optimized_model(
                model,
                MockProcessor(),
                tmpdir
            )

            assert Path(save_info["model_path"]).exists()
            assert Path(save_info["processor_path"]).exists()
            assert "model_size_mb" in save_info or "total_size_mb" in save_info

    @pytest.mark.skipif(not FBGEMM_AVAILABLE, reason="Requires FBGEMM quantization engine (Linux)")
    def test_get_optimization_metrics_with_quantization(self):
        """Test getting optimization metrics after quantization."""
        model = SimpleModel()
        optimizer = LambdaModelOptimizer()

        optimizer.quantize_model(model)
        metrics = optimizer.get_optimization_metrics()

        assert "original_size_mb" in metrics
        assert "optimized_size_mb" in metrics
        assert "optimizations_applied" in metrics
        assert "int8_quantization" in metrics["optimizations_applied"]

    def test_get_optimization_metrics_basic(self):
        """Test getting optimization metrics without quantization."""
        optimizer = LambdaModelOptimizer()
        metrics = optimizer.get_optimization_metrics()

        assert "original_size_mb" in metrics
        assert "optimized_size_mb" in metrics
        assert "optimizations_applied" in metrics

    def test_cleanup(self):
        """Test cleanup."""
        optimizer = LambdaModelOptimizer()

        optimizer.cleanup()

        assert optimizer.metrics.memory_usage_mb >= 0

    def test_calculate_optimal_batch_size(self):
        """Test batch size recommendation."""
        optimizer = LambdaModelOptimizer()
        model = SimpleModel()

        estimates = optimizer.estimate_memory_requirements(model)
        batch_size = estimates["recommended_batch_size"]

        assert batch_size >= 1
        assert batch_size <= 16

    @pytest.mark.skipif(not FBGEMM_AVAILABLE, reason="Requires FBGEMM quantization engine (Linux)")
    def test_full_optimization_pipeline_with_quantization(self):
        """Test complete optimization pipeline including quantization."""
        model = SimpleModel(input_size=1000, hidden_size=500, output_size=100)
        config = OptimizationConfig(
            enable_int8_quantization=True,
            enable_pruning=True,
            pruning_ratio=0.1,
            enable_cpu_optimization=False
        )
        optimizer = LambdaModelOptimizer(config=config)

        # Quantize
        model, quant_info = optimizer.quantize_model(model)
        assert quant_info["quantized"]

        # Prune
        model, prune_info = optimizer.prune_model(model)
        assert prune_info["pruned"]

        # Optimize for inference
        model = optimizer.optimize_for_inference(model)

        metrics = optimizer.get_optimization_metrics()
        assert len(metrics["optimizations_applied"]) >= 3

        # Model should still work
        test_input = torch.randn(1, 1000)
        with torch.no_grad():
            output = model(test_input)
            assert output.shape == (1, 100)

    def test_optimization_pipeline_without_quantization(self):
        """Test optimization pipeline without quantization (all environments)."""
        model = SimpleModel(input_size=1000, hidden_size=500, output_size=100)
        config = OptimizationConfig(
            enable_int8_quantization=False,
            enable_pruning=True,
            pruning_ratio=0.1,
            enable_cpu_optimization=False
        )
        optimizer = LambdaModelOptimizer(config=config)

        # Prune
        model, prune_info = optimizer.prune_model(model)
        assert prune_info["pruned"]

        # Optimize for inference
        model = optimizer.optimize_for_inference(model)

        metrics = optimizer.get_optimization_metrics()
        assert "eval_mode" in metrics["optimizations_applied"]

        # Model should still work
        test_input = torch.randn(1, 1000)
        with torch.no_grad():
            output = model(test_input)
            assert output.shape == (1, 100)


class TestModelSizeCalculation:
    """Test model size calculation."""

    def test_small_model_size(self):
        """Test size calculation for small model."""
        model = SimpleModel(input_size=10, hidden_size=5, output_size=2)
        optimizer = LambdaModelOptimizer()

        size = optimizer._calculate_model_size(model)

        assert size > 0
        assert size < 1  # Less than 1MB

    def test_larger_model_size(self):
        """Test size calculation for larger model."""
        model = SimpleModel(input_size=10000, hidden_size=5000, output_size=1000)
        optimizer = LambdaModelOptimizer()

        size = optimizer._calculate_model_size(model)

        assert size > 100  # More than 100MB for this size

    @pytest.mark.skipif(not FBGEMM_AVAILABLE, reason="Requires FBGEMM quantization engine (Linux)")
    def test_size_reduction_after_quantization(self):
        """Test that quantization reduces size."""
        model = SimpleModel(input_size=1000, hidden_size=500, output_size=100)
        optimizer = LambdaModelOptimizer()

        original_size = optimizer._calculate_model_size(model)
        quantized_model, _ = optimizer.quantize_model(model)
        quantized_size = optimizer._calculate_model_size(quantized_model)

        assert quantized_size < original_size
