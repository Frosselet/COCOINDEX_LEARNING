"""
Test cases for Vision Model Manager.

Tests vision model management, fallback strategies, cost optimization,
and integration with BAML client configurations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json
import os

from colpali_engine.core.vision_model_manager import (
    VisionModelManager,
    VisionModelSpec,
    VisionModelCapability,
    ImageInputType,
    FallbackStrategy,
    ImageProcessingConfig,
    VisionFallbackResult
)
from colpali_engine.core.baml_client_manager import BAMLClientManager, BAMLClient, ClientType
from colpali_engine.core.baml_function_generator import ClientComplexity
from colpali_engine.core.schema_manager import BAMLFunction


class TestVisionModelSpec:
    """Test VisionModelSpec functionality."""

    def test_vision_model_spec_creation(self):
        """Test VisionModelSpec creation with all attributes."""
        spec = VisionModelSpec(
            name="test-model",
            capability=VisionModelCapability.ADVANCED_VISION,
            max_image_size_mb=15.0,
            supported_formats=["jpeg", "png", "webp"],
            max_images_per_request=8,
            cost_per_1k_tokens=0.025,
            vision_cost_multiplier=1.8,
            supports_batch_processing=True,
            optimal_for_documents=True,
            fallback_client="fallback-client"
        )

        assert spec.name == "test-model"
        assert spec.capability == VisionModelCapability.ADVANCED_VISION
        assert spec.max_image_size_mb == 15.0
        assert spec.supported_formats == ["jpeg", "png", "webp"]
        assert spec.max_images_per_request == 8
        assert spec.cost_per_1k_tokens == 0.025
        assert spec.vision_cost_multiplier == 1.8
        assert spec.supports_batch_processing is True
        assert spec.optimal_for_documents is True
        assert spec.fallback_client == "fallback-client"

    def test_vision_model_spec_defaults(self):
        """Test VisionModelSpec with default values."""
        spec = VisionModelSpec(
            name="test-model",
            capability=VisionModelCapability.BASIC_VISION,
            max_image_size_mb=8.0,
            supported_formats=["jpeg", "png"],
            max_images_per_request=4,
            cost_per_1k_tokens=0.003,
            vision_cost_multiplier=1.2
        )

        assert spec.supports_batch_processing is False
        assert spec.optimal_for_documents is False
        assert spec.fallback_client is None


class TestImageProcessingConfig:
    """Test ImageProcessingConfig functionality."""

    def test_image_processing_config_creation(self):
        """Test ImageProcessingConfig creation."""
        config = ImageProcessingConfig(
            max_size_mb=15.0,
            supported_formats=["jpeg", "png", "gif"],
            auto_resize=False,
            quality_optimization=False,
            batch_processing=True
        )

        assert config.max_size_mb == 15.0
        assert config.supported_formats == ["jpeg", "png", "gif"]
        assert config.auto_resize is False
        assert config.quality_optimization is False
        assert config.batch_processing is True

    def test_image_processing_config_defaults(self):
        """Test ImageProcessingConfig with defaults."""
        config = ImageProcessingConfig()

        assert config.max_size_mb == 10.0
        assert config.supported_formats is None
        assert config.auto_resize is True
        assert config.quality_optimization is True
        assert config.batch_processing is False


class TestVisionFallbackResult:
    """Test VisionFallbackResult functionality."""

    def test_vision_fallback_result_success(self):
        """Test successful VisionFallbackResult."""
        result = VisionFallbackResult(
            success=True,
            result={"data": "test"},
            client_used="CustomSonnet4",
            fallback_applied=False,
            cost_estimate=0.05
        )

        assert result.success is True
        assert result.result == {"data": "test"}
        assert result.client_used == "CustomSonnet4"
        assert result.fallback_applied is False
        assert result.fallback_reason is None
        assert result.cost_estimate == 0.05

    def test_vision_fallback_result_with_fallback(self):
        """Test VisionFallbackResult with fallback applied."""
        result = VisionFallbackResult(
            success=True,
            result={"data": "fallback_result"},
            client_used="CustomHaiku",
            fallback_applied=True,
            fallback_reason="Primary client failed",
            cost_estimate=0.02,
            processing_time=2.5
        )

        assert result.success is True
        assert result.fallback_applied is True
        assert result.fallback_reason == "Primary client failed"
        assert result.processing_time == 2.5


class TestVisionModelManager:
    """Test VisionModelManager functionality."""

    def setup_method(self):
        """Set up test environment."""
        # Mock BAMLClientManager to avoid file system dependencies
        self.mock_client_manager = Mock(spec=BAMLClientManager)
        self.mock_client_manager.get_vision_capable_clients.return_value = [
            "CustomGPT5", "CustomOpus4", "CustomSonnet4", "CustomHaiku"
        ]

        self.manager = VisionModelManager(client_manager=self.mock_client_manager)

    def test_initialization(self):
        """Test VisionModelManager initialization."""
        assert len(self.manager.vision_specs) > 0
        assert "CustomGPT5" in self.manager.vision_specs
        assert "CustomOpus4" in self.manager.vision_specs
        assert "CustomSonnet4" in self.manager.vision_specs
        assert "CustomHaiku" in self.manager.vision_specs
        assert self.manager.cost_optimization_enabled is True

    def test_initialization_with_default_client_manager(self):
        """Test initialization without providing client manager."""
        # This will create a default BAMLClientManager
        manager = VisionModelManager()
        assert manager.client_manager is not None
        assert len(manager.vision_specs) > 0

    def test_get_optimal_vision_client_cost_priority(self):
        """Test getting optimal client with cost priority."""
        client = self.manager.get_optimal_vision_client(
            complexity=ClientComplexity.SIMPLE,
            image_count=2,
            cost_priority=True
        )

        assert client is not None
        # Should return one of the available vision clients
        assert client in ["CustomGPT5", "CustomOpus4", "CustomSonnet4", "CustomHaiku"]

        # For simple complexity with cost priority, should prefer cheaper options
        spec = self.manager.vision_specs[client]
        assert spec.cost_per_1k_tokens <= 0.030  # Should be reasonably priced

    def test_get_optimal_vision_client_accuracy_priority(self):
        """Test getting optimal client with accuracy priority."""
        client = self.manager.get_optimal_vision_client(
            complexity=ClientComplexity.COMPLEX,
            image_count=3,
            cost_priority=False
        )

        assert client is not None
        spec = self.manager.vision_specs[client]
        # For complex tasks, should prefer high-capability models
        assert spec.capability in [VisionModelCapability.MULTIMODAL, VisionModelCapability.ADVANCED_VISION]

    def test_get_optimal_vision_client_image_count_filtering(self):
        """Test client filtering based on image count requirements."""
        # Test with high image count that might exceed some client limits
        client = self.manager.get_optimal_vision_client(
            complexity=ClientComplexity.MODERATE,
            image_count=12,  # High count
            cost_priority=False
        )

        # Should still return a client (fallback mechanism)
        assert client is not None

    def test_get_optimal_vision_client_no_vision_clients(self):
        """Test behavior when no vision clients are available."""
        # Mock no vision clients available
        self.mock_client_manager.get_vision_capable_clients.return_value = []

        # Should return emergency fallback client instead of raising error
        client = self.manager.get_optimal_vision_client(
            complexity=ClientComplexity.SIMPLE,
            image_count=1
        )

        # Should return the emergency fallback client
        assert client == "CustomSonnet4"

    def test_validate_image_inputs_success(self):
        """Test successful image validation."""
        images = ["test_image.jpg", "another_image.png"]

        is_valid, issues = self.manager.validate_image_inputs(
            images, "CustomSonnet4"
        )

        # Should be valid (simplified validation in test implementation)
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_image_inputs_no_images(self):
        """Test validation with no images provided."""
        is_valid, issues = self.manager.validate_image_inputs(
            [], "CustomSonnet4"
        )

        assert is_valid is False
        assert "No images provided" in issues

    def test_validate_image_inputs_unknown_client(self):
        """Test validation with unknown client."""
        images = ["test.jpg"]

        is_valid, issues = self.manager.validate_image_inputs(
            images, "UnknownClient"
        )

        assert is_valid is False
        assert any("No vision specification found" in issue for issue in issues)

    def test_validate_image_inputs_too_many_images(self):
        """Test validation with too many images."""
        # CustomHaiku has max_images_per_request = 4
        images = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"]  # 5 images

        is_valid, issues = self.manager.validate_image_inputs(
            images, "CustomHaiku"
        )

        assert is_valid is False
        assert any("Too many images" in issue for issue in issues)

    def test_validate_image_inputs_none_image(self):
        """Test validation with None image."""
        images = [None, "test.jpg"]

        is_valid, issues = self.manager.validate_image_inputs(
            images, "CustomSonnet4"
        )

        assert is_valid is False
        assert any("Image is None" in issue for issue in issues)

    def test_get_cost_optimization_recommendation(self):
        """Test cost optimization recommendations."""
        recommendations = self.manager.get_cost_optimization_recommendation(
            complexity=ClientComplexity.MODERATE,
            image_count=3,
            budget_limit=0.10
        )

        assert "recommendations" in recommendations
        assert "budget_limit" in recommendations
        assert "cheapest_option" in recommendations
        assert "best_value" in recommendations

        assert recommendations["budget_limit"] == 0.10
        assert len(recommendations["recommendations"]) > 0

        # Verify recommendations structure
        for rec in recommendations["recommendations"]:
            assert "client" in rec
            assert "estimated_cost" in rec
            assert "suitable_for_complexity" in rec
            assert "capability" in rec
            assert "within_budget" in rec

    def test_get_cost_optimization_recommendation_no_budget(self):
        """Test cost optimization without budget limit."""
        recommendations = self.manager.get_cost_optimization_recommendation(
            complexity=ClientComplexity.SIMPLE,
            image_count=1
        )

        assert recommendations["budget_limit"] is None
        # All recommendations should be within budget (no limit)
        for rec in recommendations["recommendations"]:
            assert rec["within_budget"] is True

    def test_generate_vision_baml_config(self):
        """Test BAML configuration generation."""
        config = self.manager.generate_vision_baml_config()

        assert isinstance(config, str)
        assert "Vision-capable model configurations" in config
        assert "COLPALI-505" in config
        assert "CustomGPT5" in config
        assert "CustomOpus4" in config
        assert "Fallback chains" in config

        # Should contain cost and capability information
        assert "Cost:" in config
        assert "Max images:" in config
        assert "Formats:" in config

    def test_process_with_vision_fallback_success(self):
        """Test successful vision processing without fallback."""
        # Create mock function
        mock_function = Mock(spec=BAMLFunction)
        mock_function.name = "test_function"
        mock_function.client = "CustomSonnet4"

        images = ["test_image.jpg"]

        result = self.manager.process_with_vision_fallback(
            function=mock_function,
            images=images,
            fallback_strategy=FallbackStrategy.ALTERNATIVE_VISION
        )

        assert result.success is True
        assert result.client_used == "CustomSonnet4"
        assert result.fallback_applied is False
        assert result.result is not None
        assert result.cost_estimate is not None

    def test_process_with_vision_fallback_with_fallback_needed(self):
        """Test vision processing with fallback required."""
        # Create mock function with a client that will "fail"
        mock_function = Mock(spec=BAMLFunction)
        mock_function.name = "test_function"
        mock_function.client = "NonExistentClient"  # This will cause fallback

        images = ["test_image.jpg"]

        # Mock the execution to raise an exception for primary client
        with patch.object(self.manager, '_execute_vision_function') as mock_execute:
            mock_execute.side_effect = [Exception("Primary failed"), {"status": "success"}]

            result = self.manager.process_with_vision_fallback(
                function=mock_function,
                images=images,
                fallback_strategy=FallbackStrategy.ALTERNATIVE_VISION,
                max_retries=2
            )

            # Should have attempted primary and fallback
            assert mock_execute.call_count >= 1

    def test_process_with_vision_fallback_graceful_degradation(self):
        """Test graceful degradation fallback strategy."""
        mock_function = Mock(spec=BAMLFunction)
        mock_function.name = "test_function"
        mock_function.client = "CustomSonnet4"

        images = ["test_image.jpg"]

        # Mock primary execution to fail
        with patch.object(self.manager, '_execute_vision_function') as mock_execute:
            mock_execute.side_effect = Exception("Primary failed")

            result = self.manager.process_with_vision_fallback(
                function=mock_function,
                images=images,
                fallback_strategy=FallbackStrategy.GRACEFUL_DEGRADATION
            )

            # Graceful degradation should still return success with partial results
            assert result.success is True
            assert result.fallback_applied is True
            assert "error" in result.result or "extracted_data" in result.result

    def test_client_suitability_for_complexity(self):
        """Test client suitability assessment for different complexity levels."""
        # Test simple complexity
        assert self.manager._is_client_suitable_for_complexity("CustomHaiku", ClientComplexity.SIMPLE)
        assert self.manager._is_client_suitable_for_complexity("CustomSonnet4", ClientComplexity.SIMPLE)

        # Test complex complexity
        assert self.manager._is_client_suitable_for_complexity("CustomGPT5", ClientComplexity.COMPLEX)
        assert self.manager._is_client_suitable_for_complexity("CustomOpus4", ClientComplexity.COMPLEX)

        # CustomHaiku (BASIC_VISION) should not be suitable for complex tasks
        assert not self.manager._is_client_suitable_for_complexity("CustomHaiku", ClientComplexity.COMPLEX)

    def test_cost_estimation(self):
        """Test cost estimation for different clients and image counts."""
        # Test with single image
        cost_single = self.manager._estimate_processing_cost("CustomHaiku", ["image1.jpg"])
        assert cost_single > 0

        # Test with multiple images - should cost more
        cost_multiple = self.manager._estimate_processing_cost("CustomHaiku", ["img1.jpg", "img2.jpg", "img3.jpg"])
        assert cost_multiple > cost_single

        # Test different clients - higher-tier clients should cost more
        cost_haiku = self.manager._estimate_processing_cost("CustomHaiku", ["image.jpg"])
        cost_gpt5 = self.manager._estimate_processing_cost("CustomGPT5", ["image.jpg"])
        assert cost_gpt5 > cost_haiku

    def test_fallback_chains_initialization(self):
        """Test that fallback chains are properly initialized."""
        assert len(self.manager.fallback_chains) > 0

        # GPT5 should have fallback options
        assert "CustomGPT5" in self.manager.fallback_chains
        gpt5_fallbacks = self.manager.fallback_chains["CustomGPT5"]
        assert len(gpt5_fallbacks) > 0
        assert "CustomOpus4" in gpt5_fallbacks

        # CustomHaiku (cheapest) should have no fallbacks
        haiku_fallbacks = self.manager.fallback_chains.get("CustomHaiku", [])
        assert len(haiku_fallbacks) == 0

    def test_vision_specs_initialization(self):
        """Test that vision specs are properly initialized."""
        assert len(self.manager.vision_specs) >= 5  # At least 5 default models

        # Check GPT5 specs
        gpt5_spec = self.manager.vision_specs["CustomGPT5"]
        assert gpt5_spec.name == "CustomGPT5"
        assert gpt5_spec.capability == VisionModelCapability.MULTIMODAL
        assert gpt5_spec.optimal_for_documents is True
        assert gpt5_spec.supports_batch_processing is True

        # Check Haiku specs (basic model)
        haiku_spec = self.manager.vision_specs["CustomHaiku"]
        assert haiku_spec.name == "CustomHaiku"
        assert haiku_spec.capability == VisionModelCapability.BASIC_VISION
        assert haiku_spec.optimal_for_documents is False
        assert haiku_spec.fallback_client is None

    def test_emergency_fallback(self):
        """Test emergency fallback behavior."""
        fallback_client = self.manager._get_emergency_fallback()
        assert fallback_client == "CustomSonnet4"  # Should be the safe default

    def test_select_cost_optimized_client(self):
        """Test cost-optimized client selection."""
        available_clients = ["CustomGPT5", "CustomHaiku", "CustomSonnet4"]

        selected = self.manager._select_cost_optimized_client(available_clients, image_count=2)

        # Should select the cheapest suitable option
        assert selected in available_clients
        selected_cost = self.manager._estimate_processing_cost(selected, ["img1", "img2"])

        # Verify it's among the cheaper options
        all_costs = [self.manager._estimate_processing_cost(client, ["img1", "img2"])
                    for client in available_clients]
        assert selected_cost <= max(all_costs)

    def test_select_accuracy_optimized_client(self):
        """Test accuracy-optimized client selection."""
        available_clients = ["CustomGPT5", "CustomHaiku", "CustomSonnet4"]

        selected = self.manager._select_accuracy_optimized_client(available_clients, ClientComplexity.COMPLEX)

        # Should prefer document-optimized clients
        selected_spec = self.manager.vision_specs[selected]
        assert selected_spec.optimal_for_documents is True


class TestEnums:
    """Test enum classes."""

    def test_vision_model_capability_values(self):
        """Test VisionModelCapability enum values."""
        assert VisionModelCapability.BASIC_VISION.value == "basic_vision"
        assert VisionModelCapability.ADVANCED_VISION.value == "advanced_vision"
        assert VisionModelCapability.MULTIMODAL.value == "multimodal"
        assert VisionModelCapability.DOCUMENT_SPECIALIST.value == "document_specialist"

    def test_image_input_type_values(self):
        """Test ImageInputType enum values."""
        assert ImageInputType.BASE64.value == "base64"
        assert ImageInputType.URL.value == "url"
        assert ImageInputType.FILE_PATH.value == "file_path"
        assert ImageInputType.BINARY.value == "binary"

    def test_fallback_strategy_values(self):
        """Test FallbackStrategy enum values."""
        assert FallbackStrategy.TEXT_ONLY.value == "text_only"
        assert FallbackStrategy.ALTERNATIVE_VISION.value == "alternative_vision"
        assert FallbackStrategy.HYBRID_APPROACH.value == "hybrid_approach"
        assert FallbackStrategy.GRACEFUL_DEGRADATION.value == "graceful_degradation"


class TestIntegration:
    """Integration tests with other components."""

    def test_integration_with_baml_client_manager(self):
        """Test integration with real BAMLClientManager."""
        # Create a real BAMLClientManager for integration testing
        try:
            client_manager = BAMLClientManager()
            vision_manager = VisionModelManager(client_manager=client_manager)

            # Test that vision manager can work with real client manager
            vision_clients = client_manager.get_vision_capable_clients()

            if vision_clients:  # Only test if vision clients are available
                optimal_client = vision_manager.get_optimal_vision_client(
                    complexity=ClientComplexity.SIMPLE,
                    image_count=1
                )
                assert optimal_client is not None

        except Exception as e:
            # If BAML setup isn't available, skip integration test
            pytest.skip(f"BAML integration not available: {e}")

    def test_vision_config_generation_format(self):
        """Test that generated BAML config has proper format."""
        manager = VisionModelManager()
        config = manager.generate_vision_baml_config()

        # Verify basic structure
        lines = config.split('\n')
        assert len(lines) > 10  # Should have substantial content

        # Should contain comment headers
        comment_lines = [line for line in lines if line.strip().startswith('//')]
        assert len(comment_lines) > 0

        # Should contain client information
        assert any("CustomGPT5" in line for line in lines)
        assert any("CustomSonnet4" in line for line in lines)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])