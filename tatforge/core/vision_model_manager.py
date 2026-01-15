"""
Vision Model Manager - COLPALI-505 implementation.

This module manages vision-capable model configurations with fallback strategies,
cost optimization, and error handling for vision model failures.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import base64
from pathlib import Path

from .baml_client_manager import BAMLClientManager, BAMLClient, ClientType
from .baml_function_generator import ClientComplexity
from .schema_manager import BAMLFunction

logger = logging.getLogger(__name__)


class VisionModelCapability(Enum):
    """Vision model capabilities."""
    BASIC_VISION = "basic_vision"           # Can process simple images
    ADVANCED_VISION = "advanced_vision"     # Complex document understanding
    MULTIMODAL = "multimodal"              # Vision + text + reasoning
    DOCUMENT_SPECIALIST = "document_specialist"  # Optimized for documents


class ImageInputType(Enum):
    """Supported image input types."""
    BASE64 = "base64"
    URL = "url"
    FILE_PATH = "file_path"
    BINARY = "binary"


class FallbackStrategy(Enum):
    """Fallback strategies when vision models fail."""
    TEXT_ONLY = "text_only"               # Fall back to text extraction
    ALTERNATIVE_VISION = "alternative_vision"  # Try different vision model
    HYBRID_APPROACH = "hybrid_approach"   # Combine vision + text methods
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Partial results


@dataclass
class VisionModelSpec:
    """Detailed vision model specification."""
    name: str
    capability: VisionModelCapability
    max_image_size_mb: float
    supported_formats: List[str]
    max_images_per_request: int
    cost_per_1k_tokens: float
    vision_cost_multiplier: float
    supports_batch_processing: bool = False
    optimal_for_documents: bool = False
    fallback_client: Optional[str] = None


@dataclass
class ImageProcessingConfig:
    """Configuration for image processing."""
    max_size_mb: float = 10.0
    supported_formats: List[str] = None
    auto_resize: bool = True
    quality_optimization: bool = True
    batch_processing: bool = False


@dataclass
class VisionFallbackResult:
    """Result of vision processing with fallback information."""
    success: bool
    result: Optional[Any]
    client_used: str
    fallback_applied: bool = False
    fallback_reason: Optional[str] = None
    cost_estimate: Optional[float] = None
    processing_time: Optional[float] = None


class VisionModelManager:
    """
    Manages vision-capable model configurations with intelligent fallback strategies.

    Provides cost-optimized vision model selection, error handling for vision
    failures, and seamless fallback to alternative approaches.
    """

    def __init__(self, client_manager: Optional[BAMLClientManager] = None):
        self.client_manager = client_manager or BAMLClientManager()
        self.vision_specs: Dict[str, VisionModelSpec] = {}
        self.fallback_chains: Dict[str, List[str]] = {}
        self.cost_optimization_enabled = True

        self._initialize_vision_specs()
        self._initialize_fallback_chains()
        logger.info("VisionModelManager initialized with vision model support")

    def get_optimal_vision_client(
        self,
        complexity: ClientComplexity,
        image_count: int = 1,
        cost_priority: bool = False,
        preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get the optimal vision-capable client for the given requirements.

        Args:
            complexity: Schema complexity level
            image_count: Number of images to process
            cost_priority: Whether to prioritize cost over accuracy
            preferences: User preferences

        Returns:
            Optimal vision client name
        """
        try:
            # Get vision-capable clients
            vision_clients = self.client_manager.get_vision_capable_clients()

            if not vision_clients:
                raise ValueError("No vision-capable clients available")

            # Filter by requirements
            suitable_clients = self._filter_clients_by_requirements(
                vision_clients, complexity, image_count
            )

            if not suitable_clients:
                logger.warning("No clients meet requirements, using fallback")
                return self._get_fallback_client(vision_clients)

            # Apply optimization strategy
            if cost_priority:
                return self._select_cost_optimized_client(suitable_clients, image_count)
            else:
                return self._select_accuracy_optimized_client(suitable_clients, complexity)

        except Exception as e:
            logger.error(f"Failed to select vision client: {e}")
            return self._get_emergency_fallback()

    def process_with_vision_fallback(
        self,
        function: BAMLFunction,
        images: List[Any],
        fallback_strategy: FallbackStrategy = FallbackStrategy.ALTERNATIVE_VISION,
        max_retries: int = 3
    ) -> VisionFallbackResult:
        """
        Process vision task with intelligent fallback strategies.

        Args:
            function: BAML function to execute
            images: List of images to process
            fallback_strategy: Strategy to use if primary fails
            max_retries: Maximum retry attempts

        Returns:
            Processing result with fallback information
        """
        primary_client = function.client
        attempt = 1
        last_error = None

        # Try primary client first
        logger.info(f"Attempting vision processing with {primary_client}")

        try:
            result = self._execute_vision_function(function, images, primary_client)
            return VisionFallbackResult(
                success=True,
                result=result,
                client_used=primary_client,
                cost_estimate=self._estimate_processing_cost(primary_client, images)
            )
        except Exception as e:
            logger.warning(f"Primary client {primary_client} failed: {e}")
            last_error = e

        # Apply fallback strategy
        fallback_result = self._apply_fallback_strategy(
            function, images, fallback_strategy, primary_client, max_retries - 1
        )

        if fallback_result.success:
            return fallback_result
        else:
            return VisionFallbackResult(
                success=False,
                result=None,
                client_used=primary_client,
                fallback_applied=True,
                fallback_reason=f"All fallback attempts failed. Last error: {last_error}"
            )

    def validate_image_inputs(
        self,
        images: List[Any],
        client_name: str,
        config: Optional[ImageProcessingConfig] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate image inputs for a specific client.

        Args:
            images: List of image inputs to validate
            client_name: Target client name
            config: Optional processing configuration

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        config = config or ImageProcessingConfig()

        if not images:
            return False, ["No images provided"]

        vision_spec = self.vision_specs.get(client_name)
        if not vision_spec:
            return False, [f"No vision specification found for client {client_name}"]

        # Check image count
        if len(images) > vision_spec.max_images_per_request:
            issues.append(
                f"Too many images ({len(images)}). "
                f"Maximum for {client_name}: {vision_spec.max_images_per_request}"
            )

        # Validate each image
        for i, image in enumerate(images):
            image_issues = self._validate_single_image(image, vision_spec, config)
            for issue in image_issues:
                issues.append(f"Image {i+1}: {issue}")

        return len(issues) == 0, issues

    def get_cost_optimization_recommendation(
        self,
        complexity: ClientComplexity,
        image_count: int,
        budget_limit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get cost optimization recommendations for vision processing.

        Args:
            complexity: Schema complexity level
            image_count: Number of images
            budget_limit: Optional budget constraint

        Returns:
            Cost optimization recommendations
        """
        vision_clients = self.client_manager.get_vision_capable_clients()
        recommendations = []

        for client_name in vision_clients:
            spec = self.vision_specs.get(client_name)
            if not spec:
                continue

            estimated_cost = self._estimate_processing_cost(client_name, ['dummy'] * image_count)
            suitable_for_complexity = self._is_client_suitable_for_complexity(client_name, complexity)

            rec = {
                "client": client_name,
                "estimated_cost": estimated_cost,
                "suitable_for_complexity": suitable_for_complexity,
                "capability": spec.capability.value,
                "optimal_for_documents": spec.optimal_for_documents,
                "within_budget": budget_limit is None or estimated_cost <= budget_limit
            }
            recommendations.append(rec)

        # Sort by cost-effectiveness
        recommendations.sort(key=lambda x: (
            not x["suitable_for_complexity"],
            x["estimated_cost"],
            not x["optimal_for_documents"]
        ))

        return {
            "recommendations": recommendations,
            "budget_limit": budget_limit,
            "cheapest_option": recommendations[0] if recommendations else None,
            "best_value": self._find_best_value_option(recommendations, complexity)
        }

    def generate_vision_baml_config(self) -> str:
        """
        Generate BAML configuration for vision-capable models.

        Returns:
            BAML configuration snippet for vision models
        """
        config_lines = [
            "// Vision-capable model configurations - COLPALI-505",
            "// Auto-generated vision model setup with fallback strategies",
            ""
        ]

        for client_name in self.client_manager.get_vision_capable_clients():
            spec = self.vision_specs.get(client_name)
            if spec:
                config_lines.extend([
                    f"// {client_name} - {spec.capability.value}",
                    f"// Max images: {spec.max_images_per_request}, "
                    f"Max size: {spec.max_image_size_mb}MB",
                    f"// Formats: {', '.join(spec.supported_formats)}",
                    f"// Cost: ${spec.cost_per_1k_tokens}/1K tokens "
                    f"(vision: {spec.vision_cost_multiplier}x)",
                    ""
                ])

        # Add fallback chain documentation
        config_lines.extend([
            "// Fallback chains for vision model failures:",
            ""
        ])

        for primary, fallbacks in self.fallback_chains.items():
            config_lines.append(f"// {primary} → {' → '.join(fallbacks)}")

        return "\n".join(config_lines)

    def _initialize_vision_specs(self) -> None:
        """Initialize vision model specifications."""
        self.vision_specs = {
            "CustomGPT5": VisionModelSpec(
                name="CustomGPT5",
                capability=VisionModelCapability.MULTIMODAL,
                max_image_size_mb=20.0,
                supported_formats=["jpg", "jpeg", "png", "gif", "webp"],
                max_images_per_request=10,
                cost_per_1k_tokens=0.030,
                vision_cost_multiplier=2.0,
                supports_batch_processing=True,
                optimal_for_documents=True,
                fallback_client="CustomOpus4"
            ),
            "CustomGPT5Mini": VisionModelSpec(
                name="CustomGPT5Mini",
                capability=VisionModelCapability.ADVANCED_VISION,
                max_image_size_mb=10.0,
                supported_formats=["jpg", "jpeg", "png", "gif", "webp"],
                max_images_per_request=5,
                cost_per_1k_tokens=0.005,
                vision_cost_multiplier=1.5,
                supports_batch_processing=False,
                optimal_for_documents=True,
                fallback_client="CustomSonnet4"
            ),
            "CustomOpus4": VisionModelSpec(
                name="CustomOpus4",
                capability=VisionModelCapability.MULTIMODAL,
                max_image_size_mb=15.0,
                supported_formats=["jpg", "jpeg", "png", "gif", "webp"],
                max_images_per_request=8,
                cost_per_1k_tokens=0.025,
                vision_cost_multiplier=1.8,
                supports_batch_processing=True,
                optimal_for_documents=True,
                fallback_client="CustomSonnet4"
            ),
            "CustomSonnet4": VisionModelSpec(
                name="CustomSonnet4",
                capability=VisionModelCapability.ADVANCED_VISION,
                max_image_size_mb=12.0,
                supported_formats=["jpg", "jpeg", "png", "gif", "webp"],
                max_images_per_request=6,
                cost_per_1k_tokens=0.015,
                vision_cost_multiplier=1.6,
                supports_batch_processing=True,
                optimal_for_documents=True,
                fallback_client="CustomHaiku"
            ),
            "CustomHaiku": VisionModelSpec(
                name="CustomHaiku",
                capability=VisionModelCapability.BASIC_VISION,
                max_image_size_mb=8.0,
                supported_formats=["jpg", "jpeg", "png", "gif", "webp"],
                max_images_per_request=4,
                cost_per_1k_tokens=0.003,
                vision_cost_multiplier=1.2,
                supports_batch_processing=False,
                optimal_for_documents=False,
                fallback_client=None  # No fallback for cheapest option
            )
        }

    def _initialize_fallback_chains(self) -> None:
        """Initialize fallback chains for vision model failures."""
        self.fallback_chains = {
            "CustomGPT5": ["CustomOpus4", "CustomSonnet4", "CustomGPT5Mini"],
            "CustomOpus4": ["CustomSonnet4", "CustomGPT5Mini", "CustomHaiku"],
            "CustomSonnet4": ["CustomGPT5Mini", "CustomHaiku"],
            "CustomGPT5Mini": ["CustomSonnet4", "CustomHaiku"],
            "CustomHaiku": []  # No fallback for cheapest option
        }

    def _filter_clients_by_requirements(
        self,
        clients: List[str],
        complexity: ClientComplexity,
        image_count: int
    ) -> List[str]:
        """Filter clients by requirements."""
        suitable = []

        for client_name in clients:
            spec = self.vision_specs.get(client_name)
            if not spec:
                continue

            # Check image count limit
            if image_count > spec.max_images_per_request:
                continue

            # Check complexity suitability
            if not self._is_client_suitable_for_complexity(client_name, complexity):
                continue

            suitable.append(client_name)

        return suitable

    def _is_client_suitable_for_complexity(self, client_name: str, complexity: ClientComplexity) -> bool:
        """Check if client is suitable for complexity level."""
        spec = self.vision_specs.get(client_name)
        if not spec:
            return False

        # Map complexity to minimum capability requirements
        required_capability = {
            ClientComplexity.SIMPLE: VisionModelCapability.BASIC_VISION,
            ClientComplexity.MODERATE: VisionModelCapability.ADVANCED_VISION,
            ClientComplexity.COMPLEX: VisionModelCapability.MULTIMODAL,
            ClientComplexity.ADVANCED: VisionModelCapability.MULTIMODAL
        }

        required = required_capability[complexity]
        client_capability = spec.capability

        # Check if client meets minimum capability
        capability_hierarchy = {
            VisionModelCapability.BASIC_VISION: 1,
            VisionModelCapability.ADVANCED_VISION: 2,
            VisionModelCapability.MULTIMODAL: 3,
            VisionModelCapability.DOCUMENT_SPECIALIST: 3
        }

        return capability_hierarchy[client_capability] >= capability_hierarchy[required]

    def _select_cost_optimized_client(self, clients: List[str], image_count: int) -> str:
        """Select client optimized for cost."""
        best_client = None
        best_cost = float('inf')

        for client_name in clients:
            cost = self._estimate_processing_cost(client_name, ['dummy'] * image_count)
            if cost < best_cost:
                best_cost = cost
                best_client = client_name

        return best_client or clients[0]

    def _select_accuracy_optimized_client(self, clients: List[str], complexity: ClientComplexity) -> str:
        """Select client optimized for accuracy."""
        # Prioritize by capability and document optimization
        for client_name in ["CustomGPT5", "CustomOpus4", "CustomSonnet4", "CustomGPT5Mini", "CustomHaiku"]:
            if client_name in clients:
                spec = self.vision_specs.get(client_name)
                if spec and spec.optimal_for_documents:
                    return client_name

        return clients[0]

    def _get_fallback_client(self, available_clients: List[str]) -> str:
        """Get fallback client when requirements can't be met."""
        # Prefer document-optimized clients
        for client_name in available_clients:
            spec = self.vision_specs.get(client_name)
            if spec and spec.optimal_for_documents:
                return client_name

        return available_clients[0] if available_clients else "CustomSonnet4"

    def _get_emergency_fallback(self) -> str:
        """Get emergency fallback client."""
        return "CustomSonnet4"  # Safe default

    def _apply_fallback_strategy(
        self,
        function: BAMLFunction,
        images: List[Any],
        strategy: FallbackStrategy,
        failed_client: str,
        max_retries: int
    ) -> VisionFallbackResult:
        """Apply fallback strategy when primary client fails."""
        if strategy == FallbackStrategy.ALTERNATIVE_VISION:
            return self._try_alternative_vision_clients(function, images, failed_client, max_retries)
        elif strategy == FallbackStrategy.TEXT_ONLY:
            return self._fallback_to_text_only(function)
        elif strategy == FallbackStrategy.HYBRID_APPROACH:
            return self._try_hybrid_approach(function, images)
        else:  # GRACEFUL_DEGRADATION
            return self._graceful_degradation(function, images)

    def _try_alternative_vision_clients(
        self,
        function: BAMLFunction,
        images: List[Any],
        failed_client: str,
        max_retries: int
    ) -> VisionFallbackResult:
        """Try alternative vision clients from fallback chain."""
        fallback_clients = self.fallback_chains.get(failed_client, [])

        for client_name in fallback_clients[:max_retries]:
            try:
                logger.info(f"Trying fallback client: {client_name}")
                result = self._execute_vision_function(function, images, client_name)

                return VisionFallbackResult(
                    success=True,
                    result=result,
                    client_used=client_name,
                    fallback_applied=True,
                    fallback_reason=f"Primary client {failed_client} failed, used {client_name}",
                    cost_estimate=self._estimate_processing_cost(client_name, images)
                )

            except Exception as e:
                logger.warning(f"Fallback client {client_name} also failed: {e}")
                continue

        return VisionFallbackResult(
            success=False,
            result=None,
            client_used=failed_client,
            fallback_applied=True,
            fallback_reason="All alternative vision clients failed"
        )

    def _fallback_to_text_only(self, function: BAMLFunction) -> VisionFallbackResult:
        """Fallback to text-only processing."""
        # This would require implementation of text-only extraction
        # For now, return a graceful failure
        return VisionFallbackResult(
            success=False,
            result=None,
            client_used=function.client,
            fallback_applied=True,
            fallback_reason="Text-only fallback not yet implemented"
        )

    def _try_hybrid_approach(self, function: BAMLFunction, images: List[Any]) -> VisionFallbackResult:
        """Try hybrid vision + text approach."""
        # This would implement a hybrid processing strategy
        # For now, return a graceful failure
        return VisionFallbackResult(
            success=False,
            result=None,
            client_used=function.client,
            fallback_applied=True,
            fallback_reason="Hybrid approach not yet implemented"
        )

    def _graceful_degradation(self, function: BAMLFunction, images: List[Any]) -> VisionFallbackResult:
        """Implement graceful degradation strategy."""
        # Return partial results or simplified extraction
        return VisionFallbackResult(
            success=True,
            result={"error": "Vision processing failed", "extracted_data": {}},
            client_used=function.client,
            fallback_applied=True,
            fallback_reason="Graceful degradation - partial results returned"
        )

    def _execute_vision_function(self, function: BAMLFunction, images: List[Any], client_name: str) -> Any:
        """Execute vision function with specified client using BAML."""
        import json
        import tempfile
        import baml_py
        from PIL import Image

        logger.info(f"Executing {function.name} with {client_name} on {len(images)} images")

        # Validate inputs first
        is_valid, issues = self.validate_image_inputs(images, client_name)
        if not is_valid:
            raise ValueError(f"Invalid image inputs: {'; '.join(issues)}")

        try:
            # Import the generated BAML client
            from baml_client import b

            # Build extraction prompt from function definition
            extraction_prompt = self._build_extraction_prompt(function)

            # Process each image and collect results
            all_results = []
            for i, img in enumerate(images):
                logger.debug(f"Processing image {i+1}/{len(images)}")

                # Convert image to BAML image format
                baml_image = self._convert_to_baml_image(img)

                # Call the BAML extraction function
                try:
                    result = b.ExtractDocumentFields(
                        document_image=baml_image,
                        extraction_prompt=extraction_prompt
                    )
                    # Parse JSON result if it's a string
                    if isinstance(result, str):
                        try:
                            parsed_result = json.loads(result)
                        except json.JSONDecodeError:
                            parsed_result = {"raw_text": result}
                    else:
                        parsed_result = result

                    all_results.append(parsed_result)
                    logger.debug(f"Extraction result for image {i+1}: {type(parsed_result)}")

                except Exception as e:
                    logger.warning(f"Extraction failed for image {i+1}: {e}")
                    all_results.append({"error": str(e)})

            # Combine results
            if len(all_results) == 1:
                return all_results[0]
            else:
                return {"pages": all_results, "total_pages": len(all_results)}

        except ImportError as e:
            logger.error(f"BAML client not available: {e}")
            raise RuntimeError(f"BAML client not generated. Run 'baml generate' first: {e}")
        except Exception as e:
            logger.error(f"Vision function execution failed: {e}")
            raise

    def _build_extraction_prompt(self, function: BAMLFunction) -> str:
        """Build extraction prompt from BAML function definition."""
        prompt_parts = []

        # Add function description if available
        if function.description:
            prompt_parts.append(f"Task: {function.description}")

        # Add return type information
        if function.return_type:
            prompt_parts.append(f"\nExpected output structure: {function.return_type}")

        # Add prompt template if available
        if function.prompt_template:
            prompt_parts.append(f"\nInstructions:\n{function.prompt_template}")
        else:
            # Default extraction instructions
            prompt_parts.append("""
Extract all relevant information from this document image.
Return the data as a JSON object with appropriate field names and values.
If information is unclear or not present, indicate with null or "NOT_FOUND".
""")

        return "\n".join(prompt_parts)

    def _convert_to_baml_image(self, img: Any) -> "baml_py.Image":
        """Convert various image formats to BAML image."""
        import baml_py
        import tempfile
        import base64
        import io
        from PIL import Image

        # If it's already a BAML image, return as-is
        if isinstance(img, baml_py.Image):
            return img

        # If it's a file path string
        if isinstance(img, str):
            if img.startswith(('http://', 'https://')):
                return baml_py.Image.from_url(img)
            else:
                return baml_py.Image.from_file(img)

        # If it's a PIL Image
        if isinstance(img, Image.Image):
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return baml_py.Image.from_base64("image/png", img_base64)

        # If it's bytes
        if isinstance(img, bytes):
            img_base64 = base64.b64encode(img).decode('utf-8')
            return baml_py.Image.from_base64("image/png", img_base64)

        raise ValueError(f"Unsupported image type: {type(img)}")

    def _estimate_processing_cost(self, client_name: str, images: List[Any]) -> float:
        """Estimate processing cost for client and images."""
        spec = self.vision_specs.get(client_name)
        if not spec:
            return 0.0

        # Rough estimation: base cost + vision multiplier for each image
        estimated_tokens = 1000  # Rough estimate
        base_cost = (estimated_tokens / 1000) * spec.cost_per_1k_tokens
        vision_cost = base_cost * spec.vision_cost_multiplier * len(images)

        return base_cost + vision_cost

    def _validate_single_image(
        self,
        image: Any,
        spec: VisionModelSpec,
        config: ImageProcessingConfig
    ) -> List[str]:
        """Validate a single image against specifications."""
        issues = []

        # Basic validation - in real implementation, would check actual image properties
        # For now, just validate basic requirements

        if image is None:
            issues.append("Image is None")
            return issues

        # Check if it's a supported format (simplified check)
        if isinstance(image, str):
            # Assume string is file path or base64
            if image.startswith('data:image/'):
                # Base64 image
                format_part = image.split(';')[0].split('/')[1]
                if format_part not in spec.supported_formats:
                    issues.append(f"Unsupported format: {format_part}")
            elif Path(image).suffix.lower().replace('.', '') not in spec.supported_formats:
                issues.append(f"Unsupported file format: {Path(image).suffix}")

        return issues

    def _find_best_value_option(self, recommendations: List[Dict], complexity: ClientComplexity) -> Optional[Dict]:
        """Find best value option balancing cost and capability."""
        suitable_recs = [r for r in recommendations if r["suitable_for_complexity"]]
        if not suitable_recs:
            return None

        # Score based on cost, capability, and document optimization
        def calculate_value_score(rec):
            cost_score = 1.0 / (rec["estimated_cost"] + 0.001)  # Lower cost = higher score
            capability_bonus = 1.2 if rec["optimal_for_documents"] else 1.0
            return cost_score * capability_bonus

        suitable_recs.sort(key=calculate_value_score, reverse=True)
        return suitable_recs[0]