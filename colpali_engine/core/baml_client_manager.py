"""
BAML Client Manager - COLPALI-504 implementation.

This module manages integration with existing BAML client configurations
while supporting dynamic function generation and maintaining backward compatibility.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .baml_function_generator import ClientConfiguration, ClientComplexity
from .schema_manager import BAMLDefinition, BAMLFunction

logger = logging.getLogger(__name__)


class ClientType(Enum):
    """BAML client types based on capability."""
    TEXT_ONLY = "text_only"
    VISION_CAPABLE = "vision_capable"
    MULTIMODAL = "multimodal"


class RetryPolicy(Enum):
    """Available retry policies from BAML configuration."""
    EXPONENTIAL = "Exponential"
    CONSTANT = "Constant"
    NONE = "None"


@dataclass
class BAMLClient:
    """Represents a BAML client configuration."""
    name: str
    provider: str
    model: str
    api_key_env: str
    retry_policy: Optional[RetryPolicy] = None
    client_type: ClientType = ClientType.TEXT_ONLY
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    supports_vision: bool = False
    cost_tier: str = "medium"  # low, medium, high, premium
    performance_tier: str = "medium"  # fast, medium, slow, premium


@dataclass
class ClientMapping:
    """Maps complexity levels to appropriate clients."""
    simple: str
    moderate: str
    complex: str
    advanced: str


class BAMLClientManager:
    """
    Manages BAML client configurations and integrates with existing setup.

    Provides backward-compatible access to existing BAML clients while
    enabling dynamic function generation with appropriate client selection.
    """

    def __init__(self, baml_src_path: Optional[str] = None):
        self.baml_src_path = baml_src_path or self._find_baml_src_path()
        self.clients: Dict[str, BAMLClient] = {}
        self.client_mappings: Dict[str, ClientMapping] = {}
        self.available_clients: Set[str] = set()
        self.retry_policies: Dict[str, Dict[str, Any]] = {}

        self._load_existing_configuration()
        self._initialize_client_mappings()
        logger.info(f"BAMLClientManager initialized with {len(self.clients)} clients")

    def get_client_for_complexity(
        self,
        complexity: ClientComplexity,
        preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get the most appropriate client for the given complexity level.

        Args:
            complexity: Schema complexity level
            preferences: Optional user preferences

        Returns:
            Client name for the complexity level
        """
        # Check user preferences first
        if preferences and "preferred_client" in preferences:
            preferred = preferences["preferred_client"]
            if preferred in self.available_clients:
                logger.info(f"Using preferred client: {preferred}")
                return preferred

        # Use complexity-based mapping
        mapping = self.client_mappings["default"]

        if complexity == ClientComplexity.SIMPLE:
            return mapping.simple
        elif complexity == ClientComplexity.MODERATE:
            return mapping.moderate
        elif complexity == ClientComplexity.COMPLEX:
            return mapping.complex
        else:  # ADVANCED
            return mapping.advanced

    def get_client_configuration(self, client_name: str) -> Optional[BAMLClient]:
        """Get detailed configuration for a specific client."""
        return self.clients.get(client_name)

    def get_vision_capable_clients(self) -> List[str]:
        """Get list of clients that support vision/image inputs."""
        return [
            name for name, client in self.clients.items()
            if client.supports_vision
        ]

    def get_retry_policy_for_client(self, client_name: str) -> Optional[str]:
        """Get the retry policy for a specific client."""
        client = self.clients.get(client_name)
        return client.retry_policy.value if client and client.retry_policy else None

    def validate_client_availability(self, client_name: str) -> bool:
        """Validate that a client is available and properly configured."""
        if client_name not in self.available_clients:
            logger.warning(f"Client '{client_name}' is not available")
            return False

        client = self.clients[client_name]

        # Check if API key environment variable is set
        if hasattr(os.environ, client.api_key_env.replace("env.", "")):
            api_key = os.getenv(client.api_key_env.replace("env.", ""))
            if not api_key:
                logger.warning(f"API key {client.api_key_env} not set for client {client_name}")
                return False

        return True

    def register_dynamic_function(
        self,
        function: BAMLFunction,
        namespace: Optional[str] = None
    ) -> str:
        """
        Register a dynamically generated function with the BAML system.

        Args:
            function: Generated BAML function
            namespace: Optional namespace to avoid conflicts

        Returns:
            Full function name with namespace
        """
        # Generate namespaced function name
        if namespace:
            full_name = f"{namespace}_{function.name}"
        else:
            full_name = function.name

        # Validate client exists
        if not self.validate_client_availability(function.client):
            logger.error(f"Cannot register function {full_name}: invalid client {function.client}")
            raise ValueError(f"Client {function.client} is not available or properly configured")

        # Log registration
        logger.info(f"Registered dynamic function: {full_name} -> {function.client}")

        return full_name

    def generate_baml_configuration_snippet(
        self,
        definition: BAMLDefinition,
        namespace: Optional[str] = None
    ) -> str:
        """
        Generate BAML configuration snippet for dynamic functions.

        This can be appended to existing BAML files or used in temporary configurations.
        """
        snippets = []

        if namespace:
            snippets.append(f"// Generated functions for namespace: {namespace}")
        else:
            snippets.append("// Dynamically generated functions")

        for function in definition.functions:
            # Validate client is available
            if function.client not in self.available_clients:
                logger.warning(f"Function {function.name} uses unavailable client {function.client}")

            # Generate function snippet
            function_name = f"{namespace}_{function.name}" if namespace else function.name
            snippet = self._generate_function_snippet(function, function_name)
            snippets.append(snippet)

        return "\n\n".join(snippets)

    def get_cost_optimization_suggestions(self, complexity: ClientComplexity) -> Dict[str, Any]:
        """Get cost optimization suggestions based on complexity."""
        suggestions = {
            "recommended_client": self.get_client_for_complexity(complexity),
            "alternatives": [],
            "cost_analysis": {}
        }

        # Get alternative clients for cost comparison
        for client_name, client in self.clients.items():
            if client.supports_vision:
                suggestions["alternatives"].append({
                    "name": client_name,
                    "cost_tier": client.cost_tier,
                    "performance_tier": client.performance_tier,
                    "suitable_for": self._get_client_suitability(client_name, complexity)
                })

        return suggestions

    def _load_existing_configuration(self) -> None:
        """Load existing BAML client configuration from clients.baml."""
        clients_file = Path(self.baml_src_path) / "clients.baml"

        if not clients_file.exists():
            logger.warning(f"No clients.baml found at {clients_file}")
            self._create_default_clients()
            return

        try:
            with open(clients_file, 'r') as f:
                content = f.read()

            # Parse existing clients (simplified parsing)
            self._parse_clients_file(content)
            logger.info(f"Loaded {len(self.clients)} clients from {clients_file}")

        except Exception as e:
            logger.error(f"Failed to load BAML configuration: {e}")
            self._create_default_clients()

    def _parse_clients_file(self, content: str) -> None:
        """Parse the clients.baml file to extract client configurations."""
        lines = content.split('\n')
        current_client = None
        current_provider = None
        current_model = None
        current_api_key = None
        current_retry_policy = None

        for line in lines:
            line = line.strip()

            # Parse client declaration
            if line.startswith('client<llm>'):
                if current_client:
                    # Save previous client
                    self._add_parsed_client(
                        current_client, current_provider, current_model,
                        current_api_key, current_retry_policy
                    )
                # Extract client name
                current_client = line.split()[1].strip()
                current_retry_policy = None

            elif line.startswith('retry_policy'):
                current_retry_policy = line.split()[1].strip()

            elif line.startswith('provider'):
                current_provider = line.split()[1].strip()

            elif 'model' in line and '"' in line:
                # Extract model name from options
                current_model = line.split('"')[1]

            elif 'api_key' in line and 'env.' in line:
                # Extract API key environment variable
                current_api_key = line.split('env.')[1].strip()

        # Add the last client
        if current_client:
            self._add_parsed_client(
                current_client, current_provider, current_model,
                current_api_key, current_retry_policy
            )

    def _add_parsed_client(
        self,
        name: str,
        provider: str,
        model: str,
        api_key_env: str,
        retry_policy: Optional[str]
    ) -> None:
        """Add a parsed client to the configuration."""
        if not all([name, provider, model, api_key_env]):
            logger.warning(f"Incomplete client configuration for {name}")
            return

        # Determine client capabilities
        supports_vision = self._model_supports_vision(model, provider)
        client_type = ClientType.VISION_CAPABLE if supports_vision else ClientType.TEXT_ONLY

        # Determine cost and performance tiers
        cost_tier = self._determine_cost_tier(model, provider)
        performance_tier = self._determine_performance_tier(model, provider)

        # Parse retry policy
        parsed_retry_policy = None
        if retry_policy:
            try:
                parsed_retry_policy = RetryPolicy(retry_policy)
            except ValueError:
                logger.warning(f"Unknown retry policy: {retry_policy}")

        client = BAMLClient(
            name=name,
            provider=provider,
            model=model,
            api_key_env=f"env.{api_key_env}",
            retry_policy=parsed_retry_policy,
            client_type=client_type,
            supports_vision=supports_vision,
            cost_tier=cost_tier,
            performance_tier=performance_tier
        )

        self.clients[name] = client
        self.available_clients.add(name)

        logger.debug(f"Added client: {name} ({model}, vision: {supports_vision})")

    def _model_supports_vision(self, model: str, provider: str) -> bool:
        """Determine if a model supports vision inputs."""
        vision_models = {
            "gpt-5", "gpt-5-mini", "gpt-4v", "gpt-4-vision-preview",
            "claude-opus-4", "claude-sonnet-4", "claude-3-5-sonnet",
            "claude-opus-4-1-20250805", "claude-sonnet-4-20250514"
        }

        # Check if model name contains vision indicators
        model_lower = model.lower()
        return any(vm in model_lower for vm in vision_models)

    def _determine_cost_tier(self, model: str, provider: str) -> str:
        """Determine cost tier based on model and provider."""
        model_lower = model.lower()

        if "gpt-5" in model_lower:
            return "premium"
        elif "opus-4" in model_lower:
            return "high"
        elif "sonnet-4" in model_lower:
            return "medium"
        elif "haiku" in model_lower or "mini" in model_lower:
            return "low"
        else:
            return "medium"

    def _determine_performance_tier(self, model: str, provider: str) -> str:
        """Determine performance tier based on model capabilities."""
        model_lower = model.lower()

        if "opus-4" in model_lower or "gpt-5" in model_lower:
            return "premium"
        elif "sonnet-4" in model_lower:
            return "medium"
        elif "haiku" in model_lower or "mini" in model_lower:
            return "fast"
        else:
            return "medium"

    def _initialize_client_mappings(self) -> None:
        """Initialize client mappings based on available clients."""
        # Create default mapping based on available clients
        vision_clients = self.get_vision_capable_clients()

        if not vision_clients:
            logger.warning("No vision-capable clients found, using text-only clients")
            all_clients = list(self.available_clients)
            vision_clients = all_clients

        # Sort clients by performance/cost for mapping
        sorted_clients = self._sort_clients_by_capability()

        if len(sorted_clients) >= 4:
            mapping = ClientMapping(
                simple=sorted_clients[0],     # Cheapest/fastest
                moderate=sorted_clients[1],   # Balanced
                complex=sorted_clients[2],    # High performance
                advanced=sorted_clients[3]    # Premium
            )
        elif len(sorted_clients) >= 2:
            mapping = ClientMapping(
                simple=sorted_clients[0],
                moderate=sorted_clients[1],
                complex=sorted_clients[-1],
                advanced=sorted_clients[-1]
            )
        else:
            # Fallback to single client
            default_client = sorted_clients[0] if sorted_clients else "CustomSonnet4"
            mapping = ClientMapping(
                simple=default_client,
                moderate=default_client,
                complex=default_client,
                advanced=default_client
            )

        self.client_mappings["default"] = mapping
        logger.info(f"Client mapping: {mapping}")

    def _sort_clients_by_capability(self) -> List[str]:
        """Sort clients by capability (cost/performance)."""
        client_order = []

        # Preferred order based on typical usage patterns
        preferred_order = [
            "CustomHaiku",      # Fast, cheap
            "CustomSonnet4",    # Balanced
            "CustomOpus4",      # High performance
            "CustomGPT5",       # Premium
            "CustomGPT5Mini"    # Alternative fast option
        ]

        # Add clients in preferred order if they exist
        for client_name in preferred_order:
            if client_name in self.available_clients:
                client_order.append(client_name)

        # Add any remaining clients
        for client_name in self.available_clients:
            if client_name not in client_order:
                client_order.append(client_name)

        return client_order

    def _create_default_clients(self) -> None:
        """Create default client configurations if none exist."""
        logger.info("Creating default client configurations")

        defaults = [
            ("CustomSonnet4", "anthropic", "claude-sonnet-4-20250514", "ANTHROPIC_API_KEY", "Exponential"),
            ("CustomHaiku", "anthropic", "claude-3-5-haiku-20241022", "ANTHROPIC_API_KEY", "Constant"),
            ("CustomGPT5", "openai", "gpt-5", "OPENAI_API_KEY", "Exponential")
        ]

        for name, provider, model, api_key_env, retry_policy in defaults:
            self._add_parsed_client(name, provider, model, api_key_env, retry_policy)

    def _find_baml_src_path(self) -> str:
        """Find the baml_src directory path."""
        current_dir = Path.cwd()

        # Look for baml_src in current and parent directories
        for path in [current_dir, current_dir.parent]:
            baml_src = path / "baml_src"
            if baml_src.exists():
                return str(baml_src)

        # Default path
        return str(current_dir / "baml_src")

    def _generate_function_snippet(self, function: BAMLFunction, function_name: str) -> str:
        """Generate BAML function snippet."""
        input_params = []
        for param_dict in function.input_params:
            for param_name, param_type in param_dict.items():
                input_params.append(f"{param_name} {param_type}")

        params_str = ", ".join(input_params)

        return f"""function {function_name}({params_str}) -> {function.return_type} {{
    client {function.client}
    prompt #"
{function.prompt_template}
    "#
}}"""

    def _get_client_suitability(self, client_name: str, complexity: ClientComplexity) -> List[str]:
        """Get suitability assessment for client and complexity."""
        client = self.clients[client_name]
        suitability = []

        if complexity == ClientComplexity.SIMPLE and client.cost_tier == "low":
            suitability.append("cost-effective")
        if complexity == ClientComplexity.ADVANCED and client.performance_tier == "premium":
            suitability.append("high-accuracy")
        if client.supports_vision:
            suitability.append("vision-capable")

        return suitability or ["general-purpose"]