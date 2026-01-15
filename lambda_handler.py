"""
AWS Lambda handler for ColPali-BAML Vision Processing Engine.

Implements COLPALI-903: Lambda handler and API interface.
This module provides the Lambda entry point for serverless document processing
with optimized cold start performance, memory management, and comprehensive
monitoring.
"""

import asyncio
import base64
import json
import logging
import os
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Lambda utilities
from tatforge.lambda_utils import (
    LambdaModelOptimizer,
    LambdaMonitor,
    LambdaResourceManager,
    StructuredLogger
)
from tatforge.lambda_utils.model_optimizer import OptimizationConfig
from tatforge.lambda_utils.resource_manager import MemoryThresholds

# Global instances for Lambda warm starts
_pipeline = None
_resource_manager = None
_monitor = None
_is_cold_start = True


def _get_lambda_context_info(context: Any) -> Dict[str, Any]:
    """Extract information from Lambda context."""
    if context is None:
        return {}

    return {
        "function_name": getattr(context, 'function_name', None),
        "function_version": getattr(context, 'function_version', None),
        "memory_limit_mb": getattr(context, 'memory_limit_in_mb', None),
        "remaining_time_ms": (
            context.get_remaining_time_in_millis()
            if hasattr(context, 'get_remaining_time_in_millis') else None
        ),
        "aws_request_id": getattr(context, 'aws_request_id', None),
        "log_group_name": getattr(context, 'log_group_name', None),
        "log_stream_name": getattr(context, 'log_stream_name', None)
    }


def _initialize_globals(context: Any = None) -> None:
    """Initialize global instances for Lambda warm starts."""
    global _pipeline, _resource_manager, _monitor, _is_cold_start

    if _monitor is None:
        _monitor = LambdaMonitor(
            service_name="colpali-baml-engine",
            version="0.1.0"
        )

    if _resource_manager is None:
        # Get memory limit from context or environment
        memory_limit = 10240  # Default 10GB
        if context and hasattr(context, 'memory_limit_in_mb'):
            memory_limit = context.memory_limit_in_mb

        thresholds = MemoryThresholds(
            max_usage_gb=memory_limit / 1024 * 0.8  # Use 80% of limit
        )
        _resource_manager = LambdaResourceManager(memory_thresholds=thresholds)
        _resource_manager.initialize()


async def _initialize_pipeline() -> Any:
    """Initialize the vision extraction pipeline."""
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    logger.info("Initializing vision extraction pipeline...")

    try:
        # Import pipeline components
        from tatforge import VisionExtractionPipeline
        from tatforge.core.pipeline import PipelineConfig
        from tatforge.vision.colpali_client import ColPaliClient
        from tatforge.storage.qdrant_client import QdrantManager
        from tatforge.extraction.baml_interface import BAMLExecutionInterface
        from tatforge.outputs.canonical import CanonicalFormatter
        from tatforge.outputs.shaped import ShapedFormatter

        # Initialize ColPali client with Lambda optimizations
        colpali_client = ColPaliClient(
            model_name=os.environ.get("COLPALI_MODEL", "vidore/colqwen2-v0.1"),
            device="cpu",  # Lambda uses CPU
            memory_limit_gb=8,  # Conservative for Lambda
            enable_prewarming=True,
            lazy_loading=True
        )

        # Load model with optimizations
        await colpali_client.load_model()

        # Initialize other components
        qdrant_manager = QdrantManager(
            host=os.environ.get("QDRANT_HOST", "localhost"),
            port=int(os.environ.get("QDRANT_PORT", "6333"))
        )

        baml_interface = BAMLExecutionInterface()
        canonical_formatter = CanonicalFormatter()
        shaped_formatter = ShapedFormatter()

        # Create pipeline configuration
        config = PipelineConfig(
            memory_limit_gb=8,
            batch_size="auto",
            enable_shaped_output=True,
            enforce_1nf=True
        )

        # Create pipeline
        _pipeline = VisionExtractionPipeline(
            colpali_client=colpali_client,
            qdrant_manager=qdrant_manager,
            baml_interface=baml_interface,
            canonical_formatter=canonical_formatter,
            shaped_formatter=shaped_formatter,
            config=config
        )

        logger.info("Vision extraction pipeline initialized successfully")
        return _pipeline

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Pipeline initialization failed: {e}")


def _parse_request(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse and validate the incoming request.

    Args:
        event: Lambda event data

    Returns:
        Parsed request body

    Raises:
        ValueError: If request is invalid
    """
    # Handle API Gateway event format
    if 'body' in event:
        body = event['body']
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in request body: {e}")
    else:
        body = event

    # Validate required fields
    required_fields = ['document_blob', 'schema_json']
    missing = [f for f in required_fields if f not in body]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    # Decode base64 document if needed
    document_blob = body['document_blob']
    if isinstance(document_blob, str):
        try:
            document_blob = base64.b64decode(document_blob)
        except Exception as e:
            raise ValueError(f"Invalid base64 document: {e}")

    # Validate schema
    schema_json = body['schema_json']
    if isinstance(schema_json, str):
        try:
            schema_json = json.loads(schema_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON schema: {e}")

    return {
        'document_blob': document_blob,
        'schema_json': schema_json,
        'options': body.get('options', {}),
        'metadata': body.get('metadata', {})
    }


def _create_response(
    status_code: int,
    body: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Create API Gateway compatible response.

    Args:
        status_code: HTTP status code
        body: Response body
        headers: Optional response headers

    Returns:
        API Gateway response format
    """
    default_headers = {
        'Content-Type': 'application/json',
        'X-Service': 'colpali-baml-engine',
        'X-Version': '0.1.0'
    }

    if headers:
        default_headers.update(headers)

    return {
        'statusCode': status_code,
        'headers': default_headers,
        'body': json.dumps(body, default=str)
    }


def _create_error_response(
    status_code: int,
    error: str,
    message: str,
    correlation_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create error response."""
    body = {
        'error': error,
        'message': message,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }

    if correlation_id:
        body['correlation_id'] = correlation_id

    return _create_response(status_code, body)


async def _process_document_async(
    request: Dict[str, Any],
    correlation_id: str
) -> Dict[str, Any]:
    """
    Process document asynchronously.

    Args:
        request: Parsed request
        correlation_id: Request correlation ID

    Returns:
        Processing result
    """
    global _pipeline

    # Initialize pipeline if needed
    if _pipeline is None:
        await _initialize_pipeline()

    # Process document
    result = await _pipeline.process_document(
        document_blob=request['document_blob'],
        schema_json=request['schema_json'],
        document_metadata=request.get('metadata')
    )

    # Format response
    response = {
        'processing_id': result.metadata.processing_id,
        'correlation_id': correlation_id,
        'status': 'success',
        'canonical': {
            'data': result.canonical.extraction_data if result.canonical else None,
            'integrity_hash': (
                result.canonical.integrity_hash
                if result.canonical and hasattr(result.canonical, 'integrity_hash')
                else None
            )
        },
        'metadata': {
            'processing_time_seconds': result.metadata.processing_time_seconds,
            'stages_completed': len(result.metadata.lineage_steps),
            'timestamp': result.metadata.end_time.isoformat() + 'Z'
        }
    }

    # Add shaped data if available
    if result.shaped:
        response['shaped'] = {
            'data': result.shaped.transformed_data,
            'transformations': result.shaped.transformations_applied
        }

    return response


def main(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for document processing requests.

    Implements COLPALI-903: Lambda handler and API interface.

    Args:
        event: Lambda event data containing:
            - document_blob: Base64 encoded document or raw bytes
            - schema_json: JSON schema defining extraction structure
            - options: Optional processing options
            - metadata: Optional document metadata
        context: Lambda context object

    Returns:
        API Gateway response with:
            - statusCode: HTTP status code
            - headers: Response headers
            - body: JSON response body
    """
    global _is_cold_start

    start_time = datetime.utcnow()
    correlation_id = None

    try:
        # Initialize global components
        _initialize_globals(context)

        # Get context information
        context_info = _get_lambda_context_info(context)
        correlation_id = context_info.get('aws_request_id') or str(
            __import__('uuid').uuid4()
        )

        # Log request start
        logger.info(
            f"Lambda handler started",
            extra={
                'correlation_id': correlation_id,
                'cold_start': _is_cold_start,
                **context_info
            }
        )

        # Record cold start metric
        if _monitor and _is_cold_start:
            _monitor.record_metric(
                "ColdStart",
                1,
                "Count",
                function=context_info.get('function_name', 'unknown')
            )
            _is_cold_start = False

        # Check for health check request
        if event.get('path') == '/health' or event.get('healthCheck'):
            return _create_response(200, health_check())

        # Parse and validate request
        try:
            request = _parse_request(event)
        except ValueError as e:
            logger.warning(f"Invalid request: {e}")
            return _create_error_response(400, 'BadRequest', str(e), correlation_id)

        # Check resource status
        if _resource_manager:
            resource_status = _resource_manager.check_resources()
            if resource_status['overall_status'] == 'critical':
                logger.error("Critical resource status, rejecting request")
                return _create_error_response(
                    503,
                    'ServiceUnavailable',
                    'System resources critical, please retry later',
                    correlation_id
                )

        # Process document with monitoring
        with _monitor.trace_operation(
            "process_document",
            correlation_id=correlation_id,
            document_size=len(request['document_blob']),
            schema_fields=len(request['schema_json'].get('properties', {}))
        ):
            # Run async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(
                    _process_document_async(request, correlation_id)
                )
            finally:
                loop.close()

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Record metrics
        if _monitor:
            _monitor.record_metric(
                "Latency",
                processing_time * 1000,
                "Milliseconds",
                operation="process_document"
            )

        logger.info(
            f"Lambda handler completed successfully",
            extra={
                'correlation_id': correlation_id,
                'processing_time_seconds': processing_time
            }
        )

        return _create_response(
            200,
            result,
            {'X-Correlation-ID': correlation_id}
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(
            f"Lambda handler error: {error_msg}",
            extra={
                'correlation_id': correlation_id,
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
        )

        # Record error metric
        if _monitor:
            _monitor.record_metric(
                "Errors",
                1,
                "Count",
                operation="process_document",
                error_type=type(e).__name__
            )

        # Attempt cleanup on error
        if _resource_manager:
            _resource_manager.request_cleanup(aggressive=True)

        return _create_error_response(
            500,
            'InternalServerError',
            f'Processing failed: {error_msg}',
            correlation_id
        )


def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for Lambda warmup and monitoring.

    Returns:
        Health status response
    """
    global _monitor, _resource_manager

    response = {
        'status': 'healthy',
        'service': 'colpali-baml-engine',
        'version': '0.1.0',
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }

    # Add monitor health if available
    if _monitor:
        monitor_health = _monitor.health_check()
        response['metrics'] = monitor_health.get('metrics', {})
        response['system'] = monitor_health.get('system', {})
        if monitor_health.get('status') != 'healthy':
            response['status'] = monitor_health['status']

    # Add resource status if available
    if _resource_manager:
        resource_status = _resource_manager.check_resources()
        response['resources'] = {
            'memory_status': resource_status['memory']['status'],
            'memory_percent': resource_status['memory']['usage_percent']
        }
        if resource_status['overall_status'] != 'ok':
            response['status'] = resource_status['overall_status']

    return response


def warmup(event: Dict[str, Any] = None, context: Any = None) -> Dict[str, Any]:
    """
    Lambda warmup handler for provisioned concurrency.

    Args:
        event: Lambda event (ignored)
        context: Lambda context

    Returns:
        Warmup status
    """
    global _is_cold_start

    logger.info("Running Lambda warmup")
    start_time = datetime.utcnow()

    try:
        # Initialize globals
        _initialize_globals(context)

        # Initialize pipeline (this loads the model)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(_initialize_pipeline())
        finally:
            loop.close()

        warmup_time = (datetime.utcnow() - start_time).total_seconds()
        _is_cold_start = False

        logger.info(f"Lambda warmup completed in {warmup_time:.2f}s")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'warmed',
                'warmup_time_seconds': warmup_time,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        }

    except Exception as e:
        logger.error(f"Lambda warmup failed: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'warmup_failed',
                'error': str(e)
            })
        }
