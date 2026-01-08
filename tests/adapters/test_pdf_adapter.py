#!/usr/bin/env python3
"""
Test script for PDF adapter functionality.

Tests the PDFAdapter implementation against all 15 sample PDFs,
validating conversion quality, memory usage, and error handling.
"""

import asyncio
import logging
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any
import io

from colpali_engine.adapters.pdf_adapter import PDFAdapter, create_pdf_adapter
from colpali_engine.core.document_adapter import ConversionConfig, DocumentMetadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFAdapterTester:
    """Comprehensive testing suite for PDF adapter."""

    def __init__(self, pdf_directory: str = "pdfs"):
        """Initialize tester with PDF directory path."""
        self.pdf_directory = Path(pdf_directory)
        self.adapter = create_pdf_adapter(max_memory_mb=300)
        self.test_results: List[Dict[str, Any]] = []

    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive test suite.

        Returns:
            Dictionary with test results and summary
        """
        logger.info("Starting PDF adapter test suite")
        start_time = time.time()

        # Get all PDF files
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to test")

        if len(pdf_files) == 0:
            logger.error("No PDF files found in directory")
            return {"error": "No test files found"}

        # Test each PDF
        for pdf_file in pdf_files:
            await self._test_single_pdf(pdf_file)

        # Generate summary
        total_time = time.time() - start_time
        summary = self._generate_summary(total_time)

        logger.info(f"Test suite completed in {total_time:.2f} seconds")
        return summary

    async def _test_single_pdf(self, pdf_path: Path) -> None:
        """Test a single PDF file comprehensively."""
        logger.info(f"Testing: {pdf_path.name}")

        try:
            # Read file content
            with open(pdf_path, 'rb') as f:
                content = f.read()

            test_result = {
                'filename': pdf_path.name,
                'file_size_mb': len(content) / (1024 * 1024),
                'success': False,
                'errors': [],
                'warnings': [],
                'metadata': None,
                'conversion_results': {}
            }

            # Test 1: Format validation
            logger.debug(f"  Testing format validation...")
            try:
                is_valid = self.adapter.validate_format(content)
                if not is_valid:
                    test_result['errors'].append("Format validation failed")
                else:
                    logger.debug("  ✓ Format validation passed")
            except Exception as e:
                test_result['errors'].append(f"Format validation error: {str(e)}")

            # Test 2: Metadata extraction
            logger.debug(f"  Testing metadata extraction...")
            try:
                metadata = self.adapter.extract_metadata(content)
                test_result['metadata'] = {
                    'page_count': metadata.page_count,
                    'dimensions': metadata.dimensions,
                    'file_size': metadata.file_size,
                    'title': metadata.title,
                    'author': metadata.author
                }
                logger.debug(f"  ✓ Metadata: {metadata.page_count} pages")
            except Exception as e:
                test_result['errors'].append(f"Metadata extraction error: {str(e)}")

            # Test 3: Image conversion (multiple configurations)
            conversion_configs = [
                ('standard_300dpi', ConversionConfig(dpi=200, quality=95)),
                ('high_quality', ConversionConfig(dpi=300, quality=100, max_pages=2)),
                ('memory_optimized', ConversionConfig(dpi=150, quality=75, max_pages=1))
            ]

            for config_name, config in conversion_configs:
                logger.debug(f"  Testing conversion: {config_name}")
                try:
                    start_time = time.time()

                    frames = await self.adapter.convert_to_frames(content, config)

                    conversion_time = time.time() - start_time

                    # Analyze results
                    result_info = {
                        'frame_count': len(frames),
                        'conversion_time_seconds': conversion_time,
                        'config': {
                            'dpi': config.dpi,
                            'quality': config.quality,
                            'max_pages': config.max_pages
                        }
                    }

                    # Analyze frame properties
                    if frames:
                        first_frame = frames[0]
                        result_info['frame_info'] = {
                            'width': first_frame.width,
                            'height': first_frame.height,
                            'mode': first_frame.mode,
                            'format': first_frame.format
                        }

                        # Estimate memory usage
                        frame_size_mb = (first_frame.width * first_frame.height * 3) / (1024 * 1024)
                        result_info['estimated_memory_mb'] = frame_size_mb * len(frames)

                    test_result['conversion_results'][config_name] = result_info
                    logger.debug(f"  ✓ {config_name}: {len(frames)} frames in {conversion_time:.2f}s")

                except Exception as e:
                    test_result['errors'].append(f"{config_name} conversion error: {str(e)}")
                    logger.debug(f"  ✗ {config_name}: {str(e)}")

            # Determine overall success
            if not test_result['errors'] and test_result['conversion_results']:
                test_result['success'] = True

            self.test_results.append(test_result)

            status_symbol = '✓' if test_result['success'] else '✗'
            status_text = 'Success' if test_result['success'] else f"{len(test_result['errors'])} errors"
            logger.info(f"  {status_symbol} {pdf_path.name}: {status_text}")

        except Exception as e:
            logger.error(f"  ✗ {pdf_path.name}: Critical error - {str(e)}")
            self.test_results.append({
                'filename': pdf_path.name,
                'success': False,
                'errors': [f"Critical error: {str(e)}"],
                'metadata': None,
                'conversion_results': {}
            })

    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate test summary statistics."""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - successful_tests

        # Collect statistics
        total_pages = sum(
            result['metadata']['page_count']
            for result in self.test_results
            if result['metadata'] and 'page_count' in result['metadata']
        )

        total_file_size = sum(
            result['file_size_mb']
            for result in self.test_results
            if 'file_size_mb' in result
        )

        # Find performance statistics
        conversion_times = []
        for result in self.test_results:
            if result['conversion_results']:
                for config, conv_result in result['conversion_results'].items():
                    if 'conversion_time_seconds' in conv_result:
                        conversion_times.append(conv_result['conversion_time_seconds'])

        avg_conversion_time = sum(conversion_times) / len(conversion_times) if conversion_times else 0

        # Error analysis
        all_errors = []
        for result in self.test_results:
            all_errors.extend(result.get('errors', []))

        summary = {
            'test_summary': {
                'total_files': total_tests,
                'successful': successful_tests,
                'failed': failed_tests,
                'success_rate_percent': (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                'total_execution_time_seconds': total_time
            },
            'document_statistics': {
                'total_pages_processed': total_pages,
                'total_file_size_mb': total_file_size,
                'average_conversion_time_seconds': avg_conversion_time
            },
            'error_analysis': {
                'total_errors': len(all_errors),
                'unique_error_types': len(set(all_errors)),
                'common_errors': all_errors
            },
            'detailed_results': self.test_results
        }

        return summary

    def print_summary(self, summary: Dict[str, Any]) -> None:
        """Print human-readable test summary."""
        print("\n" + "="*70)
        print("PDF ADAPTER TEST SUMMARY")
        print("="*70)

        test_sum = summary['test_summary']
        print(f"Files Tested: {test_sum['total_files']}")
        print(f"Successful: {test_sum['successful']} ✓")
        print(f"Failed: {test_sum['failed']} ✗")
        print(f"Success Rate: {test_sum['success_rate_percent']:.1f}%")
        print(f"Total Time: {test_sum['total_execution_time_seconds']:.2f}s")

        doc_stats = summary['document_statistics']
        print(f"\nDocument Statistics:")
        print(f"  Total Pages: {doc_stats['total_pages_processed']}")
        print(f"  Total Size: {doc_stats['total_file_size_mb']:.1f} MB")
        print(f"  Avg Conversion Time: {doc_stats['average_conversion_time_seconds']:.2f}s per file")

        if summary['error_analysis']['total_errors'] > 0:
            print(f"\nErrors Found: {summary['error_analysis']['total_errors']}")
            print("Most Common Errors:")
            for error in set(summary['error_analysis']['common_errors'][:5]):
                print(f"  - {error}")

        print("\nDetailed Results:")
        for result in summary['detailed_results']:
            status = "✓" if result['success'] else "✗"
            pages = result['metadata']['page_count'] if result['metadata'] else "?"
            print(f"  {status} {result['filename']}: {pages} pages")

        print("="*70)


async def main():
    """Main test execution."""
    try:
        tester = PDFAdapterTester()
        summary = await tester.run_all_tests()

        # Print results
        tester.print_summary(summary)

        # Return success if most tests passed
        success_rate = summary['test_summary']['success_rate_percent']
        if success_rate >= 80:
            logger.info(f"✓ Test suite PASSED with {success_rate:.1f}% success rate")
            return 0
        else:
            logger.error(f"✗ Test suite FAILED with {success_rate:.1f}% success rate")
            return 1

    except Exception as e:
        logger.error(f"Test suite failed with critical error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)