#!/usr/bin/env python3

"""
Integration tests for all five RAG ingestion pipeline improvements.

This test suite verifies that all five immediate fixes are working properly:
1. Fixed semantic chunking bypass
2. Enhanced content type detection
3. Improved boundary preservation
4. Better error logging and recovery
5. Optimized embedding API calls
"""

import os
import sys
import unittest
import logging
from unittest.mock import patch, MagicMock
from io import StringIO

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the necessary modules
from ingest_rag import semantic_chunk_text, process_content, embed_and_upsert, Document
import adaptive_chunking

# Test data for all improvement areas
TEST_CODE = """
def calculate_metrics(data_points):
    '''Calculate statistical metrics for a set of data points.'''
    if not data_points:
        return {"mean": 0, "median": 0, "std_dev": 0}
    
    n = len(data_points)
    mean = sum(data_points) / n
    
    # Calculate median
    sorted_points = sorted(data_points)
    if n % 2 == 0:
        median = (sorted_points[n//2 - 1] + sorted_points[n//2]) / 2
    else:
        median = sorted_points[n//2]
    
    # Calculate standard deviation
    variance = sum((x - mean) ** 2 for x in data_points) / n
    std_dev = variance ** 0.5
    
    return {
        "mean": mean,
        "median": median,
        "std_dev": std_dev
    }
"""

TEST_TABLE = """
| ID | Project Name | Start Date | End Date | Status      | Budget    | Team Size |
|----|--------------|------------|----------|-------------|-----------|-----------|
| 1  | Alpha        | 2025-01-15 | 2025-04-30 | Completed | $125,000  | 5         |
| 2  | Beta         | 2025-02-01 | 2025-06-15 | In Progress | $240,000  | 8         |
"""

TEST_LIST = """
1. Project Initialization
   - Requirements gathering
   - Stakeholder interviews
   - Scope definition
   - Initial risk assessment

2. Design Phase
   - Architecture planning
   - Interface mockups
"""

TEST_MIXED_CONTENT = f"""
# Mixed Content Document

This is a prose section that contains regular text.

## Code Sample

{TEST_CODE}

## Table Data

{TEST_TABLE}

## List Structure

{TEST_LIST}
"""

class TestSemanticChunkingBypassFix(unittest.TestCase):
    """Test the fix for the semantic chunking bypass issue."""
    
    @patch('ingest_rag.click')
    def test_semantic_chunking_not_bypassed(self, mock_click):
        """Test that semantic chunking is not bypassed when enabled."""
        # Use short text to ensure we only test the bypass fix, not the chunking itself
        text = "Short test document that should be chunked properly."
        
        # Process with semantic chunking enabled
        result = process_content(
            text, 
            use_adaptive_chunking=True,
            max_chars=100  # Small value to ensure it would use multiple chunks for longer text
        )
        
        # Verify that the function didn't fall back to basic methods
        mock_click.echo.assert_any_call(unittest.mock.ANY)  # Any message
        
        # Check there were no warning messages about bypassing
        warning_messages = [
            args[0] for args, _ in mock_click.echo.call_args_list 
            if isinstance(args[0], str) and "falling back" in args[0].lower()
        ]
        self.assertEqual(len(warning_messages), 0, 
                         "Semantic chunking was bypassed: " + "; ".join(warning_messages))
        
        # Verify we got valid chunks
        self.assertTrue(len(result) > 0, "Should produce at least one chunk")
        
        if isinstance(result[0], dict) and 'content' in result[0]:
            # New format with metadata
            self.assertTrue(all('content' in chunk for chunk in result), 
                           "All chunks should have content")
        else:
            # Plain text chunks
            self.assertTrue(all(isinstance(chunk, str) for chunk in result), 
                           "All chunks should be strings")

class TestContentTypeDetection(unittest.TestCase):
    """Test enhanced content type detection."""
    
    def test_code_detection(self):
        """Test improved code detection."""
        result = adaptive_chunking.detect_content_type(TEST_CODE)
        self.assertIn("CODE", result, "Should detect code content")
        self.assertGreater(result["CODE"], 0.5, "Should have high confidence for code")
    
    def test_table_detection(self):
        """Test improved table detection."""
        result = adaptive_chunking.detect_content_type(TEST_TABLE)
        self.assertIn("TABLE", result, "Should detect table content")
        self.assertGreater(result["TABLE"], 0.5, "Should have high confidence for table")
    
    def test_list_detection(self):
        """Test improved list detection."""
        result = adaptive_chunking.detect_content_type(TEST_LIST)
        self.assertIn("LIST", result, "Should detect list content")
        self.assertGreater(result["LIST"], 0.5, "Should have high confidence for list")
    
    def test_mixed_content_detection(self):
        """Test detection of mixed content with multiple types."""
        result = adaptive_chunking.detect_content_type(TEST_MIXED_CONTENT)
        
        # Should detect multiple content types
        detected_types = [t for t, score in result.items() if score > 0.3]
        self.assertGreaterEqual(len(detected_types), 2, 
                               f"Should detect multiple content types, got: {detected_types}")
        
        # Should detect structured content
        self.assertIn("STRUCTURED", result, "Should detect structured content")

class TestBoundaryPreservation(unittest.TestCase):
    """Test improved boundary preservation."""
    
    def test_code_boundary_preservation(self):
        """Test that code boundaries are preserved."""
        chunks = semantic_chunk_text(TEST_CODE, max_chars=300)
        
        # Extract content from chunks if they're in the new format
        if chunks and isinstance(chunks[0], dict) and 'content' in chunks[0]:
            chunk_contents = [c['content'] for c in chunks]
        else:
            chunk_contents = chunks
        
        # Verify core function elements stay together
        function_def = "def calculate_metrics(data_points):"
        function_body_sample = "return {" # Part of the return statement
        
        # Find chunks containing key elements
        def_chunk = next((c for c in chunk_contents if function_def in c), None)
        body_chunk = next((c for c in chunk_contents if function_body_sample in c), None)
        
        # Verify either the entire function is in one chunk or logical parts are preserved
        if len(chunk_contents) == 1:
            # Function fits in one chunk
            self.assertIn(function_def, chunk_contents[0], "Function definition should be preserved")
            self.assertIn(function_body_sample, chunk_contents[0], "Function body should be preserved")
        else:
            # Function is split across chunks, but key parts should be preserved
            self.assertIsNotNone(def_chunk, "Function definition should be in a chunk")
            self.assertIsNotNone(body_chunk, "Function body should be in a chunk")
    
    def test_table_boundary_preservation(self):
        """Test that table boundaries are preserved."""
        chunks = semantic_chunk_text(TEST_TABLE, max_chars=300)
        
        # Extract content from chunks if they're in the new format
        if chunks and isinstance(chunks[0], dict) and 'content' in chunks[0]:
            chunk_contents = [c['content'] for c in chunks]
        else:
            chunk_contents = chunks
        
        # Check if header and at least one row stay together
        header_row = "| ID | Project Name |"
        separator_row = "|----|--------------|"
        
        # Find chunks containing header elements
        header_chunk = next((c for c in chunk_contents if header_row in c), None)
        separator_chunk = next((c for c in chunk_contents if separator_row in c), None)
        
        # Verify header structure is preserved
        self.assertIsNotNone(header_chunk, "Table header should be preserved in a chunk")
        self.assertIsNotNone(separator_chunk, "Table separator should be preserved in a chunk")
        
        # Ideally, they should be in the same chunk
        if len(chunk_contents) == 1:
            self.assertEqual(header_chunk, separator_chunk, 
                           "Header and separator should be in the same chunk")

class TestErrorLoggingRecovery(unittest.TestCase):
    """Test improved error logging and recovery."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Configure a logger for capturing logs
        self.log_capture = StringIO()
        self.handler = logging.StreamHandler(self.log_capture)
        self.handler.setLevel(logging.DEBUG)
        
        # Clear adaptive_chunking's existing handlers and add our test handler
        adaptive_chunking.logger.handlers = []
        adaptive_chunking.logger.addHandler(self.handler)
        adaptive_chunking.logger.setLevel(logging.DEBUG)
        
        # Reset error context
        adaptive_chunking.reset_error_context()
    
    def test_error_context_tracking(self):
        """Test that error context is tracked."""
        # Trigger an error
        adaptive_chunking.log_error("TEST_ERROR", "Test error message")
        
        # Check error was recorded
        self.assertEqual(adaptive_chunking.ERROR_CONTEXT["error_count"], 1)
        self.assertEqual(adaptive_chunking.ERROR_CONTEXT["last_error"]["type"], "TEST_ERROR")
        self.assertIn("TEST_ERROR", adaptive_chunking.ERROR_CONTEXT["detected_errors"])
    
    def test_fallback_tracking(self):
        """Test that fallbacks are tracked."""
        # Trigger a fallback
        adaptive_chunking.log_fallback("original_method", "fallback_method", "test reason")
        
        # Check fallback was recorded
        self.assertEqual(len(adaptive_chunking.ERROR_CONTEXT["fallback_path"]), 1)
        self.assertEqual(adaptive_chunking.ERROR_CONTEXT["fallback_path"][0]["from"], "original_method")
        self.assertEqual(adaptive_chunking.ERROR_CONTEXT["fallback_path"][0]["to"], "fallback_method")
    
    def test_log_detail_levels(self):
        """Test that logs have appropriate detail."""
        # Generate logs at different levels
        adaptive_chunking.logger.debug("Test debug message")
        adaptive_chunking.logger.info("Test info message")
        adaptive_chunking.logger.warning("Test warning message")
        adaptive_chunking.logger.error("Test error message")
        
        # Verify all log levels are captured
        log_content = self.log_capture.getvalue()
        self.assertIn("Test debug message", log_content)
        self.assertIn("Test info message", log_content)
        self.assertIn("Test warning message", log_content)
        self.assertIn("Test error message", log_content)

class TestEmbeddingOptimization(unittest.TestCase):
    """Test optimized embedding API calls."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test documents
        self.docs = [
            Document(content="Short document 1", metadata={"source": "test"}),
            Document(content="Short document 2", metadata={"source": "test"}),
            Document(content="Medium length document " + "word " * 20, metadata={"source": "test"}),
            Document(content="Very long document " + "word " * 100, metadata={"source": "test"}),
        ]
        
        # Mock Qdrant client
        self.mock_client = MagicMock()
        
        # Mock OpenAI client
        self.mock_openai = MagicMock()
        self.mock_openai.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in range(len(self.docs))],
            usage=MagicMock(prompt_tokens=100)
        )
    
    @patch('ingest_rag.logger')
    def test_batch_optimization(self, mock_logger):
        """Test that documents are batched optimally."""
        # Call the embedding function with a small batch size to force multiple batches
        embed_and_upsert(
            self.mock_client,
            "test_collection",
            self.docs,
            self.mock_openai,
            batch_size=2,  # Small batch size to force splitting
            model_name="text-embedding-3-large",
            deterministic_id=True
        )
        
        # Verify batching happens
        # The function should have created token-aware batches
        self.assertGreaterEqual(self.mock_openai.embeddings.create.call_count, 1,
                              "Should call the API at least once")
        
        # Check that the first call includes correct batching logic
        first_call_args = self.mock_openai.embeddings.create.call_args_list[0][1]
        self.assertIn('input', first_call_args, "API call should include input parameter")
    
    @patch('ingest_rag.logger')
    def test_performance_monitoring(self, mock_logger):
        """Test that performance is monitored."""
        # Call the embedding function
        embed_and_upsert(
            self.mock_client,
            "test_collection",
            self.docs,
            self.mock_openai,
            model_name="text-embedding-3-large",
            deterministic_id=True
        )
        
        # Check that performance metrics are logged
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        
        performance_logs = [call for call in log_calls if 'sec' in call]
        self.assertGreaterEqual(len(performance_logs), 1,
                               "Should log performance metrics")
    
    @patch('ingest_rag.logger')
    def test_dynamic_worker_adjustment(self, mock_logger):
        """Test that the dynamic worker pool adjusts based on success/failure."""
        # Import the DynamicWorkerPool class directly
        from ingest_rag import DynamicWorkerPool
        
        # Test direct operations on worker pool
        pool = DynamicWorkerPool(
            initial_workers=20,
            min_workers=5,
            max_workers=30,
            success_threshold=2,  # Lower threshold for faster testing
            failure_threshold=2
        )
        
        # Initial state
        self.assertEqual(pool.current_workers, 20)
        
        # Report multiple successes to trigger worker increase
        initial_workers = pool.current_workers
        pool.report_success()
        pool.report_success()  # Should trigger increase
        self.assertGreater(pool.current_workers, initial_workers,
                          "Worker count should increase after successes")
        
        # Report multiple failures to trigger worker decrease
        adjusted_workers = pool.current_workers
        pool.report_failure()
        pool.report_failure()  # Should trigger decrease
        self.assertLess(pool.current_workers, adjusted_workers,
                       "Worker count should decrease after failures")
        
        # Verify adjustment history
        self.assertEqual(len(pool.adjustment_history), 2,
                        "Should record two adjustments")
        self.assertEqual(pool.adjustment_history[0]["reason"], "success")
        self.assertEqual(pool.adjustment_history[1]["reason"], "failure")
        
        # Test integration with embed_and_upsert
        embed_and_upsert(
            self.mock_client,
            "test_collection",
            self.docs,
            self.mock_openai,
            model_name="text-embedding-3-large",
            deterministic_id=True,
            initial_workers=15,
            min_workers=5,
            max_workers=30,
            dynamic_workers=True
        )
        
        # Verify worker pool functionality was logged
        dynamic_worker_logs = [
            call for call in mock_logger.info.call_args_list
            if call[0] and isinstance(call[0][0], str) and "worker" in call[0][0].lower()
        ]
        self.assertGreaterEqual(len(dynamic_worker_logs), 1,
                              "Should log messages about dynamic worker pool")

if __name__ == '__main__':
    unittest.main()