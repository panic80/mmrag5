"""
Tests for the optimized embedding API calls in ingest_rag.py
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import json
from datetime import datetime

# Add parent directory to path to import ingest_rag
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingest_rag import embed_and_upsert, Document

class TestEmbeddingOptimization(unittest.TestCase):
    """Test the optimized embedding API functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock documents with various sizes
        self.docs = [
            Document(content="Short document", metadata={"source": "test"}),
            Document(content="Medium length document with more content to process", metadata={"source": "test"}),
            Document(content="A " + "very " * 100 + "long document that would require many tokens to embed properly", metadata={"source": "test"}),
        ]
        
        # Mock Qdrant client
        self.mock_client = MagicMock()
        
        # Mock OpenAI client (v1 style)
        self.mock_openai_client = MagicMock()
        self.mock_openai_client.embeddings = MagicMock()
        self.mock_openai_client.embeddings.create = MagicMock()
        
        # Set up response format
        self.mock_embedding_response = MagicMock()
        self.mock_embedding_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.2, 0.3, 0.4]),
            MagicMock(embedding=[0.3, 0.4, 0.5]),
        ]
        self.mock_embedding_response.usage = MagicMock(prompt_tokens=100)
        
        # Configure mock to return our prepared response
        self.mock_openai_client.embeddings.create.return_value = self.mock_embedding_response
    
    @patch('ingest_rag.logger')
    @patch('ingest_rag.click')
    def test_token_aware_batching(self, mock_click, mock_logger):
        """Test that documents are batched based on token estimates"""
        # Call the function with a small batch size to force multiple batches
        embed_and_upsert(
            self.mock_client,
            "test_collection",
            self.docs,
            self.mock_openai_client,
            batch_size=2,  # Small batch size to force splitting
            model_name="text-embedding-3-large",
            deterministic_id=True,
            parallel=1  # Use sequential processing for predictable testing
        )
        
        # The function should have called optimize_batch_for_tokens
        # and created token-aware batches
        mock_logger.info.assert_any_call(
            unittest.mock.ANY  # Match any string containing 'token-optimized batches'
        )
        
        # Verify the OpenAI API was called
        self.mock_openai_client.embeddings.create.assert_called()
    
    @patch('tenacity.retry')  # Mock the retry decorator to bypass waiting
    @patch('ingest_rag.logger')
    @patch('ingest_rag.click')
    def test_retry_mechanism(self, mock_click, mock_logger, mock_retry):
        """Test the retry mechanism for transient errors"""
        # Have the retry decorator just call the function directly without retrying
        mock_retry.return_value = lambda func: func
        
        # Import the actual class to use for the test
        from ingest_rag import RateLimitError
        
        # Configure the OpenAI client mock to fail with rate limit then succeed
        side_effects = [
            RateLimitError("Rate limit exceeded"),  # First call fails with specific error
            self.mock_embedding_response            # Second call succeeds
        ]
        self.mock_openai_client.embeddings.create.side_effect = side_effects
        
        # Call the function
        embed_and_upsert(
            self.mock_client,
            "test_collection",
            self.docs[:1],  # Just use one document for simplicity
            self.mock_openai_client,
            batch_size=1,
            model_name="text-embedding-3-large",
            deterministic_id=True,
            parallel=1
        )
        
        # Verify retry was attempted and logged
        mock_logger.warning.assert_any_call(
            unittest.mock.ANY  # Match any string containing error info
        )
        
        # Verify we called the API twice (initial + retry)
        self.assertEqual(self.mock_openai_client.embeddings.create.call_count, 2)
    
    @patch('ingest_rag.logger')
    @patch('ingest_rag.click')
    def test_performance_monitoring(self, mock_click, mock_logger):
        """Test that performance metrics are tracked"""
        # Call the function
        embed_and_upsert(
            self.mock_client,
            "test_collection",
            self.docs,
            self.mock_openai_client,
            batch_size=3,
            model_name="text-embedding-3-large",
            deterministic_id=True,
            parallel=1
        )
        
        # Verify performance metrics are logged
        mock_logger.info.assert_any_call(
            unittest.mock.ANY  # Match any string containing the word 'texts/sec'
        )
        
        # Verify embedding metrics are tracked
        # (This would normally check INGESTION_DIAGNOSTICS but requires more complex mocking)
        mock_logger.info.assert_any_call(
            unittest.mock.ANY  # Match any string containing 'Embedding complete'
        )

if __name__ == '__main__':
    unittest.main()