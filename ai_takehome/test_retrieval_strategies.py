#!/usr/bin/env python3
"""
Comprehensive test suite for retrieval strategies and synthesizers
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval_strategies import (
    BM25Strategy, TFIDFStrategy, EmbeddingStrategy,
    HybridStrategy, EnsembleStrategy, load_docs
)
from openai_synthesizer import OpenAISynthesizer, HybridSynthesizer


class TestDataLoading:
    """Test data loading functionality"""

    def test_load_docs(self):
        """Test that data loads correctly"""
        docs, df = load_docs()

        assert isinstance(docs, list)
        assert isinstance(df, pd.DataFrame)
        assert len(docs) == len(df)
        assert len(docs) > 0

        # Check required columns exist
        required_columns = ['department', 'denial_reason', 'patient_age', 'payer', 'amount']
        for col in required_columns:
            assert col in df.columns

    def test_doc_format(self):
        """Test document format is consistent"""
        docs, df = load_docs()

        # Check first document has expected format
        first_doc = docs[0]
        assert 'Department:' in first_doc
        assert 'Denial reason:' in first_doc
        assert 'Patient age:' in first_doc
        assert 'Payer:' in first_doc
        assert 'Amount:' in first_doc


class TestRetrievalStrategies:
    """Test all retrieval strategies"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        docs, df = load_docs()
        return docs[:50], df.head(50)  # Use subset for faster tests

    def test_bm25_strategy(self, sample_data):
        """Test BM25 strategy"""
        docs, df = sample_data
        strategy = BM25Strategy()

        # Test setup
        strategy.setup(docs, df)
        assert strategy.bm25 is not None
        assert strategy.df is not None

        # Test retrieval
        results, scores = strategy.retrieve("cardiology claims", top_k=5)
        assert isinstance(results, pd.DataFrame)
        assert len(results) <= 5
        assert len(scores) == len(results)

    def test_tfidf_strategy(self, sample_data):
        """Test TF-IDF strategy"""
        docs, df = sample_data
        strategy = TFIDFStrategy()

        # Test setup
        strategy.setup(docs, df)
        assert strategy.vectorizer is not None
        assert strategy.doc_vectors is not None

        # Test retrieval
        results, scores = strategy.retrieve("cardiology claims", top_k=5)
        assert isinstance(results, pd.DataFrame)
        assert len(results) <= 5
        assert len(scores) == len(results)

    @patch('retrieval_strategies.SentenceTransformer')
    def test_embedding_strategy(self, mock_transformer, sample_data):
        """Test embedding strategy with mocked SentenceTransformer"""
        docs, df = sample_data

        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(len(docs), 384)
        mock_transformer.return_value = mock_model

        strategy = EmbeddingStrategy()
        strategy.setup(docs, df)

        # Mock query encoding
        mock_model.encode.return_value = np.random.rand(1, 384)

        results, scores = strategy.retrieve("cardiology claims", top_k=5)
        assert isinstance(results, pd.DataFrame)
        assert len(results) <= 5
        assert len(scores) == len(results)

    def test_hybrid_strategy(self, sample_data):
        """Test hybrid strategy"""
        docs, df = sample_data

        with patch('retrieval_strategies.SentenceTransformer') as mock_transformer:
            # Mock the sentence transformer
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(50, 384)
            mock_transformer.return_value = mock_model

            strategy = HybridStrategy(bm25_candidates=20, final_top_k=5)
            strategy.setup(docs, df)

            # Mock query encoding for embedding part
            mock_model.encode.side_effect = [
                np.random.rand(1, 384),  # Question encoding
                np.random.rand(20, 384)  # Candidate docs encoding
            ]

            results, scores = strategy.retrieve("cardiology claims", top_k=5)
            assert isinstance(results, pd.DataFrame)
            assert len(results) <= 5
            assert len(scores) == len(results)


class TestSynthesizers:
    """Test answer synthesizers"""

    @pytest.fixture
    def sample_claims(self):
        """Create sample claims data"""
        data = {
            'department': ['Cardiology', 'Cardiology', 'Radiology'],
            'denial_reason': ['Coding error', 'Coding error', 'Invalid CPT'],
            'patient_age': [45, 67, 32],
            'payer': ['Aetna', 'Medicare', 'BCBS'],
            'amount': [1500.0, 2300.0, 800.0]
        }
        return pd.DataFrame(data)

    def test_hybrid_synthesizer_rule_based(self, sample_claims):
        """Test rule-based synthesis"""
        synthesizer = HybridSynthesizer()

        result = synthesizer.synthesize_answer(
            "Why are cardiology claims denied?",
            sample_claims,
            use_openai=False
        )

        assert 'rule_based' in result
        assert 'used' in result
        assert result['used'] == 'rule_based'
        assert 'Cardiology' in result['rule_based']

    @patch('openai_synthesizer.OpenAI')
    def test_openai_synthesizer(self, mock_openai_class, sample_claims):
        """Test OpenAI synthesizer with mocking"""
        # Mock OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Cardiology claims are denied due to coding errors."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        synthesizer = OpenAISynthesizer(api_key="test_key")
        result = synthesizer.synthesize_answer("Why are cardiology claims denied?", sample_claims)

        assert isinstance(result, str)
        assert "coding errors" in result.lower()

    def test_openai_synthesizer_no_api_key(self, sample_claims):
        """Test OpenAI synthesizer without API key"""
        synthesizer = OpenAISynthesizer(api_key=None)
        result = synthesizer.synthesize_answer("Why are cardiology claims denied?", sample_claims)

        assert "OpenAI API key not configured" in result

    def test_hybrid_synthesizer_fallback(self, sample_claims):
        """Test hybrid synthesizer fallback behavior"""
        synthesizer = HybridSynthesizer(openai_api_key=None)

        result = synthesizer.synthesize_answer(
            "Why are cardiology claims denied?",
            sample_claims,
            use_openai=True
        )

        assert 'rule_based' in result
        assert result['used'] == 'rule_based'

    def test_synthesizer_department_filtering(self, sample_claims):
        """Test department-specific filtering in synthesizer"""
        synthesizer = HybridSynthesizer()

        # Test cardiology question
        result = synthesizer.synthesize_answer(
            "Why are cardiology claims denied most often?",
            sample_claims,
            use_openai=False
        )

        assert 'Cardiology' in result['rule_based']
        assert 'Coding error' in result['rule_based']

    def test_synthesizer_special_patterns(self):
        """Test special pattern handling in synthesizer"""
        synthesizer = HybridSynthesizer()

        # Create test data with duplicate claims
        duplicate_data = pd.DataFrame({
            'department': ['General', 'General'],
            'denial_reason': ['Duplicate claim', 'Duplicate claim'],
            'patient_age': [30, 40],
            'payer': ['Aetna', 'BCBS'],
            'amount': [100.0, 200.0]
        })

        result = synthesizer.synthesize_answer(
            "Top duplicate claim issues?",
            duplicate_data,
            use_openai=False
        )

        assert 'Duplicate claim: 2' in result['rule_based']


class TestIntegration:
    """Integration tests for the complete system"""

    def test_eval_compatibility(self):
        """Test that our enhanced system maintains eval.py compatibility"""
        # Import the baseline agent answer function
        from baseline_agent import answer

        # Test that it still works with the evaluation questions
        test_cases = [
            "Why are cardiology claims denied most often?",
            "List common denial reasons for radiology.",
            "Top duplicate claim issues?",
            "Why do we have expired coverage denials?",
            "What missing info causes Pediatrics denials?"
        ]

        for question in test_cases:
            result = answer(question)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_multi_strategy_agent_import(self):
        """Test that multi-strategy agent can be imported and used"""
        from enhanced_agent import answer, get_strategy

        # Test strategy factory
        strategies = ['bm25', 'tfidf', 'embeddings', 'hybrid', 'ensemble']
        for strategy_name in strategies:
            strategy = get_strategy(strategy_name)
            assert strategy is not None
            assert hasattr(strategy, 'setup')
            assert hasattr(strategy, 'retrieve')


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])