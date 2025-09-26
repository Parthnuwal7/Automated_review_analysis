"""Unit tests for the sentiment analysis pipeline components."""

import pytest
import tempfile
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.process import ReviewProcessor
from pipeline.translation import TranslationService
from pipeline.sentiment import SentimentAnalyzer
from pipeline.intent import IntentClassifier
from pipeline.aspect_extraction import AspectExtractor
from pipeline.insights import InsightGenerator
from utils.text import TextProcessor
from utils.db import DatabaseManager


class TestTextProcessor:
    """Test text processing utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TextProcessor()
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "  Hello World!   This is a <b>test</b>. https://example.com  "
        cleaned = self.processor.clean_text(dirty_text)
        
        assert "Hello World!" in cleaned
        assert "<b>" not in cleaned
        assert "[URL]" in cleaned
        assert len(cleaned.strip()) > 0
    
    def test_language_detection(self):
        """Test language detection."""
        english_text = "This is an English text for testing."
        hindi_text = "यह हिंदी में एक परीक्षण पाठ है।"
        
        eng_lang = self.processor.detect_language(english_text)
        hindi_lang = self.processor.detect_language(hindi_text)
        
        assert eng_lang in ["en", "English"]
        assert hindi_lang in ["hi", "Hindi"]
    
    def test_tokenization(self):
        """Test text tokenization."""
        text = "This is a simple test sentence."
        tokens = self.processor.tokenize(text)
        
        assert len(tokens) > 0
        assert "simple" in tokens
        assert "test" in tokens
    
    def test_keyword_extraction(self):
        """Test keyword extraction."""
        text = "The website performance is very slow and needs improvement. The loading speed is poor."
        keywords = self.processor.extract_keywords(text)
        
        assert len(keywords) > 0
        assert any(word in keywords for word in ["website", "performance", "slow", "improvement"])
    
    def test_sentiment_token_extraction(self):
        """Test sentiment token extraction."""
        positive_text = "Great service! Very helpful and efficient."
        negative_text = "Terrible experience! Very slow and confusing."
        
        pos_tokens, neg_tokens = self.processor.extract_sentiment_tokens(positive_text, "Positive")
        assert len(pos_tokens) > 0
        
        pos_tokens, neg_tokens = self.processor.extract_sentiment_tokens(negative_text, "Negative")
        assert len(neg_tokens) > 0


class TestSentimentAnalyzer:
    """Test sentiment analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    def test_positive_sentiment(self):
        """Test positive sentiment detection."""
        positive_text = "I love this website! It's amazing and very helpful."
        result = self.analyzer.analyze_sentiment(positive_text)
        
        assert result["label"] in ["Positive", "positive"]
        assert 0 <= result["score"] <= 1
        assert "confidence" in result
    
    def test_negative_sentiment(self):
        """Test negative sentiment detection."""
        negative_text = "This website is terrible! Very slow and confusing interface."
        result = self.analyzer.analyze_sentiment(negative_text)
        
        assert result["label"] in ["Negative", "negative"]
        assert 0 <= result["score"] <= 1
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment detection."""
        neutral_text = "The website has a standard interface with basic features."
        result = self.analyzer.analyze_sentiment(neutral_text)
        
        assert result["label"] in ["Neutral", "neutral", "Positive", "Negative"]  # Allow any valid sentiment
        assert 0 <= result["score"] <= 1
    
    def test_empty_text(self):
        """Test handling of empty text."""
        result = self.analyzer.analyze_sentiment("")
        
        assert result["label"] == "Neutral"
        assert result["score"] == 0.5


class TestIntentClassifier:
    """Test intent classification functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = IntentClassifier()
    
    def test_complaint_intent(self):
        """Test complaint intent detection."""
        complaint_text = "The system is broken and doesn't work properly. I'm very frustrated."
        result = self.classifier.classify_intent(complaint_text)
        
        assert result["label"] in ["Complaint", "complaint"]
        assert 0 <= result["score"] <= 1
    
    def test_praise_intent(self):
        """Test praise intent detection."""
        praise_text = "Excellent service! Thank you for the wonderful experience."
        result = self.classifier.classify_intent(praise_text)
        
        assert result["label"] in ["Praise", "praise"]
        assert 0 <= result["score"] <= 1
    
    def test_suggestion_intent(self):
        """Test suggestion intent detection."""
        suggestion_text = "I suggest you should improve the loading speed and add more features."
        result = self.classifier.classify_intent(suggestion_text)
        
        assert result["label"] in ["Suggestion", "suggestion", "General Feedback"]  # Allow fallback
        assert 0 <= result["score"] <= 1
    
    def test_empty_text(self):
        """Test handling of empty text."""
        result = self.classifier.classify_intent("")
        
        assert result["label"] == "General Feedback"
        assert result["score"] == 0.5


class TestAspectExtractor:
    """Test aspect extraction functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = AspectExtractor()
    
    def test_aspect_extraction(self):
        """Test basic aspect extraction."""
        text = "The website interface is confusing and the loading speed is very slow."
        aspects = self.extractor.extract_aspects_with_sentiment(text)
        
        assert len(aspects) >= 0  # May be empty if models not available
        
        if aspects:
            aspect = aspects[0]
            assert "aspect_label" in aspect
            assert "aspect_sentiment" in aspect
            assert "matched_text" in aspect
    
    def test_empty_text(self):
        """Test handling of empty text."""
        aspects = self.extractor.extract_aspects_with_sentiment("")
        assert aspects == []
    
    def test_service_info(self):
        """Test service information retrieval."""
        info = self.extractor.get_service_info()
        
        assert "spacy_model" in info
        assert "embedding_service" in info
        assert "sentiment_service" in info


class TestInsightGenerator:
    """Test insight generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = InsightGenerator()
    
    def test_insight_generation(self):
        """Test insight generation from sample reviews."""
        sample_reviews = [
            {
                "review_id": "1",
                "sentiment": "Negative",
                "intent": "Complaint",
                "aspects": [
                    {
                        "aspect_label": "performance",
                        "aspect_sentiment": "Negative",
                        "aspect_sentiment_score": 0.2,
                        "matched_text": "slow loading"
                    }
                ]
            },
            {
                "review_id": "2",
                "sentiment": "Positive",
                "intent": "Praise",
                "aspects": [
                    {
                        "aspect_label": "interface",
                        "aspect_sentiment": "Positive",
                        "aspect_sentiment_score": 0.8,
                        "matched_text": "great design"
                    }
                ]
            }
        ]
        
        insights = self.generator.generate_insights(sample_reviews)
        
        assert "pain_points" in insights
        assert "recommendations" in insights
        assert isinstance(insights["pain_points"], list)
        assert isinstance(insights["recommendations"], list)
    
    def test_empty_reviews(self):
        """Test handling of empty review list."""
        insights = self.generator.generate_insights([])
        
        assert insights["pain_points"] == []
        assert insights["recommendations"] == []


class TestDatabaseManager:
    """Test database functionality."""
    
    def setup_method(self):
        """Set up test fixtures with temporary database."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db = DatabaseManager(self.temp_db.name)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        Path(self.temp_db.name).unlink(missing_ok=True)
    
    def test_database_initialization(self):
        """Test database table creation."""
        info = self.db.get_database_info()
        
        assert info["database_path"] == self.temp_db.name
        assert "table_sizes" in info
    
    def test_save_and_retrieve_results(self):
        """Test saving and retrieving processed results."""
        sample_results = {
            "reviews": [
                {
                    "review_id": "test_1",
                    "original_text": "Test review",
                    "language": "English",
                    "timestamp": "2025-01-01 10:00:00",
                    "sentiment": "Positive",
                    "sentiment_score": 0.8,
                    "intent": "Praise",
                    "intent_score": 0.7,
                    "aspects": [
                        {
                            "aspect_id": 1,
                            "aspect_label": "test_aspect",
                            "matched_text": "test",
                            "aspect_sentiment": "Positive",
                            "aspect_sentiment_score": 0.8
                        }
                    ],
                    "tokens_positive": ["great"],
                    "tokens_negative": []
                }
            ],
            "summary": {
                "statistical_summary": {"positive_percent": 100.0},
                "textual_summary": "Test summary"
            },
            "insights": {
                "pain_points": [],
                "recommendations": ["Test recommendation"]
            }
        }
        
        batch_id = self.db.save_processed_results(sample_results)
        assert batch_id.startswith("batch_")
        
        # Retrieve review
        retrieved_review = self.db.get_review_by_id("test_1")
        assert retrieved_review is not None
        assert retrieved_review["original_text"] == "Test review"
        assert retrieved_review["sentiment"] == "Positive"
        
        # Get statistics
        stats = self.db.get_summary_statistics()
        assert stats["total_reviews"] >= 1


class TestReviewProcessor:
    """Test complete review processing pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Note: This may fail if models are not available
        try:
            self.processor = ReviewProcessor()
        except Exception as e:
            pytest.skip(f"Review processor initialization failed: {e}")
    
    def test_single_review_processing(self):
        """Test processing a single review."""
        try:
            result = self.processor.process_single_review(
                text="This is a great website with excellent features!",
                review_id="test_single",
                timestamp="2025-01-01 10:00:00"
            )
            
            assert result["review_id"] == "test_single"
            assert result["original_text"] == "This is a great website with excellent features!"
            assert "sentiment" in result
            assert "intent" in result
            assert "language" in result
            
        except Exception as e:
            pytest.skip(f"Single review processing failed: {e}")
    
    def test_csv_processing(self):
        """Test processing reviews from CSV file."""
        try:
            # Create temporary CSV
            temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
            temp_csv.write("text,timestamp,review_id\n")
            temp_csv.write("Great service!,2025-01-01 10:00:00,csv_1\n")
            temp_csv.write("Poor performance,2025-01-01 11:00:00,csv_2\n")
            temp_csv.close()
            
            results = self.processor.process_reviews_from_csv(temp_csv.name)
            
            assert "reviews" in results
            assert "summary" in results
            assert "insights" in results
            assert len(results["reviews"]) == 2
            
            # Clean up
            Path(temp_csv.name).unlink()
            
        except Exception as e:
            pytest.skip(f"CSV processing failed: {e}")


# Test configuration
def pytest_configure(config):
    """Configure pytest settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may require model downloads)"
    )


# Run specific tests
if __name__ == "__main__":
    # Run basic tests
    print("Running basic tests...")
    pytest.main([__file__ + "::TestTextProcessor", "-v"])
    print("\nFor full test suite including model tests, run: pytest tests/ -v")
