"""Smoke tests for Streamlit application components."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def test_app_imports():
    """Test that all app modules can be imported."""
    try:
        import app.main
        import app.ui
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import app modules: {e}")


def test_config_imports():
    """Test that configuration can be imported."""
    try:
        import config
        assert hasattr(config, 'MODEL_CONFIGS')
        assert hasattr(config, 'PROCESSING_CONFIG')
        assert hasattr(config, 'STREAMLIT_CONFIG')
    except ImportError as e:
        pytest.fail(f"Failed to import config: {e}")


def test_pipeline_imports():
    """Test that pipeline modules can be imported."""
    try:
        import pipeline.process
        import pipeline.translation
        import pipeline.sentiment
        import pipeline.intent
        import pipeline.aspect_extraction
        import pipeline.insights
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import pipeline modules: {e}")


def test_utils_imports():
    """Test that utility modules can be imported."""
    try:
        import utils.text
        import utils.db
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import utils modules: {e}")


def test_models_imports():
    """Test that model modules can be imported."""
    try:
        import models.embedding
        import models.gemini_client
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import model modules: {e}")


@patch('streamlit.set_page_config')
@patch('streamlit.title')
@patch('streamlit.markdown')
@patch('streamlit.sidebar')
def test_main_app_execution(mock_sidebar, mock_markdown, mock_title, mock_config):
    """Test that main app can be executed without errors."""
    try:
        # Mock Streamlit components
        mock_sidebar.header = Mock()
        mock_sidebar.selectbox = Mock(return_value="📊 Dashboard")
        
        # Import and test main function
        from app.main import main
        
        # This should not raise any exceptions
        with patch('app.ui.SentimentDashboard') as mock_dashboard:
            mock_dashboard_instance = Mock()
            mock_dashboard.return_value = mock_dashboard_instance
            
            # Should be able to call main without errors
            main()
            
            # Verify mocks were called
            mock_config.assert_called_once()
            mock_title.assert_called_once()
        
    except Exception as e:
        pytest.fail(f"Main app execution failed: {e}")


def test_dashboard_initialization():
    """Test that dashboard can be initialized."""
    try:
        with patch('pipeline.process.ReviewProcessor'):
            with patch('utils.db.DatabaseManager'):
                with patch('models.gemini_client.GeminiClient'):
                    from app.ui import SentimentDashboard
                    
                    dashboard = SentimentDashboard()
                    assert dashboard is not None
                    
    except Exception as e:
        pytest.fail(f"Dashboard initialization failed: {e}")


def test_sample_data_generation():
    """Test that sample data can be generated."""
    try:
        from tools.generate_sample_data import generate_sample_reviews
        
        reviews = generate_sample_reviews()
        
        assert len(reviews) > 0
        assert all("review_id" in review for review in reviews)
        assert all("text" in review for review in reviews)
        assert all("timestamp" in review for review in reviews)
        
        # Check for both English and Hindi reviews
        has_english = any("EN_" in review["review_id"] for review in reviews)
        has_hindi = any("HI_" in review["review_id"] for review in reviews)
        
        assert has_english, "No English reviews found"
        assert has_hindi, "No Hindi reviews found"
        
    except Exception as e:
        pytest.fail(f"Sample data generation failed: {e}")


def test_database_schema():
    """Test that database schema can be created."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.db') as temp_db:
            from utils.db import DatabaseManager
            
            db = DatabaseManager(temp_db.name)
            info = db.get_database_info()
            
            assert info["database_path"] == temp_db.name
            assert "table_sizes" in info
            
            # Test that all expected tables exist
            expected_tables = ['reviews', 'aspects', 'tokens', 'summaries', 'insights']
            table_sizes = info["table_sizes"]
            
            for table in expected_tables:
                assert table in table_sizes, f"Table {table} not found"
                
    except Exception as e:
        pytest.fail(f"Database schema test failed: {e}")


@pytest.mark.parametrize("text,expected_lang", [
    ("This is English text", "en"),
    ("यह हिंदी टेक्स्ट है", "hi"),
    ("", "unknown")
])
def test_language_detection_basic(text, expected_lang):
    """Test basic language detection functionality."""
    try:
        from utils.text import TextProcessor
        
        processor = TextProcessor()
        detected_lang = processor.detect_language(text)
        
        if expected_lang == "unknown":
            assert detected_lang in ["unknown", "en"]  # Default fallback
        else:
            # Allow for some flexibility in detection
            assert detected_lang in [expected_lang, "en", "hi"]
            
    except Exception as e:
        pytest.fail(f"Language detection test failed: {e}")


def test_text_cleaning():
    """Test text cleaning functionality."""
    try:
        from utils.text import TextProcessor
        
        processor = TextProcessor()
        
        # Test HTML removal
        html_text = "This is <b>bold</b> text with <a href='link'>link</a>"
        cleaned = processor.clean_text(html_text)
        assert "<b>" not in cleaned
        assert "<a" not in cleaned
        
        # Test URL removal
        url_text = "Visit https://example.com for more info"
        cleaned = processor.clean_text(url_text)
        assert "[URL]" in cleaned
        assert "https://example.com" not in cleaned
        
    except Exception as e:
        pytest.fail(f"Text cleaning test failed: {e}")


def test_configuration_values():
    """Test that configuration has expected values."""
    try:
        import config
        
        # Test model configurations
        assert "translation" in config.MODEL_CONFIGS
        assert "sentiment" in config.MODEL_CONFIGS
        assert "intent" in config.MODEL_CONFIGS
        assert "embedding" in config.MODEL_CONFIGS
        
        # Test processing configuration
        assert "batch_size" in config.PROCESSING_CONFIG
        assert "max_aspects" in config.PROCESSING_CONFIG
        
        # Test Streamlit configuration
        assert "page_title" in config.STREAMLIT_CONFIG
        
    except Exception as e:
        pytest.fail(f"Configuration test failed: {e}")


# Performance smoke test
def test_basic_processing_performance():
    """Test that basic processing doesn't take too long."""
    import time
    
    try:
        from utils.text import TextProcessor
        
        processor = TextProcessor()
        
        start_time = time.time()
        
        # Process multiple texts
        texts = [
            "This is a test sentence for performance testing.",
            "Another test sentence with different content.",
            "Third test sentence to check processing speed."
        ] * 10  # 30 texts total
        
        for text in texts:
            processor.clean_text(text)
            processor.tokenize(text)
            processor.detect_language(text)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 30 short texts in under 5 seconds
        assert processing_time < 5.0, f"Processing took too long: {processing_time:.2f}s"
        
    except Exception as e:
        pytest.fail(f"Performance test failed: {e}")


if __name__ == "__main__":
    print("Running Streamlit app smoke tests...")
    pytest.main([__file__, "-v"])
