"""Aspect-based sentiment analysis (ABSA) using pyABSA library."""

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import numpy as np

# Test pyABSA availability with the correct API
try:
    from pyabsa import ATEPCCheckpointManager, TaskCodeOption
    PYABSA_AVAILABLE = True
except ImportError:
    PYABSA_AVAILABLE = False

from config import MODEL_CONFIGS, PROCESSING_CONFIG
from utils.text import TextProcessor

logger = logging.getLogger(__name__)

# Log pyABSA availability status
if PYABSA_AVAILABLE:
    logger.info("✅ pyABSA library available with ATEPC API")
else:
    logger.warning("⚠️ pyABSA not available. Install with: pip install pyabsa")


class AspectExtractor:
    """Aspect extraction and sentiment analysis using pyABSA."""
    
    def __init__(self):
        """Initialize aspect extraction components."""
        self.text_processor = TextProcessor()
        
        # Configuration - always set these regardless of pyABSA availability
        self.model_config = MODEL_CONFIGS.get("absa", {})
        self.model_name = self.model_config.get("model_name", "multilingual")
        self.cache_dir = self.model_config.get("cache_dir", "./models/cache/absa")
        self.max_seq_len = self.model_config.get("max_seq_len", 512)
        self.batch_size = self.model_config.get("batch_size", 16)
        self.device = self.model_config.get("device", "auto")
        
        # Processing config
        self.min_frequency = PROCESSING_CONFIG["min_aspect_frequency"]
        self.max_aspects = PROCESSING_CONFIG["max_aspects"]
        
        # Initialize models
        self.apc_model = None  # Aspect Polarity Classification (legacy)
        self.ate_model = None  # Aspect Term Extraction (legacy)
        self.aspect_extractor = None  # ATEPC extractor
        self._available = False
        
        if not PYABSA_AVAILABLE:
            logger.warning("pyABSA is not available. Please install it with: pip install pyabsa")
            logger.info("Using fallback aspect extraction methods")
        else:
            self._load_models()
        
        logger.info(f"AspectExtractor initialized (pyABSA available: {PYABSA_AVAILABLE}, fallback ready: True)")
    
    def _load_models(self):
        """Load pyABSA models using the correct ATEPC API with robust error handling."""
        try:
            if not PYABSA_AVAILABLE:
                logger.info("pyABSA not available - using fallback methods only")
                self.aspect_extractor = None
                self._available = False
                return
            
            logger.info("Attempting to load pyABSA models with ATEPC API...")
            
            try:
                # Use the working API with careful error handling
                from pyabsa import ATEPCCheckpointManager
                
                logger.info("Loading ATEPC extractor (this may take a moment)...")
                self.aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
                    checkpoint='multilingual',
                    auto_device=True,
                    task_code='ATEPC'
                )
                
                # Simple test to verify it works
                logger.info("Testing ATEPC extractor...")
                test_result = self.aspect_extractor.extract_aspect(
                    ["Good service"],
                    pred_sentiment=True
                )
                
                self._available = True
                logger.info("✅ pyABSA ATEPC models loaded and tested successfully!")
                
            except Exception as e:
                logger.warning(f"pyABSA ATEPC loading failed: {str(e)}")
                logger.info("Using fallback methods for aspect extraction")
                self.aspect_extractor = None
                self._available = False
                
        except ImportError:
            logger.info("pyABSA import failed, using fallback methods")
            self.aspect_extractor = None
            self._available = False
        except Exception as e:
            logger.error(f"❌ Unexpected error loading pyABSA: {str(e)}")
            self.aspect_extractor = None
            self._available = False
    
    def extract_aspects_with_sentiment(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract aspects and their sentiments from text using pyABSA.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of aspect dictionaries with sentiment information
        """
        if not text or not text.strip():
            return []
        
        if not self._available:
            logger.warning("pyABSA models not available, using fallback")
            return self._fallback_aspect_extraction(text)
        
        try:
            # Clean text
            processed_text = self.text_processor.clean_text(text)
            
            # Truncate if too long
            if len(processed_text) > self.max_seq_len:
                processed_text = processed_text[:self.max_seq_len]
            
            # Extract aspects and sentiments using pyABSA
            aspects = self._extract_with_pyabsa(processed_text)
            
            return aspects
            
        except Exception as e:
            logger.error(f"pyABSA aspect extraction error for text '{text[:50]}...': {str(e)}")
            return self._fallback_aspect_extraction(text)
    
    def _extract_with_pyabsa(self, text: str) -> List[Dict[str, Any]]:
        """Extract aspects using pyABSA ATEPC API."""
        try:
            if not self.aspect_extractor:
                logger.warning("ATEPC extractor not initialized, using fallback")
                return self._fallback_aspect_extraction(text)
            
            # Use the working pyABSA ATEPC API
            results = self.aspect_extractor.extract_aspect(
                [text],  # Input as list
                pred_sentiment=True  # Enable sentiment prediction
            )
            
            aspects = []
            
            if results and len(results) > 0:
                result = results[0]  # Get first result
                logger.info(f"pyABSA result: {result}")
                
                # Handle different result formats
                if isinstance(result, dict):
                    # Extract aspects and sentiments from result
                    aspect_terms = result.get('aspect', [])
                    sentiments = result.get('sentiment', [])
                    
                    if isinstance(aspect_terms, list) and isinstance(sentiments, list):
                        for i, (aspect_term, sentiment) in enumerate(zip(aspect_terms, sentiments)):
                            if aspect_term and aspect_term.strip():
                                # Find position in text
                                aspect_pos = text.lower().find(aspect_term.lower())
                                position = [aspect_pos, aspect_pos + len(aspect_term)] if aspect_pos >= 0 else [0, len(text)]
                                
                                aspect_info = {
                                    'aspect_id': i,
                                    'aspect_label': aspect_term.strip(),
                                    'matched_text': text[max(0, position[0]-20):position[1]+20] if position[0] >= 0 else aspect_term,
                                    'aspect_sentiment': self._map_sentiment(sentiment),
                                    'aspect_sentiment_score': self._get_sentiment_confidence(sentiment),
                                    'position': position
                                }
                                aspects.append(aspect_info)
                    
                    # Also check if result has direct aspect-sentiment pairs
                    elif 'aspects' in result:
                        for i, aspect_data in enumerate(result['aspects']):
                            if isinstance(aspect_data, dict):
                                aspect_term = aspect_data.get('term', aspect_data.get('aspect', ''))
                                sentiment = aspect_data.get('sentiment', 'Neutral')
                                
                                if aspect_term and aspect_term.strip():
                                    aspect_pos = text.lower().find(aspect_term.lower())
                                    position = [aspect_pos, aspect_pos + len(aspect_term)] if aspect_pos >= 0 else [0, len(text)]
                                    
                                    aspect_info = {
                                        'aspect_id': i,
                                        'aspect_label': aspect_term.strip(),
                                        'matched_text': text[max(0, position[0]-20):position[1]+20] if position[0] >= 0 else aspect_term,
                                        'aspect_sentiment': self._map_sentiment(sentiment),
                                        'aspect_sentiment_score': self._get_sentiment_confidence(sentiment),
                                        'position': position
                                    }
                                    aspects.append(aspect_info)
            
            # If we found aspects, return them
            if aspects:
                logger.info(f"✅ pyABSA found {len(aspects)} aspects")
                return aspects[:self.max_aspects]
            else:
                logger.info("No aspects found by pyABSA, using fallback")
                return self._fallback_aspect_extraction(text)
                
        except Exception as e:
            logger.error(f"Error in pyABSA extraction: {str(e)}")
            logger.info("Falling back to keyword-based extraction")
            return self._fallback_aspect_extraction(text)
            
            # Use the predictor to analyze text
            # pyABSA 2.x expects format: "text$T$aspect" for aspect-level analysis
            # For aspect extraction, we'll use a different approach
            
            try:
                # Try direct text analysis
                result = predictor.predict(text)
                
                aspects = []
                if isinstance(result, dict):
                    # Single result format
                    aspect_info = {
                        'aspect_id': 0,
                        'aspect_label': result.get('aspect', 'general'),
                        'matched_text': result.get('text', text)[:100],
                        'aspect_sentiment': self._map_sentiment(result.get('sentiment', 'Neutral')),
                        'aspect_sentiment_score': result.get('confidence', 0.8),
                        'position': [0, len(text)]
                    }
                    aspects.append(aspect_info)
                elif isinstance(result, list):
                    # Multiple results
                    for i, res in enumerate(result):
                        if isinstance(res, dict):
                            aspect_info = {
                                'aspect_id': i,
                                'aspect_label': res.get('aspect', f'aspect_{i}'),
                                'matched_text': res.get('text', text)[:100],
                                'aspect_sentiment': self._map_sentiment(res.get('sentiment', 'Neutral')),
                                'aspect_sentiment_score': res.get('confidence', 0.8),
                                'position': res.get('position', [0, len(text)])
                            }
                            aspects.append(aspect_info)
                
                return aspects
                
            except Exception as pred_error:
                logger.warning(f"Predictor.predict failed: {pred_error}, trying alternative approach")
                return self._extract_with_simple_predict(text)
                
        except Exception as e:
            logger.error(f"Error in pyABSA extraction: {str(e)}")
            return []
    
    def _extract_with_simple_predict(self, text: str) -> List[Dict[str, Any]]:
        """Simple prediction using pyABSA with common aspects."""
        try:
            # Extract common aspects and predict their sentiment
            common_aspects = ['service', 'quality', 'price', 'support', 'design', 'performance', 'interface', 'system']
            aspects = []
            
            import pyabsa
            
            for i, aspect in enumerate(common_aspects):
                if aspect.lower() in text.lower():
                    # For pyABSA 2.x, we'll use a simplified approach
                    # Create the format expected by pyABSA: "text$T$aspect"
                    apc_input = f"{text}$T${aspect}"
                    
                    try:
                        # Try to use the new API structure
                        from pyabsa import APCPredictor
                        predictor = APCPredictor(checkpoint=self.apc_checkpoint or 'english')
                        result = predictor.predict(apc_input)
                        
                        sentiment = self._map_sentiment(result.get('sentiment', 'Neutral') if isinstance(result, dict) else 'Neutral')
                        confidence = result.get('confidence', 0.5) if isinstance(result, dict) else 0.5
                        
                    except Exception:
                        # Fallback to simple sentiment analysis
                        sentiment = self._simple_aspect_sentiment(text, aspect)
                        confidence = sentiment['score']
                        sentiment = sentiment['label']
                    
                    aspect_info = {
                        'aspect_id': i,
                        'aspect_label': aspect,
                        'matched_text': f"...{aspect}...",
                        'aspect_sentiment': sentiment,
                        'aspect_sentiment_score': confidence,
                        'position': [text.lower().find(aspect.lower()), text.lower().find(aspect.lower()) + len(aspect)]
                    }
                    aspects.append(aspect_info)
            
            return aspects[:5]  # Limit to 5 aspects
            
        except Exception as e:
            logger.error(f"Simple predict also failed: {str(e)}")
            return []
    
    def _extract_with_ate_apc(self, text: str) -> List[Dict[str, Any]]:
        """Alternative extraction using fallback approach."""
        try:
            # Since pyABSA 2.4.2 API has changed significantly,
            # we'll use a hybrid approach with keyword detection + sentiment analysis
            aspects = []
            
            # Step 1: Extract aspect terms using simple keyword approach
            aspect_terms = self._extract_simple_aspects(text)
            
            if not aspect_terms:
                return []
            
            # Step 2: For each aspect, try to get sentiment using pyABSA if possible
            for i, aspect_term in enumerate(aspect_terms):
                try:
                    # Try to use pyABSA for sentiment analysis
                    # Format: "text$T$aspect_term" as expected by pyABSA
                    apc_input = f"{text}$T${aspect_term}"
                    
                    # Try different ways to get sentiment
                    sentiment_result = None
                    confidence = 0.5
                    
                    try:
                        from pyabsa import APCPredictor
                        predictor = APCPredictor(checkpoint=self.apc_checkpoint or 'english')
                        sentiment_result = predictor.predict(apc_input)
                        
                        if isinstance(sentiment_result, dict):
                            sentiment = sentiment_result.get('sentiment', 'Neutral')
                            confidence = sentiment_result.get('confidence', 0.5)
                        else:
                            sentiment = 'Neutral'
                            confidence = 0.5
                            
                    except Exception as pyabsa_error:
                        logger.debug(f"pyABSA prediction failed for '{aspect_term}': {pyabsa_error}")
                        # Fallback to simple rule-based sentiment
                        sentiment_result = self._simple_aspect_sentiment(text, aspect_term)
                        sentiment = sentiment_result['label']
                        confidence = sentiment_result['score']
                    
                    # Find position of aspect in text
                    aspect_pos = text.lower().find(aspect_term.lower())
                    position = [aspect_pos, aspect_pos + len(aspect_term)] if aspect_pos >= 0 else [0, len(text)]
                    
                    aspect_info = {
                        'aspect_id': i,
                        'aspect_label': aspect_term,
                        'matched_text': text[max(0, position[0]-20):position[1]+20] if position[0] >= 0 else aspect_term,
                        'aspect_sentiment': self._map_sentiment(sentiment),
                        'aspect_sentiment_score': confidence,
                        'position': position
                    }
                    aspects.append(aspect_info)
                    
                except Exception as aspect_error:
                    logger.warning(f"Error processing aspect '{aspect_term}': {aspect_error}")
                    # Add with neutral sentiment as fallback
                    aspects.append({
                        'aspect_id': i,
                        'aspect_label': aspect_term,
                        'matched_text': aspect_term,
                        'aspect_sentiment': 'Neutral',
                        'aspect_sentiment_score': 0.5,
                        'position': [0, len(text)]
                    })
            
            return aspects
            
        except Exception as e:
            logger.error(f"Error in ATE+APC extraction: {str(e)}")
            return []
    
    def _simple_aspect_sentiment(self, text: str, aspect_term: str) -> Dict[str, Any]:
        """Simple rule-based aspect sentiment analysis as fallback."""
        try:
            # Create a focused context around the aspect term
            aspect_pos = text.lower().find(aspect_term.lower())
            if aspect_pos >= 0:
                # Extract context window around aspect (±50 characters)
                start = max(0, aspect_pos - 50)
                end = min(len(text), aspect_pos + len(aspect_term) + 50)
                context = text[start:end]
            else:
                context = text
            
            # Simple keyword-based sentiment
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'perfect', 'love', 'best', 'awesome', 'fantastic', 'satisfied', 'happy']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'poor', 'disappointing', 'frustrated', 'angry', 'sad', 'problem', 'issue', 'broken']
            
            context_lower = context.lower()
            positive_count = sum(1 for word in positive_words if word in context_lower)
            negative_count = sum(1 for word in negative_words if word in context_lower)
            
            if positive_count > negative_count:
                return {'label': 'Positive', 'score': min(0.7, 0.5 + positive_count * 0.1)}
            elif negative_count > positive_count:
                return {'label': 'Negative', 'score': min(0.7, 0.5 + negative_count * 0.1)}
            else:
                return {'label': 'Neutral', 'score': 0.5}
                
        except Exception as e:
            logger.warning(f"Simple sentiment analysis failed: {e}")
            return {'label': 'Neutral', 'score': 0.5}

    def _extract_simple_aspects(self, text: str) -> List[str]:
        """Simple aspect extraction using keywords when ATE fails."""
        from config import ASPECT_KEYWORDS
        
        text_lower = text.lower()
        found_aspects = []
        
        for aspect_label, keywords in ASPECT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_aspects.append(aspect_label)
                    break  # Only one aspect per category
        
        # Also try to extract some common nouns
        common_aspects = ['website', 'app', 'system', 'service', 'support', 'interface', 'feature', 'function']
        for aspect in common_aspects:
            if aspect in text_lower and aspect not in found_aspects:
                found_aspects.append(aspect)
        
        return found_aspects[:5]  # Limit to 5 aspects
    
    def _map_sentiment(self, sentiment: Any) -> str:
        """Map pyABSA sentiment labels to standard format."""
        if isinstance(sentiment, str):
            sentiment_lower = sentiment.lower()
            if 'pos' in sentiment_lower or sentiment_lower == 'positive':
                return 'Positive'
            elif 'neg' in sentiment_lower or sentiment_lower == 'negative':
                return 'Negative'
            else:
                return 'Neutral'
        elif isinstance(sentiment, (int, float)):
            # Numeric sentiment scores
            if sentiment > 0.1:
                return 'Positive'
            elif sentiment < -0.1:
                return 'Negative'
            else:
                return 'Neutral'
        else:
            return 'Neutral'
    
    def _get_sentiment_confidence(self, sentiment: Any) -> float:
        """Get confidence score for sentiment."""
        # pyABSA doesn't always provide confidence scores in the result
        # For now, return a reasonable default based on sentiment
        if isinstance(sentiment, str):
            sentiment_lower = sentiment.lower()
            if 'positive' in sentiment_lower or 'negative' in sentiment_lower:
                return 0.75  # Higher confidence for clear sentiments
            else:
                return 0.5   # Lower confidence for neutral
        elif isinstance(sentiment, (int, float)):
            # For numeric scores, use absolute value as confidence
            return min(0.9, abs(float(sentiment)) + 0.5)
        return 0.5
    
    def _fallback_aspect_extraction(self, text: str) -> List[Dict[str, Any]]:
        """
        Fallback aspect extraction when pyABSA is not available.
        Uses simple keyword matching from config.
        """
        from config import ASPECT_KEYWORDS
        
        text_lower = text.lower()
        aspects = []
        aspect_id = 0
        
        for aspect_label, keywords in ASPECT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Simple rule-based sentiment for the aspect
                    aspect_sentiment = self._simple_aspect_sentiment(text, keyword)
                    
                    aspects.append({
                        'aspect_id': aspect_id,
                        'aspect_label': aspect_label,
                        'matched_text': f"...{keyword}...",
                        'aspect_sentiment': aspect_sentiment['label'],
                        'aspect_sentiment_score': aspect_sentiment['score'],
                        'position': [text_lower.find(keyword), text_lower.find(keyword) + len(keyword)]
                    })
                    aspect_id += 1
                    break  # Only one match per aspect category
        
        return aspects[:self.max_aspects]  # Limit results
    
    def _simple_aspect_sentiment(self, text: str, aspect_keyword: str) -> Dict[str, Any]:
        """Simple rule-based sentiment analysis for an aspect."""
        text_lower = text.lower()
        
        # Look for sentiment indicators near the aspect keyword
        aspect_pos = text_lower.find(aspect_keyword)
        if aspect_pos == -1:
            return {'label': 'Neutral', 'score': 0.5}
        
        # Check surrounding context (50 chars before and after)
        context_start = max(0, aspect_pos - 50)
        context_end = min(len(text), aspect_pos + len(aspect_keyword) + 50)
        context = text_lower[context_start:context_end]
        
        positive_words = ['good', 'great', 'excellent', 'easy', 'fast', 'helpful', 'nice']
        negative_words = ['bad', 'slow', 'difficult', 'poor', 'terrible', 'annoying', 'confusing']
        
        pos_count = sum(1 for word in positive_words if word in context)
        neg_count = sum(1 for word in negative_words if word in context)
        
        if pos_count > neg_count:
            return {'label': 'Positive', 'score': 0.7}
        elif neg_count > pos_count:
            return {'label': 'Negative', 'score': 0.7}
        else:
            return {'label': 'Neutral', 'score': 0.5}
    
    def analyze_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Analyze aspects for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of aspect analysis results for each text
        """
        if not texts:
            return []
        
        if not self._available:
            # Use fallback for all texts
            return [self._fallback_aspect_extraction(text) for text in texts]
        
        try:
            results = []
            
            # Process in smaller batches to avoid memory issues
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_results = []
                
                for text in batch:
                    result = self.extract_aspects_with_sentiment(text)
                    batch_results.append(result)
                
                results.extend(batch_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch aspect analysis error: {str(e)}")
            # Fallback to individual processing
            return [self.extract_aspects_with_sentiment(text) for text in texts]
    
    def get_aspect_statistics(self) -> Dict[str, Any]:
        """Get statistics about the ABSA service."""
        return {
            "pyabsa_available": PYABSA_AVAILABLE,
            "models_loaded": self._available,
            "model_config": self.model_config,
            "processing_config": {
                "min_frequency": self.min_frequency,
                "max_aspects": self.max_aspects,
                "batch_size": self.batch_size
            }
        }
    
    def is_service_available(self) -> bool:
        """Check if the aspect extraction service is available."""
        return self._available or True  # Fallback is always available
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the aspect extraction service."""
        return {
            "service_type": "pyABSA" if self._available else "Fallback",
            "pyabsa_version": getattr(__import__('pyabsa'), '__version__', 'unknown') if PYABSA_AVAILABLE else None,
            "model_name": self.model_name,
            "cache_dir": self.cache_dir,
            "max_seq_len": self.max_seq_len,
            "service_available": self.is_service_available(),
            "models_loaded": self._available
        }
    
    def update_aspect_clusters(self, new_aspects: List[str], new_embeddings=None):
        """
        Legacy method for compatibility.
        pyABSA doesn't use manual clustering, so this is a no-op.
        """
        logger.info("update_aspect_clusters called but not needed with pyABSA")
        pass
    
    def extract_aspects(self, text: str) -> List[Dict[str, Any]]:
        """
        Main method to extract aspects and their sentiments.
        This is the primary interface method.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of aspect dictionaries with sentiment information
        """
        return self.extract_aspects_with_sentiment(text)
