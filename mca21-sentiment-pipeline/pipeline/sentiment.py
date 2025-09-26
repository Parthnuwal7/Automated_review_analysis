"""Sentiment analysis using multilingual transformer models."""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Optional
import logging
import numpy as np
from functools import lru_cache

from config import MODEL_CONFIGS
from utils.text import TextProcessor

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Multilingual sentiment analysis using transformer models."""
    
    def __init__(self):
        """Initialize sentiment analysis components."""
        self.model_config = MODEL_CONFIGS["sentiment"]
        self.model_name = self.model_config["model_name"]
        self.cache_dir = self.model_config["cache_dir"]
        self.labels_map = self.model_config["labels_map"]
        
        self._pipeline = None
        self._text_processor = TextProcessor()
        
        logger.info(f"SentimentAnalyzer initialized with model: {self.model_name}")
    
    @property
    def sentiment_pipeline(self):
        """Lazy load sentiment analysis pipeline."""
        if self._pipeline is None:
            self._load_pipeline()
        return self._pipeline
    
    def _load_pipeline(self):
        """Load the sentiment analysis pipeline."""
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            
            # Remove cache_dir from pipeline initialization - it's not supported
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1,
                top_k=None  # Updated parameter name
            )
            
            logger.info("Sentiment analysis pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading sentiment model: {str(e)}")
            # Load fallback pipeline
            self._load_fallback_pipeline()

    def _load_fallback_pipeline(self):
        """Load a fallback sentiment model."""
        try:
            fallback_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            logger.info(f"Loading fallback sentiment model: {fallback_model}")
            
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=fallback_model,
                device=0 if torch.cuda.is_available() else -1,
                top_k=None  # Updated parameter name instead of return_all_scores
            )
            
            # Update labels map for fallback model
            self.labels_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
            
            logger.info("Fallback sentiment analysis pipeline loaded")
            
        except Exception as e:
            logger.error(f"Error loading fallback sentiment model: {str(e)}")
            self._pipeline = None

    
    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment label, score, and confidence
        """
        if not text or not text.strip():
            return {
                "label": "Neutral",
                "score": 0.5,
                "confidence": 0.0,
                "all_scores": {}
            }
        
        try:
            # Preprocess text
            processed_text = self._text_processor.clean_text(text)
            
            # Use ML model if available
            if self.sentiment_pipeline is not None:
                return self._analyze_with_model(processed_text)
            else:
                # Fallback to rule-based analysis
                logger.warning("ML model not available, using rule-based sentiment analysis")
                return self._analyze_rule_based(processed_text)
                
        except Exception as e:
            logger.error(f"Sentiment analysis error for text '{text[:50]}...': {str(e)}")
            return self._analyze_rule_based(text)
    
    def _analyze_with_model(self, text: str) -> Dict[str, any]:
        """Analyze sentiment using the transformer model."""
        try:
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            # Get predictions
            results = self.sentiment_pipeline(text)
            
            if not results or len(results) == 0:
                return self._get_neutral_result()
            
            # Process results (return_all_scores=True gives list of all labels)
            all_scores = {}
            best_result = None
            best_score = 0
            
            for result in results[0]:  # results is a list of lists
                label = result["label"]
                score = result["score"]
                
                # Map label to readable format
                readable_label = self.labels_map.get(label, label)
                all_scores[readable_label] = score
                
                if score > best_score:
                    best_score = score
                    best_result = {
                        "label": readable_label,
                        "score": score
                    }
            
            if best_result is None:
                return self._get_neutral_result()
            
            return {
                "label": best_result["label"],
                "score": best_result["score"],
                "confidence": best_result["score"],
                "all_scores": all_scores
            }
            
        except Exception as e:
            logger.error(f"Model sentiment analysis error: {str(e)}")
            raise
    
    def _analyze_rule_based(self, text: str) -> Dict[str, any]:
        """
        Fallback rule-based sentiment analysis.
        Uses simple keyword matching for basic sentiment detection.
        """
        text_lower = text.lower()
        
        # Define sentiment keywords
        positive_keywords = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic", "awesome",
            "love", "like", "best", "perfect", "nice", "beautiful", "helpful", "easy",
            "fast", "quick", "smooth", "efficient", "useful", "clear", "simple",
            "अच्छा", "बेहतरीन", "शानदार", "अति", "सुंदर", "आसान", "तेज़", "सुविधाजनक"
        }
        
        negative_keywords = {
            "bad", "terrible", "awful", "horrible", "worst", "hate", "dislike", "poor",
            "slow", "difficult", "hard", "confusing", "broken", "error", "problem",
            "issue", "bug", "crash", "fail", "wrong", "annoying", "frustrating",
            "बुरा", "खराब", "गलत", "मुश्किल", "धीमा", "परेशानी", "समस्या", "त्रुटि"
        }
        
        neutral_keywords = {
            "okay", "ok", "fine", "average", "normal", "standard", "typical",
            "ठीक", "सामान्य", "औसत"
        }
        
        # Count keyword occurrences
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        neutral_count = sum(1 for word in neutral_keywords if word in text_lower)
        
        # Determine sentiment
        total_count = positive_count + negative_count + neutral_count
        
        if total_count == 0:
            return {
                "label": "Neutral",
                "score": 0.5,
                "confidence": 0.1,
                "all_scores": {"Positive": 0.33, "Negative": 0.33, "Neutral": 0.34}
            }
        
        positive_score = positive_count / total_count if total_count > 0 else 0
        negative_score = negative_count / total_count if total_count > 0 else 0
        neutral_score = neutral_count / total_count if total_count > 0 else 0
        
        # Adjust scores based on simple rules
        if positive_score > negative_score and positive_score > neutral_score:
            label = "Positive"
            score = 0.5 + (positive_score * 0.4)  # Scale to 0.5-0.9
        elif negative_score > positive_score and negative_score > neutral_score:
            label = "Negative" 
            score = 0.5 + (negative_score * 0.4)  # Scale to 0.5-0.9
        else:
            label = "Neutral"
            score = 0.5 + (neutral_score * 0.3)  # Scale to 0.5-0.8
        
        return {
            "label": label,
            "score": min(score, 0.95),  # Cap at 0.95
            "confidence": min(total_count * 0.1, 0.8),  # Higher confidence with more keywords
            "all_scores": {
                "Positive": positive_score,
                "Negative": negative_score,
                "Neutral": neutral_score
            }
        }
    
    def _get_neutral_result(self) -> Dict[str, any]:
        """Return neutral sentiment result."""
        return {
            "label": "Neutral",
            "score": 0.5,
            "confidence": 0.5,
            "all_scores": {"Positive": 0.33, "Negative": 0.33, "Neutral": 0.34}
        }
    
    @lru_cache(maxsize=1000)
    def analyze_sentiment_cached(self, text: str) -> Dict[str, any]:
        """Cached sentiment analysis for frequently used texts."""
        return self.analyze_sentiment(text)
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        if not texts:
            return []
        
        results = []
        
        try:
            # Process in smaller batches to avoid memory issues
            batch_size = 32
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_results = []
                
                for text in batch_texts:
                    result = self.analyze_sentiment(text)
                    batch_results.append(result)
                
                results.extend(batch_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch sentiment analysis error: {str(e)}")
            # Fallback to individual processing
            return [self.analyze_sentiment(text) for text in texts]
    
    def is_model_available(self) -> bool:
        """Check if the sentiment model is available."""
        return self._pipeline is not None
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "cache_dir": self.cache_dir,
            "labels_map": self.labels_map,
            "model_loaded": self._pipeline is not None,
            "device": "GPU" if torch.cuda.is_available() else "CPU"
        }
    
    def analyze_aspect_sentiment(self, text: str, aspect_context: str = "") -> Dict[str, any]:
        """
        Analyze sentiment of text in the context of a specific aspect.
        
        Args:
            text: Text to analyze
            aspect_context: Context about the aspect being analyzed
            
        Returns:
            Sentiment analysis result
        """
        # Combine text with aspect context for better analysis
        if aspect_context:
            combined_text = f"{aspect_context}: {text}"
        else:
            combined_text = text
        
        return self.analyze_sentiment(combined_text)
