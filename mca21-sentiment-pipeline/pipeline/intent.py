"""Intent classification using zero-shot learning."""

import torch
from transformers import pipeline
from typing import Dict, List, Optional
import logging
import numpy as np
from functools import lru_cache

from config import MODEL_CONFIGS
from utils.text import TextProcessor

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Zero-shot intent classification for e-consultation reviews."""
    
    def __init__(self):
        """Initialize intent classification components."""
        self.model_config = MODEL_CONFIGS["intent"]
        self.model_name = self.model_config["model_name"]
        self.cache_dir = self.model_config["cache_dir"]
        self.candidate_labels = self.model_config["candidate_labels"]
        self.threshold = self.model_config["threshold"]
        
        self._pipeline = None
        self._text_processor = TextProcessor()
        
        logger.info(f"IntentClassifier initialized with model: {self.model_name}")
    
    @property
    def classification_pipeline(self):
        """Lazy load zero-shot classification pipeline."""
        if self._pipeline is None:
            self._load_pipeline()
        return self._pipeline
    
    def _load_pipeline(self):
        """Load the zero-shot classification pipeline."""
        try:
            logger.info(f"Loading intent classification model: {self.model_name}")
            
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                cache_dir=self.cache_dir,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Intent classification pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading intent model: {str(e)}")
            # Try fallback model
            self._load_fallback_pipeline()
    
    def _load_fallback_pipeline(self):
        """Load a fallback zero-shot classification model."""
        try:
            fallback_model = "typeform/distilbert-base-uncased-mnli"
            logger.info(f"Loading fallback intent model: {fallback_model}")
            
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=fallback_model,
                cache_dir=self.cache_dir,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Fallback intent classification pipeline loaded")
            
        except Exception as e:
            logger.error(f"Error loading fallback intent model: {str(e)}")
            self._pipeline = None
    
    def classify_intent(self, text: str, custom_labels: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Classify intent of input text.
        
        Args:
            text: Input text to classify
            custom_labels: Custom intent labels (optional)
            
        Returns:
            Dictionary with intent label, score, and all scores
        """
        if not text or not text.strip():
            return {
                "label": "General Feedback",
                "score": 0.5,
                "all_scores": {}
            }
        
        try:
            # Preprocess text
            processed_text = self._text_processor.clean_text(text)
            
            # Use custom labels if provided
            labels = custom_labels or self.candidate_labels
            
            # Use ML model if available
            if self.classification_pipeline is not None:
                return self._classify_with_model(processed_text, labels)
            else:
                # Fallback to rule-based classification
                logger.warning("ML model not available, using rule-based intent classification")
                return self._classify_rule_based(processed_text, labels)
                
        except Exception as e:
            logger.error(f"Intent classification error for text '{text[:50]}...': {str(e)}")
            return self._classify_rule_based(text, self.candidate_labels)
    
    def _classify_with_model(self, text: str, labels: List[str]) -> Dict[str, any]:
        """Classify intent using the zero-shot model."""
        try:
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            # Get predictions
            result = self.classification_pipeline(text, labels)
            
            if not result:
                return self._get_default_result()
            
            # Process results
            all_scores = {}
            for label, score in zip(result["labels"], result["scores"]):
                all_scores[label] = float(score)
            
            top_label = result["labels"][0]
            top_score = float(result["scores"][0])
            
            # Apply threshold
            if top_score < self.threshold:
                top_label = "General Feedback"
                top_score = self.threshold
            
            return {
                "label": top_label,
                "score": top_score,
                "all_scores": all_scores
            }
            
        except Exception as e:
            logger.error(f"Model intent classification error: {str(e)}")
            raise
    
    def _classify_rule_based(self, text: str, labels: List[str]) -> Dict[str, any]:
        """
        Fallback rule-based intent classification.
        Uses keyword matching to determine intent.
        """
        text_lower = text.lower()
        
        # Define intent keywords
        intent_keywords = {
            "Complaint": {
                "bad", "terrible", "awful", "horrible", "worst", "hate", "dislike", "poor",
                "slow", "difficult", "hard", "confusing", "broken", "error", "problem",
                "issue", "bug", "crash", "fail", "wrong", "annoying", "frustrating",
                "complain", "complaint", "disappointed", "unsatisfied", "angry",
                "बुरा", "खराब", "गलत", "मुश्किल", "धीमा", "परेशानी", "समस्या", "त्रुटि", "शिकायत"
            },
            
            "Praise": {
                "good", "great", "excellent", "amazing", "wonderful", "fantastic", "awesome",
                "love", "like", "best", "perfect", "nice", "beautiful", "helpful", "easy",
                "fast", "quick", "smooth", "efficient", "useful", "clear", "simple",
                "thank", "thanks", "appreciate", "impressed", "satisfied", "happy",
                "अच्छा", "बेहतरीन", "शानदार", "धन्यवाद", "खुश", "संतुष्ट", "प्रशंसा"
            },
            
            "Suggestion": {
                "suggest", "suggestion", "recommend", "should", "could", "would",
                "improve", "improvement", "better", "enhance", "add", "include",
                "feature", "option", "consider", "maybe", "perhaps", "idea",
                "सुझाव", "सुधार", "बेहतर", "जोड़ना", "शामिल", "विचार", "सुझाना"
            },
            
            "General Feedback": {
                "feedback", "opinion", "think", "feel", "experience", "review",
                "comment", "note", "mention", "observe", "notice", "find",
                "प्रतिक्रिया", "राय", "अनुभव", "समीक्षा", "टिप्पणी"
            }
        }
        
        # Calculate scores for each intent
        intent_scores = {}
        total_matches = 0
        
        for intent_label in labels:
            if intent_label in intent_keywords:
                keywords = intent_keywords[intent_label]
                matches = sum(1 for keyword in keywords if keyword in text_lower)
                intent_scores[intent_label] = matches
                total_matches += matches
            else:
                intent_scores[intent_label] = 0
        
        # Normalize scores
        if total_matches == 0:
            # Default to General Feedback if no keywords found
            for label in labels:
                intent_scores[label] = 1.0 / len(labels)
            top_intent = "General Feedback"
            top_score = intent_scores.get("General Feedback", 0.25)
        else:
            # Convert to probabilities
            for label in labels:
                intent_scores[label] = intent_scores[label] / total_matches
            
            # Find top intent
            top_intent = max(intent_scores, key=intent_scores.get)
            top_score = intent_scores[top_intent]
        
        # Apply minimum confidence
        confidence_boost = min(total_matches * 0.1, 0.3)
        top_score = max(top_score, 0.25) + confidence_boost
        top_score = min(top_score, 0.95)  # Cap at 0.95
        
        return {
            "label": top_intent,
            "score": top_score,
            "all_scores": intent_scores
        }
    
    def _get_default_result(self) -> Dict[str, any]:
        """Return default intent result."""
        default_scores = {}
        for label in self.candidate_labels:
            default_scores[label] = 1.0 / len(self.candidate_labels)
        
        return {
            "label": "General Feedback",
            "score": default_scores.get("General Feedback", 0.25),
            "all_scores": default_scores
        }
    
    @lru_cache(maxsize=1000)
    def classify_intent_cached(self, text: str) -> Dict[str, any]:
        """Cached intent classification for frequently used texts."""
        return self.classify_intent(text)
    
    def classify_batch(self, texts: List[str], custom_labels: Optional[List[str]] = None) -> List[Dict[str, any]]:
        """
        Classify intent for a batch of texts.
        
        Args:
            texts: List of texts to classify
            custom_labels: Custom intent labels (optional)
            
        Returns:
            List of intent classification results
        """
        if not texts:
            return []
        
        results = []
        labels = custom_labels or self.candidate_labels
        
        try:
            # Process individually for now (batch processing can be added later)
            for text in texts:
                result = self.classify_intent(text, labels)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch intent classification error: {str(e)}")
            # Return default results
            return [self._get_default_result() for _ in texts]
    
    def is_model_available(self) -> bool:
        """Check if the intent classification model is available."""
        return self._pipeline is not None
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "cache_dir": self.cache_dir,
            "candidate_labels": self.candidate_labels,
            "threshold": self.threshold,
            "model_loaded": self._pipeline is not None,
            "device": "GPU" if torch.cuda.is_available() else "CPU"
        }
    
    def add_custom_intent_keywords(self, intent_label: str, keywords: List[str]):
        """
        Add custom keywords for a specific intent (for rule-based fallback).
        
        Args:
            intent_label: Intent label to add keywords for
            keywords: List of keywords to associate with the intent
        """
        # This could be implemented to enhance rule-based classification
        # For now, it's a placeholder for future enhancement
        logger.info(f"Custom keywords for {intent_label}: {keywords}")
    
    def evaluate_intent_confidence(self, text: str, predicted_intent: str) -> float:
        """
        Evaluate confidence in the predicted intent based on text characteristics.
        
        Args:
            text: Original text
            predicted_intent: Predicted intent label
            
        Returns:
            Confidence score (0-1)
        """
        text_lower = text.lower()
        
        # Basic confidence factors
        length_factor = min(len(text.split()) / 20.0, 1.0)  # Longer text = higher confidence
        
        # Intent-specific confidence indicators
        if predicted_intent == "Complaint":
            negative_indicators = ["not", "don't", "can't", "won't", "never", "bad", "slow", "error"]
            indicator_count = sum(1 for indicator in negative_indicators if indicator in text_lower)
            return min(0.5 + (indicator_count * 0.1) + length_factor * 0.3, 0.95)
        
        elif predicted_intent == "Praise":
            positive_indicators = ["good", "great", "love", "excellent", "thank", "amazing"]
            indicator_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
            return min(0.5 + (indicator_count * 0.1) + length_factor * 0.3, 0.95)
        
        elif predicted_intent == "Suggestion":
            suggestion_indicators = ["should", "could", "suggest", "recommend", "improve", "add"]
            indicator_count = sum(1 for indicator in suggestion_indicators if indicator in text_lower)
            return min(0.5 + (indicator_count * 0.15) + length_factor * 0.2, 0.95)
        
        else:  # General Feedback
            return 0.6 + length_factor * 0.2
