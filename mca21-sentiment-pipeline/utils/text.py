"""Text processing utilities for multilingual sentiment analysis."""

import re
import string
from typing import List, Dict, Set, Optional, Tuple
import logging
try:
    from langdetect import detect, LangDetectError
except ImportError:
    from langdetect import detect
    # Fallback for different langdetect versions
    class LangDetectError(Exception):
        pass
import unicodedata

from config import HINDI_STOPWORDS, LANGUAGE_CONFIG

logger = logging.getLogger(__name__)


class TextProcessor:
    """Comprehensive text processing utilities for multilingual text."""
    
    def __init__(self):
        """Initialize text processor with language-specific resources."""
        self.hindi_stopwords = HINDI_STOPWORDS
        self.english_stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
        
        # Patterns for cleaning
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'[\+]?[1-9]?[0-9]{7,15}')
        self.html_pattern = re.compile(r'<[^<]+?>')
        self.extra_whitespace_pattern = re.compile(r'\s+')
        
        logger.info("TextProcessor initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        try:
            # Remove HTML tags
            text = self.html_pattern.sub(' ', text)
            
            # Remove URLs
            text = self.url_pattern.sub(' [URL] ', text)
            
            # Remove email addresses
            text = self.email_pattern.sub(' [EMAIL] ', text)
            
            # Remove phone numbers
            text = self.phone_pattern.sub(' [PHONE] ', text)
            
            # Normalize Unicode characters
            text = unicodedata.normalize('NFKD', text)
            
            # Remove excessive punctuation (more than 2 consecutive)
            text = re.sub(r'([.!?]){3,}', r'\1\1', text)
            
            # Normalize whitespace
            text = self.extra_whitespace_pattern.sub(' ', text)
            
            # Strip and return
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text.strip() if text else ""
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code ('en', 'hi', etc.) or 'unknown'
        """
        if not text or not text.strip():
            return "unknown"
        
        try:
            # Clean text for better detection
            cleaned_text = self.clean_text(text)
            
            # Remove short words and numbers for better detection
            words = cleaned_text.split()
            filtered_words = [word for word in words if len(word) > 2 and not word.isdigit()]
            
            if len(filtered_words) < 3:
                # Too few words, try with original text
                detection_text = cleaned_text
            else:
                detection_text = ' '.join(filtered_words)
            
            # Use langdetect with error handling
            detected_lang = detect(detection_text)
            
            # Map to supported languages
            if detected_lang in LANGUAGE_CONFIG["supported_languages"]:
                return detected_lang
            elif detected_lang in ['hi', 'ur', 'bn', 'mr', 'gu', 'ta', 'te']:
                return 'hi'  # Treat other Indic languages as Hindi
            else:
                return 'en'  # Default to English
                
        except (LangDetectError, Exception) as e:
            logger.warning(f"Language detection failed: {str(e)}")
            # Fallback: check for Devanagari script
            return self._detect_script_based(text)
    
    def _detect_script_based(self, text: str) -> str:
        """Fallback language detection based on script."""
        try:
            # Count Devanagari characters
            devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
            total_chars = len([char for char in text if char.isalpha()])
            
            if total_chars == 0:
                return "en"
            
            devanagari_ratio = devanagari_count / total_chars
            
            if devanagari_ratio > 0.3:  # 30% threshold for Hindi
                return "hi"
            else:
                return "en"
                
        except Exception as e:
            logger.error(f"Script-based detection error: {str(e)}")
            return "en"
    
    def tokenize(self, text: str, language: str = "auto") -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            language: Language of the text ('en', 'hi', 'auto')
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        try:
            if language == "auto":
                language = self.detect_language(text)
            
            # Clean text first
            cleaned_text = self.clean_text(text)
            
            # Basic tokenization
            # Handle punctuation
            tokens = re.findall(r'\b\w+\b', cleaned_text.lower())
            
            # Filter out very short tokens and numbers
            tokens = [token for token in tokens if len(token) > 1 and not token.isdigit()]
            
            return tokens
            
        except Exception as e:
            logger.error(f"Tokenization error: {str(e)}")
            return text.split() if text else []
    
    def remove_stopwords(self, tokens: List[str], language: str = "auto") -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            language: Language of the tokens
            
        Returns:
            Filtered tokens without stopwords
        """
        if not tokens:
            return []
        
        try:
            if language == "auto":
                # Detect language from first few tokens
                sample_text = ' '.join(tokens[:10])
                language = self.detect_language(sample_text)
            
            # Select appropriate stopwords
            if language == "hi":
                stopwords = self.hindi_stopwords
            else:
                stopwords = self.english_stopwords
            
            # Filter tokens
            filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
            
            return filtered_tokens
            
        except Exception as e:
            logger.error(f"Stopword removal error: {str(e)}")
            return tokens
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text using frequency analysis.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords sorted by importance
        """
        try:
            # Tokenize and clean
            tokens = self.tokenize(text)
            
            # Remove stopwords
            keywords = self.remove_stopwords(tokens)
            
            # Filter by length and content
            keywords = [
                keyword for keyword in keywords 
                if len(keyword) >= 3 and keyword.isalpha()
            ]
            
            # Count frequency
            from collections import Counter
            keyword_counts = Counter(keywords)
            
            # Get most common keywords
            top_keywords = [keyword for keyword, count in keyword_counts.most_common(max_keywords)]
            
            return top_keywords
            
        except Exception as e:
            logger.error(f"Keyword extraction error: {str(e)}")
            return []
    
    def extract_sentiment_tokens(self, text: str, sentiment_label: str) -> Tuple[List[str], List[str]]:
        """
        Extract positive and negative sentiment tokens.
        
        Args:
            text: Text to analyze
            sentiment_label: Overall sentiment of the text
            
        Returns:
            Tuple of (positive_tokens, negative_tokens)
        """
        try:
            tokens = self.tokenize(text)
            filtered_tokens = self.remove_stopwords(tokens)
            
            # Define sentiment indicators
            positive_indicators = {
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome',
                'love', 'like', 'best', 'perfect', 'nice', 'beautiful', 'helpful', 'easy',
                'fast', 'quick', 'smooth', 'efficient', 'useful', 'clear', 'simple', 'thanks',
                'appreciate', 'satisfied', 'happy', 'pleased', 'impressed'
            }
            
            negative_indicators = {
                'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike', 'poor',
                'slow', 'difficult', 'hard', 'confusing', 'broken', 'error', 'problem',
                'issue', 'bug', 'crash', 'fail', 'wrong', 'annoying', 'frustrating',
                'disappointed', 'angry', 'upset', 'useless', 'complicated'
            }
            
            positive_tokens = []
            negative_tokens = []
            
            for token in filtered_tokens:
                token_lower = token.lower()
                if token_lower in positive_indicators:
                    positive_tokens.append(token)
                elif token_lower in negative_indicators:
                    negative_tokens.append(token)
                else:
                    # Contextual assignment based on overall sentiment
                    if sentiment_label == "Positive" and len(token) > 3:
                        positive_tokens.append(token)
                    elif sentiment_label == "Negative" and len(token) > 3:
                        negative_tokens.append(token)
            
            # Remove duplicates while preserving order
            positive_tokens = list(dict.fromkeys(positive_tokens))
            negative_tokens = list(dict.fromkeys(negative_tokens))
            
            return positive_tokens[:20], negative_tokens[:20]  # Limit to top 20 each
            
        except Exception as e:
            logger.error(f"Sentiment token extraction error: {str(e)}")
            return [], []
    
    def normalize_text(self, text: str, language: str = "auto") -> str:
        """
        Normalize text for consistent processing.
        
        Args:
            text: Text to normalize
            language: Language of the text
            
        Returns:
            Normalized text
        """
        try:
            if not text:
                return ""
            
            # Clean text
            normalized = self.clean_text(text)
            
            # Convert to lowercase
            normalized = normalized.lower()
            
            # Remove extra punctuation
            normalized = re.sub(r'[^\w\s\u0900-\u097F]', ' ', normalized)
            
            # Normalize whitespace
            normalized = self.extra_whitespace_pattern.sub(' ', normalized)
            
            return normalized.strip()
            
        except Exception as e:
            logger.error(f"Text normalization error: {str(e)}")
            return text
    
    def get_text_stats(self, text: str) -> Dict[str, any]:
        """
        Get comprehensive statistics about the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with text statistics
        """
        try:
            if not text:
                return {}
            
            # Basic stats
            char_count = len(text)
            word_count = len(text.split())
            sentence_count = len(re.split(r'[.!?]+', text))
            
            # Language detection
            language = self.detect_language(text)
            
            # Tokenization
            tokens = self.tokenize(text)
            unique_tokens = len(set(tokens))
            
            # Keywords
            keywords = self.extract_keywords(text, max_keywords=5)
            
            return {
                "character_count": char_count,
                "word_count": word_count,
                "sentence_count": max(1, sentence_count),
                "token_count": len(tokens),
                "unique_tokens": unique_tokens,
                "language": language,
                "keywords": keywords,
                "avg_word_length": sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0,
                "lexical_diversity": unique_tokens / len(tokens) if tokens else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting text stats: {str(e)}")
            return {"error": str(e)}
    
    def is_meaningful_text(self, text: str, min_length: int = 3) -> bool:
        """
        Check if text contains meaningful content.
        
        Args:
            text: Text to check
            min_length: Minimum character length
            
        Returns:
            True if text is meaningful
        """
        if not text or len(text.strip()) < min_length:
            return False
        
        # Check if text is not just punctuation or whitespace
        cleaned = re.sub(r'[^\w\u0900-\u097F]', '', text)
        return len(cleaned) >= min_length
    
    def extract_phrases(self, text: str, min_length: int = 2, max_length: int = 4) -> List[str]:
        """
        Extract meaningful phrases from text.
        
        Args:
            text: Text to extract phrases from
            min_length: Minimum phrase length in words
            max_length: Maximum phrase length in words
            
        Returns:
            List of extracted phrases
        """
        try:
            # Clean and tokenize
            tokens = self.tokenize(text)
            filtered_tokens = self.remove_stopwords(tokens)
            
            phrases = []
            
            # Extract n-grams
            for n in range(min_length, max_length + 1):
                for i in range(len(filtered_tokens) - n + 1):
                    phrase = ' '.join(filtered_tokens[i:i + n])
                    if len(phrase) > 3:  # Minimum phrase character length
                        phrases.append(phrase)
            
            # Remove duplicates and return
            return list(set(phrases))
            
        except Exception as e:
            logger.error(f"Phrase extraction error: {str(e)}")
            return []
