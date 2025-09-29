"""Hindi to English translation service using Helsinki-NLP models."""

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, pipeline
from typing import Optional, List
import logging
from functools import lru_cache

from config import MODEL_CONFIGS

logger = logging.getLogger(__name__)


class TranslationService:
    """Service for translating Hindi text to English."""
    
    def __init__(self):
        """Initialize translation model."""
        self.model_config = MODEL_CONFIGS["translation"]
        self.model_name = self.model_config["model_name"]
        self.cache_dir = self.model_config["cache_dir"]
        self.max_length = self.model_config["max_length"]
        self.batch_size = self.model_config.get("batch_size", 8)
        self.src_lang = self.model_config.get("src_lang", "hi")
        self.tgt_lang = self.model_config.get("tgt_lang", "en")
        
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        
        logger.info(f"TranslationService initialized with model: {self.model_name}")
    
    @property
    def model(self):
        """Lazy load translation model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    @property
    def translation_pipeline(self):
        """Lazy load translation pipeline."""
        if self._pipeline is None:
            self._load_pipeline()
        return self._pipeline
    
    def _load_model(self):
        """Load the translation model and tokenizer."""
        try:
            logger.info(f"Loading translation model: {self.model_name}")
            
            self._tokenizer = M2M100Tokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self._model = M2M100ForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(device)
            
            # Set source language
            self._tokenizer.src_lang = self.src_lang
            
            logger.info(f"Translation model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Error loading translation model: {str(e)}")
            raise
    
    def _load_pipeline(self):
        """Load translation pipeline as fallback."""
        try:
            self._pipeline = pipeline(
                "translation",
                model=self.model_name,
                tokenizer=self.model_name,
                cache_dir=self.cache_dir,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Translation pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Error loading translation pipeline: {str(e)}")
            self._pipeline = None
    
    def translate_hindi_to_english(self, hindi_text: str) -> str:
        """
        Translate Hindi text to English.
        
        Args:
            hindi_text: Input Hindi text
            
        Returns:
            Translated English text
        """
        if not hindi_text or not hindi_text.strip():
            return ""
        
        try:
            # First try with direct model
            if self._model is not None and self._tokenizer is not None:
                return self._translate_with_model(hindi_text)
            
            # Fallback to pipeline
            elif self._pipeline is not None:
                return self._translate_with_pipeline(hindi_text)
            
            else:
                # Last resort: rule-based fallback
                logger.warning("Translation models not available, using fallback")
                return self._translate_fallback(hindi_text)
                
        except Exception as e:
            logger.error(f"Translation error for text '{hindi_text[:50]}...': {str(e)}")
            return hindi_text  # Return original text if translation fails
    
    def _translate_with_model(self, text: str) -> str:
        """Translate using direct model inference."""
        try:
            # Set source language
            self._tokenizer.src_lang = self.src_lang
            
            # Split text into sentences for better translation quality
            # Simple sentence splitting for now
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if not sentences:
                sentences = [text]
            
            translations = []
            
            with torch.no_grad():
                for i in range(0, len(sentences), self.batch_size):
                    batch = sentences[i:i + self.batch_size]
                    
                    # Tokenize batch
                    encoded = self._tokenizer(
                        batch, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=self.max_length
                    ).to(self._model.device)
                    
                    # Generate translation with forced target language
                    generated_tokens = self._model.generate(
                        **encoded,
                        forced_bos_token_id=self._tokenizer.get_lang_id(self.tgt_lang),
                        max_length=self.max_length,
                        num_return_sequences=1,
                        do_sample=False
                    )
                    
                    # Decode batch
                    translated_batch = self._tokenizer.batch_decode(
                        generated_tokens, 
                        skip_special_tokens=True
                    )
                    translations.extend(translated_batch)
            
            # Join translations
            final_text = " ".join(translations)
            return final_text.strip()
            
        except Exception as e:
            logger.error(f"Model translation error: {str(e)}")
            raise
    
    def _translate_with_pipeline(self, text: str) -> str:
        """Translate using HuggingFace pipeline."""
        try:
            result = self.translation_pipeline(
                text,
                max_length=self.max_length,
                truncation=True
            )
            
            if result and len(result) > 0:
                return result[0]["translation_text"].strip()
            else:
                return text
                
        except Exception as e:
            logger.error(f"Pipeline translation error: {str(e)}")
            raise
    
    def _translate_fallback(self, text: str) -> str:
        """
        Simple fallback translation using basic word mapping.
        This is a very basic implementation for emergency cases.
        """
        # Basic Hindi to English word mappings
        word_mappings = {
            "है": "is",
            "हैं": "are", 
            "और": "and",
            "का": "of",
            "के": "of",
            "की": "of",
            "को": "to",
            "से": "from",
            "में": "in",
            "पर": "on",
            "यह": "this",
            "वह": "that",
            "अच्छा": "good",
            "बुरा": "bad",
            "बहुत": "very",
            "धीमा": "slow",
            "तेज": "fast",
            "आसान": "easy",
            "मुश्किल": "difficult",
            "वेबसाइट": "website",
            "फॉर्म": "form",
            "सिस्टम": "system",
            "लॉगिन": "login",
            "पासवर्ड": "password"
        }
        
        words = text.split()
        translated_words = []
        
        for word in words:
            # Remove punctuation for mapping
            clean_word = word.strip(".,!?;:")
            mapped_word = word_mappings.get(clean_word, clean_word)
            
            # Preserve punctuation
            if clean_word != word:
                mapped_word = mapped_word + word[len(clean_word):]
            
            translated_words.append(mapped_word)
        
        return " ".join(translated_words)
    
    @lru_cache(maxsize=1000)
    def translate_cached(self, text: str) -> str:
        """Cached translation for frequently used texts."""
        return self.translate_hindi_to_english(text)
    
    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of Hindi texts to translate
            
        Returns:
            List of translated English texts
        """
        if not texts:
            return []
        
        try:
            # Filter empty texts
            non_empty_texts = [(i, text) for i, text in enumerate(texts) if text.strip()]
            
            if not non_empty_texts:
                return [""] * len(texts)
            
            # Translate non-empty texts
            results = [""] * len(texts)
            
            if self._model is not None and self._tokenizer is not None:
                # Use direct model for better batch processing
                batch_texts = [text for _, text in non_empty_texts]
                
                # Set source language
                self._tokenizer.src_lang = self.src_lang
                
                translations = []
                with torch.no_grad():
                    for i in range(0, len(batch_texts), self.batch_size):
                        batch = batch_texts[i:i + self.batch_size]
                        
                        # Tokenize batch
                        encoded = self._tokenizer(
                            batch, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True,
                            max_length=self.max_length
                        ).to(self._model.device)
                        
                        # Generate translation
                        generated_tokens = self._model.generate(
                            **encoded,
                            forced_bos_token_id=self._tokenizer.get_lang_id(self.tgt_lang),
                            max_length=self.max_length,
                            num_return_sequences=1,
                            do_sample=False
                        )
                        
                        # Decode batch
                        translated_batch = self._tokenizer.batch_decode(
                            generated_tokens, 
                            skip_special_tokens=True
                        )
                        translations.extend(translated_batch)
                
                # Map results back to original positions
                for (original_idx, _), translation in zip(non_empty_texts, translations):
                    results[original_idx] = translation.strip()
            
            elif self._pipeline is not None:
                # Fallback to pipeline for batch processing
                batch_texts = [text for _, text in non_empty_texts]
                batch_results = self.translation_pipeline(
                    batch_texts,
                    max_length=self.max_length,
                    truncation=True,
                    batch_size=self.batch_size
                )
                
                # Map results back to original positions
                for (original_idx, _), result in zip(non_empty_texts, batch_results):
                    results[original_idx] = result["translation_text"].strip()
            
            else:
                # Fallback to individual translation
                for original_idx, text in non_empty_texts:
                    results[original_idx] = self.translate_hindi_to_english(text)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch translation error: {str(e)}")
            # Fallback to individual translation
            return [self.translate_hindi_to_english(text) for text in texts]
    
    def is_translation_available(self) -> bool:
        """Check if translation service is available."""
        try:
            # Try to access model or pipeline
            return (self._model is not None and self._tokenizer is not None) or self._pipeline is not None
        except:
            return False
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "cache_dir": self.cache_dir,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "model_loaded": self._model is not None,
            "pipeline_loaded": self._pipeline is not None,
            "device": str(self.model.device) if self._model else "N/A"
        }
