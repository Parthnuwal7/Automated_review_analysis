"""Aspect-based sentiment analysis (ABSA) using embedding clustering and rule-based approaches."""

import spacy
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import logging
import re

from models.embedding import EmbeddingService
from pipeline.sentiment import SentimentAnalyzer
from utils.text import TextProcessor
from config import ASPECT_KEYWORDS, PROCESSING_CONFIG

logger = logging.getLogger(__name__)


class AspectExtractor:
    """Aspect extraction and sentiment analysis using hybrid approach."""
    
    def __init__(self):
        """Initialize aspect extraction components."""
        self.embedding_service = EmbeddingService()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.text_processor = TextProcessor()
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Configuration
        self.min_frequency = PROCESSING_CONFIG["min_aspect_frequency"]
        self.max_aspects = PROCESSING_CONFIG["max_aspects"]
        self.clustering_threshold = PROCESSING_CONFIG["clustering_threshold"]
        
        # Aspect mapping
        self.aspect_clusters = {}
        self.cluster_labels = {}
        
        logger.info("AspectExtractor initialized")
    
    def extract_aspects_with_sentiment(self, text: str) -> List[Dict[str, any]]:
        """
        Extract aspects and their sentiments from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of aspect dictionaries with sentiment information
        """
        if not text or not text.strip():
            return []
        
        try:
            # Step 1: Extract potential aspects from text
            aspect_candidates = self._extract_aspect_candidates(text)
            
            if not aspect_candidates:
                return []
            
            # Step 2: Match with known aspects or create new ones
            matched_aspects = self._match_aspects(aspect_candidates, text)
            
            # Step 3: Analyze sentiment for each aspect
            aspect_sentiments = []
            for aspect_info in matched_aspects:
                sentiment_result = self._analyze_aspect_sentiment(
                    text, 
                    aspect_info["matched_text"],
                    aspect_info["aspect_label"]
                )
                
                aspect_sentiments.append({
                    "aspect_id": aspect_info["aspect_id"],
                    "aspect_label": aspect_info["aspect_label"],
                    "matched_text": aspect_info["matched_text"],
                    "aspect_sentiment": sentiment_result["label"],
                    "aspect_sentiment_score": sentiment_result["score"]
                })
            
            return aspect_sentiments
            
        except Exception as e:
            logger.error(f"Aspect extraction error for text '{text[:50]}...': {str(e)}")
            return []
    
    def _extract_aspect_candidates(self, text: str) -> List[str]:
        """Extract potential aspect candidates from text."""
        candidates = []
        
        try:
            # Method 1: Use spaCy for noun phrase extraction
            if self.nlp:
                candidates.extend(self._extract_noun_phrases(text))
            
            # Method 2: Use predefined aspect keywords
            candidates.extend(self._extract_keyword_matches(text))
            
            # Method 3: Use rule-based patterns
            candidates.extend(self._extract_pattern_matches(text))
            
            # Clean and deduplicate candidates
            cleaned_candidates = []
            seen = set()
            
            for candidate in candidates:
                cleaned = self.text_processor.clean_text(candidate.lower().strip())
                if (cleaned and 
                    len(cleaned.split()) <= 3 and  # Max 3 words
                    len(cleaned) > 2 and  # Min 2 characters
                    cleaned not in seen):
                    cleaned_candidates.append(cleaned)
                    seen.add(cleaned)
            
            return cleaned_candidates
            
        except Exception as e:
            logger.error(f"Error extracting aspect candidates: {str(e)}")
            return []
    
    def _extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases using spaCy."""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            noun_phrases = []
            
            # Extract noun chunks
            for chunk in doc.noun_chunks:
                if (len(chunk.text.split()) <= 3 and  # Max 3 words
                    not chunk.root.is_stop and        # Not stopword
                    chunk.root.pos_ in ["NOUN", "PROPN"]):  # Noun or proper noun
                    noun_phrases.append(chunk.text.lower())
            
            # Extract individual nouns
            for token in doc:
                if (token.pos_ in ["NOUN", "PROPN"] and 
                    not token.is_stop and 
                    not token.is_punct and
                    len(token.text) > 2):
                    noun_phrases.append(token.text.lower())
            
            return noun_phrases
            
        except Exception as e:
            logger.error(f"Error extracting noun phrases: {str(e)}")
            return []
    
    def _extract_keyword_matches(self, text: str) -> List[str]:
        """Extract aspects using predefined keywords."""
        text_lower = text.lower()
        matches = []
        
        for aspect_label, keywords in ASPECT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    matches.append(aspect_label)
                    # Also add the actual keyword found
                    matches.append(keyword)
        
        return matches
    
    def _extract_pattern_matches(self, text: str) -> List[str]:
        """Extract aspects using pattern matching."""
        patterns = [
            # "the X is/was" patterns
            r"the (\w+(?:\s+\w+){0,2}) (?:is|was|are|were)",
            # "X problem/issue" patterns
            r"(\w+(?:\s+\w+){0,2}) (?:problem|issue|error|bug)",
            # "when I X" patterns
            r"when I (\w+(?:\s+\w+){0,2})",
            # "difficulty with X" patterns
            r"difficulty with (\w+(?:\s+\w+){0,2})",
            # "X feature/function" patterns
            r"(\w+(?:\s+\w+){0,2}) (?:feature|function|functionality)",
            # "X works/doesn't work" patterns
            r"(\w+(?:\s+\w+){0,2}) (?:works|doesn't work|not working)",
            # "using X" patterns
            r"using (\w+(?:\s+\w+){0,2})",
            # "X takes too long" patterns
            r"(\w+(?:\s+\w+){0,2}) (?:takes|taking) (?:too )?long",
        ]
        
        matches = []
        for pattern in patterns:
            found_matches = re.findall(pattern, text, re.IGNORECASE)
            for match in found_matches:
                if isinstance(match, tuple):
                    # Handle multiple groups in regex
                    matches.extend([m.strip() for m in match if m.strip()])
                else:
                    matches.append(match.strip())
        
        return matches
    
    def _match_aspects(self, candidates: List[str], original_text: str) -> List[Dict[str, any]]:
        """Match candidates with existing aspect clusters or create new ones."""
        matched_aspects = []
        
        try:
            # Get embeddings for candidates
            candidate_embeddings = self.embedding_service.get_embeddings(candidates)
            
            if not self.aspect_clusters:
                # First time: create initial clusters
                self._initialize_clusters(candidates, candidate_embeddings, original_text)
            
            # Match each candidate to existing clusters
            for i, candidate in enumerate(candidates):
                best_cluster = self._find_best_cluster(candidate_embeddings[i])
                
                if best_cluster is not None:
                    cluster_id, similarity = best_cluster
                    aspect_info = {
                        "aspect_id": cluster_id,
                        "aspect_label": self.cluster_labels.get(cluster_id, f"aspect_{cluster_id}"),
                        "matched_text": self._extract_context_sentence(candidate, original_text),
                        "similarity": similarity
                    }
                    matched_aspects.append(aspect_info)
            
            return matched_aspects
            
        except Exception as e:
            logger.error(f"Error matching aspects: {str(e)}")
            # Fallback to simple keyword matching
            return self._simple_keyword_matching(candidates, original_text)
    
    def _initialize_clusters(self, candidates: List[str], embeddings: np.ndarray, original_text: str):
        """Initialize aspect clusters from candidates."""
        if len(candidates) == 0 or embeddings.size == 0:
            return
        
        try:
            # Use clustering if we have enough candidates
            if len(candidates) >= 3:
                # Compute similarity matrix
                similarity_matrix = cosine_similarity(embeddings)
                
                # Convert to distance matrix
                distance_matrix = 1 - similarity_matrix
                
                # Perform clustering
                n_clusters = min(len(candidates) // 2, self.max_aspects)
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='precomputed',
                    linkage='average'
                )
                cluster_labels = clustering.fit_predict(distance_matrix)
                
                # Create cluster mappings
                for i, label in enumerate(cluster_labels):
                    if label not in self.aspect_clusters:
                        self.aspect_clusters[label] = {
                            "centroid": embeddings[i],
                            "candidates": [candidates[i]],
                            "count": 1
                        }
                        # Generate cluster label
                        self.cluster_labels[label] = self._generate_cluster_label(candidates[i])
                    else:
                        # Update cluster
                        self.aspect_clusters[label]["candidates"].append(candidates[i])
                        self.aspect_clusters[label]["count"] += 1
                        # Update centroid
                        current_centroid = self.aspect_clusters[label]["centroid"]
                        new_centroid = (current_centroid + embeddings[i]) / 2
                        self.aspect_clusters[label]["centroid"] = new_centroid
            
            else:
                # Few candidates: create individual clusters
                for i, candidate in enumerate(candidates):
                    cluster_id = i
                    self.aspect_clusters[cluster_id] = {
                        "centroid": embeddings[i],
                        "candidates": [candidate],
                        "count": 1
                    }
                    self.cluster_labels[cluster_id] = self._generate_cluster_label(candidate)
            
            logger.info(f"Initialized {len(self.aspect_clusters)} aspect clusters")
            
        except Exception as e:
            logger.error(f"Error initializing clusters: {str(e)}")
            # Fallback: create simple mapping
            for i, candidate in enumerate(candidates):
                self.aspect_clusters[i] = {
                    "centroid": embeddings[i] if i < len(embeddings) else np.zeros(384),
                    "candidates": [candidate],
                    "count": 1
                }
                self.cluster_labels[i] = self._generate_cluster_label(candidate)
    
    def _find_best_cluster(self, candidate_embedding: np.ndarray) -> Optional[Tuple[int, float]]:
        """Find the best matching cluster for a candidate."""
        if not self.aspect_clusters:
            return None
        
        best_cluster = None
        best_similarity = -1
        
        for cluster_id, cluster_info in self.aspect_clusters.items():
            centroid = cluster_info["centroid"]
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                candidate_embedding.reshape(1, -1),
                centroid.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity and similarity > self.clustering_threshold:
                best_similarity = similarity
                best_cluster = (cluster_id, similarity)
        
        return best_cluster
    
    def _generate_cluster_label(self, representative_term: str) -> str:
        """Generate a human-readable label for a cluster."""
        # Check if it matches a predefined aspect
        for aspect_label, keywords in ASPECT_KEYWORDS.items():
            if any(keyword in representative_term.lower() for keyword in keywords):
                return aspect_label
        
        # Generate generic label
        cleaned_term = representative_term.replace("_", " ").title()
        return cleaned_term
    
    def _extract_context_sentence(self, aspect_term: str, text: str) -> str:
        """Extract the sentence containing the aspect term."""
        sentences = text.split('.')
        
        for sentence in sentences:
            if aspect_term.lower() in sentence.lower():
                return sentence.strip()
        
        # Fallback: return the aspect term itself
        return aspect_term
    
    def _analyze_aspect_sentiment(self, full_text: str, aspect_context: str, aspect_label: str) -> Dict[str, any]:
        """Analyze sentiment for a specific aspect in context."""
        try:
            # Use the aspect context for more focused sentiment analysis
            context_text = aspect_context if aspect_context else full_text
            
            # Get sentiment using the sentiment analyzer
            sentiment_result = self.sentiment_analyzer.analyze_aspect_sentiment(
                context_text, aspect_label
            )
            
            return sentiment_result
            
        except Exception as e:
            logger.error(f"Error analyzing aspect sentiment: {str(e)}")
            return {
                "label": "Neutral",
                "score": 0.5,
                "confidence": 0.1
            }
    
    def _simple_keyword_matching(self, candidates: List[str], original_text: str) -> List[Dict[str, any]]:
        """Fallback simple keyword matching when clustering fails."""
        matched_aspects = []
        
        for i, candidate in enumerate(candidates):
            # Simple mapping based on predefined keywords
            aspect_label = "General"
            
            for predefined_aspect, keywords in ASPECT_KEYWORDS.items():
                if any(keyword in candidate.lower() for keyword in keywords):
                    aspect_label = predefined_aspect
                    break
            
            matched_aspects.append({
                "aspect_id": i,
                "aspect_label": aspect_label,
                "matched_text": self._extract_context_sentence(candidate, original_text),
                "similarity": 0.5  # Default similarity
            })
        
        return matched_aspects
    
    def get_aspect_statistics(self) -> Dict[str, any]:
        """Get statistics about extracted aspects."""
        if not self.aspect_clusters:
            return {"total_clusters": 0, "cluster_details": {}}
        
        stats = {
            "total_clusters": len(self.aspect_clusters),
            "cluster_details": {}
        }
        
        for cluster_id, cluster_info in self.aspect_clusters.items():
            label = self.cluster_labels.get(cluster_id, f"cluster_{cluster_id}")
            stats["cluster_details"][label] = {
                "count": cluster_info["count"],
                "candidates": cluster_info["candidates"]
            }
        
        return stats
    
    def update_aspect_clusters(self, new_aspects: List[str], new_embeddings: np.ndarray):
        """Update existing clusters with new aspects."""
        try:
            if not new_aspects or new_embeddings.size == 0:
                return
            
            for i, aspect in enumerate(new_aspects):
                best_cluster = self._find_best_cluster(new_embeddings[i])
                
                if best_cluster is not None:
                    cluster_id, similarity = best_cluster
                    # Update existing cluster
                    cluster_info = self.aspect_clusters[cluster_id]
                    cluster_info["candidates"].append(aspect)
                    cluster_info["count"] += 1
                    
                    # Update centroid
                    current_centroid = cluster_info["centroid"]
                    new_centroid = (current_centroid + new_embeddings[i]) / 2
                    cluster_info["centroid"] = new_centroid
                    
                else:
                    # Create new cluster
                    new_cluster_id = max(self.aspect_clusters.keys()) + 1 if self.aspect_clusters else 0
                    self.aspect_clusters[new_cluster_id] = {
                        "centroid": new_embeddings[i],
                        "candidates": [aspect],
                        "count": 1
                    }
                    self.cluster_labels[new_cluster_id] = self._generate_cluster_label(aspect)
        
        except Exception as e:
            logger.error(f"Error updating aspect clusters: {str(e)}")
    
    def is_service_available(self) -> bool:
        """Check if the aspect extraction service is fully available."""
        return (self.embedding_service.is_available() and 
                self.sentiment_analyzer.is_model_available() and
                self.nlp is not None)
    
    def get_service_info(self) -> Dict[str, any]:
        """Get information about the aspect extraction service."""
        return {
            "spacy_model": "en_core_web_sm" if self.nlp else None,
            "embedding_service": self.embedding_service.get_model_info(),
            "sentiment_service": self.sentiment_analyzer.get_model_info(),
            "clustering_config": {
                "min_frequency": self.min_frequency,
                "max_aspects": self.max_aspects,
                "clustering_threshold": self.clustering_threshold
            },
            "active_clusters": len(self.aspect_clusters),
            "service_available": self.is_service_available()
        }
