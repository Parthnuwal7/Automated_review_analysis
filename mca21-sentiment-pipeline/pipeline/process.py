"""Main processing pipeline for review analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import logging
from tqdm import tqdm

from pipeline.translation import TranslationService
from pipeline.sentiment import SentimentAnalyzer
from pipeline.intent import IntentClassifier
from pipeline.aspect_extraction import AspectExtractor
from pipeline.insights import InsightGenerator
from utils.text import TextProcessor
from config import PROCESSING_CONFIG

logger = logging.getLogger(__name__)


class ReviewProcessor:
    """Main pipeline for processing e-consultation reviews."""
    
    def __init__(self):
        """Initialize all pipeline components."""
        logger.info("Initializing ReviewProcessor components...")
        
        self.text_processor = TextProcessor()
        self.translation_service = TranslationService()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.intent_classifier = IntentClassifier()
        self.aspect_extractor = AspectExtractor()
        self.insight_generator = InsightGenerator()
        
        logger.info("ReviewProcessor initialized successfully")
    
    def process_reviews_from_csv(self, csv_path: str) -> Dict[str, Any]:
        """
        Process reviews from CSV file.
        
        Args:
            csv_path: Path to CSV file with columns: text, timestamp, review_id
            
        Returns:
            Dictionary with processed results
        """
        logger.info(f"Processing reviews from {csv_path}")
        
        # Load data
        df = pd.read_csv(csv_path)
        required_cols = ["text", "timestamp", "review_id"]
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Process in batches
        batch_size = PROCESSING_CONFIG["batch_size"]
        all_results = []
        
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch_df = df.iloc[i:i+batch_size]
            batch_results = self._process_batch(batch_df)
            all_results.extend(batch_results)
        
        # Generate aggregated results
        results = self._aggregate_results(all_results)
        
        logger.info(f"Processed {len(all_results)} reviews successfully")
        return results
    
    def process_single_review(self, text: str, review_id: str, timestamp: str = None) -> Dict[str, Any]:
        """
        Process a single review.
        
        Args:
            text: Review text
            review_id: Unique review identifier
            timestamp: Review timestamp (ISO format)
            
        Returns:
            Processed review dictionary
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        return self._process_review_text(text, review_id, timestamp)
    
    def _process_batch(self, batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process a batch of reviews."""
        results = []
        
        for _, row in batch_df.iterrows():
            try:
                result = self._process_review_text(
                    text=row["text"],
                    review_id=str(row["review_id"]),
                    timestamp=str(row["timestamp"])
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing review {row['review_id']}: {str(e)}")
                # Add error placeholder
                results.append({
                    "review_id": str(row["review_id"]),
                    "original_text": row["text"],
                    "error": str(e),
                    "timestamp": str(row["timestamp"]),
                    "language": "Unknown",
                    "sentiment": "Neutral",
                    "intent": "General Feedback"
                })
        
        return results
    
    def _process_review_text(self, text: str, review_id: str, timestamp: str) -> Dict[str, Any]:
        """Process individual review text through the complete pipeline."""
        
        # 1. Text preprocessing and language detection
        processed_text = self.text_processor.clean_text(text)
        language = self.text_processor.detect_language(processed_text)
        
        # 2. Translation (if needed)
        translated_text = None
        if language == "Hindi":
            translated_text = self.translation_service.translate_hindi_to_english(processed_text)
            analysis_text = translated_text
        else:
            analysis_text = processed_text
        
        # 3. Sentiment analysis
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(analysis_text)
        
        # 4. Intent classification
        intent_result = self.intent_classifier.classify_intent(analysis_text)
        
        # 5. Aspect extraction and sentiment
        aspects = self.aspect_extractor.extract_aspects_with_sentiment(analysis_text)
        
        # 6. Token analysis
        tokens_positive, tokens_negative = self.text_processor.extract_sentiment_tokens(analysis_text, sentiment_result["label"])
        
        # 7. Compile results
        result = {
            "review_id": review_id,
            "original_text": text,
            "translated_text": translated_text,
            "language": "Hindi" if language == "Hindi" else "English",
            "timestamp": timestamp,
            "sentiment": sentiment_result["label"],
            "sentiment_score": sentiment_result["score"],
            "intent": intent_result["label"],
            "intent_score": intent_result["score"],
            "aspects": aspects,
            "tokens_positive": tokens_positive,
            "tokens_negative": tokens_negative
        }
        
        return result
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate individual results into summary statistics and insights."""
        
        # Filter out error results
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return {
                "reviews": results,
                "summary": {"error": "No valid reviews to process"},
                "wordcloud": {},
                "insights": {}
            }
        
        # Calculate statistical summary
        sentiments = [r["sentiment"] for r in valid_results]
        total_reviews = len(valid_results)
        
        statistical_summary = {
            "total_reviews": total_reviews,
            "positive_percent": (sentiments.count("Positive") / total_reviews * 100) if total_reviews > 0 else 0,
            "negative_percent": (sentiments.count("Negative") / total_reviews * 100) if total_reviews > 0 else 0,
            "neutral_percent": (sentiments.count("Neutral") / total_reviews * 100) if total_reviews > 0 else 0,
        }
        
        # Generate aspect summaries
        aspect_summaries = self._generate_aspect_summaries(valid_results)
        
        # Generate word cloud data
        wordcloud_data = self._generate_wordcloud_data(valid_results)
        
        # Generate insights
        insights = self.insight_generator.generate_insights(valid_results)
        
        return {
            "reviews": results,
            "summary": {
                "statistical_summary": statistical_summary,
                "textual_summary": "",  # To be filled by Gemini if enabled
                "aspect_summaries": aspect_summaries
            },
            "wordcloud": wordcloud_data,
            "insights": insights
        }
    
    def _generate_aspect_summaries(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate summaries for each aspect."""
        aspect_data = {}
        
        # Collect aspect information
        for result in results:
            for aspect in result.get("aspects", []):
                aspect_label = aspect["aspect_label"]
                if aspect_label not in aspect_data:
                    aspect_data[aspect_label] = {
                        "total": 0,
                        "positive": 0,
                        "negative": 0,
                        "neutral": 0,
                        "examples": []
                    }
                
                aspect_data[aspect_label]["total"] += 1
                aspect_data[aspect_label][aspect["aspect_sentiment"].lower()] += 1
                
                # Store examples
                if len(aspect_data[aspect_label]["examples"]) < 3:
                    aspect_data[aspect_label]["examples"].append(aspect.get("matched_text", ""))
        
        # Generate summaries
        summaries = {}
        for aspect_label, data in aspect_data.items():
            total = data["total"]
            pos_pct = (data["positive"] / total * 100) if total > 0 else 0
            neg_pct = (data["negative"] / total * 100) if total > 0 else 0
            
            summary = f"{aspect_label}: {total} mentions ({pos_pct:.1f}% positive, {neg_pct:.1f}% negative)"
            summaries[aspect_label] = summary
        
        return summaries
    
    def _generate_wordcloud_data(self, results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Generate word cloud token lists."""
        positive_tokens = []
        negative_tokens = []
        aspect_keywords = {}
        
        for result in results:
            # Collect sentiment tokens
            positive_tokens.extend(result.get("tokens_positive", []))
            negative_tokens.extend(result.get("tokens_negative", []))
            
            # Collect aspect keywords
            for aspect in result.get("aspects", []):
                aspect_label = aspect["aspect_label"]
                if aspect_label not in aspect_keywords:
                    aspect_keywords[aspect_label] = []
                
                # Extract keywords from matched text
                matched_text = aspect.get("matched_text", "")
                keywords = self.text_processor.extract_keywords(matched_text)
                aspect_keywords[aspect_label].extend(keywords)
        
        # Get top tokens
        from collections import Counter
        
        positive_counter = Counter(positive_tokens)
        negative_counter = Counter(negative_tokens)
        
        # Filter and get top tokens
        top_positive = [word for word, count in positive_counter.most_common(50) if count >= 2]
        top_negative = [word for word, count in negative_counter.most_common(50) if count >= 2]
        
        # Get top keywords per aspect
        aspect_top_keywords = {}
        for aspect, keywords in aspect_keywords.items():
            keyword_counter = Counter(keywords)
            aspect_top_keywords[aspect] = [word for word, count in keyword_counter.most_common(10)]
        
        return {
            "positive_keywords": top_positive,
            "negative_keywords": top_negative,
            "aspect_keywords": aspect_top_keywords
        }
    
    def save_results_to_file(self, results: Dict[str, Any], output_path: str):
        """Save processing results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to JSON serializable
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean data for JSON serialization
        cleaned_results = json.loads(json.dumps(results, default=convert_numpy))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
