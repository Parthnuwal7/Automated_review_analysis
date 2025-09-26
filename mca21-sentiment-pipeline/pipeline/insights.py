"""Actionable insights generation from processed reviews."""

import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import logging
from datetime import datetime, timedelta

from utils.text import TextProcessor

logger = logging.getLogger(__name__)


class InsightGenerator:
    """Generate actionable insights from processed review data."""
    
    def __init__(self):
        """Initialize insight generation components."""
        self.text_processor = TextProcessor()
        logger.info("InsightGenerator initialized")
    
    def generate_insights(self, processed_reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive insights from processed reviews.
        
        Args:
            processed_reviews: List of processed review dictionaries
            
        Returns:
            Dictionary containing pain points and recommendations
        """
        if not processed_reviews:
            return {"pain_points": [], "recommendations": []}
        
        try:
            # Generate pain points
            pain_points = self._generate_pain_points(processed_reviews)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(processed_reviews, pain_points)
            
            # Additional insights
            trend_insights = self._generate_trend_insights(processed_reviews)
            language_insights = self._generate_language_insights(processed_reviews)
            
            return {
                "pain_points": pain_points,
                "recommendations": recommendations,
                "trends": trend_insights,
                "language_breakdown": language_insights,
                "summary_stats": self._generate_summary_stats(processed_reviews)
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return {"pain_points": [], "recommendations": [], "error": str(e)}
    
    def _generate_pain_points(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate ranked pain points based on aspect frequency and negativity."""
        aspect_data = defaultdict(lambda: {
            "total_mentions": 0,
            "negative_mentions": 0,
            "neutral_mentions": 0,
            "positive_mentions": 0,
            "examples": [],
            "avg_sentiment_score": 0.0,
            "sentiment_scores": []
        })
        
        # Collect aspect data
        for review in reviews:
            for aspect in review.get("aspects", []):
                aspect_label = aspect["aspect_label"]
                aspect_sentiment = aspect["aspect_sentiment"]
                sentiment_score = aspect.get("aspect_sentiment_score", 0.0)
                matched_text = aspect.get("matched_text", "")
                
                # Update counters
                aspect_data[aspect_label]["total_mentions"] += 1
                aspect_data[aspect_label]["sentiment_scores"].append(sentiment_score)
                
                if aspect_sentiment == "Negative":
                    aspect_data[aspect_label]["negative_mentions"] += 1
                elif aspect_sentiment == "Neutral":
                    aspect_data[aspect_label]["neutral_mentions"] += 1
                else:
                    aspect_data[aspect_label]["positive_mentions"] += 1
                
                # Store examples (limit to 5 per aspect)
                if len(aspect_data[aspect_label]["examples"]) < 5:
                    aspect_data[aspect_label]["examples"].append({
                        "text": matched_text,
                        "sentiment": aspect_sentiment,
                        "review_id": review.get("review_id", "unknown")
                    })
        
        # Calculate pain point scores
        pain_points = []
        for aspect_label, data in aspect_data.items():
            if data["total_mentions"] < 2:  # Skip aspects with too few mentions
                continue
            
            # Calculate metrics
            negative_ratio = data["negative_mentions"] / data["total_mentions"]
            frequency_weight = min(np.log(data["total_mentions"] + 1), 3.0)  # Cap at log(4)
            
            # Pain point score: negative ratio weighted by frequency
            pain_score = negative_ratio * frequency_weight
            
            # Average sentiment score
            avg_sentiment = np.mean(data["sentiment_scores"]) if data["sentiment_scores"] else 0.5
            
            pain_points.append({
                "aspect": aspect_label,
                "score": pain_score,
                "total_mentions": data["total_mentions"],
                "negative_mentions": data["negative_mentions"],
                "negative_ratio": negative_ratio,
                "avg_sentiment_score": avg_sentiment,
                "examples": [ex["text"] for ex in data["examples"][:3]],
                "severity": self._categorize_severity(pain_score, negative_ratio)
            })
        
        # Sort by pain score and return top pain points
        pain_points.sort(key=lambda x: x["score"], reverse=True)
        return pain_points[:10]  # Top 10 pain points
    
    def _categorize_severity(self, pain_score: float, negative_ratio: float) -> str:
        """Categorize the severity of a pain point."""
        if pain_score > 2.0 and negative_ratio > 0.7:
            return "Critical"
        elif pain_score > 1.5 and negative_ratio > 0.5:
            return "High"
        elif pain_score > 1.0 and negative_ratio > 0.3:
            return "Medium"
        else:
            return "Low"
    
    def _generate_recommendations(self, reviews: List[Dict[str, Any]], pain_points: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on pain points."""
        recommendations = []
        
        # General statistics
        total_reviews = len(reviews)
        negative_reviews = len([r for r in reviews if r.get("sentiment") == "Negative"])
        complaint_reviews = len([r for r in reviews if r.get("intent") == "Complaint"])
        
        # High-level recommendations based on overall sentiment
        if negative_reviews / total_reviews > 0.4:
            recommendations.append(
                f"🚨 High negative sentiment detected ({negative_reviews}/{total_reviews} reviews). "
                "Consider immediate review of user experience and system performance."
            )
        
        if complaint_reviews / total_reviews > 0.3:
            recommendations.append(
                f"📞 Significant complaint volume ({complaint_reviews}/{total_reviews} reviews). "
                "Establish dedicated support channels and FAQ resources."
            )
        
        # Specific recommendations based on top pain points
        for i, pain_point in enumerate(pain_points[:5], 1):
            aspect = pain_point["aspect"]
            severity = pain_point["severity"]
            
            if aspect.lower() in ["performance", "speed", "loading"]:
                recommendations.append(
                    f"{i}. 🚀 **Performance Issues**: {severity} priority - "
                    f"Optimize system performance and loading times. "
                    f"({pain_point['negative_mentions']} complaints)"
                )
            
            elif aspect.lower() in ["user_interface", "ui", "design", "navigation"]:
                recommendations.append(
                    f"{i}. 🎨 **UI/UX Improvements**: {severity} priority - "
                    f"Redesign interface for better usability and navigation. "
                    f"({pain_point['negative_mentions']} complaints)"
                )
            
            elif aspect.lower() in ["functionality", "feature", "function"]:
                recommendations.append(
                    f"{i}. ⚙️ **Functionality Issues**: {severity} priority - "
                    f"Review and fix core functionality problems. "
                    f"({pain_point['negative_mentions']} complaints)"
                )
            
            elif aspect.lower() in ["support", "help", "customer service"]:
                recommendations.append(
                    f"{i}. 🤝 **Support Enhancement**: {severity} priority - "
                    f"Improve customer support response times and quality. "
                    f"({pain_point['negative_mentions']} complaints)"
                )
            
            elif aspect.lower() in ["documentation", "manual", "guide"]:
                recommendations.append(
                    f"{i}. 📚 **Documentation Update**: {severity} priority - "
                    f"Create clearer documentation and user guides. "
                    f"({pain_point['negative_mentions']} complaints)"
                )
            
            else:
                recommendations.append(
                    f"{i}. 🔧 **{aspect.title()} Issues**: {severity} priority - "
                    f"Address specific concerns related to {aspect}. "
                    f"({pain_point['negative_mentions']} complaints)"
                )
        
        # Language-specific recommendations
        hindi_reviews = [r for r in reviews if r.get("language") == "Hindi"]
        if len(hindi_reviews) > total_reviews * 0.3:
            recommendations.append(
                f"🌐 **Multilingual Support**: Consider enhancing Hindi language support "
                f"({len(hindi_reviews)}/{total_reviews} reviews in Hindi)."
            )
        
        # Intent-based recommendations
        suggestion_reviews = [r for r in reviews if r.get("intent") == "Suggestion"]
        if len(suggestion_reviews) > 10:
            recommendations.append(
                f"💡 **User Suggestions**: Review {len(suggestion_reviews)} user suggestions "
                "for potential feature improvements and enhancements."
            )
        
        return recommendations[:8]  # Limit to 8 key recommendations
    
    def _generate_trend_insights(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights about trends over time."""
        if not reviews:
            return {}
        
        try:
            # Parse timestamps and group by date
            daily_data = defaultdict(lambda: {
                "total": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0
            })
            
            for review in reviews:
                try:
                    # Parse timestamp
                    timestamp_str = review.get("timestamp", "")
                    if timestamp_str:
                        # Handle different timestamp formats
                        if "T" in timestamp_str:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:
                            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        
                        date_key = timestamp.date().isoformat()
                        sentiment = review.get("sentiment", "Neutral")
                        
                        daily_data[date_key]["total"] += 1
                        daily_data[date_key][sentiment.lower()] += 1
                
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error parsing timestamp '{timestamp_str}': {e}")
                    continue
            
            if not daily_data:
                return {"error": "No valid timestamps found"}
            
            # Calculate trends
            dates = sorted(daily_data.keys())
            if len(dates) > 1:
                recent_sentiment = self._calculate_recent_sentiment_trend(daily_data, dates)
                volume_trend = self._calculate_volume_trend(daily_data, dates)
                
                return {
                    "date_range": {"start": dates[0], "end": dates[-1]},
                    "total_days": len(dates),
                    "sentiment_trend": recent_sentiment,
                    "volume_trend": volume_trend,
                    "daily_averages": {
                        "total": sum(data["total"] for data in daily_data.values()) / len(dates),
                        "positive_ratio": sum(data["positive"] for data in daily_data.values()) / sum(data["total"] for data in daily_data.values()),
                        "negative_ratio": sum(data["negative"] for data in daily_data.values()) / sum(data["total"] for data in daily_data.values())
                    }
                }
            else:
                return {"single_day_data": daily_data}
                
        except Exception as e:
            logger.error(f"Error generating trend insights: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_recent_sentiment_trend(self, daily_data: Dict, dates: List[str]) -> str:
        """Calculate recent sentiment trend."""
        if len(dates) < 3:
            return "Insufficient data"
        
        # Compare last 3 days with previous 3 days
        recent_dates = dates[-3:]
        previous_dates = dates[-6:-3] if len(dates) >= 6 else dates[:-3]
        
        recent_negative_ratio = sum(daily_data[date]["negative"] for date in recent_dates) / sum(daily_data[date]["total"] for date in recent_dates)
        
        if previous_dates:
            previous_negative_ratio = sum(daily_data[date]["negative"] for date in previous_dates) / sum(daily_data[date]["total"] for date in previous_dates)
            
            if recent_negative_ratio > previous_negative_ratio + 0.1:
                return "Worsening"
            elif recent_negative_ratio < previous_negative_ratio - 0.1:
                return "Improving"
            else:
                return "Stable"
        else:
            return "Recent data only"
    
    def _calculate_volume_trend(self, daily_data: Dict, dates: List[str]) -> str:
        """Calculate volume trend."""
        if len(dates) < 2:
            return "Insufficient data"
        
        volumes = [daily_data[date]["total"] for date in dates]
        
        # Simple trend calculation
        if len(volumes) >= 3:
            recent_avg = sum(volumes[-3:]) / 3
            earlier_avg = sum(volumes[:-3]) / len(volumes[:-3]) if len(volumes) > 3 else volumes[0]
            
            if recent_avg > earlier_avg * 1.2:
                return "Increasing"
            elif recent_avg < earlier_avg * 0.8:
                return "Decreasing"
            else:
                return "Stable"
        else:
            return "Limited data"
    
    def _generate_language_insights(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights about language distribution and patterns."""
        language_data = defaultdict(lambda: {
            "count": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "intents": defaultdict(int),
            "avg_sentiment_score": []
        })
        
        for review in reviews:
            language = review.get("language", "Unknown")
            sentiment = review.get("sentiment", "Neutral")
            intent = review.get("intent", "General Feedback")
            sentiment_score = review.get("sentiment_score", 0.5)
            
            language_data[language]["count"] += 1
            language_data[language][sentiment.lower()] += 1
            language_data[language]["intents"][intent] += 1
            language_data[language]["avg_sentiment_score"].append(sentiment_score)
        
        # Process language insights
        insights = {}
        for language, data in language_data.items():
            total = data["count"]
            insights[language] = {
                "total_reviews": total,
                "sentiment_distribution": {
                    "positive": data["positive"] / total,
                    "negative": data["negative"] / total,
                    "neutral": data["neutral"] / total
                },
                "top_intent": max(data["intents"].items(), key=lambda x: x[1])[0] if data["intents"] else "Unknown",
                "avg_sentiment_score": np.mean(data["avg_sentiment_score"]) if data["avg_sentiment_score"] else 0.5
            }
        
        return insights
    
    def _generate_summary_stats(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_reviews = len(reviews)
        
        if total_reviews == 0:
            return {}
        
        # Sentiment distribution
        sentiments = [review.get("sentiment", "Neutral") for review in reviews]
        sentiment_counter = Counter(sentiments)
        
        # Intent distribution
        intents = [review.get("intent", "General Feedback") for review in reviews]
        intent_counter = Counter(intents)
        
        # Language distribution
        languages = [review.get("language", "Unknown") for review in reviews]
        language_counter = Counter(languages)
        
        # Aspect statistics
        all_aspects = []
        for review in reviews:
            all_aspects.extend([aspect["aspect_label"] for aspect in review.get("aspects", [])])
        
        aspect_counter = Counter(all_aspects)
        
        return {
            "total_reviews": total_reviews,
            "sentiment_distribution": dict(sentiment_counter),
            "intent_distribution": dict(intent_counter),
            "language_distribution": dict(language_counter),
            "top_aspects": dict(aspect_counter.most_common(10)),
            "avg_aspects_per_review": len(all_aspects) / total_reviews if total_reviews > 0 else 0,
            "unique_aspects": len(set(all_aspects))
        }
    
    def generate_executive_summary(self, insights: Dict[str, Any]) -> str:
        """Generate an executive summary from insights."""
        try:
            summary_parts = []
            
            # Overall statistics
            stats = insights.get("summary_stats", {})
            total_reviews = stats.get("total_reviews", 0)
            
            if total_reviews > 0:
                summary_parts.append(f"Analyzed {total_reviews} reviews across the e-consultation platform.")
                
                # Sentiment summary
                sentiment_dist = stats.get("sentiment_distribution", {})
                negative_pct = (sentiment_dist.get("Negative", 0) / total_reviews) * 100
                positive_pct = (sentiment_dist.get("Positive", 0) / total_reviews) * 100
                
                summary_parts.append(
                    f"Overall sentiment: {positive_pct:.1f}% positive, {negative_pct:.1f}% negative."
                )
                
                # Top pain points
                pain_points = insights.get("pain_points", [])
                if pain_points:
                    top_pain = pain_points[0]
                    summary_parts.append(
                        f"Primary concern: {top_pain['aspect']} with {top_pain['negative_mentions']} negative mentions."
                    )
                
                # Recommendations count
                recommendations = insights.get("recommendations", [])
                if recommendations:
                    summary_parts.append(f"Generated {len(recommendations)} actionable recommendations for improvement.")
                
                # Language insights
                lang_dist = stats.get("language_distribution", {})
                if len(lang_dist) > 1:
                    hindi_pct = (lang_dist.get("Hindi", 0) / total_reviews) * 100
                    summary_parts.append(f"Language distribution: {hindi_pct:.1f}% Hindi, {100-hindi_pct:.1f}% English.")
                
                return " ".join(summary_parts)
            
            else:
                return "No review data available for analysis."
                
        except Exception as e:
            logger.error(f"Error generating executive summary: {str(e)}")
            return "Error generating summary."
