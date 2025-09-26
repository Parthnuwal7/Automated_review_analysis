"""Streamlit UI components and layouts for the sentiment analysis dashboard."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import base64

from pipeline.process import ReviewProcessor
from utils.db import DatabaseManager
from models.gemini_client import GeminiClient
from config import GEMINI_CONFIG, PROCESSING_CONFIG


class SentimentDashboard:
    """Main dashboard class for Streamlit UI."""
    
    _processor = None  # Class-level cache
    _db = None
    _gemini_client = None
    
    def __init__(self):
        """Initialize dashboard components with caching."""
        # Use class-level caching to prevent reinitialization
        if SentimentDashboard._processor is None:
            SentimentDashboard._processor = ReviewProcessor()
        
        if SentimentDashboard._db is None:
            SentimentDashboard._db = DatabaseManager()
        
        if SentimentDashboard._gemini_client is None and GEMINI_CONFIG["use_gemini"]:
            SentimentDashboard._gemini_client = GeminiClient()
        
        self.processor = SentimentDashboard._processor
        self.db = SentimentDashboard._db
        self.gemini_client = SentimentDashboard._gemini_client
        
        # Initialize session state
        if "processed_data" not in st.session_state:
            st.session_state.processed_data = None
        if "filter_state" not in st.session_state:
            st.session_state.filter_state = {}
    
    def show_dashboard(self):
        """Display main dashboard with analytics."""
        st.header("Analytics Dashboard")
        
        # Load processed data
        data = self._load_processed_data()
        if data is None:
            self._show_empty_state()
            return
        
        # Filters
        filtered_data = self._apply_filters(data)
        
        # KPIs
        self._display_kpis(filtered_data)
        
        # Charts
        col1, col2 = st.columns(2)
        with col1:
            self._display_sentiment_chart(filtered_data)
            self._display_aspect_chart(filtered_data)
        
        with col2:
            self._display_timeline_chart(filtered_data)
            self._display_intent_chart(filtered_data)
        
        # Word clouds and insights
        self._display_wordclouds(filtered_data)
        self._display_insights(filtered_data)
        
        # Sample reviews
        self._display_sample_reviews(filtered_data)
    
    def show_upload_process(self):
        """Display upload and processing interface."""
        st.header("Upload & Process Reviews")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with reviews",
            type=["csv"],
            help="File should contain columns: text, timestamp, review_id"
        )
        
        if uploaded_file is not None:
            # Preview data
            df = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Validate columns
            required_cols = ["text", "timestamp", "review_id"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return
            
            # Processing options
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.slider("Batch Size", 10, 100, PROCESSING_CONFIG["batch_size"])
                use_gemini = st.checkbox(
                    "Enable Gemini Summarization", 
                    value=GEMINI_CONFIG["use_gemini"],
                    help="Requires GEMINI_API_KEY in environment"
                )
            
            with col2:
                max_reviews = st.slider(
                    "Max Reviews to Process", 
                    100, 5000, 
                    min(len(df), PROCESSING_CONFIG["max_reviews_per_batch"])
                )
                save_to_db = st.checkbox("Save to Database", value=True)
            
            # Process button
            if st.button("🚀 Process Reviews", type="primary"):
                self._process_reviews(df, batch_size, use_gemini, max_reviews, save_to_db)
        
        else:
            # Show option to use sample data
            st.info("No file uploaded. You can use sample data to test the system.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 Generate Sample Data", type="secondary"):
                    self._generate_and_process_sample_data()
            
            with col2:
                if st.button("📊 Load Existing Data", type="secondary"):
                    self._load_existing_data()
    
    def show_settings(self):
        """Display settings and configuration options."""
        st.header("Settings & Configuration")
        
        # Model settings
        with st.expander("Model Configuration", expanded=True):
            st.markdown("**Current Models:**")
            st.code("""
            Translation: Helsinki-NLP/opus-mt-hi-en
            Sentiment: cardiffnlp/twitter-xlm-roberta-base-sentiment-latest
            Intent: facebook/bart-large-mnli
            Embeddings: sentence-transformers/all-MiniLM-L6-v2
            """)
        
        # Processing settings
        with st.expander("Processing Settings"):
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("Batch Size", value=PROCESSING_CONFIG["batch_size"])
                st.number_input("Min Aspect Frequency", value=PROCESSING_CONFIG["min_aspect_frequency"])
            with col2:
                st.number_input("Max Aspects", value=PROCESSING_CONFIG["max_aspects"])
                st.slider("Sentiment Threshold", 0.0, 1.0, PROCESSING_CONFIG["sentiment_threshold"])
        
        # API settings
        with st.expander("API Configuration"):
            gemini_key_set = "✅" if GEMINI_CONFIG["api_key"] else "❌"
            st.markdown(f"**Gemini API Key:** {gemini_key_set}")
            st.checkbox("Use Gemini Summarization", value=GEMINI_CONFIG["use_gemini"])
        
        # Database settings
        with st.expander("Database"):
            db_info = self.db.get_database_info()
            st.json(db_info)
            
            if st.button("Clear Database"):
                # Use a unique key for the confirm button to avoid conflicts
                confirm_key = f"confirm_clear_{hash(str(datetime.now()))}"
                if st.button("⚠️ Confirm Clear Database", type="secondary", key=confirm_key):
                    self.db.clear_all_data()
                    st.session_state.processed_data = None  # Clear session state too
                    st.success("Database cleared successfully!")
                    st.rerun()
    
    def show_about(self):
        """Display about page with documentation."""
        st.header("About MCA21 Analysis Pipeline")
        
        st.markdown("""
        ### 🎯 Purpose
        This application analyzes e-consultation reviews with comprehensive sentiment analysis,
        intent detection, and aspect-based sentiment analysis (ABSA) for both English and Hindi text.
        
        ### 🔧 Features
        - **Multilingual Support**: Automatic Hindi→English translation
        - **Sentiment Analysis**: Positive/Negative/Neutral classification
        - **Intent Detection**: Complaint/Praise/Suggestion/General categorization
        - **Aspect Extraction**: Identifies key topics and their sentiment
        - **Actionable Insights**: Ranked pain points and recommendations
        - **Interactive Visualizations**: Charts, word clouds, and trend analysis
        
        ### 🤖 Technology Stack
        - **Translation**: Helsinki-NLP OPUS models
        - **Sentiment**: Cardiff NLP multilingual RoBERTa
        - **Intent**: Facebook BART zero-shot classification
        - **Embeddings**: Sentence Transformers MiniLM
        - **UI**: Streamlit with Plotly visualizations
        
        ### 📊 Output Format
        Each review is analyzed and structured with:
        - Original and translated text
        - Language detection
        - Overall sentiment and confidence
        - Intent classification and confidence
        - Extracted aspects with individual sentiments
        - Positive/negative token lists
        """)
        
        # Model performance info
        with st.expander("Model Performance"):
            st.markdown("""
            | Component | Model | Languages | Accuracy |
            |-----------|-------|-----------|----------|
            | Translation | Helsinki-NLP/opus-mt-hi-en | Hindi→English | ~90% BLEU |
            | Sentiment | Cardiff NLP XLM-RoBERTa | 8 languages | ~85% F1 |
            | Intent | Facebook BART-large | Zero-shot | ~80% accuracy |
            | Embeddings | MiniLM-L6-v2 | 50+ languages | Semantic similarity |
            """)
        
        # Usage examples
        with st.expander("Usage Examples"):
            st.code("""
            # Example Hindi review
            "यह वेबसाइट बहुत धीमी है और फॉर्म भरना मुश्किल है।"
            
            # Analysis result:
            {
                "language": "Hindi",
                "translated_text": "This website is very slow and filling forms is difficult.",
                "sentiment": "Negative",
                "intent": "Complaint",
                "aspects": [
                    {"aspect_label": "performance", "sentiment": "Negative"},
                    {"aspect_label": "usability", "sentiment": "Negative"}
                ]
            }
            """)
    
    def _load_processed_data(self) -> Dict[str, Any]:
        """Load processed data from database or session state."""
        # Use session state caching instead of @st.cache_data
        if st.session_state.processed_data:
            return st.session_state.processed_data
        
        # Try loading from database
        try:
            reviews = self.db.get_all_reviews()
            if reviews:
                # Convert to expected format
                data = self._format_db_data(reviews)
                st.session_state.processed_data = data  # Cache in session state
                return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
        
        return None
    
    def _apply_filters(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply sidebar filters to data."""
        if not data or not data.get("reviews"):
            return data
        
        # Sidebar filters
        st.sidebar.subheader("Filters")
        
        # Date range filter
        reviews_df = pd.DataFrame(data["reviews"])
        
        # Handle timestamp parsing
        try:
            reviews_df["timestamp"] = pd.to_datetime(reviews_df["timestamp"])
            
            min_date = reviews_df["timestamp"].min().date()
            max_date = reviews_df["timestamp"].max().date()
            
            date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        except Exception as e:
            st.sidebar.error(f"Error parsing dates: {e}")
            date_range = None
        
        # Language filter
        languages = reviews_df["language"].unique()
        selected_languages = st.sidebar.multiselect(
            "Languages", languages, default=list(languages)
        )
        
        # Sentiment filter
        sentiments = reviews_df["sentiment"].unique()
        selected_sentiments = st.sidebar.multiselect(
            "Sentiments", sentiments, default=list(sentiments)
        )
        
        # Intent filter
        intents = reviews_df["intent"].unique()
        selected_intents = st.sidebar.multiselect(
            "Intents", intents, default=list(intents)
        )
        
        # Apply filters
        filtered_reviews = reviews_df.copy()
        
        # Apply date filter if available
        if date_range and len(date_range) == 2:
            try:
                filtered_reviews = filtered_reviews[
                    (filtered_reviews["timestamp"].dt.date >= date_range[0]) &
                    (filtered_reviews["timestamp"].dt.date <= date_range[1])
                ]
            except:
                pass  # Skip date filtering if there's an error
        
        # Apply other filters
        filtered_reviews = filtered_reviews[
            (filtered_reviews["language"].isin(selected_languages)) &
            (filtered_reviews["sentiment"].isin(selected_sentiments)) &
            (filtered_reviews["intent"].isin(selected_intents))
        ]
        
        # Update data structure
        filtered_data = data.copy()
        filtered_data["reviews"] = filtered_reviews.to_dict("records")
        
        return filtered_data
    
    def _display_kpis(self, data: Dict[str, Any]):
        """Display key performance indicators with better error handling."""
        if not data or not data.get("reviews"):
            st.info("No data available for KPIs")
            return
        
        try:
            reviews = data["reviews"]
            total_reviews = len(reviews)
            
            if total_reviews == 0:
                st.warning("No reviews to display")
                return
            
            # Calculate KPIs safely
            sentiments = [r.get("sentiment", "Neutral") for r in reviews if r.get("sentiment")]
            sentiment_counts = pd.Series(sentiments).value_counts() if sentiments else pd.Series()
            
            pos_pct = (sentiment_counts.get("Positive", 0) / total_reviews * 100) if total_reviews > 0 else 0
            neg_pct = (sentiment_counts.get("Negative", 0) / total_reviews * 100) if total_reviews > 0 else 0
            neu_pct = (sentiment_counts.get("Neutral", 0) / total_reviews * 100) if total_reviews > 0 else 0
            
            # Display KPIs
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Reviews", f"{total_reviews:,}")
            
            with col2:
                st.metric("😊 Positive", f"{pos_pct:.1f}%", delta=None)
            
            with col3:
                st.metric("😔 Negative", f"{neg_pct:.1f}%", delta=None)
            
            with col4:
                st.metric("😐 Neutral", f"{neu_pct:.1f}%", delta=None)
            
            with col5:
                # Calculate top aspect safely
                all_aspects = []
                for review in reviews:
                    aspects = review.get("aspects", [])
                    if aspects and isinstance(aspects, list):
                        all_aspects.extend([a.get("aspect_label", "") for a in aspects if a.get("aspect_label")])
                
                if all_aspects:
                    top_aspect = pd.Series(all_aspects).value_counts().index[0]
                    st.metric("🏷️ Top Aspect", top_aspect)
                else:
                    st.metric("🏷️ Top Aspect", "None found")
                    
        except Exception as e:
            st.error(f"Error displaying KPIs: {e}")

    
    def _display_sentiment_chart(self, data: Dict[str, Any]):
        """Display sentiment distribution chart."""
        st.subheader("Sentiment Distribution")
        
        if not data or not data.get("reviews"):
            st.info("No data available")
            return
        
        # Prepare data
        sentiments = [r.get("sentiment", "Neutral") for r in data["reviews"]]
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        # Create pie chart
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color_discrete_map={
                "Positive": "#2ecc71",
                "Negative": "#e74c3c", 
                "Neutral": "#95a5a6"
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True, height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_timeline_chart(self, data: Dict[str, Any]):
        """Display sentiment timeline chart."""
        st.subheader("Sentiment Timeline")
        
        if not data or not data.get("reviews"):
            st.info("No data available")
            return
        
        try:
            # Prepare data
            df = pd.DataFrame(data["reviews"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["date"] = df["timestamp"].dt.date
            
            # Group by date and sentiment
            timeline_data = df.groupby(["date", "sentiment"]).size().reset_index(name="count")
            
            # Create line chart
            fig = px.line(
                timeline_data, x="date", y="count", color="sentiment",
                color_discrete_map={
                    "Positive": "#2ecc71",
                    "Negative": "#e74c3c",
                    "Neutral": "#95a5a6"
                }
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Reviews",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating timeline chart: {e}")
    
    def _display_aspect_chart(self, data: Dict[str, Any]):
        """Display aspect sentiment heatmap."""
        st.subheader("Aspect Analysis")
        
        if not data or not data.get("reviews"):
            st.info("No data available")
            return
        
        # Prepare aspect data
        aspect_sentiment_data = []
        for review in data["reviews"]:
            aspects = review.get("aspects", [])
            for aspect in aspects:
                aspect_sentiment_data.append({
                    "aspect": aspect.get("aspect_label", "Unknown"),
                    "sentiment": aspect.get("aspect_sentiment", "Neutral")
                })
        
        if not aspect_sentiment_data:
            st.info("No aspects found")
            return
        
        try:
            # Create heatmap data
            aspect_df = pd.DataFrame(aspect_sentiment_data)
            heatmap_data = aspect_df.pivot_table(
                index="aspect", 
                columns="sentiment", 
                aggfunc="size", 
                fill_value=0
            )
            
            # Create heatmap
            fig = px.imshow(
                heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                color_continuous_scale="RdYlBu_r",
                aspect="auto"
            )
            fig.update_layout(
                title="Aspect-Sentiment Heatmap",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating aspect chart: {e}")
    
    def _display_intent_chart(self, data: Dict[str, Any]):
        """Display intent distribution chart."""
        st.subheader("Intent Distribution")
        
        if not data or not data.get("reviews"):
            st.info("No data available")
            return
        
        # Prepare data
        intents = [r.get("intent", "General Feedback") for r in data["reviews"]]
        intent_counts = pd.Series(intents).value_counts()
        
        # Create bar chart
        fig = px.bar(
            x=intent_counts.index,
            y=intent_counts.values,
            color=intent_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(
            xaxis_title="Intent",
            yaxis_title="Number of Reviews",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_wordclouds(self, data: Dict[str, Any]):
        """Display word clouds for positive and negative tokens."""
        st.subheader("Word Clouds")
        
        if not data or not data.get("wordcloud"):
            st.info("No word cloud data available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Positive Keywords**")
            pos_keywords = data["wordcloud"].get("positive_keywords", [])
            if pos_keywords:
                # Display as tags (simplified word cloud)
                keywords_text = " • ".join(pos_keywords[:20])
                st.success(keywords_text)
            else:
                st.info("No positive keywords found")
        
        with col2:
            st.markdown("**Negative Keywords**")
            neg_keywords = data["wordcloud"].get("negative_keywords", [])
            if neg_keywords:
                keywords_text = " • ".join(neg_keywords[:20])
                st.error(keywords_text)
            else:
                st.info("No negative keywords found")
    
    def _display_insights(self, data: Dict[str, Any]):
        """Display actionable insights."""
        st.subheader("🎯 Actionable Insights")
        
        if not data:
            st.info("No insights available")
            return
        
        # Display Gemini Summary if available
        summary = data.get("summary", {})
        textual_summary = summary.get("textual_summary", "")
        
        if textual_summary and textual_summary.strip() and textual_summary != "":
            st.subheader("🤖 AI-Powered Summary (Gemini)")
            with st.expander("📝 Detailed Analysis", expanded=True):
                st.markdown(textual_summary)
        elif GEMINI_CONFIG["use_gemini"] and GEMINI_CONFIG["api_key"]:
            st.info("🤖 Gemini summary will appear here after processing reviews with Gemini enabled")
        
        insights = data.get("insights", {})
        
        # Pain points
        if insights.get("pain_points"):
            st.markdown("**🔥 Top Pain Points:**")
            for i, pain_point in enumerate(insights["pain_points"][:5], 1):
                severity = pain_point.get('severity', 'Unknown')
                severity_color = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(severity, "⚪")
                
                with st.expander(f"{severity_color} {i}. {pain_point.get('aspect', 'Unknown')} (Score: {pain_point.get('score', 0):.2f})"):
                    st.write(f"**Severity:** {severity}")
                    st.write(f"**Total Mentions:** {pain_point.get('total_mentions', 0)}")
                    st.write(f"**Negative Mentions:** {pain_point.get('negative_mentions', 0)}")
                    st.write("**Examples:**")
                    for example in pain_point.get("examples", [])[:3]:
                        st.write(f"• {example}")
        
        # Recommendations
        if insights.get("recommendations"):
            st.markdown("**💡 Recommendations:**")
            for i, rec in enumerate(insights["recommendations"], 1):
                st.write(f"{i}. {rec}")
    
    def _display_sample_reviews(self, data: Dict[str, Any]):
        """Display sample processed reviews."""
        st.subheader("Sample Processed Reviews")
        
        if not data or not data.get("reviews"):
            st.info("No reviews available")
            return
        
        # Show first few reviews
        sample_reviews = data["reviews"][:5]
        
        for i, review in enumerate(sample_reviews, 1):
            with st.expander(f"Review {i} - {review.get('language', 'Unknown')} - {review.get('sentiment', 'Unknown')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Text:**")
                    st.write(review.get("original_text", "N/A"))
                    
                    if review.get("translated_text"):
                        st.write("**Translated:**")
                        st.write(review["translated_text"])
                
                with col2:
                    st.write(f"**Sentiment:** {review.get('sentiment', 'Unknown')} ({review.get('sentiment_score', 0):.2f})")
                    st.write(f"**Intent:** {review.get('intent', 'Unknown')} ({review.get('intent_score', 0):.2f})")
                    
                    aspects = review.get("aspects", [])
                    if aspects:
                        st.write("**Aspects:**")
                        for aspect in aspects:
                            st.write(f"• {aspect.get('aspect_label', 'Unknown')}: {aspect.get('aspect_sentiment', 'Unknown')}")
    
    def _show_empty_state(self):
        """Display empty state when no data is available."""
        st.info("No processed data available. Please upload and process reviews first.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### Quick Start Options:")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("📤 Go to Upload Page", type="primary"):
                    st.session_state.current_page = "📤 Upload & Process"
                    st.rerun()
            
            with col_b:
                if st.button("📄 Try Sample Data", type="secondary"):
                    self._generate_and_process_sample_data()
    
    def _generate_and_process_sample_data(self):
        """Generate and process sample data."""
        try:
            with st.spinner("Generating and processing sample data..."):
                # Import sample data generator
                from tools.generate_sample_data import generate_sample_reviews
                
                # Generate sample reviews
                sample_reviews = generate_sample_reviews()
                
                # Convert to DataFrame
                df = pd.DataFrame(sample_reviews)
                
                # Process the data
                import tempfile
                import os
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_file:
                    df.to_csv(temp_file.name, index=False)
                    temp_file_path = temp_file.name
                
                # Process reviews
                results = self.processor.process_reviews_from_csv(temp_file_path)
                
                # Save to database
                self.db.save_processed_results(results)
                
                # Update session state
                st.session_state.processed_data = results
                
                # Clean up
                os.unlink(temp_file_path)
                
                st.success(f"Successfully generated and processed {len(results.get('reviews', []))} sample reviews!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error generating sample data: {str(e)}")
    
    def _load_existing_data(self):
        """Load existing data from database."""
        try:
            data = self._load_processed_data()
            if data:
                st.success("Existing data loaded successfully!")
                st.rerun()
            else:
                st.info("No existing data found in database.")
        except Exception as e:
            st.error(f"Error loading existing data: {str(e)}")
    
    def _process_reviews(self, df: pd.DataFrame, batch_size: int, use_gemini: bool, max_reviews: int, save_to_db: bool):
        """Process uploaded reviews."""
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Limit reviews
            df_limited = df.head(max_reviews)
            
            # Save to temporary CSV
            temp_file = "temp_upload.csv"
            df_limited.to_csv(temp_file, index=False)
            
            # Process reviews
            status_text.text("Processing reviews...")
            progress_bar.progress(20)
            
            results = self.processor.process_reviews_from_csv(temp_file)
            progress_bar.progress(60)
            
            # Generate summary with Gemini if enabled
            if use_gemini and self.gemini_client:
                status_text.text("Generating AI summary...")
                summary_text = self.gemini_client.summarize_reviews(
                    [r.get("translated_text") or r.get("original_text", "") for r in results.get("reviews", [])]
                )
                if "summary" not in results:
                    results["summary"] = {}
                results["summary"]["textual_summary"] = summary_text
                progress_bar.progress(80)
            
            # Save to database
            if save_to_db:
                status_text.text("Saving to database...")
                self.db.save_processed_results(results)
                progress_bar.progress(90)
            
            # Update session state
            st.session_state.processed_data = results
            progress_bar.progress(100)
            
            # Show success
            st.success(f"Successfully processed {len(results.get('reviews', []))} reviews!")
            
            # Show download buttons
            self._show_download_buttons(results)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"Error processing reviews: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        
        finally:
            # Cleanup
            import os
            if os.path.exists("temp_upload.csv"):
                os.remove("temp_upload.csv")
    
    def _show_download_buttons(self, results: Dict[str, Any]):
        """Show download buttons for processed results."""
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON download
            json_str = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"processed_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # CSV download
            reviews = results.get("reviews", [])
            if reviews:
                df = pd.DataFrame(reviews)
                csv_str = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_str,
                    file_name=f"processed_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    def _format_db_data(self, reviews: List[Dict]) -> Dict[str, Any]:
        """Format database data to expected structure."""
        # This would convert DB format to the expected pipeline output format
        # Implementation depends on DB schema
        return {
            "reviews": reviews,
            "summary": {},
            "wordcloud": {},
            "insights": {}
        }
