"""Main Streamlit application for MCA21 sentiment analysis pipeline."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.ui import SentimentDashboard
from pipeline.process import ReviewProcessor
from utils.db import DatabaseManager
from config import STREAMLIT_CONFIG
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title=STREAMLIT_CONFIG["page_title"],
        page_icon=STREAMLIT_CONFIG["page_icon"],
        layout=STREAMLIT_CONFIG["layout"],
        initial_sidebar_state=STREAMLIT_CONFIG["sidebar_state"],
    )
    
    # Initialize components
    dashboard = SentimentDashboard()
    
    # Header
    st.title("🔍 MCA21 E-Consultation Analysis Dashboard")
    st.markdown("Multilingual sentiment analysis with Hindi-English support")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose Page",
        ["📊 Dashboard", "📤 Upload & Process", "⚙️ Settings", "📖 About"]
    )
    
    # Route to different pages
    if page == "📊 Dashboard":
        dashboard.show_dashboard()
    elif page == "📤 Upload & Process":
        dashboard.show_upload_process()
    elif page == "⚙️ Settings":
        dashboard.show_settings()
    elif page == "📖 About":
        dashboard.show_about()

if __name__ == "__main__":
    main()
