"""Configuration settings for the MCA21 sentiment analysis pipeline."""

import os
from pathlib import Path
from typing import Dict, List

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = MODELS_DIR / "cache"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for dir_path in [DATA_DIR, CACHE_DIR, OUTPUTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Database configuration
DB_PATH = os.getenv("DB_PATH", str(DATA_DIR / "reviews.db"))

# Model configurations
# Model configurations
MODEL_CONFIGS = {
    "translation": {
        "model_name": "Helsinki-NLP/opus-mt-hi-en",
        "cache_dir": str(CACHE_DIR / "translation"),
        "max_length": 512,
    },
    "sentiment": {
        "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "cache_dir": str(CACHE_DIR / "sentiment"),
        "labels_map": {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"},
    },
    "intent": {
        "model_name": "typeform/distilbert-base-uncased-mnli",
        "cache_dir": str(CACHE_DIR / "intent"), 
        "candidate_labels": ["Complaint", "Praise", "Suggestion", "General Feedback"],
        "threshold": 0.3,
    },
    "embedding": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "cache_dir": str(CACHE_DIR / "embeddings"),
        "dimension": 384,
    },
    "spacy": {
        "model_name": "en_core_web_sm",
    },
}

# Processing parameters
PROCESSING_CONFIG = {
    "batch_size": int(os.getenv("BATCH_SIZE", "32")),
    "max_reviews_per_batch": int(os.getenv("MAX_REVIEWS_PER_BATCH", "1000")),
    "min_aspect_frequency": 3,
    "max_aspects": 20,
    "sentiment_threshold": 0.1,
    "clustering_threshold": 0.7,
}

# Gemini API configuration
GEMINI_CONFIG = {
    "api_key": os.getenv("GEMINI_API_KEY"),
    "use_gemini": os.getenv("USE_GEMINI", "false").lower() == "true",
    "model_name": "gemini-2.5-flash-lite",
    "max_tokens": 1000,
    "temperature": 0.3,
}

# Language detection configuration
LANGUAGE_CONFIG = {
    "supported_languages": ["en", "hi"],
    "default_language": "en",
    "confidence_threshold": 0.8,
}

# Hindi stopwords and common words
HINDI_STOPWORDS = {
    "और", "का", "के", "की", "को", "से", "में", "पर", "है", "हैं", "था", "थे", "थी", 
    "हो", "होना", "होने", "होते", "होती", "हुए", "हुआ", "हुई", "गया", "गयी", "गये",
    "जा", "जाना", "जाने", "जाते", "जाती", "आ", "आना", "आने", "आते", "आती", "यह", 
    "वह", "ये", "वे", "इस", "उस", "इन", "उन", "एक", "दो", "तीन", "भी", "नहीं", 
    "कि", "तो", "ही", "अब", "तब", "यहाँ", "वहाँ", "कहाँ", "कैसे", "क्यों", "कब"
}

# Aspect extraction keywords (domain-specific)
ASPECT_KEYWORDS = {
    "user_interface": ["ui", "interface", "design", "layout", "screen", "page", "website", "portal"],
    "functionality": ["feature", "function", "work", "working", "operation", "process", "system"],
    "performance": ["speed", "fast", "slow", "performance", "loading", "time", "response"],
    "usability": ["easy", "difficult", "user", "friendly", "simple", "complex", "navigation"],
    "support": ["help", "support", "customer", "service", "assistance", "guidance"],
    "documentation": ["manual", "guide", "instruction", "documentation", "help", "tutorial"],
    "security": ["secure", "security", "safe", "password", "login", "authentication"],
    "mobile": ["mobile", "phone", "app", "application", "android", "ios"],
    "payment": ["payment", "pay", "money", "transaction", "billing", "cost", "price"],
    "registration": ["register", "registration", "signup", "account", "profile"],
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "MCA21 E-Consultation Analysis",
    "page_icon": "📊",
    "layout": "wide",
    "sidebar_state": "expanded",
    "theme": os.getenv("STREAMLIT_THEME", "light"),
}

# Logging configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "file": os.getenv("LOG_FILE", str(LOGS_DIR / "pipeline.log")),
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    "rotation": "10 MB",
    "retention": "30 days",
}

# File upload configuration
UPLOAD_CONFIG = {
    "max_file_size": "200MB",
    "allowed_extensions": [".csv", ".xlsx", ".json"],
    "required_columns": ["text", "timestamp", "review_id"],
}
