# Architecture and Design Decisions

## Overview

The MCA21 E-Consultation Sentiment Analysis Pipeline is designed as a modular, production-ready system for analyzing multilingual (Hindi-English) user reviews. The architecture emphasizes scalability, maintainability, and extensibility while providing comprehensive sentiment analysis capabilities.

## Core Architecture

### 1. Modular Design

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Streamlit UI    │────│ Pipeline Core   │────│ Model Services  │
│ (app/)          │    │   (pipeline/)   │    │ (models/)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
│                       │                       │
└───────────────────────┼───────────────────────┘
                        │
                ┌─────────────────┐
                │ Utilities       │
                │ (utils/)        │
                └─────────────────┘

text

### 2. Processing Pipeline

The analysis pipeline follows a sequential processing model:

1. **Text Preprocessing** → Language Detection → Translation (if needed)
2. **Sentiment Analysis** → Intent Classification → Aspect Extraction
3. **Insight Generation** → Visualization → Storage

## Model Selection Rationale

### Translation Models

**Choice: Helsinki-NLP/opus-mt-hi-en**
- **Pros**: Specialized Hindi→English translation, good BLEU scores, optimized for speed
- **Cons**: Limited to Hindi-English pair, may struggle with code-mixed text
- **Alternative Considered**: Google Translate API (rejected due to cost and dependency)

### Sentiment Analysis

**Choice: cardiffnlp/twitter-xlm-roberta-base-sentiment-latest**
- **Pros**: Multilingual support, trained on social media text, handles informal language
- **Cons**: Larger model size (~500MB), may be overkill for formal text
- **Alternative Considered**: VADER (rejected for multilingual requirements)

### Intent Classification

**Choice: facebook/bart-large-mnli (Zero-shot)**
- **Pros**: No training data required, flexible label addition, good general performance
- **Cons**: Larger model size, may be less accurate than specialized models
- **Alternative Considered**: Custom classifier (rejected due to lack of labeled data)

### Embeddings

**Choice: sentence-transformers/all-MiniLM-L6-v2**
- **Pros**: Compact size (80MB), fast inference, good semantic understanding
- **Cons**: Limited to 384 dimensions, may miss some nuances
- **Alternative Considered**: all-mpnet-base-v2 (rejected due to size constraints)

## Design Decisions

### 1. Hybrid Aspect Extraction

**Decision**: Combine rule-based patterns, keyword matching, and embedding clustering

**Rationale**:
- Rule-based patterns catch explicit mentions
- Keywords handle domain-specific terms
- Embeddings group semantically similar aspects
- Provides fallback when one method fails

### 2. Optional Gemini Integration

**Decision**: Make Gemini API optional with environment variable control

**Rationale**:
- Reduces dependency on external paid services
- Allows graceful degradation without API key
- Provides enhanced summarization when available
- Maintains system functionality for all users

### 3. SQLite for Persistence

**Decision**: Use SQLite over other databases

**Rationale**:
- Serverless, no setup required
- Sufficient for MVP scale (thousands of reviews)
- Easy backup and migration
- Suitable for single-instance deployment

### 4. Caching Strategy

**Decision**: Implement multiple levels of caching

**Implementation**:
- **Model Caching**: Disk-based model storage with lazy loading
- **Embedding Caching**: Pickle-based cache for computed embeddings
- **Streamlit Caching**: `@st.cache_data` for UI components
- **Translation Caching**: LRU cache for frequently translated texts

## Performance Considerations

### 1. Memory Management

- **Model Loading**: Lazy initialization to reduce startup memory
- **Batch Processing**: Configurable batch sizes to prevent memory overflow
- **GPU Utilization**: Automatic GPU detection and usage when available

### 2. Processing Optimization

- **Pipeline Parallelization**: Independent components can run concurrently
- **Embedding Reuse**: Cache embeddings for repeated texts
- **Early Stopping**: Skip processing for empty or invalid inputs

### 3. Scalability Features

- **Configurable Limits**: Batch sizes, maximum aspects, clustering parameters
- **Graceful Degradation**: Fallback mechanisms when models fail
- **Resource Monitoring**: Memory and processing time tracking

## Security and Privacy

### 1. Data Handling

- **No External Data Sharing**: All processing happens locally except optional Gemini
- **Secure Storage**: SQLite database with proper file permissions
- **Input Sanitization**: HTML/script tag removal, URL anonymization

### 2. API Security

- **Environment Variables**: Sensitive keys stored in environment, not code
- **Optional Services**: External APIs are optional, not required
- **Rate Limiting**: Built-in retry mechanisms with exponential backoff

## Extensibility Points

### 1. Adding New Languages
In config.py
LANGUAGE_CONFIG = {
"supported_languages": ["en", "hi", "ta", "bn"], # Add new languages
# ...
}

In translation.py
def get_translation_model(source_lang, target_lang):
model_map = {
("ta", "en"): "Helsinki-NLP/opus-mt-ta-en",
("bn", "en"): "Helsinki-NLP/opus-mt-bn-en",
# Add new model mappings
}

text

### 2. Custom Aspect Categories

In config.py
CUSTOM_ASPECTS = {
"domain_specific": ["custom_keyword1", "custom_keyword2"],
# Add domain-specific aspects
}

text

### 3. Alternative Models

Models can be swapped by updating `config.py`:

MODEL_CONFIGS = {
"sentiment": {
"model_name": "new-sentiment-model", # Change model
# Update configuration accordingly
}
}

text

## Deployment Options

### 1. Local Development
./scripts/run_local.sh

text

### 2. Docker Container
docker build -t mca21-sentiment .
docker run -p 8501:8501 mca21-sentiment

text

### 3. Cloud Deployment
- **Streamlit Cloud**: Direct GitHub integration
- **AWS/GCP**: Container-based deployment
- **Azure**: Web Apps for Containers

## Monitoring and Logging

### 1. Application Logging

- **Structured Logging**: JSON format with timestamps and levels
- **Component-Specific Logs**: Separate loggers for each pipeline component
- **Error Tracking**: Detailed error messages with context

### 2. Performance Metrics

- **Processing Time**: Track time per review and batch
- **Model Performance**: Monitor accuracy and confidence scores
- **Resource Usage**: Memory and CPU utilization tracking

### 3. User Analytics

- **Usage Patterns**: Track most common analysis types
- **Error Rates**: Monitor and alert on processing failures
- **Data Quality**: Track input text characteristics

## Testing Strategy

### 1. Unit Tests

- **Component Testing**: Individual pipeline components
- **Mock Dependencies**: External services and models
- **Edge Cases**: Empty inputs, invalid data, network failures

### 2. Integration Tests

- **End-to-End**: Complete pipeline processing
- **Database Operations**: CRUD operations and data integrity
- **API Integration**: External service interactions

### 3. Performance Tests

- **Load Testing**: Large batch processing
- **Memory Profiling**: Resource usage under load
- **Latency Benchmarks**: Response time measurements

## Future Enhancements

### 1. Advanced Analytics

- **Trend Analysis**: Time-series sentiment tracking
- **Comparative Analysis**: Cross-period comparisons
- **Predictive Insights**: Forecasting sentiment trends

### 2. Enhanced NLP

- **Custom Models**: Domain-specific fine-tuned models
- **Multi-modal Analysis**: Image and text combined analysis
- **Real-time Processing**: Streaming data analysis

### 3. User Experience

- **Interactive Dashboards**: Drill-down capabilities
- **Export Options**: Multiple format support
- **Collaborative Features**: Team-based analysis workflows

### 4. Enterprise Features

- **API Endpoints**: REST API for system integration
- **SSO Integration**: Enterprise authentication
- **Audit Trails**: Comprehensive logging and tracking
- **Multi-tenancy**: Organization-based data separation

## Conclusion

This architecture provides a robust foundation for multilingual sentiment analysis while maintaining flexibility for future enhancements. The modular design enables independent component updates, and the comprehensive caching strategy ensures good performance even with resource constraints.

The system successfully balances functionality, performance, and maintainability to deliver a production-ready MVP for MCA21 e-consultation analysis.
