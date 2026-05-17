# Industrial Audio RAG - User Manual

**Complete guide to using the Industrial Audio RAG system for intelligent industrial audio analysis**

---

## 📚 Table of Contents

1. [System Overview](#system-overview)
2. [Getting Started](#getting-started)
3. [Core Features](#core-features)
4. [API Reference](#api-reference)
5. [Web Interface](#web-interface)
6. [Query Examples](#query-examples)
7. [Advanced Usage](#advanced-usage)
8. [Monitoring & Metrics](#monitoring--metrics)
9. [Security Features](#security-features)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

---

## 📋 System Overview

### What is Industrial Audio RAG?

Industrial Audio RAG is an AI-powered system that allows you to ask natural language questions about industrial audio data. It combines:

- **Audio Feature Extraction**: Automated analysis of acoustic properties
- **Vector Search**: Similarity-based retrieval of relevant audio clips
- **Large Language Model**: Intelligent answer generation with context
- **Production Infrastructure**: Enterprise-ready deployment and monitoring

### Key Capabilities

| Feature | Description | Example |
|---------|-------------|---------|
| **Natural Language Queries** | Ask questions in plain English | "Find bearing clips with high frequency noise" |
| **Acoustic Analysis** | Automatic feature extraction from audio | RMS, dominant frequency, SNR analysis |
| **Similarity Search** | Find acoustically similar clips | "Show clips similar to this anomalous bearing" |
| **Context-Aware Answers** | AI responses with source attribution | Answer + relevant audio file references |
| **Real-time Performance** | Sub-3 second response times | Suitable for interactive use |

### Supported Audio Types

- **Industrial Machinery**: Bearings, valves, gearboxes, pumps
- **Audio Formats**: WAV files (16-bit, 44.1kHz recommended)
- **Duration**: Optimized for 1-second clips (configurable)
- **Dataset**: Currently trained on DCASE 2024 Task-2 data

---

## 🚀 Getting Started

### Option 1: Local Installation

#### Prerequisites
```bash
# System requirements
- Python 3.10+
- Docker (for Qdrant database)
- 8GB RAM minimum
- 50GB free disk space (for full dataset)
```

#### Quick Setup
```bash
# 1. Clone repository
git clone https://github.com/sylvainbonnot/industrial-audio-rag
cd industrial-audio-rag

# 2. Create environment
conda env create -f env.yml
conda activate ml_py310

# 3. Start database
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:v1.8.1

# 4. Install dependencies
make install

# 5. Setup development environment (downloads data)
make dev-setup

# 6. Start API server
make run
```

#### Access Points
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics

---

## 🎯 Core Features

### 1. Audio Feature Extraction

The system automatically extracts acoustic features from audio files:

#### Extracted Features
| Feature | Description | Use Case |
|---------|-------------|-----------|
| **RMS (Root Mean Square)** | Overall loudness/energy | Identify loud anomalies |
| **Dominant Frequency** | Peak frequency in spectrum | Find resonant frequencies |
| **SNR (Signal-to-Noise Ratio)** | Audio quality metric | Detect noisy/clean signals |
| **Duration** | Actual audio length | Quality control |
| **Spectral Statistics** | Frequency domain analysis | Advanced acoustic analysis |

#### Feature Computation
```python
# Example of extracted features for a bearing clip
{
    "rms": 0.0234,                    # Loudness
    "dominant_freq_hz": 157.3,        # Main frequency peak
    "snr_db": 28.4,                   # Signal quality
    "duration_sec": 1.0,              # Clip length
    "file": "bearing_01_normal_000123.wav"
}
```

### 2. Vector Search Engine

#### How It Works
1. **Indexing**: Audio features + metadata → vector embeddings
2. **Storage**: Embeddings stored in Qdrant vector database
3. **Search**: Query → embedding → find similar vectors
4. **Retrieval**: Return most relevant audio clips with scores

#### Search Types
- **Semantic Search**: "Find clips with bearing problems"
- **Technical Search**: "Show clips with dominant frequency above 500 Hz"
- **Comparative Search**: "Find clips similar to abnormal valve sounds"
- **Filtered Search**: "Get all section 00 clips with high RMS"

### 3. AI Answer Generation

#### Process Flow
1. **Query Processing**: Clean and validate user input
2. **Context Retrieval**: Find relevant audio clips via vector search
3. **Context Preparation**: Format clip metadata for LLM
4. **Answer Generation**: LLM generates response using context
5. **Response Formatting**: Structure answer with sources

#### LLM Integration
- **Primary**: Ollama (local Mistral 7B)
- **Alternative**: OpenAI API compatible models
- **Customization**: Configurable prompts and parameters

---

## 🔌 API Reference

### Authentication

#### API Key Authentication (Optional)
```bash
# Enable authentication
export ENABLE_API_KEY_AUTH=true
export API_KEY="your-secure-key"

# Use in requests
curl -H "Authorization: Bearer your-secure-key" \
     "http://localhost:8000/ask"
```

### Core Endpoints

#### 1. Ask Questions - `/ask`
**POST** - Main query endpoint

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Find bearing clips with high frequency noise",
       "max_results": 5,
       "filters": {
         "machine_type": "bearing",
         "section": "00"
       }
     }'
```

**Request Parameters:**
```json
{
  "query": "string (required)",           // Natural language question
  "max_results": "integer (1-20)",       // Number of results to return
  "max_tokens": "integer (100-2000)",    // Max response length
  "filters": {                           // Optional metadata filters
    "machine_type": "bearing|valve|fan|gearbox|slider|toycar",
    "section": "00|01|02|03|04|05|06",
    "domain": "source|target",
    "state": "normal|anomaly"
  }
}
```

**Response Format:**
```json
{
  "answer": "string",                     // AI-generated answer
  "sources": [                           // Retrieved audio clips
    {
      "filename": "bearing_01_normal_000123.wav",
      "score": 0.924,                    // Relevance score (0-1)
      "metadata": {
        "machine_type": "bearing",
        "section": "01",
        "rms": 0.0234,
        "dominant_freq_hz": 157.3
      }
    }
  ],
  "metadata": {
    "response_time": 1.847,              // Total response time (s)
    "embedding_time": 0.023,             // Embedding generation (s)
    "search_time": 0.156,                // Vector search (s) 
    "llm_time": 1.654,                   // LLM generation (s)
    "total_sources": 847,                // Total matching sources
    "model_version": "v1.2.0"
  }
}
```

#### 2. System Information - `/info`
**GET** - System status and configuration

```bash
curl "http://localhost:8000/info"
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.2.0",
  "model_info": {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "mistral:7b",
    "embedding_dimension": 384
  },
  "data_info": {
    "total_documents": 23454,
    "collection_name": "dcase24_audio",
    "last_updated": "2024-01-15T10:30:00Z"
  },
  "performance": {
    "avg_response_time": 2.1,
    "total_queries": 1247,
    "uptime_seconds": 86400
  }
}
```

#### 3. Health Check - `/health`
**GET** - Service health status

```bash
curl "http://localhost:8000/health"
```

#### 4. Metrics - `/metrics`
**GET** - Prometheus metrics

```bash
curl "http://localhost:8000/metrics"
```

#### 5. Security Status - `/security`
**GET** - Security configuration and stats

```bash
curl "http://localhost:8000/security"
```

### Rate Limiting

Default rate limits (configurable):
- `/ask`: 10 requests per minute
- `/info`: 30 requests per minute
- `/health`: 100 requests per minute

**Rate Limit Headers:**
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1640995200
```

---

## 💻 Web Interface

### Interactive Demo Features

#### Query Interface
- **Text Input**: Natural language question box
- **Sample Questions**: Pre-loaded example queries
- **Parameters**: Adjustable max results and filters
- **Submit Options**: Button or Enter key

#### Response Display
- **Answer Section**: Formatted AI response with markdown
- **Performance Metrics**: Real-time timing information
- **Sources Section**: Retrieved audio files with scores
- **System Information**: Live API status and configuration

#### Sample Questions Available
1. "Which bearing clips in section 00 show dominant frequency above 900 Hz?"
2. "Find all clips with bearing anomalies in the training dataset"
3. "What are the frequency characteristics of normal vs abnormal bearings?"
4. "Show me clips with high energy content in the 1-2 kHz range"
5. "Which audio files contain bearing defects similar to outer race damage?"
6. "Find clips with spectral patterns indicating bearing wear"
7. "What's the difference between normal and abnormal bearing signatures?"
8. "Show bearing clips with unusual vibration patterns"

---

## 📝 Query Examples

### Basic Queries

#### 1. Find Specific Conditions
```
Query: "Find bearing clips with anomalies"
Expected: Lists abnormal bearing clips with technical details
```

#### 2. Frequency Analysis
```
Query: "Which clips have dominant frequency above 1000 Hz?"
Expected: Clips with high-frequency content and frequency values
```

#### 3. Loudness Analysis  
```
Query: "Show me the loudest valve clips"
Expected: Valve clips sorted by RMS values with loudness metrics
```

### Advanced Queries

#### 4. Comparative Analysis
```
Query: "What are the differences between normal and abnormal bearings in section 00?"
Expected: Comparative analysis with statistics and examples
```

#### 5. Technical Specifications
```
Query: "Find gearbox clips with SNR below 20 dB and dominant frequency between 100-500 Hz"
Expected: Filtered results matching technical criteria
```

#### 6. Pattern Recognition
```
Query: "Show clips with similar acoustic patterns to bearing outer race defects"
Expected: Clips with similar spectral characteristics
```

### Domain-Specific Queries

#### 7. Machine Type Analysis
```
Query: "Compare noise characteristics across different machine types"
Expected: Cross-machine analysis with statistical comparisons
```

#### 8. Section-Based Analysis
```
Query: "Why do section 02 clips have lower SNR than section 01?"
Expected: Analysis of environmental or recording differences
```

#### 9. Temporal Analysis
```
Query: "Find clips that show signs of progressive bearing wear"
Expected: Clips indicating degradation patterns
```

### Query Best Practices

#### ✅ Good Queries
- **Specific**: "Find bearing clips with dominant frequency above 500 Hz"
- **Technical**: "Show valve clips with high SNR and low RMS"
- **Contextual**: "Compare normal vs abnormal gearbox patterns"

#### ❌ Avoid These
- **Too Vague**: "Show me some clips"
- **Non-audio**: "What's the weather today?"
- **Overly Complex**: 200+ word queries

---

## 🔧 Advanced Usage

### Environment Configuration

#### Core Settings
```bash
# Database
export QDRANT_URL="http://localhost:6333"
export QDRANT_COLLECTION_NAME="audio_embeddings"

# Models  
export EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
export LLM_MODEL_NAME="mistral:7b"

# API Configuration
export MAX_QUERY_LENGTH="1000"
export MAX_RESULTS="20"
export DEFAULT_MAX_RESULTS="5"
```

#### Security Settings
```bash
# Authentication
export ENABLE_API_KEY_AUTH="true"
export API_KEY="your-secure-api-key"

# Rate Limiting
export ENABLE_RATE_LIMITING="true"
export RATE_LIMIT_PER_MINUTE="10"

# Content Safety
export ENABLE_PII_DETECTION="true"
export MAX_QUERY_LENGTH="1000"
```

#### Monitoring Settings
```bash
# Metrics
export ENABLE_METRICS="true"
export METRICS_PORT="8000"

# Tracing
export OTEL_EXPORTER_OTLP_ENDPOINT="http://jaeger:14268"
export ENABLE_TRACING="true"
```

### Custom Data Ingestion

#### Prepare Your Audio Data
```bash
# Expected structure
data/
├── machine_type_1/
│   ├── section_00/
│   │   ├── domain_source/
│   │   │   ├── split_train/
│   │   │   │   ├── state_normal/
│   │   │   │   │   ├── clip_000001.wav
│   │   │   │   │   └── clip_000002.wav
│   │   │   │   └── state_anomaly/
│   │   │   └── split_test/
│   │   └── domain_target/
│   └── section_01/
└── machine_type_2/
```

#### Index Custom Data
```bash
# Run indexer on your data
python -m rag_audio.indexer \
  --data /path/to/your/audio/data \
  --collection-name "your_collection" \
  --batch-size 100
```

### Batch Processing

#### Process Multiple Queries
```python
import requests
import json

queries = [
    "Find bearing anomalies",
    "Show high frequency clips", 
    "Compare normal vs abnormal patterns"
]

results = []
for query in queries:
    response = requests.post(
        "http://localhost:8000/ask",
        json={"query": query, "max_results": 3}
    )
    results.append(response.json())

# Save batch results
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## 📊 Monitoring & Metrics

### Performance Metrics

#### API Performance
- **Request Rate**: Requests per second
- **Response Time**: P50, P95, P99 percentiles
- **Error Rate**: Failed requests percentage
- **Throughput**: Successful queries per minute

#### RAG Pipeline Metrics
- **Embedding Time**: Vector generation duration
- **Search Time**: Qdrant query duration  
- **LLM Time**: Answer generation duration
- **End-to-End Time**: Total response time

#### System Health
- **Memory Usage**: RAM consumption
- **CPU Usage**: Processor utilization
- **Storage Usage**: Disk space consumption
- **Database Health**: Qdrant connection status

### Accessing Metrics

#### Prometheus Metrics
```bash
# Raw metrics endpoint
curl http://localhost:8000/metrics

# Key metrics
http_requests_total                    # Total requests
http_request_duration_seconds          # Response times  
rag_embedding_duration_seconds         # Embedding times
rag_search_duration_seconds           # Search times
rag_llm_duration_seconds              # LLM times
```

#### Grafana Dashboard

If monitoring is enabled, access Grafana:
```bash
# Port forward to Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Default credentials
# Username: admin
# Password: admin123
```

**Dashboard Panels:**
- API Request Rate (requests/second)
- Response Time Distribution (P50, P95, P99)
- RAG Pipeline Breakdown (embedding, search, LLM)
- Error Rate and Status Codes
- System Resource Usage
- Query Success Rate

---

## 🛡️ Security Features

### Authentication & Authorization

#### API Key Authentication
```bash
# Enable authentication
export ENABLE_API_KEY_AUTH=true
export API_KEY="secure-random-key-here"

# Use in requests
curl -H "Authorization: Bearer secure-random-key-here" \
     "http://localhost:8000/ask" \
     -d '{"query": "test query"}'
```

### Rate Limiting

#### Default Limits
- `/ask`: 10 requests/minute per IP
- `/info`: 30 requests/minute per IP
- `/health`: 100 requests/minute per IP

#### Custom Configuration
```bash
export RATE_LIMIT_ASK="20/minute"      # Custom ask limit
export RATE_LIMIT_INFO="60/minute"     # Custom info limit
export ENABLE_RATE_LIMITING="true"
```

### Input Validation & Sanitization

#### Automatic Protection
- **HTML/Script Injection**: Automatic sanitization with bleach
- **Query Length Limits**: Configurable maximum query length
- **Pattern Detection**: XSS and script injection prevention
- **Content Filtering**: Dangerous pattern blocking

#### PII Detection & Redaction
```bash
# Enable PII protection
export ENABLE_PII_DETECTION=true

# Detected PII types
- Email addresses
- Phone numbers  
- Social Security Numbers
- Names and addresses
- Credit card numbers
```

### Security Monitoring

#### Security Metrics
- Rate limit violations by IP and endpoint
- Authentication failures by reason
- PII detections by type and frequency
- Blocked requests by security category
- Input validation failures

#### Security Endpoint
```bash
# Check security status
curl "http://localhost:8000/security"

# Response includes
{
  "security_config": {
    "authentication_enabled": true,
    "rate_limiting_enabled": true,
    "pii_detection_enabled": true
  },
  "security_stats": {
    "total_blocked_requests": 23,
    "rate_limit_violations": 12,
    "pii_detections": 3
  }
}
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. API Connection Errors

**Symptoms:**
- "Connection refused" errors
- Timeout errors
- 502/503 HTTP status codes

**Solutions:**
```bash
# Check service status
curl http://localhost:8000/health

# Verify database connection
docker ps | grep qdrant

# Check logs
docker logs qdrant
tail -f logs/api.log
```

#### 2. Slow Response Times

**Symptoms:**
- Responses taking >10 seconds
- Timeout errors
- High P95/P99 latencies

**Solutions:**
```bash
# Check system resources
htop
df -h

# Optimize query
# Use specific filters
{"query": "bearing clips", "filters": {"section": "00"}}

# Reduce max_results
{"query": "test", "max_results": 3}

# Check database performance
curl http://localhost:6333/collections/audio_embeddings
```

#### 3. Empty or Poor Results

**Symptoms:**
- No sources returned
- Irrelevant answers
- Low similarity scores

**Solutions:**
```bash
# Check data availability
curl http://localhost:8000/info

# Verify collection status
curl http://localhost:6333/collections/audio_embeddings

# Try broader queries
"bearing clips" instead of "bearing clips with specific technical parameters"

# Check embedding model
curl http://localhost:8000/info | jq '.model_info'
```

#### 4. Authentication Issues

**Symptoms:**
- 401 Unauthorized errors
- "Invalid API key" messages

**Solutions:**
```bash
# Verify API key format
export API_KEY="correct-format-key"

# Check authentication setting
curl http://localhost:8000/security

# Use correct header format
curl -H "Authorization: Bearer your-api-key"
```

### Debug Mode

#### Enable Detailed Logging
```bash
export LOG_LEVEL="DEBUG"
export ENABLE_TRACE_LOGS="true"

# Restart service
make run
```

#### View Debug Information
```bash
# API logs
tail -f logs/api.log

# Database logs  
docker logs qdrant -f

# System metrics
curl http://localhost:8000/metrics | grep -E "(response_time|error_rate)"
```

---

## 💡 Best Practices

### Query Optimization

#### Effective Query Patterns
1. **Be Specific**: Include technical terms and constraints
2. **Use Filters**: Narrow search with metadata filters
3. **Reasonable Scope**: Don't request too many results at once
4. **Domain Language**: Use industrial/acoustic terminology

#### Example Optimizations
```bash
# Instead of:
"Show me some clips"

# Use:
"Find bearing clips with dominant frequency above 500 Hz in section 00"

# With filters:
{
  "query": "bearing anomalies",
  "filters": {"state": "anomaly", "section": "00"},
  "max_results": 5
}
```

### Performance Optimization

#### Client-Side
- **Cache Results**: Store frequently used queries
- **Batch Processing**: Group related queries
- **Connection Reuse**: Maintain persistent connections
- **Error Handling**: Implement retry logic with backoff

#### Server-Side Configuration
```bash
# Optimize for your use case
export MAX_RESULTS="10"              # Balance quality vs speed
export EMBEDDING_BATCH_SIZE="32"     # Batch embedding processing
export LLM_MAX_TOKENS="1000"         # Control response length
```

### Data Management

#### Audio File Preparation
- **Format**: WAV, 16-bit, 44.1kHz
- **Duration**: 1-second clips (optimal)
- **Naming**: Descriptive, consistent naming scheme
- **Quality**: Clean recordings without corruption

#### Indexing Strategy
- **Batch Size**: Process 100-500 files at once
- **Incremental Updates**: Add new data without full reindex
- **Backup**: Regular Qdrant snapshots
- **Monitoring**: Track indexing progress and errors

### Security Best Practices

#### Production Deployment
1. **Strong API Keys**: Use cryptographically secure keys
2. **Rate Limiting**: Implement appropriate limits for your use case
3. **Network Security**: Use HTTPS, restrict access by IP
4. **Input Validation**: Always validate and sanitize inputs
5. **Monitoring**: Log and monitor all security events

#### Development
1. **Environment Separation**: Different keys for dev/staging/prod
2. **Secret Management**: Use environment variables, not hardcoded keys
3. **Regular Updates**: Keep dependencies updated
4. **Access Control**: Limit who can access production systems

---

## 📞 Support & Resources

### Getting Help

#### Documentation
- **User Manual**: This document
- **API Documentation**: http://localhost:8000/docs
- **Deployment Guide**: [infra/DEPLOYMENT.md](../infra/DEPLOYMENT.md)
- **Evaluation Guide**: [eval/README.md](../eval/README.md)

#### Community
- **GitHub Issues**: [Report bugs and request features](https://github.com/sylvainbonnot/industrial-audio-rag/issues)
- **Discussions**: [Community Q&A](https://github.com/sylvainbonnot/industrial-audio-rag/discussions)

### Version Information

#### Current Version: v1.2.0
- **Release Date**: January 2024
- **Key Features**: Production-ready RAG with monitoring
- **Compatibility**: Python 3.10+, Docker, Kubernetes

#### What's New
- Enhanced security features
- Comprehensive monitoring
- Cloud deployment support
- Performance optimizations
- Extended evaluation framework

---

**🎯 This manual covers all major features of the Industrial Audio RAG system. For specific technical questions or advanced use cases, please refer to the API documentation or reach out to our support team.**