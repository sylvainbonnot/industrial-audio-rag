# src/rag_audio/api.py
"""
FastAPI service exposing /ask and /info endpoints for the indexed audio vectors.

Start with:

    uvicorn rag_audio.api:app --host 0.0.0.0 --port 8000

Environment variables (with defaults):
    QDRANT_URL        http://localhost:6333
    COLLECTION        dcase24_bearing
    EMBED_MODEL       mixedbread-ai/mxbai-embed-large-v1
    OLLAMA_URL        http://localhost:11434
    OLLAMA_MODEL      mistral
    SEARCH_LIMIT      6
"""

from __future__ import annotations

import json
import os
import time
import logging
import requests
from typing import Any, Optional, List, Dict, Callable

from fastapi import FastAPI, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# Monitoring imports
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# OpenTelemetry imports
try:
    from opentelemetry import trace, metrics
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logging.warning("OpenTelemetry not available. Monitoring will be limited to Prometheus metrics.")

# Security imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import HTTPException, Depends, status
from jose import JWTError, jwt
from passlib.context import CryptContext
import bleach
import re
from typing import Union

# PII Detection imports
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PII_DETECTION_AVAILABLE = True
except ImportError:
    PII_DETECTION_AVAILABLE = False
    logging.warning("Presidio not available. PII detection will be disabled.")


# Try importing ollama; fallback to requests-based client if not available
try:
    from ollama import Client as OllamaClient  # type: ignore

    def _llm_chat(prompt: str, model: str, host: str) -> str:
        oc = OllamaClient(host=host)
        res = oc.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return res["message"]["content"]

except ImportError:

    def _llm_chat(prompt: str, model: str, host: str) -> str:
        r = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("response", "")


# ---------------------------------------------------------------------------
# Environment Variables with default values
COLLECTION: str = os.getenv("COLLECTION", "dcase24_bearing")
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
SEARCH_LIMIT: int = int(os.getenv("SEARCH_LIMIT", "6"))

# Monitoring configuration
METRICS_ENABLED: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"
JAEGER_ENDPOINT: str = os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces")

# Security configuration
API_KEY: str = os.getenv("API_KEY", "")
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM: str = "HS256"
JWT_EXPIRATION_HOURS: int = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
RATE_LIMIT_PER_MINUTE: str = os.getenv("RATE_LIMIT", "10/minute")
ENABLE_API_KEY_AUTH: bool = os.getenv("ENABLE_API_KEY_AUTH", "false").lower() == "true"
ENABLE_PII_DETECTION: bool = os.getenv("ENABLE_PII_DETECTION", "true").lower() == "true"
MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "1000"))

# ---------------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------------

# Request metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

REQUEST_SIZE = Histogram(
    'api_request_size_bytes',
    'Request size in bytes',
    ['method', 'endpoint']
)

RESPONSE_SIZE = Histogram(
    'api_response_size_bytes',
    'Response size in bytes',
    ['method', 'endpoint']
)

# RAG-specific metrics
RAG_QUERY_COUNT = Counter(
    'rag_queries_total',
    'Total number of RAG queries',
    ['collection']
)

RAG_QUERY_DURATION = Histogram(
    'rag_query_duration_seconds',
    'RAG query processing time',
    ['collection', 'stage'],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
)

EMBEDDING_DURATION = Histogram(
    'embedding_duration_seconds',
    'Embedding generation time',
    ['model'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
)

VECTOR_SEARCH_DURATION = Histogram(
    'vector_search_duration_seconds',
    'Vector search time',
    ['collection'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5)
)

LLM_DURATION = Histogram(
    'llm_generation_duration_seconds',
    'LLM response generation time',
    ['model'],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0)
)

# System metrics
ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections'
)

CACHE_HITS = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

CACHE_MISSES = Counter(
    'cache_misses_total', 
    'Total cache misses',
    ['cache_type']
)

# Security metrics
RATE_LIMIT_EXCEEDED = Counter(
    'rate_limit_exceeded_total',
    'Total rate limit violations',
    ['endpoint', 'client_ip']
)

AUTH_FAILURES = Counter(
    'auth_failures_total',
    'Total authentication failures',
    ['reason']
)

PII_DETECTIONS = Counter(
    'pii_detections_total',
    'Total PII detections and redactions',
    ['pii_type']
)

BLOCKED_REQUESTS = Counter(
    'blocked_requests_total',
    'Total blocked requests',
    ['reason']
)

# ---------------------------------------------------------------------------
# OpenTelemetry Setup
# ---------------------------------------------------------------------------

if OTEL_AVAILABLE and METRICS_ENABLED:
    # Setup tracing
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Setup Jaeger exporter if endpoint is configured
    if JAEGER_ENDPOINT:
        jaeger_exporter = JaegerExporter(
            endpoint=JAEGER_ENDPOINT,
        )
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

# ---------------------------------------------------------------------------
# Security Setup
# ---------------------------------------------------------------------------

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# API Key authentication
security = HTTPBearer(auto_error=False)

# PII Detection engines
if PII_DETECTION_AVAILABLE and ENABLE_PII_DETECTION:
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
else:
    analyzer = None
    anonymizer = None

# Input validation patterns
DANGEROUS_PATTERNS = [
    r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
    r'javascript:',  # JavaScript URLs
    r'on\w+\s*=',  # Event handlers
    r'data:text\/html',  # Data URLs
    r'vbscript:',  # VBScript
]

def validate_and_sanitize_input(text: str) -> str:
    """Validate and sanitize user input."""
    if not text or len(text.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input cannot be empty"
        )
    
    if len(text) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input too long. Maximum {MAX_QUERY_LENGTH} characters allowed."
        )
    
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            if METRICS_ENABLED:
                BLOCKED_REQUESTS.labels(reason="dangerous_pattern").inc()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input contains potentially dangerous content"
            )
    
    # Sanitize HTML
    sanitized = bleach.clean(text, tags=[], attributes={}, strip=True)
    
    # PII Detection and redaction
    if analyzer and anonymizer and ENABLE_PII_DETECTION:
        results = analyzer.analyze(text=sanitized, language='en')
        if results:
            anonymized = anonymizer.anonymize(text=sanitized, analyzer_results=results)
            for result in results:
                if METRICS_ENABLED:
                    PII_DETECTIONS.labels(pii_type=result.entity_type).inc()
            sanitized = anonymized.text
    
    return sanitized

def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> bool:
    """Verify API key if authentication is enabled."""
    if not ENABLE_API_KEY_AUTH:
        return True
    
    if not API_KEY:
        # If auth is enabled but no key is set, allow requests (dev mode)
        return True
    
    if not credentials:
        if METRICS_ENABLED:
            AUTH_FAILURES.labels(reason="missing_credentials").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.credentials != API_KEY:
        if METRICS_ENABLED:
            AUTH_FAILURES.labels(reason="invalid_api_key").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return True

# Initialize clients
client = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(EMBED_MODEL)

app = FastAPI(
    title="Industrial‑Audio RAG API",
    description="Query factory machine sounds with natural language",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Custom rate limit exception handler with metrics
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    if METRICS_ENABLED:
        client_ip = get_remote_address(request)
        RATE_LIMIT_EXCEEDED.labels(endpoint=request.url.path, client_ip=client_ip).inc()
    
    response = JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "Rate limit exceeded",
            "detail": f"Rate limit exceeded: {exc.detail}",
            "retry_after": getattr(exc, 'retry_after', None)
        }
    )
    return response

# ---------------------------------------------------------------------------
# Middleware for metrics collection
# ---------------------------------------------------------------------------

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect metrics for all HTTP requests."""
    if not METRICS_ENABLED:
        return await call_next(request)
    
    start_time = time.time()
    
    # Get request size
    request_size = 0
    if hasattr(request, "headers") and "content-length" in request.headers:
        request_size = int(request.headers["content-length"])
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Extract endpoint info
    method = request.method
    endpoint = request.url.path
    status = str(response.status_code)
    
    # Update metrics
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    REQUEST_SIZE.labels(method=method, endpoint=endpoint).observe(request_size)
    
    # Get response size if available
    if hasattr(response, "headers") and "content-length" in response.headers:
        response_size = int(response.headers["content-length"])
        RESPONSE_SIZE.labels(method=method, endpoint=endpoint).observe(response_size)
    
    return response

# ---------------------------------------------------------------------------
# Instrument with OpenTelemetry
# ---------------------------------------------------------------------------

if OTEL_AVAILABLE and METRICS_ENABLED:
    FastAPIInstrumentor.instrument_app(app)
    RequestsInstrumentor().instrument()


# ---------------------------------------------------------------------------


def _rag_answer(question: str) -> str:
    """
    Generate an answer using RAG over industrial audio logs with detailed metrics.

    Args:
        question (str): Natural language query about machine sounds

    Returns:
        str: LLM-generated answer based on retrieved snippets
    """
    start_time = time.time()
    
    # Increment query counter
    if METRICS_ENABLED:
        RAG_QUERY_COUNT.labels(collection=COLLECTION).inc()
    
    # Embedding stage
    embedding_start = time.time()
    if OTEL_AVAILABLE and METRICS_ENABLED:
        with tracer.start_as_current_span("embedding_generation") as span:
            span.set_attribute("model", EMBED_MODEL)
            span.set_attribute("question_length", len(question))
            vec: List[float] = embedder.encode(question).tolist()
    else:
        vec: List[float] = embedder.encode(question).tolist()
    
    if METRICS_ENABLED:
        embedding_duration = time.time() - embedding_start
        EMBEDDING_DURATION.labels(model=EMBED_MODEL).observe(embedding_duration)
        RAG_QUERY_DURATION.labels(collection=COLLECTION, stage="embedding").observe(embedding_duration)
    
    # Vector search stage
    search_start = time.time()
    if OTEL_AVAILABLE and METRICS_ENABLED:
        with tracer.start_as_current_span("vector_search") as span:
            span.set_attribute("collection", COLLECTION)
            span.set_attribute("search_limit", SEARCH_LIMIT)
            span.set_attribute("vector_dimension", len(vec))
            hits = client.search(
                collection_name=COLLECTION, query_vector=vec, limit=SEARCH_LIMIT
            )
            span.set_attribute("hits_found", len(hits))
    else:
        hits = client.search(
            collection_name=COLLECTION, query_vector=vec, limit=SEARCH_LIMIT
        )
    
    if METRICS_ENABLED:
        search_duration = time.time() - search_start
        VECTOR_SEARCH_DURATION.labels(collection=COLLECTION).observe(search_duration)
        RAG_QUERY_DURATION.labels(collection=COLLECTION, stage="search").observe(search_duration)

    if not hits:
        total_duration = time.time() - start_time
        if METRICS_ENABLED:
            RAG_QUERY_DURATION.labels(collection=COLLECTION, stage="total").observe(total_duration)
        return "No matching audio snippets found."

    # Context preparation
    context: str = "\n".join(json.dumps(h.payload) for h in hits)
    prompt: str = (
        "You are an industrial‑AI assistant. Use only the sensor snippets below.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n"
    )
    
    # LLM generation stage
    llm_start = time.time()
    if OTEL_AVAILABLE and METRICS_ENABLED:
        with tracer.start_as_current_span("llm_generation") as span:
            span.set_attribute("model", OLLAMA_MODEL)
            span.set_attribute("prompt_length", len(prompt))
            span.set_attribute("context_snippets", len(hits))
            answer = _llm_chat(prompt, OLLAMA_MODEL, OLLAMA_URL)
            span.set_attribute("answer_length", len(answer))
    else:
        answer = _llm_chat(prompt, OLLAMA_MODEL, OLLAMA_URL)
    
    if METRICS_ENABLED:
        llm_duration = time.time() - llm_start
        total_duration = time.time() - start_time
        LLM_DURATION.labels(model=OLLAMA_MODEL).observe(llm_duration)
        RAG_QUERY_DURATION.labels(collection=COLLECTION, stage="llm").observe(llm_duration)
        RAG_QUERY_DURATION.labels(collection=COLLECTION, stage="total").observe(total_duration)
    
    return answer


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Industrial‑Audio RAG API alive",
        "version": "0.1.0",
        "endpoints": ["/ask", "/info", "/health", "/metrics", "/security"],
        "docs": "/docs",
        "security": {
            "authentication_required": ENABLE_API_KEY_AUTH,
            "rate_limiting_enabled": True,
            "pii_detection_enabled": ENABLE_PII_DETECTION and PII_DETECTION_AVAILABLE
        }
    }


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint for container orchestration."""
    try:
        # Check Qdrant connection
        qdrant_health = client.get_collections()
        qdrant_status = "healthy"
    except Exception as e:
        qdrant_status = f"unhealthy: {str(e)}"
    
    # Check if embedding model is loaded
    try:
        embedder.encode("test")
        embedding_status = "healthy"
    except Exception as e:
        embedding_status = f"unhealthy: {str(e)}"
    
    overall_status = "healthy" if all(
        status == "healthy" for status in [qdrant_status, embedding_status]
    ) else "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": time.time(),
        "services": {
            "qdrant": qdrant_status,
            "embedding_model": embedding_status,
        },
        "config": {
            "collection": COLLECTION,
            "embedding_model": EMBED_MODEL,
            "llm_model": OLLAMA_MODEL,
            "metrics_enabled": METRICS_ENABLED
        }
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if not METRICS_ENABLED:
        return {"error": "Metrics are disabled"}
    
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/ask")
@limiter.limit(RATE_LIMIT_PER_MINUTE)
async def ask(
    request: Request,
    q: str = Query(..., description="Natural-language question"),
    authenticated: bool = Depends(verify_api_key)
) -> dict[str, Any]:
    """
    Ask a natural-language question about industrial audio data.
    
    Requires API key authentication if enabled.
    Rate limited to prevent abuse.

    Returns:
        dict: Question + generated answer with metadata
    """
    start_time = time.time()
    
    # Validate and sanitize input
    try:
        sanitized_question = validate_and_sanitize_input(q)
    except HTTPException as e:
        # Re-raise the HTTPException with proper status code
        raise e
    
    # Process the sanitized question
    answer = _rag_answer(sanitized_question)
    processing_time = time.time() - start_time
    
    response = {
        "question": sanitized_question,  # Return sanitized version
        "answer": answer,
        "metadata": {
            "processing_time_seconds": round(processing_time, 3),
            "collection": COLLECTION,
            "embedding_model": EMBED_MODEL,
            "llm_model": OLLAMA_MODEL,
            "sanitized": sanitized_question != q,  # Indicate if input was modified
            "security": {
                "authenticated": authenticated and ENABLE_API_KEY_AUTH,
                "rate_limited": True,
                "pii_detection_enabled": ENABLE_PII_DETECTION and PII_DETECTION_AVAILABLE
            }
        }
    }
    
    if METRICS_ENABLED:
        response["metadata"]["metrics_enabled"] = True
    
    return response


@app.get("/info")
@limiter.limit("30/minute")  # More generous rate limit for info endpoint
async def info(request: Request) -> dict[str, Any]:
    """
    Return metadata about the current RAG setup.

    Returns:
        dict: Info including vector count, models used, etc.
    """
    try:
        count = client.count(collection_name=COLLECTION).count
        collection_info = client.get_collection(collection_name=COLLECTION)
    except Exception as e:
        count = 0
        collection_info = {"error": str(e)}
    
    return {
        "collection": COLLECTION,
        "vectors": count,
        "embedding_model": EMBED_MODEL,
        "llm_model": OLLAMA_MODEL,
        "search_limit": SEARCH_LIMIT,
        "metrics_enabled": METRICS_ENABLED,
        "collection_info": collection_info.dict() if hasattr(collection_info, 'dict') else collection_info,
        "opentelemetry_available": OTEL_AVAILABLE,
        "security": {
            "api_key_auth_enabled": ENABLE_API_KEY_AUTH,
            "pii_detection_enabled": ENABLE_PII_DETECTION and PII_DETECTION_AVAILABLE,
            "rate_limiting_enabled": True,
            "max_query_length": MAX_QUERY_LENGTH,
            "rate_limit": RATE_LIMIT_PER_MINUTE
        }
    }


@app.get("/security")
@limiter.limit("10/minute")
async def security_status(
    request: Request,
    authenticated: bool = Depends(verify_api_key)
) -> dict[str, Any]:
    """
    Return security configuration and status.
    Requires authentication if API key auth is enabled.

    Returns:
        dict: Security configuration and metrics
    """
    security_metrics = {}
    
    if METRICS_ENABLED:
        # Get security-related metrics (simplified - in production you'd query Prometheus)
        security_metrics = {
            "rate_limit_violations": "Available via /metrics endpoint",
            "auth_failures": "Available via /metrics endpoint", 
            "pii_detections": "Available via /metrics endpoint",
            "blocked_requests": "Available via /metrics endpoint"
        }
    
    return {
        "security_config": {
            "api_key_authentication": {
                "enabled": ENABLE_API_KEY_AUTH,
                "configured": bool(API_KEY) if ENABLE_API_KEY_AUTH else False
            },
            "rate_limiting": {
                "enabled": True,
                "default_limit": RATE_LIMIT_PER_MINUTE,
                "ask_endpoint_limit": RATE_LIMIT_PER_MINUTE,
                "info_endpoint_limit": "30/minute",
                "security_endpoint_limit": "10/minute"
            },
            "input_validation": {
                "enabled": True,
                "max_query_length": MAX_QUERY_LENGTH,
                "html_sanitization": True,
                "dangerous_pattern_detection": True
            },
            "pii_detection": {
                "enabled": ENABLE_PII_DETECTION,
                "available": PII_DETECTION_AVAILABLE,
                "auto_redaction": ENABLE_PII_DETECTION and PII_DETECTION_AVAILABLE
            },
            "monitoring": {
                "security_metrics": METRICS_ENABLED,
                "opentelemetry_tracing": OTEL_AVAILABLE and METRICS_ENABLED
            }
        },
        "security_metrics": security_metrics,
        "recommendations": [
            "Enable API key authentication in production" if not ENABLE_API_KEY_AUTH else None,
            "Configure proper CORS origins for production" if "*" in str(app.user_middleware) else None,
            "Set up proper secrets management" if JWT_SECRET_KEY == "your-secret-key-change-in-production" else None,
            "Monitor security metrics via /metrics endpoint" if METRICS_ENABLED else "Enable metrics for security monitoring"
        ]
    }
