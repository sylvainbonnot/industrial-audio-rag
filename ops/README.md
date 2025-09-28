# Industrial Audio RAG - Monitoring & Observability

This directory contains the monitoring and observability configuration for the Industrial Audio RAG system.

## 🚀 Quick Start

Start the full monitoring stack:

```bash
# Start with monitoring profile
docker-compose --profile monitoring up -d

# Or using the new compose command
docker compose --profile monitoring up -d
```

## 📊 Access Dashboards

- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus Metrics**: http://localhost:9090
- **API Metrics**: http://localhost:8000/metrics
- **API Health**: http://localhost:8000/health

## 🔍 Monitoring Stack

### Prometheus (Port 9090)
- Scrapes metrics from the RAG API every 15 seconds
- Collects Qdrant database metrics
- Stores time-series data with 30-day retention
- Configured alerting rules for performance and availability

### Grafana (Port 3000)
- Pre-configured dashboard for Industrial Audio RAG
- Real-time monitoring of:
  - API request rate and response times
  - RAG query performance (embedding, search, LLM stages)
  - System health and error rates
  - HTTP status code distribution

### OpenTelemetry (Optional)
- Distributed tracing for request flows
- Jaeger exporter for trace visualization
- Automatic instrumentation of FastAPI and requests

## 📈 Key Metrics

### API Performance
- `api_requests_total`: Total API requests by method, endpoint, status
- `api_request_duration_seconds`: Request latency histogram
- `api_request_size_bytes`: Request payload size
- `api_response_size_bytes`: Response payload size

### RAG Pipeline
- `rag_queries_total`: Total RAG queries by collection
- `rag_query_duration_seconds`: End-to-end query processing time
- `embedding_duration_seconds`: Embedding generation time
- `vector_search_duration_seconds`: Vector similarity search time
- `llm_generation_duration_seconds`: LLM response generation time

### System Health
- `up`: Service availability (1=up, 0=down)
- `active_connections`: Current active connections
- `cache_hits_total` / `cache_misses_total`: Cache performance

## 🚨 Alerting Rules

Configured alerts in `prometheus/alert_rules.yml`:

- **Critical**: API/Qdrant service down
- **Warning**: High error rate (>10%)
- **Warning**: High response time (95th percentile > 10s)
- **Warning**: Slow embedding/search/LLM generation
- **Info**: High query rate (>100 req/s)

## 🛠 Configuration

### Environment Variables
```bash
# Enable/disable metrics collection
METRICS_ENABLED=true

# Jaeger tracing endpoint
JAEGER_ENDPOINT=http://localhost:14268/api/traces

# Grafana admin password
GRAFANA_PASSWORD=admin

# Prometheus retention
PROMETHEUS_RETENTION=30d
```

### Customization

1. **Add Custom Metrics**: Modify `src/rag_audio/api.py`
2. **Update Dashboards**: Edit `grafana/dashboards/industrial-audio-rag.json`
3. **Configure Alerts**: Modify `prometheus/alert_rules.yml`
4. **Adjust Scraping**: Update `prometheus/prometheus.yml`

## 🔧 Troubleshooting

### Metrics Not Appearing
1. Check if metrics are enabled: `curl http://localhost:8000/metrics`
2. Verify Prometheus targets: http://localhost:9090/targets
3. Check container logs: `docker-compose logs prometheus grafana`

### Dashboard Not Loading
1. Verify Grafana provisioning: Check container logs
2. Confirm datasource connection: Grafana → Configuration → Data Sources
3. Import dashboard manually if needed

### Performance Issues
1. Reduce scrape intervals in `prometheus.yml`
2. Adjust retention policies
3. Monitor resource usage of monitoring stack

## 📚 Further Reading

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [OpenTelemetry Python](https://opentelemetry-python.readthedocs.io/)
- [FastAPI Monitoring Best Practices](https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/)