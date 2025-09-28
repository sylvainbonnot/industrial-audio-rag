# Security Features & Configuration

The Industrial Audio RAG API implements comprehensive security measures to protect against common threats and ensure safe operation in production environments.

## 🔒 Security Features

### 1. API Key Authentication
- **Status**: Optional (disabled by default)
- **Environment Variable**: `ENABLE_API_KEY_AUTH=true`
- **Configuration**: Set `API_KEY` environment variable
- **Usage**: Include `Authorization: Bearer <your-api-key>` header

```bash
# Enable API key authentication
export ENABLE_API_KEY_AUTH=true
export API_KEY=your-secure-api-key-here

# Test with authentication
curl -H "Authorization: Bearer your-secure-api-key-here" \
     "http://localhost:8000/ask?q=test+query"
```

### 2. Rate Limiting
- **Status**: Always enabled
- **Default Limit**: 10 requests per minute per IP
- **Configuration**: `RATE_LIMIT=10/minute`
- **Endpoints**:
  - `/ask`: Configurable rate limit (default: 10/min)
  - `/info`: 30/minute
  - `/security`: 10/minute

```bash
# Configure rate limiting
export RATE_LIMIT=20/minute  # Increase to 20 requests per minute
```

### 3. Input Validation & Sanitization
- **HTML/Script Injection Protection**: Removes dangerous HTML tags and scripts
- **Pattern Detection**: Blocks potentially malicious patterns
- **Length Limits**: Configurable maximum query length
- **Configuration**: `MAX_QUERY_LENGTH=1000`

**Blocked Patterns**:
- `<script>` tags
- JavaScript URLs (`javascript:`)
- Event handlers (`onclick=`, etc.)
- Data URLs with HTML content
- VBScript

### 4. PII Detection & Redaction
- **Status**: Enabled by default (if Presidio is available)
- **Engine**: Microsoft Presidio
- **Configuration**: `ENABLE_PII_DETECTION=true`
- **Auto-redaction**: Automatically replaces detected PII with placeholders

**Detected PII Types**:
- Email addresses
- Phone numbers
- Credit card numbers
- Social Security Numbers
- Names (person)
- Addresses

```bash
# Disable PII detection
export ENABLE_PII_DETECTION=false
```

### 5. Security Monitoring
- **Metrics**: Comprehensive security metrics via Prometheus
- **Alerting**: Pre-configured alerts for security events
- **Tracing**: OpenTelemetry integration for request tracing

**Security Metrics**:
- `rate_limit_exceeded_total`: Rate limit violations by endpoint and IP
- `auth_failures_total`: Authentication failures by reason
- `pii_detections_total`: PII detections by type
- `blocked_requests_total`: Blocked requests by reason

## 🛡️ Security Endpoints

### GET /security
Returns comprehensive security configuration and status.

**Requires authentication if API key auth is enabled.**

```bash
curl "http://localhost:8000/security"
```

**Response**:
```json
{
  "security_config": {
    "api_key_authentication": {
      "enabled": false,
      "configured": false
    },
    "rate_limiting": {
      "enabled": true,
      "default_limit": "10/minute"
    },
    "input_validation": {
      "enabled": true,
      "max_query_length": 1000,
      "html_sanitization": true
    },
    "pii_detection": {
      "enabled": true,
      "available": true,
      "auto_redaction": true
    }
  },
  "recommendations": [
    "Enable API key authentication in production"
  ]
}
```

## 🔧 Production Security Checklist

### Essential Security Configuration
- [ ] **Enable API Key Authentication**: `ENABLE_API_KEY_AUTH=true`
- [ ] **Set Strong API Key**: Generate cryptographically secure API key
- [ ] **Configure JWT Secret**: `JWT_SECRET=<secure-random-string>`
- [ ] **Adjust Rate Limits**: Set appropriate limits for your use case
- [ ] **Configure CORS**: Restrict origins in production
- [ ] **Enable HTTPS**: Use reverse proxy (nginx/traefik) with TLS

### Environment Variables for Production
```bash
# Security
ENABLE_API_KEY_AUTH=true
API_KEY=<generate-secure-key>
JWT_SECRET=<generate-secure-secret>
RATE_LIMIT=30/minute
MAX_QUERY_LENGTH=500

# PII Protection
ENABLE_PII_DETECTION=true

# Monitoring
METRICS_ENABLED=true
```

### Network Security
```yaml
# docker-compose.prod.yml
services:
  api:
    environment:
      - ENABLE_API_KEY_AUTH=true
      - API_KEY=${API_KEY}
    # Remove external port exposure
    # ports: ["8000:8000"]  # Remove this
    networks:
      - internal

  nginx:
    image: nginx:alpine
    ports: ["443:443", "80:80"]
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    networks:
      - internal
```

### Monitoring & Alerting
```yaml
# Prometheus alerting rules
- alert: HighSecurityViolations
  expr: rate(rate_limit_exceeded_total[5m]) > 10
  for: 2m
  annotations:
    summary: "High rate of security violations detected"

- alert: AuthenticationFailures
  expr: rate(auth_failures_total[5m]) > 5
  for: 1m
  annotations:
    summary: "Multiple authentication failures detected"
```

## 🚨 Security Incident Response

### Rate Limit Violations
1. Monitor `/metrics` endpoint for `rate_limit_exceeded_total`
2. Check source IPs in logs
3. Consider IP blocking at reverse proxy level

### Authentication Failures
1. Monitor `auth_failures_total` metric
2. Rotate API keys if suspicious activity
3. Review access logs for patterns

### PII Detection Alerts
1. Monitor `pii_detections_total` for unusual spikes
2. Review sanitized logs to understand data patterns
3. Consider additional input validation rules

## 🔍 Testing Security Features

### Rate Limiting Test
```bash
# Test rate limiting
for i in {1..15}; do
  curl "http://localhost:8000/ask?q=test$i"
  echo "Request $i"
done
```

### Input Validation Test
```bash
# Test dangerous input blocking
curl "http://localhost:8000/ask?q=<script>alert('xss')</script>"

# Test PII redaction
curl "http://localhost:8000/ask?q=My email is john@example.com"
```

### Authentication Test
```bash
# Test without API key (should fail if auth enabled)
curl "http://localhost:8000/ask?q=test"

# Test with invalid API key
curl -H "Authorization: Bearer invalid-key" \
     "http://localhost:8000/ask?q=test"

# Test with valid API key
curl -H "Authorization: Bearer your-api-key" \
     "http://localhost:8000/ask?q=test"
```

## 📚 Security Best Practices

1. **API Keys**: Use long, random keys (>32 characters)
2. **Secrets**: Never commit secrets to version control
3. **Logs**: Ensure no sensitive data in logs
4. **Updates**: Keep dependencies updated via Dependabot
5. **Scanning**: Regular security scans in CI/CD
6. **Monitoring**: Active monitoring of security metrics
7. **Backup**: Regular backups of vector database
8. **Access**: Principle of least privilege for all systems