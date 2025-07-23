# 🚀 LLM Inference Server Monitoring

Comprehensive monitoring system for the LLM inference pipeline with Prometheus metrics, health checks, and performance tracking.

## 📊 Monitoring Features

### ✅ **Implemented Features**

1. **Prometheus Metrics Integration**
   - Real-time metric collection
   - Custom business metrics
   - Performance monitoring
   - Error tracking

2. **Health & Status Endpoints**
   - Service health checks
   - Detailed status information
   - Model cache monitoring
   - Device information

3. **Performance Tracking**
   - Response time monitoring
   - Token generation metrics
   - Request/response size tracking
   - Memory usage monitoring

4. **Business Metrics**
   - Successful/failed inferences
   - Model version tracking
   - Domain-specific metrics
   - Cache hit/miss ratios

## 🔧 Quick Start

### 1. Start the Mock Server (for testing)

```bash
cd ml_pipeline
python test_server.py
```

### 2. Test Monitoring Endpoints

```bash
python test_monitoring.py
```

### 3. Start Production Server

```bash
python start_server.py
```

## 📈 Available Metrics

### **Standard API Metrics**

| Metric | Type | Description | Labels |
|--------|------|-------------|---------|
| `api_requests_total` | Counter | Total API requests | endpoint, status, model, version |
| `api_response_time_seconds` | Histogram | Response time | endpoint, model, version |
| `api_request_size_bytes` | Histogram | Request size | endpoint |
| `api_response_size_bytes` | Histogram | Response size | endpoint |

### **Model Performance Metrics**

| Metric | Type | Description | Labels |
|--------|------|-------------|---------|
| `model_load_time_seconds` | Histogram | Model loading time | model, version |
| `model_memory_usage_bytes` | Gauge | Model memory usage | model, version |
| `token_generation_time_seconds` | Histogram | Time per token | model, version |
| `tokens_generated_total` | Counter | Total tokens generated | model, version |

### **Cache Performance Metrics**

| Metric | Type | Description | Labels |
|--------|------|-------------|---------|
| `model_cache_hits_total` | Counter | Cache hits | model, version |
| `model_cache_misses_total` | Counter | Cache misses | model, version |

### **Business Metrics**

| Metric | Type | Description | Labels |
|--------|------|-------------|---------|
| `successful_inferences_total` | Counter | Successful inferences | model, version, domain |
| `failed_inferences_total` | Counter | Failed inferences | model, version, error_type |

## 🌐 API Endpoints

### **Health Check**
```http
GET /health
```
**Response:**
```json
{
  "status": "ok",
  "timestamp": 1703123456.789
}
```

### **Service Status**
```http
GET /status
```
**Response:**
```json
{
  "service": "LLM Inference API",
  "version": "1.0.0",
  "status": "healthy",
  "timestamp": 1703123456.789,
  "prometheus_available": true,
  "cached_models": 2,
  "device": "cuda",
  "cached_model_info": [
    {
      "base_model": "microsoft/DialoGPT-medium",
      "version": "v1.0",
      "loaded": true
    }
  ]
}
```

### **Prometheus Metrics**
```http
GET /prometheus
```
**Response:** Prometheus-formatted metrics
```
# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total{endpoint="/infer",status="success",model="microsoft/DialoGPT-medium",version="v1.0"} 42

# HELP api_response_time_seconds Response time in seconds
# TYPE api_response_time_seconds histogram
api_response_time_seconds_bucket{endpoint="/infer",model="microsoft/DialoGPT-medium",version="v1.0",le="0.1"} 35
```

### **List Models**
```http
GET /models
```
**Response:**
```json
{
  "models": [
    {
      "base_model": "microsoft/DialoGPT-medium",
      "versions": ["v1.0", "v1.1"],
      "description": "DialoGPT model for conversation"
    }
  ]
}
```

### **Inference Endpoint**
```http
POST /infer
Content-Type: application/json
X-API-Key: your-api-key

{
  "prompt": "What is the capital of France?",
  "max_new_tokens": 32,
  "adapter_version": "v1.0",
  "base_model": "microsoft/DialoGPT-medium"
}
```
**Response:**
```json
{
  "result": "The capital of France is Paris.",
  "metadata": {
    "model": "microsoft/DialoGPT-medium",
    "version": "v1.0",
    "tokens_generated": 8,
    "response_time": 0.245,
    "generation_time": 0.198
  }
}
```

## 📊 Monitoring Dashboard

### **Key Performance Indicators (KPIs)**

1. **Response Time**
   - Average: `< 500ms`
   - 95th percentile: `< 1s`
   - 99th percentile: `< 2s`

2. **Throughput**
   - Requests per second: `> 10`
   - Tokens per second: `> 100`

3. **Reliability**
   - Success rate: `> 99%`
   - Error rate: `< 1%`

4. **Resource Usage**
   - Memory usage: `< 80%`
   - Cache hit rate: `> 90%`

### **Alerting Rules**

```yaml
# High response time
- alert: HighResponseTime
  expr: histogram_quantile(0.95, api_response_time_seconds) > 1
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High response time detected"

# High error rate
- alert: HighErrorRate
  expr: rate(api_requests_total{status="error"}[5m]) / rate(api_requests_total[5m]) > 0.05
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "High error rate detected"

# Model loading failures
- alert: ModelLoadingFailures
  expr: rate(model_load_time_seconds_count[5m]) > 0
  for: 1m
  labels:
    severity: warning
  annotations:
    summary: "Model loading failures detected"
```

## 🔍 Troubleshooting

### **Common Issues**

1. **Prometheus Metrics Not Available**
   ```bash
   pip install prometheus-fastapi-instrumentator prometheus-client
   ```

2. **High Response Times**
   - Check model cache hit rate
   - Monitor memory usage
   - Verify GPU availability

3. **High Error Rates**
   - Check API key authentication
   - Verify model availability
   - Monitor system resources

### **Debug Endpoints**

```bash
# Check service health
curl http://localhost:8000/health

# Get detailed status
curl http://localhost:8000/status

# View Prometheus metrics
curl http://localhost:8000/prometheus

# List available models
curl http://localhost:8000/models
```

## 🚀 Production Deployment

### **Docker Configuration**

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "deployment.inference_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **Kubernetes Configuration**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-inference
  template:
    metadata:
      labels:
        app: llm-inference
    spec:
      containers:
      - name: inference-server
        image: llm-inference:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
```

### **Prometheus Configuration**

```yaml
scrape_configs:
  - job_name: 'llm-inference'
    static_configs:
      - targets: ['llm-inference:8000']
    metrics_path: '/prometheus'
    scrape_interval: 15s
```

## 📈 Monitoring Best Practices

1. **Set up Grafana Dashboards**
   - Response time trends
   - Error rate monitoring
   - Resource usage tracking
   - Business metrics visualization

2. **Configure Alerting**
   - Response time thresholds
   - Error rate alerts
   - Resource usage warnings
   - Business metric alerts

3. **Log Aggregation**
   - Centralized logging
   - Structured log format
   - Error correlation
   - Performance analysis

4. **Performance Optimization**
   - Model caching
   - Batch processing
   - Resource monitoring
   - Auto-scaling

## 🔧 Configuration

### **Environment Variables**

```bash
# Server configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# Monitoring configuration
PROMETHEUS_ENABLED=true
METRICS_PATH=/prometheus

# Model configuration
DEFAULT_MODEL=microsoft/DialoGPT-medium
DEFAULT_VERSION=v1.0
CACHE_SIZE=5

# Security configuration
API_KEY_HEADER=X-API-Key
RATE_LIMIT=5/minute
```

### **Configuration File**

```yaml
server:
  host: 0.0.0.0
  port: 8000
  log_level: info

monitoring:
  prometheus_enabled: true
  metrics_path: /prometheus
  custom_metrics: true

models:
  default_model: microsoft/DialoGPT-medium
  default_version: v1.0
  cache_size: 5

security:
  api_key_header: X-API-Key
  rate_limit: 5/minute
```

## 📚 Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [FastAPI Monitoring](https://fastapi.tiangolo.com/)
- [Prometheus FastAPI Instrumentator](https://github.com/trallnag/prometheus-fastapi-instrumentator)
- [Grafana Dashboards](https://grafana.com/grafana/dashboards/)

---

**🎉 Your LLM inference server is now fully monitored and production-ready!** 