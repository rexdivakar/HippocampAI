# HippocampAI Monitoring Integration Guide

## Overview

HippocampAI includes comprehensive monitoring with **Prometheus** metrics collection and **Grafana** dashboards for real-time visibility into:

- API performance and request rates
- Memory operations (CRUD, search, retrieval)
- Vector database (Qdrant) performance
- Cache hit rates and efficiency
- Memory health and conflicts
- Background tasks and system errors

**Status**: ✅ Fully Integrated

---

## Architecture

```
┌─────────────────┐
│  HippocampAI    │
│  FastAPI API    │──┐
│  (Port 8000)    │  │
│  /metrics       │  │
└─────────────────┘  │
                     │ scrapes metrics
┌─────────────────┐  │ every 10s
│     Qdrant      │──┤
│  (Port 6333)    │  │
│  /metrics       │  │
└─────────────────┘  │
                     ▼
┌─────────────────┐  ┌──────────────────┐
│     Redis       │  │   Prometheus     │
│  (Port 6379)    │──│   (Port 9090)    │
│  /metrics       │  │  - Collects      │
└─────────────────┘  │  - Stores        │
                     │  - Queries       │
                     └────────┬─────────┘
                              │
                              │ datasource
                              ▼
                     ┌──────────────────┐
                     │     Grafana      │
                     │   (Port 3000)    │
                     │  - Dashboards    │
                     │  - Alerts        │
                     │  - Visualizations│
                     └──────────────────┘
```

---

## Metrics Collected

### API Performance Metrics

```
# HTTP Request Metrics
hippocampai_http_requests_total{method, endpoint, status}
hippocampai_http_request_duration_seconds{method, endpoint}
hippocampai_http_requests_in_progress{method, endpoint}

# Example Queries:
rate(hippocampai_http_requests_total[1m])
histogram_quantile(0.95, rate(hippocampai_http_request_duration_seconds_bucket[5m]))
```

### Memory Operation Metrics

```
# CRUD Operations
hippocampai_memory_operations_total{operation, status}
hippocampai_memory_operation_duration_seconds{operation}

# Memory Store
hippocampai_total_memories{user_id, memory_type}
hippocampai_memories_created_total{memory_type}
hippocampai_memories_deleted_total{memory_type}

# Example Queries:
rate(hippocampai_memory_operations_total[5m])
sum(hippocampai_total_memories) by (memory_type)
```

### Search & Retrieval Metrics

```
# Search Performance
hippocampai_search_requests_total{search_type, status}
hippocampai_search_duration_seconds{search_type}
hippocampai_search_results_count

# Cache Performance
hippocampai_cache_hits_total{cache_type}
hippocampai_cache_misses_total{cache_type}
hippocampai_cache_hit_rate{cache_type}

# Example Queries:
hippocampai_cache_hit_rate
histogram_quantile(0.99, rate(hippocampai_search_duration_seconds_bucket[5m]))
```

### Advanced Feature Metrics

```
# Conflict Resolution
hippocampai_conflicts_detected_total{conflict_type}
hippocampai_conflicts_resolved_total{resolution_strategy}

# Memory Health
hippocampai_memory_health_score{user_id}
hippocampai_stale_memories_count{user_id}
hippocampai_duplicate_memories_count{user_id}

# Temporal Features
hippocampai_memory_freshness_score{memory_id}
hippocampai_memory_age_days

# Example Queries:
avg(hippocampai_memory_health_score)
rate(hippocampai_conflicts_detected_total[1h])
```

### Vector Database Metrics

```
# Qdrant Operations
hippocampai_qdrant_operations_total{operation, collection, status}
hippocampai_qdrant_operation_duration_seconds{operation, collection}
hippocampai_vector_search_latency_seconds{collection}

# Example Queries:
rate(hippocampai_qdrant_operations_total[5m])
histogram_quantile(0.95, rate(hippocampai_vector_search_latency_seconds_bucket[5m]))
```

### System & Error Metrics

```
# Application Info
hippocampai_app_info{version, environment}

# Connections & Errors
hippocampai_active_connections
hippocampai_errors_total{error_type, operation}

# Background Tasks
hippocampai_background_tasks_total{task_name, status}
hippocampai_background_task_duration_seconds{task_name}

# Example Queries:
rate(hippocampai_errors_total[5m])
sum(hippocampai_active_connections)
```

---

## Quick Start

### 1. Start Monitoring Stack

```bash
# Start all services including monitoring
docker-compose up -d

# Verify services are running
docker-compose ps

# Check logs
docker-compose logs -f prometheus
docker-compose logs -f grafana
```

### 2. Access Monitoring UIs

**Prometheus**:
- URL: http://localhost:9090
- Query metrics and targets
- View active scrape configs

**Grafana**:
- URL: http://localhost:3000
- Username: `admin`
- Password: `admin` (change on first login)
- Pre-configured Prometheus datasource

**Flower (Celery)**:
- URL: http://localhost:5555
- Monitor background tasks
- View task history

### 3. Verify Metrics Endpoint

```bash
# Check API metrics endpoint
curl http://localhost:8000/metrics

# Should return Prometheus format metrics:
# TYPE hippocampai_http_requests_total counter
# hippocampai_http_requests_total{method="GET",endpoint="/health",status="200"} 42
```

### 4. Test Metrics Collection

```bash
# Generate some traffic
curl http://localhost:8000/health
curl -X POST http://localhost:8000/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"text": "Test memory", "user_id": "test"}'

# Wait 15 seconds for Prometheus scrape

# Check in Prometheus
open http://localhost:9090/graph
# Query: rate(hippocampai_http_requests_total[1m])
```

---

## Grafana Dashboards

### Pre-configured Dashboards

**1. HippocampAI Overview** (`hippocampai_overview.json`)
- API request rate and latency
- Memory operations rate
- Search performance
- Cache hit rates
- Memory health scores
- Conflict detection
- Error rates

Location: `monitoring/grafana/dashboards/hippocampai_overview.json`

### Accessing Dashboards

1. Open Grafana: http://localhost:3000
2. Login (admin/admin)
3. Navigate to: Dashboards → Browse
4. Select "HippocampAI Overview"

### Dashboard Panels

**Performance**:
- API Request Rate (requests/sec)
- API Response Time (p95, p99)
- Memory Operations Rate
- Search Latency

**Cache & Database**:
- Cache Hit Rate
- Qdrant Operation Latency
- Vector Search Performance

**Health & Quality**:
- Memory Health Score (0-100)
- Stale Memories Count
- Duplicate Detection
- Conflicts Detected

**Errors & System**:
- Error Rate by Type
- Active Connections
- Background Task Status

---

## Custom Queries

### Common PromQL Queries

**API Performance**:
```promql
# Request rate by endpoint
sum(rate(hippocampai_http_requests_total[5m])) by (endpoint)

# Slowest endpoints (p99)
histogram_quantile(0.99,
  sum(rate(hippocampai_http_request_duration_seconds_bucket[5m]))
  by (endpoint, le)
)

# Error rate
sum(rate(hippocampai_http_requests_total{status=~"5.."}[5m]))
```

**Memory Operations**:
```promql
# Total memories per user
sum(hippocampai_total_memories) by (user_id)

# Memory creation rate
rate(hippocampai_memories_created_total[5m])

# Average operation duration
avg(rate(hippocampai_memory_operation_duration_seconds_sum[5m]))
  /
avg(rate(hippocampai_memory_operation_duration_seconds_count[5m]))
```

**Cache Efficiency**:
```promql
# Overall cache hit rate
sum(rate(hippocampai_cache_hits_total[5m]))
  /
(sum(rate(hippocampai_cache_hits_total[5m])) + sum(rate(hippocampai_cache_misses_total[5m])))

# Cache hits per second
sum(rate(hippocampai_cache_hits_total[5m])) by (cache_type)
```

**System Health**:
```promql
# Average memory health
avg(hippocampai_memory_health_score) by (user_id)

# Total conflicts per hour
increase(hippocampai_conflicts_detected_total[1h])

# Error rate by operation
sum(rate(hippocampai_errors_total[5m])) by (operation)
```

---

## Alerting (Optional)

### Prometheus Alertmanager

Add to `monitoring/prometheus.yml`:

```yaml
rule_files:
  - 'alerts.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093
```

### Sample Alert Rules

Create `monitoring/alerts.yml`:

```yaml
groups:
  - name: hippocampai_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(hippocampai_errors_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/sec"

      - alert: SlowAPIResponse
        expr: histogram_quantile(0.95, rate(hippocampai_http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        annotations:
          summary: "API response time degraded"
          description: "P95 latency is {{ $value }}s"

      - alert: LowCacheHitRate
        expr: hippocampai_cache_hit_rate < 0.7
        for: 10m
        annotations:
          summary: "Cache hit rate below threshold"
          description: "Hit rate is {{ $value }}"

      - alert: MemoryHealthDegraded
        expr: avg(hippocampai_memory_health_score) < 50
        for: 15m
        annotations:
          summary: "Memory health score low"
          description: "Average health score is {{ $value }}/100"
```

---

## Troubleshooting

### Metrics Not Appearing

```bash
# 1. Check if metrics endpoint is accessible
curl http://localhost:8000/metrics

# 2. Check Prometheus targets
open http://localhost:9090/targets
# All targets should show "UP"

# 3. Check Prometheus logs
docker-compose logs prometheus | grep -i error

# 4. Verify prometheus-client is installed
docker exec hippocampai-api pip list | grep prometheus
```

### Grafana Dashboard Not Loading

```bash
# 1. Check Grafana datasource
# Go to: Configuration → Data Sources → Prometheus
# Click "Test" - should show "Data source is working"

# 2. Check dashboard provisioning
docker exec hippocampai-grafana ls -la /etc/grafana/provisioning/dashboards/

# 3. Restart Grafana
docker-compose restart grafana
```

### High Cardinality Warnings

If Prometheus warns about high cardinality:

```yaml
# Add to prometheus.yml
global:
  metric_relabel_configs:
    - source_labels: [user_id]
      regex: '^.{32,}$'  # Drop very long user IDs
      action: drop
```

---

## Best Practices

### 1. Metric Retention

```yaml
# prometheus.yml
global:
  scrape_interval: 15s  # Default scrape
  evaluation_interval: 15s

# Retention in docker-compose.yml
command:
  - '--storage.tsdb.retention.time=30d'  # Keep 30 days
  - '--storage.tsdb.retention.size=50GB'  # Max size
```

### 2. Query Performance

- Use `rate()` for counters
- Use recording rules for expensive queries
- Limit time range for large queries
- Use `by()` to reduce cardinality

### 3. Dashboard Organization

- Group related metrics together
- Use variables for filtering (user_id, endpoint)
- Set appropriate refresh intervals (10-30s)
- Add annotations for deployments

### 4. Alerting

- Set appropriate thresholds
- Use `for:` to avoid flapping
- Include context in annotations
- Test alerts before production

---

## Integration with Application Code

### Recording Custom Metrics

```python
from hippocampai.monitoring.prometheus_metrics import (
    record_memory_operation,
    record_search_operation,
    update_memory_health_metrics,
)

# Record a memory operation
start = time.time()
# ... perform operation ...
duration = time.time() - start
record_memory_operation("create", duration, "success", "fact")

# Record a search
record_search_operation("hybrid", search_time, len(results), "success")

# Update health metrics
update_memory_health_metrics("user123", {
    "overall_score": 85.5,
    "stale_memories": [],
    "duplicates": []
})
```

### Viewing Metrics in Real-time

```bash
# Watch metrics live
watch -n 2 'curl -s http://localhost:8000/metrics | grep hippocampai_http_requests_total'

# Stream Grafana dashboard
# Open: http://localhost:3000/d/hippocampai-overview
# Set refresh: 5s (top-right corner)
```

---

## Production Recommendations

1. **Use Prometheus Operator** for Kubernetes
2. **Enable Authentication** on Grafana
3. **Set up Alertmanager** for notifications
4. **Use Recording Rules** for expensive queries
5. **Export Dashboards** as code (JSON)
6. **Monitor Prometheus itself** (meta-monitoring)
7. **Set up Remote Storage** for long-term retention
8. **Use Service Discovery** instead of static configs

---

## Summary

✅ **Prometheus** - Metrics collection from all services
✅ **Grafana** - Visualization and dashboards
✅ **Comprehensive Metrics** - API, Memory, Search, Cache, Health
✅ **Pre-built Dashboards** - Ready to use
✅ **Docker Integration** - Fully automated setup
✅ **Production Ready** - Scalable and reliable

**Access Points**:
- Metrics: http://localhost:8000/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Flower: http://localhost:5555

**Next Steps**:
1. Deploy with `docker-compose up -d`
2. Access Grafana dashboards
3. Generate load to see metrics
4. Set up alerts (optional)
5. Customize dashboards as needed

---

**Documentation Version**: 1.0
**Date**: 2026-02-11
**Status**: Production Ready ✅
