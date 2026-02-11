# HippocampAI SaaS Platform Guide

**Complete guide to deploying, configuring, and using HippocampAI as a SaaS platform**

Last Updated: 2026-02-11

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Deployment Modes](#deployment-modes)
4. [Authentication & User Management](#authentication--user-management)
5. [API Reference](#api-reference)
6. [Automation & Background Tasks](#automation--background-tasks)
7. [Monitoring & Observability](#monitoring--observability)
8. [Rate Limiting & Tiers](#rate-limiting--tiers)
9. [Security Best Practices](#security-best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

HippocampAI supports two operational modes:

- **Local Mode** (`USER_AUTH_ENABLED=false`) - No authentication, ideal for development and self-hosted deployments
- **SaaS Mode** (`USER_AUTH_ENABLED=true`) - Full authentication, rate limiting, and multi-tenant support

### Key Features

**Unified SaaS + Library Control:**
- Library users control ALL features programmatically
- Single configuration works in both library and SaaS modes
- Policy-based automation
- Flexible execution (immediate or background)
- No vendor lock-in

**SaaS Platform Features:**
- Multi-user authentication and API key management
- Tier-based rate limiting
- Usage tracking and analytics
- Admin dashboard
- Background task processing with Celery
- Real-time monitoring with Flower, Prometheus, and Grafana

---

## Quick Start

### 1. Start Infrastructure

```bash
# Clone repository
git clone https://github.com/rexdivakar/HippocampAI.git
cd HippocampAI

# Start all services with Docker Compose
docker-compose up -d

# Verify services are running
docker-compose ps
```

**Services Started:**
- API Server: http://localhost:8000 (FastAPI with Swagger UI)
- Admin Dashboard: http://localhost:3001
- Flower (Celery): http://localhost:5555
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Qdrant: http://localhost:6333
- Redis: port 6379

### 2. Verify PostgreSQL Schema

```bash
# Verify database initialization (uses docker exec - no local psql needed)
./scripts/verify_postgres_docker.sh

# Alternative if you have psql installed locally
# ./scripts/verify_postgres.sh
```

**Note**: PostgreSQL schema is automatically initialized on first container start.

### 3. Access Admin Dashboard

Open http://localhost:3001

**Default Admin Credentials:**
- Email: `admin@hippocampai.com`
- Password: `admin123`

**IMPORTANT**: Change the default admin password in production!

### 4. Create Users and API Keys

**Via Admin UI:**
1. Go to "Users" tab → Click "Create User"
2. Fill in user details (email, password, tier)
3. Click on user → "API Key" button
4. Copy the API key (shown only once!)

**Via API:**
```bash
# Create user
curl -X POST http://localhost:8000/admin/users \
  -H "X-User-Auth: false" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword",
    "tier": "pro"
  }'

# Create API key for user
curl -X POST http://localhost:8000/admin/users/{user_id}/api-keys \
  -H "X-User-Auth: false" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production Key",
    "rate_limit_tier": "pro",
    "expires_in_days": 365
  }'
```

---

## Deployment Modes

### Mode 1: Local Mode (Development/Self-Hosted)

**Configuration:**
```bash
USER_AUTH_ENABLED=false  # or not set (defaults to false)
```

**Behavior:**
- No authentication required
- No rate limiting
- All endpoints publicly accessible
- Full admin-level access

**Client Library Usage:**
```python
from hippocampai import MemoryClient

# Option 1: Explicit local mode
client = MemoryClient(user_auth=False)

# Option 2: Default (user_auth defaults to False)
client = MemoryClient()

# All operations work without authentication
client.add_memory(
    user_id="user123",
    text="Paris is the capital of France",
    memory_type="fact"
)
```

**API Requests:**
```bash
# No headers needed
curl -X POST http://localhost:8000/v1/memories:remember \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "text": "Paris is the capital of France",
    "type": "fact"
  }'
```

**Use Cases:**
- Local development
- Self-hosted deployments
- Internal team usage
- Testing without authentication overhead

### Mode 2: SaaS Mode (Production/Multi-Tenant)

**Configuration:**
```bash
USER_AUTH_ENABLED=true
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=hippocampai
POSTGRES_USER=hippocampai
POSTGRES_PASSWORD=your_secure_password
```

**Behavior:**
- Authentication required for all API requests
- Tier-based rate limiting enforced
- Multi-user support with usage tracking
- Admin endpoints require admin session

**Client Library Usage:**
```python
from hippocampai import MemoryClient
import os

# Option 1: Explicit API key
client = MemoryClient(
    user_auth=True,
    api_key="hc_live_xxxxxxxxxxxxxxxxxxxxx"
)

# Option 2: Environment variables
os.environ["HIPPOCAMPAI_USER_AUTH"] = "true"
os.environ["HIPPOCAMPAI_API_KEY"] = "hc_live_xxxxxxxxxxxxxxxxxxxxx"
client = MemoryClient()

# All operations now use authentication and are rate-limited
client.add_memory(
    user_id="user123",
    text="Paris is the capital of France",
    memory_type="fact"
)
```

**API Requests:**
```bash
curl -X POST http://localhost:8000/v1/memories:remember \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer hc_live_xxxxxxxxxxxxxxxxxxxxx" \
  -d '{
    "user_id": "user123",
    "text": "Paris is the capital of France",
    "type": "fact"
  }'
```

**Response Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1699564800
```

---

## Authentication & User Management

### User Tiers

| Tier       | Per Minute | Per Hour  | Per Day     |
|------------|------------|-----------|-------------|
| Free       | 10         | 100       | 1,000       |
| Pro        | 100        | 10,000    | 100,000     |
| Enterprise | 1,000      | 100,000   | 1,000,000   |
| Admin      | 10,000     | 1,000,000 | 10,000,000  |

### Creating Users

**Via Admin Dashboard:**
1. Login at http://localhost:3001
2. Navigate to "Users" tab
3. Click "Create User"
4. Fill in details and select tier
5. Submit

**Via API:**
```bash
curl -X POST http://localhost:8000/admin/users \
  -H "X-Session-Token: your-session-token" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "userpassword123",
    "full_name": "John Doe",
    "tier": "pro"
  }'
```

### API Key Management

**Generate API Key:**
```bash
curl -X POST http://localhost:8000/admin/users/{user_id}/api-keys \
  -H "X-Session-Token: your-session-token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production Key",
    "rate_limit_tier": "pro",
    "expires_in_days": 365
  }'
```

**Response:**
```json
{
  "api_key": {
    "id": "key-uuid",
    "key_prefix": "hc_live",
    "rate_limit_tier": "pro"
  },
  "secret_key": "hc_live_xxxxxxxxxxxxxxxxxxxxx"
}
```

**IMPORTANT**: Save the `secret_key` immediately - it's only shown once!

### User Management Operations

**Enable/Disable User:**
```bash
# Disable user (revokes all API access)
curl -X PATCH http://localhost:8000/admin/users/{user_id} \
  -H "X-Session-Token: your-session-token" \
  -H "Content-Type: application/json" \
  -d '{"is_active": false}'
```

**Change User Tier:**
```bash
curl -X PATCH http://localhost:8000/admin/users/{user_id} \
  -H "X-Session-Token: your-session-token" \
  -H "Content-Type: application/json" \
  -d '{"tier": "enterprise"}'
```

### Metadata Tracking

All users automatically get metadata tracked:

```json
{
  "id": "uuid",
  "email": "user@example.com",
  "tier": "pro",
  "is_active": true,
  "country": "US",
  "region": "CA",
  "city": "San Francisco",
  "timezone": "America/Los_Angeles",
  "signup_ip": "203.0.113.45",
  "last_login_ip": "203.0.113.46",
  "user_agent": "Mozilla/5.0...",
  "referrer": "https://google.com",
  "metadata": {
    "utm_source": "google",
    "utm_campaign": "november_promo"
  }
}
```

---

## API Reference

### Service Overview

| Service | Port | Purpose | URL |
|---------|------|---------|-----|
| **FastAPI** | 8000 | Core memory operations | `http://localhost:8000` |
| **Celery Tasks** | 8000 | Async task management | `http://localhost:8000/api/v1/tasks` |
| **Intelligence** | 8000 | Advanced AI features | `http://localhost:8000/v1/intelligence` |
| **Flower** | 5555 | Celery monitoring UI | `http://localhost:5555` |
| **Prometheus** | 9090 | Metrics collection | `http://localhost:9090` |
| **Grafana** | 3000 | Dashboards | `http://localhost:3000` |

### Core Memory Operations

**Store Memory:**
```bash
POST /v1/memories:remember

{
  "text": "I prefer oat milk in my coffee",
  "user_id": "alice",
  "session_id": "session_123",
  "type": "preference",
  "importance": 8.0,
  "tags": ["food", "beverages"],
  "ttl_days": 30
}
```

**Recall Memories:**
```bash
POST /v1/memories:recall

{
  "query": "What are my coffee preferences?",
  "user_id": "alice",
  "limit": 10
}
```

**Get All Memories:**
```bash
GET /v1/memories?user_id=alice&limit=100
```

**Update Memory:**
```bash
PATCH /v1/memories/{memory_id}

{
  "text": "Updated text",
  "importance": 9.0,
  "tags": ["updated"]
}
```

**Delete Memory:**
```bash
DELETE /v1/memories/{memory_id}
```

### Complete API Documentation

For complete API reference including all 102+ library methods and 56 REST endpoints, see:
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Library Complete Reference](LIBRARY_COMPLETE_REFERENCE.md) - All library methods
- Interactive Swagger UI: http://localhost:8000/docs

---

## Automation & Background Tasks

### Unified Library & SaaS Control

HippocampAI provides **unified control** where library users can programmatically configure and control SaaS automation features.

### Architecture

```
┌─────────────────────────────────────────┐
│         LIBRARY USER                     │
│      (Python Application)                │
│                                          │
│  automation.create_policy(policy)        │
│  automation.run_summarization()          │
└──────────────┬───────────────────────────┘
               │
               │ Configure
               ▼
    ┌──────────────────────────┐
    │  AutomationController    │
    │  (Unified Control Plane) │
    └──────┬────────────┬──────┘
           │            │
     ┌─────▼────┐  ┌───▼────────┐
     │ Library  │  │   SaaS     │
     │  Mode    │  │  Workers   │
     └──────────┘  └────────────┘
```

### Basic Setup

```python
from hippocampai import (
    MemoryClient,
    AutomationController,
    AutomationPolicy,
    PolicyType
)
from hippocampai.adapters import GroqLLM
from hippocampai.embed.embedder import Embedder

# Initialize
llm = GroqLLM(api_key="your-api-key")
client = MemoryClient(llm_provider=llm)
embedder = Embedder()

automation = AutomationController(
    memory_service=client.memory_service,
    llm=llm,
    embedder=embedder
)
```

### Create Automation Policy

```python
policy = AutomationPolicy(
    user_id="user123",
    policy_type=PolicyType.THRESHOLD,

    # Enable features
    auto_summarization=True,
    auto_consolidation=True,
    health_monitoring=True,

    # Configure thresholds
    summarization_threshold=500,  # Run when >500 memories
    consolidation_threshold=300,

    # Configure settings
    summarization_age_days=30,
    consolidation_similarity=0.85,
    decay_half_life_days=90
)

# Register policy
automation.create_policy(policy)
```

### Execution Modes

**1. Library Mode (Immediate):**
```python
# Run immediately
result = automation.run_summarization("user123")
print(f"Summaries created: {result['summaries_created']}")
```

**2. SaaS Mode (Background):**
```python
from hippocampai.saas import TaskManager

task_manager = TaskManager()
task = task_manager.submit_task("user123", "summarization")
# Queued for background execution
```

**3. Hybrid (Both):**
```python
# Quick ops: immediate
health = automation.run_health_check("user123")

# Heavy ops: background
task_manager.submit_task("user123", "summarization")
```

### Scheduled Background Tasks

```python
# Policy with schedules (for SaaS cron)
policy = AutomationPolicy(
    user_id="user123",
    auto_summarization=True,
    summarization_schedule=AutomationSchedule(
        enabled=True,
        cron_expression="0 2 * * *",  # 2 AM daily
    )
)

automation.create_policy(policy)
# Celery beat will schedule automatically
```

### Celery Deployment

**Create celeryconfig.py:**
```python
broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/0'

beat_schedule = {
    'run-optimizations': {
        'task': 'tasks.run_scheduled_tasks',
        'schedule': crontab(hour=2, minute=0),  # 2 AM daily
    }
}
```

**Create celery_tasks.py:**
```python
from celery import Celery
from hippocampai import AutomationController

app = Celery('tasks')
app.config_from_object('celeryconfig')

automation = AutomationController(...)

@app.task
def run_scheduled_tasks():
    """Run for all users with policies."""
    for user_id in automation.policies.keys():
        policy = automation.get_policy(user_id)
        if policy.enabled and automation.should_run_summarization(user_id):
            automation.run_summarization(user_id, force=True)
```

**Start Workers:**
```bash
celery -A celery_tasks worker --beat --loglevel=info
```

---

## Monitoring & Observability

### Flower UI

Access Celery task monitoring at http://localhost:5555

**Features:**
- Active tasks per worker
- Task success/failure rates
- Queue lengths
- Worker resource usage
- Task history & logs

### Prometheus Metrics

Access metrics at http://localhost:9090

**Available Metrics:**
- Request rates
- Response times
- Error rates
- Memory operations
- Task queue lengths

### Grafana Dashboards

Access dashboards at http://localhost:3000

**Default Credentials:**
- Username: `admin`
- Password: `admin`

**Pre-configured Dashboards:**
- System Overview
- API Performance
- Celery Tasks
- Memory Operations
- User Analytics

### Admin Dashboard

Access at http://localhost:3001

**Features:**
- User management
- API key management
- Usage statistics
- Real-time metrics

---

## Rate Limiting & Tiers

### Rate Limit Details

| Tier       | Per Minute | Per Hour  | Per Day     |
|------------|------------|-----------|-------------|
| Free       | 10         | 100       | 1,000       |
| Pro        | 100        | 10,000    | 100,000     |
| Enterprise | 1,000      | 100,000   | 1,000,000   |
| Admin      | 10,000     | 1,000,000 | 10,000,000  |

### Rate Limit Headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1699564800
```

### Rate Limit Exceeded (429)

```json
{
  "error": "Rate limit exceeded",
  "message": "Too many requests. Please try again later.",
  "retry_after": 60
}
```

**Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1699564860
Retry-After: 60
```

---

## Security Best Practices

### 1. Change Default Credentials

```bash
# Update admin password immediately
# Use strong passwords (min 8 characters)
```

### 2. API Key Management

- Store API keys securely (environment variables, secrets manager)
- Never commit keys to version control
- Rotate keys regularly
- Use key expiration for temporary access

### 3. Rate Limiting

- Set appropriate tier limits for your use case
- Monitor usage to detect abuse
- Use Redis persistence to prevent limit resets

### 4. Database Security

- Change PostgreSQL credentials in production
- Enable SSL for database connections
- Regular backups of postgres_data volume

### 5. Network Security

- Use HTTPS in production (reverse proxy with TLS)
- Restrict database and Redis ports to internal network
- Enable CORS only for trusted origins

### Production Deployment

**Enable Production Mode:**
```bash
# .env file
USER_AUTH_ENABLED=true
POSTGRES_PASSWORD=<strong-password>
```

**Deploy with TLS:**
- Use nginx or Caddy as reverse proxy
- Get SSL certificate (Let's Encrypt)
- Redirect HTTP to HTTPS

---

## Troubleshooting

### Authentication Issues

**401 Unauthorized:**
- Check API key is correct
- Verify key hasn't expired or been revoked
- Ensure `Authorization: Bearer <key>` header is set

**USER_AUTH_ENABLED=false but still getting 401:**
```bash
# Check environment variable
docker exec hippocampai-api env | grep USER_AUTH_ENABLED

# Restart container after changing
docker-compose restart hippocampai
```

### Rate Limiting

**429 Rate Limit Exceeded:**
- Check current tier limits
- Wait for rate limit window to reset
- Consider upgrading tier
- Use exponential backoff for retries

**Rate limits not being enforced:**
```bash
# Check middleware is loaded
docker logs hippocampai-api 2>&1 | grep -i "authmiddleware\|ratelimiter"

# Verify USER_AUTH_ENABLED=true
docker exec hippocampai-api env | grep USER_AUTH
```

### Admin UI Not Loading

```bash
# Check if admin service is running
docker-compose ps admin

# Verify port 3001 is not in use
# Check nginx logs
docker-compose logs admin
```

### PostgreSQL Connection Errors

```bash
# Wait for PostgreSQL healthcheck to pass
# Check connection string in environment variables

# Verify schema was initialized
./scripts/verify_postgres_docker.sh

# View logs
docker-compose logs postgres

# If schema failed to initialize, recreate database
docker-compose stop postgres
docker-compose rm -f postgres
docker volume rm hippocampai_postgres_data
docker-compose up -d postgres
```

### API Key Not Working

```sql
-- Verify key is active
SELECT * FROM api_keys WHERE key_prefix = 'hc_live';

-- Check user is active
SELECT u.is_active, u.email
FROM users u
JOIN api_keys k ON u.id = k.user_id
WHERE k.key_prefix = 'hc_live';

-- Check key hasn't expired
SELECT expires_at FROM api_keys
WHERE key_prefix = 'hc_live' AND (expires_at IS NULL OR expires_at > NOW());
```

---

## Switching Between Modes

### From Local to SaaS

```bash
# 1. Update environment
export USER_AUTH_ENABLED=true

# 2. Restart services
docker-compose restart hippocampai

# 3. Verify middleware is active
docker logs hippocampai-api | grep AuthMiddleware
# Should see: "AuthMiddleware configured (enabled=True)"

# 4. Test authentication is required
curl http://localhost:8000/v1/memories:remember \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "text": "test"}'
# This should fail with 401
```

### From SaaS to Local

```bash
# 1. Update environment
export USER_AUTH_ENABLED=false

# 2. Restart services
docker-compose restart hippocampai

# 3. Verify middleware is in bypass mode
docker logs hippocampai-api | grep AuthMiddleware
# Should see: "AuthMiddleware configured (enabled=False)"

# 4. Test authentication is not required
curl http://localhost:8000/v1/memories:remember \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "text": "test", "type": "fact"}'
# This should succeed without API key
```

---

## Utility Scripts

### Verify Schema (Docker - Recommended)

Check if the PostgreSQL schema is properly initialized using docker exec:

```bash
./scripts/verify_postgres_docker.sh
```

### Verify Schema (Local psql)

If you have PostgreSQL client installed locally:

```bash
./scripts/verify_postgres.sh
```

This will check:
- Database connection
- All required tables exist
- Views and functions created
- Default admin user exists
- Database statistics

### Initialize/Reset Schema

Manually initialize or reset the PostgreSQL schema:

```bash
./scripts/init_postgres.sh
```

This will:
- Connect to PostgreSQL
- Check for existing tables
- Ask for confirmation before dropping (if tables exist)
- Run the full schema initialization
- Create default admin user
- Verify everything was created

---

## Summary

| Feature | Local Mode | SaaS Mode |
|---------|------------|-----------|
| Authentication | ❌ Not required | ✅ Required |
| API Keys | ❌ Not needed | ✅ Mandatory |
| Rate Limiting | ❌ Disabled | ✅ Enforced |
| User Management | ❌ No users | ✅ Full user system |
| Metadata Tracking | ❌ No tracking | ✅ Full tracking |
| Admin Dashboard | ⚠️ Direct access | ✅ Login required |
| Multi-Tenancy | ❌ Single tenant | ✅ Multi-tenant |
| Usage Analytics | ❌ None | ✅ Per-user stats |
| Cost | 🆓 Free | 💰 Tiered pricing |

**Recommendation:**
- Use **Local Mode** for development, self-hosted, internal usage
- Use **SaaS Mode** for production, multi-user, commercial deployments

Both modes use the same codebase and can be switched with a single environment variable!

---

## Next Steps

1. **Integrate into FastAPI App** - See [Integration Guide](SAAS_INTEGRATION_GUIDE.md)
2. **Set Up Monitoring** - Configure Prometheus and Grafana
3. **Enable Production Mode** - Set USER_AUTH_ENABLED=true
4. **Deploy with TLS** - Use nginx or Caddy as reverse proxy

---

## Related Documentation

- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Library Complete Reference](LIBRARY_COMPLETE_REFERENCE.md) - All 102+ library methods
- [Celery Guide](CELERY_GUIDE.md) - Background task processing
- [Monitoring Guide](MONITORING.md) - Observability and metrics
- [Security Guide](SECURITY.md) - Security best practices

---

**Built with ❤️ by the HippocampAI community**
