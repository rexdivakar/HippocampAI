# HippocampAI SaaS Modes Guide

## Overview

HippocampAI supports two operational modes:
1. **Local Mode** (`USER_AUTH_ENABLED=false`) - No authentication, no rate limits
2. **SaaS Mode** (`USER_AUTH_ENABLED=true`) - Full authentication, rate limiting, and user management

## Mode 1: Local Mode (Development/Self-Hosted)

### Configuration

**Environment Variable**:
```bash
USER_AUTH_ENABLED=false  # or not set (defaults to false)
```

**Docker Compose**:
```yaml
hippocampai:
  environment:
    USER_AUTH_ENABLED: "false"
```

### Behavior

✅ **No Authentication Required**
- API works without API keys
- No login needed
- All endpoints publicly accessible

✅ **No Rate Limiting**
- Unlimited requests
- No throttling
- Full admin-level access

✅ **Client Library Usage**:
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

results = client.search_memories(
    user_id="user123",
    query="What is the capital of France?",
    k=5
)
```

✅ **API Requests**:
```bash
# No headers needed
curl -X POST http://localhost:8000/v1/memories:remember \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "text": "Paris is the capital of France",
    "type": "fact"
  }'

# Or with X-User-Auth: false header for clarity
curl -X POST http://localhost:8000/v1/memories:remember \
  -H "Content-Type: application/json" \
  -H "X-User-Auth: false" \
  -d '{
    "user_id": "user123",
    "text": "Paris is the capital of France",
    "type": "fact"
  }'
```

✅ **Admin Dashboard**:
- Accessible without login when using `X-User-Auth: false` header
- Use for managing users/keys even in local mode

### Use Cases

- Local development
- Self-hosted deployments
- Internal team usage
- Testing without authentication overhead

---

## Mode 2: SaaS Mode (Production/Multi-Tenant)

### Configuration

**Environment Variable**:
```bash
USER_AUTH_ENABLED=true
```

**Docker Compose**:
```yaml
hippocampai:
  environment:
    USER_AUTH_ENABLED: "true"
    POSTGRES_HOST: postgres
    POSTGRES_PORT: 5432
    POSTGRES_DB: hippocampai
    POSTGRES_USER: hippocampai
    POSTGRES_PASSWORD: your_secure_password
```

### Behavior

🔒 **Authentication Required**
- All API requests require valid API key
- 401 Unauthorized without proper authentication
- Admin endpoints require admin session

🚦 **Rate Limiting Enforced**
- Tier-based request limits
- 429 Too Many Requests when exceeded
- Rate limit info in response headers

📊 **User Management**
- Multi-user support
- Usage tracking per user
- Metadata collection (IP, location, etc.)

### Setup Steps

#### 1. Start Services

```bash
docker-compose up -d
```

#### 2. Verify PostgreSQL Schema

```bash
./scripts/verify_postgres_docker.sh
```

#### 3. Create Users (Admin Dashboard)

**Login to Admin Dashboard**:
```bash
curl -X POST http://localhost:8000/admin/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@hippocampai.com",
    "password": "admin123"
  }'
```

**Response**:
```json
{
  "user": {...},
  "session_token": "your-session-token",
  "expires_in": 604800
}
```

**Create User**:
```bash
curl -X POST http://localhost:8000/admin/users \
  -H "Content-Type: application/json" \
  -H "X-Session-Token: your-session-token" \
  -d '{
    "email": "user@example.com",
    "password": "userpassword123",
    "full_name": "John Doe",
    "tier": "pro"
  }'
```

#### 4. Generate API Key

```bash
curl -X POST http://localhost:8000/admin/users/USER_UUID/api-keys \
  -H "Content-Type: application/json" \
  -H "X-Session-Token: your-session-token" \
  -d '{
    "name": "Production Key",
    "rate_limit_tier": "pro",
    "expires_in_days": 365
  }'
```

**Response**:
```json
{
  "api_key": {
    "id": "key-uuid",
    "key_prefix": "hc_live",
    "rate_limit_tier": "pro",
    ...
  },
  "secret_key": "hc_live_xxxxxxxxxxxxxxxxxxxxx"
}
```

⚠️ **IMPORTANT**: Save the `secret_key` immediately - it's only shown once!

### Client Library Usage (SaaS Mode)

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

# Rate limit info available in responses
results = client.search_memories(
    user_id="user123",
    query="capital of France",
    k=5
)
```

### API Requests (SaaS Mode)

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

**Response Headers Include**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1699564800
```

### Rate Limiting Details

| Tier       | Per Minute | Per Hour  | Per Day     |
|------------|------------|-----------|-------------|
| Free       | 10         | 100       | 1,000       |
| Pro        | 100        | 10,000    | 100,000     |
| Enterprise | 1,000      | 100,000   | 1,000,000   |
| Admin      | 10,000     | 1,000,000 | 10,000,000  |

**Rate Limit Exceeded (429)**:
```json
{
  "error": "Rate limit exceeded",
  "message": "Too many requests. Please try again later.",
  "retry_after": 60
}
```

**Headers**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1699564860
Retry-After: 60
```

### User Management

#### Enable/Disable Users

**Disable user** (revokes all API access):
```bash
curl -X PATCH http://localhost:8000/admin/users/USER_UUID \
  -H "Content-Type: application/json" \
  -H "X-Session-Token: your-session-token" \
  -d '{
    "is_active": false
  }'
```

**Enable user**:
```bash
curl -X PATCH http://localhost:8000/admin/users/USER_UUID \
  -H "Content-Type: application/json" \
  -H "X-Session-Token: your-session-token" \
  -d '{
    "is_active": true
  }'
```

#### Change User Tier

```bash
curl -X PATCH http://localhost:8000/admin/users/USER_UUID \
  -H "Content-Type: application/json" \
  -H "X-Session-Token: your-session-token" \
  -d '{
    "tier": "enterprise"
  }'
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

## Switching Between Modes

### From Local to SaaS

1. Update environment:
   ```bash
   export USER_AUTH_ENABLED=true
   ```

2. Restart services:
   ```bash
   docker-compose restart hippocampai
   ```

3. Verify middleware is active:
   ```bash
   docker logs hippocampai-api | grep AuthMiddleware
   # Should see: "AuthMiddleware configured (enabled=True)"
   ```

4. Test authentication is required:
   ```bash
   # This should fail with 401
   curl http://localhost:8000/v1/memories:remember \
     -H "Content-Type: application/json" \
     -d '{"user_id": "test", "text": "test"}'
   ```

### From SaaS to Local

1. Update environment:
   ```bash
   export USER_AUTH_ENABLED=false
   ```

2. Restart services:
   ```bash
   docker-compose restart hippocampai
   ```

3. Verify middleware is in bypass mode:
   ```bash
   docker logs hippocampai-api | grep AuthMiddleware
   # Should see: "AuthMiddleware configured (enabled=False)"
   ```

4. Test authentication is not required:
   ```bash
   # This should succeed without API key
   curl http://localhost:8000/v1/memories:remember \
     -H "Content-Type: application/json" \
     -d '{"user_id": "test", "text": "test", "type": "fact"}'
   ```

---

## Testing Both Modes

### Test Script: Local Mode

```python
from hippocampai import MemoryClient

# Local mode - no auth
client = MemoryClient(user_auth=False)

# Should work without API key
memory = client.add_memory(
    user_id="test_user",
    text="Testing local mode",
    memory_type="fact"
)
print(f"✅ Local mode works: {memory.id}")

results = client.search_memories(
    user_id="test_user",
    query="testing",
    k=5
)
print(f"✅ Search works: {len(results)} results")
```

### Test Script: SaaS Mode

```python
from hippocampai import MemoryClient
import os

# SaaS mode - requires auth
os.environ["HIPPOCAMPAI_USER_AUTH"] = "true"
os.environ["HIPPOCAMPAI_API_KEY"] = "hc_live_your_key_here"

client = MemoryClient()

try:
    memory = client.add_memory(
        user_id="test_user",
        text="Testing SaaS mode",
        memory_type="fact"
    )
    print(f"✅ SaaS mode works: {memory.id}")
except Exception as e:
    if "401" in str(e) or "Unauthorized" in str(e):
        print("❌ Authentication failed - check API key")
    elif "429" in str(e):
        print("⚠️ Rate limit exceeded")
    else:
        raise
```

---

## Troubleshooting

### "USER_AUTH_ENABLED=false but still getting 401"

Check the actual environment variable:
```bash
docker exec hippocampai-api env | grep USER_AUTH_ENABLED
```

Restart container after changing:
```bash
docker-compose restart hippocampai
```

### "Rate limits not being enforced"

Check middleware is loaded:
```bash
docker logs hippocampai-api 2>&1 | grep -i "authmiddleware\|ratelimiter"
```

Verify USER_AUTH_ENABLED=true:
```bash
docker exec hippocampai-api env | grep USER_AUTH
```

### "API key not working"

1. Verify key is active:
   ```sql
   SELECT * FROM api_keys WHERE key_prefix = 'hc_live';
   ```

2. Check user is active:
   ```sql
   SELECT u.is_active, u.email
   FROM users u
   JOIN api_keys k ON u.id = k.user_id
   WHERE k.key_prefix = 'hc_live';
   ```

3. Check key hasn't expired:
   ```sql
   SELECT expires_at FROM api_keys
   WHERE key_prefix = 'hc_live' AND (expires_at IS NULL OR expires_at > NOW());
   ```

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

**Recommendation**:
- Use **Local Mode** for development, self-hosted, internal usage
- Use **SaaS Mode** for production, multi-user, commercial deployments

Both modes use the same codebase and can be switched with a single environment variable!
