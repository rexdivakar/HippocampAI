# HippocampAI SaaS Quickstart Guide

Complete SaaS authentication infrastructure has been implemented! This guide shows you how to get started.

## 🚀 Quick Start

### 1. Start the Infrastructure

```bash
# Start all services including PostgreSQL and Admin UI
docker-compose up -d

# Check service status
docker-compose ps

# Verify PostgreSQL schema was initialized (uses docker exec - no local psql needed)
./scripts/verify_postgres_docker.sh

# Alternative: If you have psql installed locally
# ./scripts/verify_postgres.sh
```

**Note**: The PostgreSQL schema is automatically initialized on first container start. The schema file (`src/hippocampai/auth/schema.sql`) is mounted to `/docker-entrypoint-initdb.d/` which PostgreSQL executes automatically.

Services will be available at:
- **API**: http://localhost:8000
- **Admin Dashboard**: http://localhost:3001
- **Flower (Celery)**: http://localhost:5555
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090

### 2. Access Admin Dashboard

Open http://localhost:3001 in your browser.

**Default Admin Credentials:**
- Email: `admin@hippocampai.com`
- Password: `admin123`

⚠️ **IMPORTANT**: Change the default admin password in production!

### 3. Create Users and API Keys

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

## 📚 Using the Client Library

### Local Mode (No Authentication)

```python
from hippocampai import MemoryClient

# Local mode - bypasses authentication
client = MemoryClient(
    user_auth=False  # Default: false
)

# All operations work without API key
client.add_memory(
    user_id="user123",
    text="Paris is the capital of France",
    memory_type="fact"
)
```

### Remote Mode (With Authentication)

```python
from hippocampai import MemoryClient

# Remote mode - requires API key
client = MemoryClient(
    user_auth=True,
    api_key="hc_live_your_api_key_here"
)

# Or use environment variable
import os
os.environ["HIPPOCAMPAI_API_KEY"] = "hc_live_your_api_key_here"
os.environ["HIPPOCAMPAI_USER_AUTH"] = "true"

client = MemoryClient()  # Reads from environment

# All operations use authenticated requests
client.add_memory(
    user_id="user123",
    text="Paris is the capital of France",
    memory_type="fact"
)
```

## 🔒 Rate Limits by Tier

| Tier       | Per Minute | Per Hour | Per Day   |
|------------|------------|----------|-----------|
| Free       | 10         | 100      | 1,000     |
| Pro        | 100        | 10,000   | 100,000   |
| Enterprise | 1,000      | 100,000  | 1,000,000 |
| Admin      | 10,000     | 1,000,000| 10,000,000|

Rate limit info is returned in response headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1635724800
```

When rate limit is exceeded, you'll receive a 429 status code with retry-after header.

## 🔧 Configuration

### Environment Variables

```bash
# Authentication
USER_AUTH_ENABLED=false          # Set to 'true' for production
HIPPOCAMPAI_API_KEY=hc_live_... # Your API key

# PostgreSQL
POSTGRES_DB=hippocampai
POSTGRES_USER=hippocampai
POSTGRES_PASSWORD=hippocampai_secret
POSTGRES_PORT=5432

# Admin UI
ADMIN_PORT=3001
```

### Docker Compose

The `docker-compose.yml` includes:
- ✅ PostgreSQL database with schema auto-initialization
- ✅ Admin UI (nginx serving static HTML)
- ✅ Redis for rate limiting and caching
- ✅ Qdrant for vector storage
- ✅ HippocampAI API with authentication middleware

## 📊 Admin Dashboard Features

### Users Tab
- Create new users
- View all users with tier and status
- Update user tier and permissions
- Delete users (cascades to API keys)

### API Keys Tab
- View all API keys across users
- Filter by user
- Create new API keys with custom expiration
- Revoke or delete keys
- See last usage timestamps

### Usage Statistics Tab
- Total users, API keys, and requests
- Per-user statistics (requests, tokens, last activity)
- Real-time metrics

## 🏗️ Architecture

```
┌─────────────────┐
│  Client Library │ ← user_auth + api_key
└────────┬────────┘
         │
         v
┌─────────────────┐
│ AuthMiddleware  │ ← Validates API key, checks rate limits
└────────┬────────┘
         │
         v
┌─────────────────┐
│   FastAPI API   │ ← Your application logic
└────────┬────────┘
         │
    ┌────┴────┬─────────┬─────────┐
    v         v         v         v
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│ PG  │  │Redis│  │Qdrant│  │Other│
└─────┘  └─────┘  └─────┘  └─────┘
```

### Components Implemented

1. **src/hippocampai/auth/schema.sql** - PostgreSQL database schema
2. **src/hippocampai/auth/models.py** - Pydantic models
3. **src/hippocampai/auth/auth_service.py** - User and API key management
4. **src/hippocampai/auth/rate_limiter.py** - Redis-based rate limiting
5. **src/hippocampai/api/middleware.py** - FastAPI authentication middleware
6. **src/hippocampai/api/admin_routes.py** - Admin API endpoints
7. **admin_ui/index.html** - Admin dashboard (no dependencies!)
8. **src/hippocampai/client.py** - Updated with user_auth support

## 🔐 Security Best Practices

1. **Change Default Credentials**
   - Update admin password immediately
   - Use strong passwords (min 8 characters)

2. **API Key Management**
   - Store API keys securely (environment variables, secrets manager)
   - Never commit keys to version control
   - Rotate keys regularly
   - Use key expiration for temporary access

3. **Rate Limiting**
   - Set appropriate tier limits for your use case
   - Monitor usage to detect abuse
   - Use Redis persistence to prevent limit resets

4. **Database Security**
   - Change PostgreSQL credentials in production
   - Enable SSL for database connections
   - Regular backups of postgres_data volume

5. **Network Security**
   - Use HTTPS in production (reverse proxy with TLS)
   - Restrict database and Redis ports to internal network
   - Enable CORS only for trusted origins

## 📝 Example Workflows

### Workflow 1: New User Onboarding

1. Create user via admin UI or API
2. Generate API key with appropriate tier
3. Send key to user securely (one-time link, encrypted email)
4. User configures client library with API key
5. User starts making requests (rate limited by tier)

### Workflow 2: Upgrading User Tier

1. Admin updates user tier in dashboard
2. Create new API key with higher tier
3. Revoke old API key
4. User gets higher rate limits immediately

### Workflow 3: Monitoring Usage

1. Check "Usage Statistics" tab for overview
2. View per-user request counts and token usage
3. Identify heavy users or potential abuse
4. Adjust tier or rate limits as needed

## 🚨 Troubleshooting

### 401 Unauthorized
- Check API key is correct
- Verify key hasn't expired or been revoked
- Ensure `Authorization: Bearer <key>` header is set

### 429 Rate Limit Exceeded
- Check current tier limits
- Wait for rate limit window to reset (see X-RateLimit-Reset header)
- Consider upgrading tier
- Use exponential backoff for retries

### Admin UI Not Loading
- Check if admin service is running: `docker-compose ps admin`
- Verify port 3001 is not in use by another service
- Check nginx logs: `docker-compose logs admin`

### PostgreSQL Connection Errors
- Wait for PostgreSQL healthcheck to pass
- Check connection string in environment variables
- Verify schema was initialized: `./scripts/verify_postgres_docker.sh`
- View logs: `docker-compose logs postgres`
- If schema failed to initialize, recreate database:
  ```bash
  docker-compose stop postgres
  docker-compose rm -f postgres
  docker volume rm hippocampai_postgres_data
  docker-compose up -d postgres
  ```

## 🛠️ Utility Scripts

Three helper scripts are provided to manage the PostgreSQL database:

### Verify Schema (Docker - Recommended)

Check if the PostgreSQL schema is properly initialized using docker exec (no local psql needed):

```bash
./scripts/verify_postgres_docker.sh
```

### Verify Schema (Local psql)

If you have PostgreSQL client installed locally:

```bash
./scripts/verify_postgres.sh
```

This will check:
- ✓ Database connection
- ✓ All required tables exist
- ✓ Views and functions created
- ✓ Default admin user exists
- ✓ Database statistics

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

**When to use:**
- First time setup (though docker-compose does this automatically)
- After database corruption
- To reset to clean state for testing
- To upgrade schema after changes

## 📚 Next Steps

1. **Integrate into FastAPI App**
   ```python
   from hippocampai.api.middleware import AuthMiddleware
   from hippocampai.auth.auth_service import AuthService
   from hippocampai.auth.rate_limiter import RateLimiter

   # In your FastAPI app setup
   app.state.auth_service = AuthService(db_pool)
   app.state.rate_limiter = RateLimiter(redis)

   app.add_middleware(
       AuthMiddleware,
       auth_service=auth_service,
       rate_limiter=rate_limiter,
       user_auth_enabled=True
   )

   # Include admin routes
   from hippocampai.api.admin_routes import router
   app.include_router(router)
   ```

2. **Set Up Monitoring**
   - Configure Prometheus to scrape API metrics
   - Import Grafana dashboards for visualization
   - Set up alerts for rate limit violations

3. **Enable Production Mode**
   ```bash
   # .env file
   USER_AUTH_ENABLED=true
   POSTGRES_PASSWORD=<strong-password>
   ```

4. **Deploy with TLS**
   - Use nginx or Caddy as reverse proxy
   - Get SSL certificate (Let's Encrypt)
   - Redirect HTTP to HTTPS

## 🎉 You're Ready!

Your HippocampAI SaaS infrastructure is now fully set up with:
- ✅ User authentication
- ✅ API key management
- ✅ Multi-tier rate limiting
- ✅ Admin dashboard
- ✅ Usage tracking
- ✅ Local & remote modes

For detailed API documentation, see `SAAS_AUTH_IMPLEMENTATION_GUIDE.md`.
