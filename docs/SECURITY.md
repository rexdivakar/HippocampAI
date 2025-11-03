# Security Best Practices for HippocampAI

This guide covers security best practices for deploying and operating HippocampAI in production environments.

## Table of Contents

- [Security Overview](#security-overview)
- [Authentication & Authorization](#authentication--authorization)
- [Secrets Management](#secrets-management)
- [Network Security](#network-security)
- [Data Protection](#data-protection)
- [API Security](#api-security)
- [Container Security](#container-security)
- [Multi-Tenant Security](#multi-tenant-security)
- [Audit & Compliance](#audit--compliance)
- [Security Checklist](#security-checklist)
- [Incident Response](#incident-response)

---

## Security Overview

### Threat Model

HippocampAI handles sensitive data including:
- User conversations and personal information
- Behavioral patterns and preferences
- Organizational knowledge
- API keys and credentials

**Key Security Objectives:**
1. **Confidentiality**: Protect sensitive memory data from unauthorized access
2. **Integrity**: Ensure memories are not tampered with
3. **Availability**: Maintain service availability and resilience
4. **Auditability**: Track all access and modifications

---

## Authentication & Authorization

### API Authentication

**⚠️ NOT CURRENTLY IMPLEMENTED (v0.2.5):**

Authentication and authorization are **NOT currently implemented** in HippocampAI. The configuration options mentioned below are **placeholders for future implementation**.

**For production deployments TODAY**, you must implement authentication at the infrastructure level using:
- Reverse proxy (Nginx, Traefik) with authentication
- API Gateway with built-in auth
- Network-level security (VPN, private networks)

**Planned for future releases:** The approaches below are planned but not yet implemented:

#### Option 1: API Key Authentication (PLANNED - Not Implemented)

**⚠️ This is a placeholder - not currently functional**

1. **Placeholder configuration (not functional):**

```bash
# .env (commented out - not implemented)
# ENABLE_AUTH=true
# API_KEY_HEADER=X-API-Key
```

2. **Generate secure API keys:**

```python
import secrets

# Generate a cryptographically secure API key
api_key = secrets.token_urlsafe(32)
print(f"API Key: {api_key}")
```

3. **Store API keys securely:**

```bash
# Use environment variables or secrets manager
export HIPPOCAMPAI_API_KEY="your-generated-key"
```

4. **Implement in your application:**

```python
import os
import requests

api_key = os.environ.get("HIPPOCAMPAI_API_KEY")

headers = {
    "X-API-Key": api_key,
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:8000/api/memories",
    headers=headers,
    json={"text": "memory", "user_id": "alice"}
)
```

---

#### Option 2: JWT Authentication (PLANNED - Not Implemented)

**⚠️ This is a placeholder - not currently functional**

1. **Placeholder configuration (not functional):**

```bash
# .env (commented out - not implemented)
# ENABLE_AUTH=true
# AUTH_TYPE=jwt
# JWT_SECRET=your-super-secret-jwt-key-min-32-chars
# JWT_ALGORITHM=HS256
# JWT_EXPIRATION_HOURS=24
```

2. **Generate secure JWT secret:**

```bash
openssl rand -hex 32
```

3. **Implement token generation (your auth service):**

```python
import jwt
from datetime import datetime, timedelta

def generate_token(user_id: str, secret: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=24),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, secret, algorithm="HS256")
```

4. **Use tokens in requests:**

```python
token = generate_token("alice", jwt_secret)

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:8000/api/memories",
    headers=headers,
    json={"text": "memory", "user_id": "alice"}
)
```

---

#### Option 3: OAuth 2.0 / OpenID Connect (PLANNED - Not Implemented)

**⚠️ This is a placeholder - not currently functional**

For enterprise deployments with existing identity providers (planned):

```bash
# .env (not implemented)
# ENABLE_AUTH=true
# AUTH_TYPE=oauth2
# OAUTH_PROVIDER=auth0  # or okta, keycloak, etc.
# OAUTH_CLIENT_ID=your-client-id
# OAUTH_CLIENT_SECRET=your-client-secret
# OAUTH_ISSUER=https://your-domain.auth0.com/
```

**Integration with Auth0 example:**

```python
from authlib.integrations.requests_client import OAuth2Session

session = OAuth2Session(
    client_id="your-client-id",
    client_secret="your-client-secret",
    token_endpoint="https://your-domain.auth0.com/oauth/token"
)

token = session.fetch_token()

headers = {
    "Authorization": f"Bearer {token['access_token']}",
    "Content-Type": "application/json"
}
```

---

### User Authorization

Implement role-based access control (RBAC):

```python
# Define user roles
class Role:
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"

# In your JWT payload
payload = {
    "user_id": "alice",
    "role": Role.USER,
    "permissions": ["read", "write"],
    "exp": datetime.utcnow() + timedelta(hours=24)
}
```

**Permission Matrix:**

| Role | Read Memories | Write Memories | Delete Memories | Admin Operations |
|------|---------------|----------------|-----------------|------------------|
| Admin | ✅ | ✅ | ✅ | ✅ |
| User | ✅ (own) | ✅ (own) | ✅ (own) | ❌ |
| Readonly | ✅ (own) | ❌ | ❌ | ❌ |

---

### Agent Authorization

For multi-agent systems, implement agent-level permissions:

```python
# Register agent with permissions
client.register_agent(
    agent_id="agent_001",
    name="Customer Support Bot",
    metadata={
        "permissions": ["read", "write"],
        "access_scope": "customer_data",
        "max_memory_limit": 10000
    }
)

# Check permissions before operations
if client.check_agent_permission(agent_id, "write"):
    client.remember(text, user_id, agent_id=agent_id)
```

---

## Secrets Management

### Environment Variables

**❌ Never commit secrets to version control:**

```bash
# Add to .gitignore
.env
.env.local
.env.production
secrets/
*.pem
*.key
```

---

### Secrets Managers (Production Recommendation)

#### AWS Secrets Manager

```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager', region_name='us-west-2')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Load secrets
secrets = get_secret('hippocampai/prod/credentials')
os.environ['OPENAI_API_KEY'] = secrets['openai_api_key']
os.environ['JWT_SECRET'] = secrets['jwt_secret']
```

---

#### HashiCorp Vault

```python
import hvac

client = hvac.Client(url='http://vault:8200')
client.token = os.environ['VAULT_TOKEN']

# Read secrets
secret = client.secrets.kv.v2.read_secret_version(
    path='hippocampai/config'
)

os.environ['OPENAI_API_KEY'] = secret['data']['data']['openai_api_key']
```

---

#### Kubernetes Secrets

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: hippocampai-secrets
type: Opaque
data:
  openai-api-key: <base64-encoded-key>
  jwt-secret: <base64-encoded-secret>
```

```yaml
# deployment.yaml
env:
  - name: OPENAI_API_KEY
    valueFrom:
      secretKeyRef:
        name: hippocampai-secrets
        key: openai-api-key
```

---

### API Key Rotation

Implement regular key rotation:

```bash
# Rotate API keys every 90 days
# 1. Generate new key
NEW_KEY=$(openssl rand -hex 32)

# 2. Add new key (support both keys during transition)
export HIPPOCAMPAI_API_KEY_PRIMARY=$NEW_KEY
export HIPPOCAMPAI_API_KEY_SECONDARY=$OLD_KEY

# 3. Update clients to use new key
# 4. After transition period, remove old key
unset HIPPOCAMPAI_API_KEY_SECONDARY
```

---

## Network Security

### TLS/SSL Configuration

**Always use HTTPS in production:**

#### Option 1: Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/hippocampai
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

---

#### Option 2: Traefik (Docker)

```yaml
# docker-compose.yml
version: '3.8'

services:
  traefik:
    image: traefik:v2.10
    command:
      - "--providers.docker=true"
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.letsencrypt.acme.email=admin@yourdomain.com"
      - "--certificatesresolvers.letsencrypt.acme.storage=/letsencrypt/acme.json"
      - "--certificatesresolvers.letsencrypt.acme.httpchallenge.entrypoint=web"
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./letsencrypt:/letsencrypt

  hippocampai:
    image: hippocampai:latest
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.hippocampai.rule=Host(`api.yourdomain.com`)"
      - "traefik.http.routers.hippocampai.entrypoints=websecure"
      - "traefik.http.routers.hippocampai.tls.certresolver=letsencrypt"
```

---

### Firewall Configuration

**Allow only necessary ports:**

```bash
# UFW (Ubuntu)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 443/tcp  # HTTPS
sudo ufw allow 22/tcp   # SSH (restrict to specific IPs)
sudo ufw enable

# Restrict Qdrant and Redis to localhost only
sudo ufw deny 6333  # Qdrant
sudo ufw deny 6379  # Redis
```

---

### Network Isolation

**Docker network isolation:**

```yaml
# docker-compose.yml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No external access

services:
  hippocampai:
    networks:
      - frontend
      - backend

  qdrant:
    networks:
      - backend  # Only accessible from backend network

  redis:
    networks:
      - backend  # Only accessible from backend network
```

---

### Rate Limiting

**⚠️ NOT CURRENTLY IMPLEMENTED (v0.2.5):**

Rate limiting is **NOT currently implemented** at the application level. For production, implement rate limiting using:
- Nginx (shown below)
- API Gateway
- Cloud provider WAF/DDoS protection

**Planned for future releases:** Application-level rate limiting

Protect against DDoS and abuse with infrastructure-level solutions:

#### Nginx Rate Limiting

```nginx
# Define rate limit zone
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

server {
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;
        proxy_pass http://localhost:8000;
    }
}
```

---

#### Application-Level Rate Limiting (PLANNED - Not Implemented)

**⚠️ This is a placeholder - not currently functional**

```bash
# .env (not implemented)
# ENABLE_RATE_LIMITING=true
# RATE_LIMIT_PER_MINUTE=60
# RATE_LIMIT_BURST=10
```

For now, implement custom rate limiting if needed:

```python
# Custom implementation with Redis (you must add this yourself)
from redis import Redis
from datetime import datetime, timedelta

def check_rate_limit(user_id: str, redis_client: Redis) -> bool:
    key = f"rate_limit:{user_id}:{datetime.now().strftime('%Y%m%d%H%M')}"
    current = redis_client.incr(key)

    if current == 1:
        redis_client.expire(key, 60)  # Expire after 1 minute

    return current <= 60  # 60 requests per minute
```

---

## Data Protection

### Data Encryption at Rest

#### Qdrant Encryption

Qdrant doesn't provide built-in encryption. Use:

1. **Encrypted file systems:**
```bash
# LUKS encryption (Linux)
cryptsetup luksFormat /dev/sdb
cryptsetup luksOpen /dev/sdb qdrant_encrypted
mkfs.ext4 /dev/mapper/qdrant_encrypted
mount /dev/mapper/qdrant_encrypted /var/lib/qdrant
```

2. **Cloud provider encryption:**
   - AWS EBS encryption
   - GCP persistent disk encryption
   - Azure disk encryption

---

#### Redis Encryption

```bash
# redis.conf
requirepass your-strong-redis-password

# Enable TLS (Redis 6+)
tls-port 6380
port 0  # Disable non-TLS
tls-cert-file /path/to/redis.crt
tls-key-file /path/to/redis.key
tls-ca-cert-file /path/to/ca.crt
```

```python
# Connect with TLS
import redis

redis_client = redis.Redis(
    host='localhost',
    port=6380,
    password='your-strong-redis-password',
    ssl=True,
    ssl_cert_reqs='required',
    ssl_ca_certs='/path/to/ca.crt'
)
```

---

### Data Encryption in Transit

**Always use TLS for:**
- Client → API communication (HTTPS)
- API → Qdrant (if remote)
- API → Redis (Redis TLS)
- API → LLM providers (all providers use HTTPS)

---

### Sensitive Data Handling

**PII (Personally Identifiable Information) Protection:**

```python
# Hash user IDs instead of using plain emails
import hashlib

def hash_user_id(email: str) -> str:
    return hashlib.sha256(email.encode()).hexdigest()[:16]

user_id = hash_user_id("alice@example.com")  # "7b8f9a3c2d1e4f5a"
memory = client.remember("sensitive info", user_id=user_id)
```

**Redact sensitive data before storage:**

```python
import re

def redact_pii(text: str) -> str:
    # Redact credit cards
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[REDACTED_CC]', text)

    # Redact SSNs
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', text)

    # Redact emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', text)

    return text

text = redact_pii("My card is 1234-5678-9012-3456")
memory = client.remember(text, user_id="alice")
```

---

## API Security

### Input Validation

**Validate all inputs:**

```python
from pydantic import BaseModel, validator, constr

class MemoryRequest(BaseModel):
    text: constr(min_length=1, max_length=10000)
    user_id: constr(min_length=1, max_length=100)
    importance: float

    @validator('importance')
    def validate_importance(cls, v):
        if not 0 <= v <= 10:
            raise ValueError('Importance must be between 0 and 10')
        return v

    @validator('text')
    def sanitize_text(cls, v):
        # Remove potentially dangerous characters
        return v.strip()
```

---

### SQL Injection Prevention

HippocampAI uses Qdrant (vector DB) and Redis, which don't use SQL. However:

**For metadata filters, sanitize inputs:**

```python
# ❌ Bad - user input directly in filter
user_input = request.get("filter")  # Could be malicious
filter_dict = eval(user_input)  # NEVER DO THIS

# ✅ Good - validate and construct safely
allowed_fields = {"type", "tags", "importance"}
filter_key = request.get("filter_key")

if filter_key in allowed_fields:
    filter_dict = {filter_key: request.get("filter_value")}
```

---

### XSS Prevention

**Sanitize memory text before displaying:**

```python
import bleach

def sanitize_for_display(text: str) -> str:
    # Allow only safe tags
    allowed_tags = ['p', 'br', 'strong', 'em']
    return bleach.clean(text, tags=allowed_tags, strip=True)

# Before sending to frontend
safe_text = sanitize_for_display(memory.text)
```

---

### CORS Configuration

**Configure CORS for frontend access:**

```python
# In FastAPI app (api/async_app.py)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "https://app.yourdomain.com"
    ],  # Don't use "*" in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

---

### Security Headers

**Add security headers:**

```python
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response

app.add_middleware(SecurityHeadersMiddleware)
```

---

## Container Security

### Docker Security Best Practices

**1. Use non-root user:**

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 hippocampai
USER hippocampai

WORKDIR /app
COPY --chown=hippocampai:hippocampai . .

CMD ["python", "-m", "uvicorn", "api.async_app:app", "--host", "0.0.0.0"]
```

---

**2. Scan for vulnerabilities:**

```bash
# Trivy scanner
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image hippocampai:latest

# Grype scanner
grype hippocampai:latest
```

---

**3. Use minimal base images:**

```dockerfile
# Use distroless or alpine
FROM python:3.11-slim AS builder
# Build dependencies

FROM gcr.io/distroless/python3-debian11
COPY --from=builder /app /app
```

---

**4. Set resource limits:**

```yaml
# docker-compose.yml
services:
  hippocampai:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          memory: 2G
```

---

### Kubernetes Security

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hippocampai
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000

      containers:
      - name: hippocampai
        image: hippocampai:latest
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
              - ALL

        resources:
          limits:
            cpu: 2000m
            memory: 4Gi
          requests:
            cpu: 500m
            memory: 1Gi
```

---

## Multi-Tenant Security

### User Isolation

**Ensure strict user_id filtering:**

```python
# ✅ Always filter by user_id
results = client.recall(query="test", user_id=authenticated_user_id)

# ❌ Never allow user to specify different user_id
# user_id = request.get("user_id")  # Attacker could access others' data
```

---

### Agent Isolation

**Enforce agent permissions:**

```python
def verify_agent_access(agent_id: str, user_id: str, operation: str) -> bool:
    agent = client.get_agent(agent_id)

    # Check if agent is allowed to access user's data
    if user_id not in agent.metadata.get("allowed_users", []):
        return False

    # Check if agent has permission for operation
    if operation not in agent.metadata.get("permissions", []):
        return False

    return True
```

---

### Namespace Isolation

**Use separate Qdrant collections per tenant:**

```python
def get_collection_name(tenant_id: str, collection_type: str) -> str:
    return f"tenant_{tenant_id}_{collection_type}"

# Create tenant-specific client
tenant_client = MemoryClient(
    config=Config(
        collection_facts=get_collection_name(tenant_id, "facts"),
        collection_prefs=get_collection_name(tenant_id, "prefs")
    )
)
```

---

## Audit & Compliance

### Audit Logging

**⚠️ NOT CURRENTLY IMPLEMENTED (v0.2.5):**

Audit logging is **NOT currently implemented** at the application level. The configuration options below are placeholders.

**For production**, implement audit logging yourself using:
- Application logging with structured logs
- Centralized logging (ELK, Loki, CloudWatch)
- SIEM integration

**Planned configuration (not functional):**

```bash
# .env (not implemented)
# ENABLE_AUDIT_LOGS=true
# AUDIT_LOG_FILE=/var/log/hippocampai/audit.log
# AUDIT_LOG_LEVEL=INFO
```

**Implement custom audit logging yourself:**

```python
import logging
from datetime import datetime

# You must set this up yourself
audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)
handler = logging.FileHandler("/var/log/hippocampai/audit.log")
audit_logger.addHandler(handler)

def audit_log(user_id: str, operation: str, resource: str, result: str):
    audit_logger.info({
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "operation": operation,
        "resource": resource,
        "result": result
    })

# Example usage
audit_log("alice", "DELETE_MEMORY", "mem_123", "SUCCESS")
```

---

### GDPR Compliance

**Right to be forgotten:**

```python
def delete_user_data(user_id: str):
    # Delete all memories
    memories = client.get_memories(user_id=user_id, limit=10000)
    client.delete_memories([m.id for m in memories])

    # Delete sessions
    sessions = client.get_user_sessions(user_id=user_id)
    for session in sessions:
        client.delete_session(session.session_id)

    # Delete from audit logs (if required)
    # Remove personal data from logs

    audit_log("system", "GDPR_DELETE_USER", user_id, "SUCCESS")
```

**Data export:**

```python
def export_user_data(user_id: str) -> dict:
    return {
        "user_id": user_id,
        "memories": [m.to_dict() for m in client.get_memories(user_id=user_id, limit=10000)],
        "sessions": [s.to_dict() for s in client.get_user_sessions(user_id=user_id)],
        "export_date": datetime.utcnow().isoformat()
    }
```

---

### SOC 2 / ISO 27001

**Access control documentation:**
- Maintain user access logs
- Implement least privilege principle
- Regular access reviews

**Incident response plan:**
- Define security incident procedures
- Maintain contact list
- Regular security drills

---

## Security Checklist

### Pre-Production Security Checklist

- [ ] **Authentication & Authorization**
  - [ ] Enable authentication (API keys or JWT)
  - [ ] Implement role-based access control
  - [ ] Configure session timeout
  - [ ] Set up API key rotation schedule

- [ ] **Secrets Management**
  - [ ] Move all secrets to environment variables
  - [ ] Use secrets manager (Vault, AWS Secrets Manager)
  - [ ] Remove hardcoded credentials
  - [ ] Rotate all default passwords

- [ ] **Network Security**
  - [ ] Enable HTTPS/TLS
  - [ ] Configure firewall rules
  - [ ] Set up network isolation
  - [ ] Implement rate limiting

- [ ] **Data Protection**
  - [ ] Enable encryption at rest
  - [ ] Configure TLS for Redis
  - [ ] Implement PII redaction
  - [ ] Set up data backup encryption

- [ ] **API Security**
  - [ ] Validate all inputs
  - [ ] Configure CORS properly
  - [ ] Add security headers
  - [ ] Implement request size limits

- [ ] **Container Security**
  - [ ] Run as non-root user
  - [ ] Scan images for vulnerabilities
  - [ ] Set resource limits
  - [ ] Use minimal base images

- [ ] **Monitoring & Audit**
  - [ ] Enable audit logging
  - [ ] Set up security monitoring
  - [ ] Configure alerting
  - [ ] Define retention policies

- [ ] **Compliance**
  - [ ] Implement GDPR compliance features
  - [ ] Document security procedures
  - [ ] Conduct security review
  - [ ] Perform penetration testing

---

## Incident Response

### Security Incident Response Plan

**1. Detection & Triage**
```bash
# Monitor logs for suspicious activity
grep "FAILED_AUTH" /var/log/hippocampai/audit.log
grep "RATE_LIMIT_EXCEEDED" /var/log/hippocampai/api.log
```

**2. Containment**
```python
# Immediately revoke compromised credentials
client.revoke_api_key(compromised_key)

# Block suspicious IPs
# Add to firewall
sudo ufw deny from 192.168.1.100
```

**3. Investigation**
```python
# Review audit trail
trail = client.get_audit_trail(
    user_id=suspected_user,
    limit=1000
)

# Analyze access patterns
for entry in trail:
    print(f"{entry.timestamp}: {entry.operation} on {entry.resource}")
```

**4. Recovery**
```python
# Restore from backup if data compromised
client.import_graph_from_json("backup_20241101.json")

# Reset user passwords
# Rotate all API keys
```

**5. Post-Incident**
- Document incident details
- Update security procedures
- Conduct lessons learned review
- Implement additional controls

---

### Contact Information

**Security Vulnerabilities:**
Report security issues to: security@hippocampai.dev (placeholder - update with actual contact)

**Do NOT** report security vulnerabilities in public GitHub issues.

---

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [GDPR Compliance Guide](https://gdpr.eu/)

---

**Version:** v0.2.5
**Last Updated:** November 2025
