# Backup & Recovery Guide for HippocampAI

Complete guide for backing up and recovering HippocampAI data in production environments.

## Table of Contents

- [Overview](#overview)
- [What to Backup](#what-to-backup)
- [Backup Strategies](#backup-strategies)
- [Backup Procedures](#backup-procedures)
- [Recovery Procedures](#recovery-procedures)
- [Disaster Recovery](#disaster-recovery)
- [Automated Backups](#automated-backups)
- [Cloud Storage Integration](#cloud-storage-integration)
- [Best Practices](#best-practices)
- [Testing & Validation](#testing--validation)

---

## Overview

### Backup Objectives

1. **Data Protection**: Prevent data loss from hardware failures, software bugs, or human error
2. **Business Continuity**: Minimize downtime during incidents
3. **Compliance**: Meet regulatory requirements for data retention
4. **Point-in-Time Recovery**: Restore to specific timestamps

### Recovery Time Objective (RTO) & Recovery Point Objective (RPO)

| Deployment Type | RTO Target | RPO Target | Backup Frequency |
|-----------------|------------|------------|------------------|
| **Development** | 1-2 hours | 24 hours | Daily |
| **Staging** | 30 minutes | 6 hours | Every 6 hours |
| **Production** | 15 minutes | 1 hour | Hourly + continuous |
| **Mission-Critical** | 5 minutes | 5 minutes | Continuous replication |

---

## What to Backup

### Critical Data

1. **Qdrant Vector Database**
   - Memory vectors and payloads
   - Collection configurations
   - Index structures

2. **Redis Cache** (Optional but recommended)
   - Session data
   - Cached results
   - Temporary state

3. **Memory Metadata**
   - User information
   - Session metadata
   - Agent configurations

4. **Application Configuration**
   - `.env` files
   - `docker-compose.yml`
   - Custom configurations

5. **Audit Logs**
   - Access logs
   - Change history
   - Security events

### Data NOT to Backup

- LLM provider API keys (use secrets manager)
- Temporary cache files
- Log rotations older than retention period
- Test/development data

---

## Backup Strategies

### Strategy 1: Full Backup

**Description:** Complete snapshot of all data

**Pros:**
- Simple to implement
- Fast recovery
- Complete data integrity

**Cons:**
- Large storage requirements
- Longer backup time
- Higher network usage

**Recommended for:**
- Weekly baseline backups
- Small deployments (<10GB)
- Pre-upgrade snapshots

---

### Strategy 2: Incremental Backup

**Description:** Only changes since last backup

**Pros:**
- Fast backup time
- Minimal storage
- Low network usage

**Cons:**
- Complex recovery (need chain)
- Risk of chain corruption

**Recommended for:**
- Daily/hourly backups
- Large deployments
- High-frequency changes

---

### Strategy 3: Continuous Replication

**Description:** Real-time replication to standby

**Pros:**
- Near-zero data loss
- Fast failover
- No backup windows

**Cons:**
- Higher cost
- Complex setup
- Requires dedicated infrastructure

**Recommended for:**
- Mission-critical production
- High availability requirements
- Large enterprises

---

## Backup Procedures

### Manual Backup

#### 1. Backup Qdrant Vector Database

**Method 1: Snapshot API (Recommended)**

```python
from hippocampai import MemoryClient
from datetime import datetime

client = MemoryClient()

# Create snapshot
snapshot_name = client.create_snapshot(collection="hippocampai_facts")
print(f"Created snapshot: {snapshot_name}")

# Snapshot is stored in Qdrant's snapshots directory
# Default: /qdrant/storage/snapshots/
```

**Method 2: Docker Volume Backup**

```bash
#!/bin/bash
# backup_qdrant.sh

BACKUP_DIR="/backups/qdrant"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="qdrant_backup_${TIMESTAMP}.tar.gz"

# Stop Qdrant to ensure consistency (optional)
docker-compose stop qdrant

# Create backup
docker run --rm \
  --volumes-from hippocampai_qdrant_1 \
  -v ${BACKUP_DIR}:/backup \
  ubuntu:22.04 \
  tar czf /backup/${BACKUP_FILE} /qdrant/storage

# Restart Qdrant
docker-compose start qdrant

echo "Backup saved to: ${BACKUP_DIR}/${BACKUP_FILE}"
```

**Method 3: Direct File System Backup**

```bash
#!/bin/bash
# For non-Docker deployments

QDRANT_DATA="/var/lib/qdrant/storage"
BACKUP_DIR="/backups/qdrant"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Stop Qdrant service
sudo systemctl stop qdrant

# Create backup with rsync
sudo rsync -av --delete \
  ${QDRANT_DATA}/ \
  ${BACKUP_DIR}/qdrant_${TIMESTAMP}/

# Restart Qdrant
sudo systemctl start qdrant

echo "Backup completed: ${BACKUP_DIR}/qdrant_${TIMESTAMP}"
```

---

#### 2. Backup Redis Data

**Method 1: Redis SAVE Command**

```bash
#!/bin/bash
# backup_redis.sh

BACKUP_DIR="/backups/redis"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Trigger Redis save
redis-cli SAVE

# Copy RDB file
docker cp hippocampai_redis_1:/data/dump.rdb \
  ${BACKUP_DIR}/redis_${TIMESTAMP}.rdb

echo "Redis backup: ${BACKUP_DIR}/redis_${TIMESTAMP}.rdb"
```

**Method 2: Redis BGSAVE (Non-blocking)**

```bash
#!/bin/bash
# Non-blocking background save

redis-cli BGSAVE

# Wait for completion
while [ $(redis-cli LASTSAVE) -eq $LAST_SAVE ]; do
  sleep 1
done

# Copy RDB file
cp /var/lib/redis/dump.rdb /backups/redis/dump_$(date +%Y%m%d_%H%M%S).rdb
```

**Method 3: Redis AOF Backup**

```bash
# If using AOF persistence
redis-cli BGREWRITEAOF

# Copy AOF file
cp /var/lib/redis/appendonly.aof /backups/redis/aof_$(date +%Y%m%d_%H%M%S).aof
```

---

#### 3. Export Memory Graph (Application-Level)

```python
#!/usr/bin/env python3
# export_memories.py

from hippocampai import MemoryClient
from datetime import datetime
import os

def backup_memories(output_dir="/backups/memories"):
    """Export all memories to JSON"""
    client = MemoryClient()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"memories_backup_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Export graph
    stats = client.export_graph_to_json(filepath)

    print(f"✅ Backup completed: {filepath}")
    print(f"   Memories: {stats['total_memories']}")
    print(f"   Users: {stats['total_users']}")
    print(f"   Size: {stats['file_size_mb']:.2f} MB")

    return filepath

if __name__ == "__main__":
    backup_memories()
```

Run the backup:
```bash
python export_memories.py
```

---

#### 4. Backup Configuration Files

```bash
#!/bin/bash
# backup_config.sh

BACKUP_DIR="/backups/config"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_BACKUP="${BACKUP_DIR}/config_${TIMESTAMP}.tar.gz"

# Create backup of configuration
tar czf ${CONFIG_BACKUP} \
  .env \
  docker-compose.yml \
  prometheus.yml \
  monitoring/ \
  --exclude='*.pyc' \
  --exclude='__pycache__'

echo "Configuration backup: ${CONFIG_BACKUP}"
```

---

### Complete Backup Script

```bash
#!/bin/bash
# complete_backup.sh - Full system backup

set -e  # Exit on error

BACKUP_ROOT="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${BACKUP_ROOT}/${TIMESTAMP}"

mkdir -p ${BACKUP_DIR}/{qdrant,redis,config,memories,logs}

echo "=== HippocampAI Backup Started: ${TIMESTAMP} ==="

# 1. Backup Qdrant
echo "Backing up Qdrant..."
docker run --rm \
  --volumes-from hippocampai_qdrant_1 \
  -v ${BACKUP_DIR}/qdrant:/backup \
  ubuntu:22.04 \
  tar czf /backup/qdrant.tar.gz /qdrant/storage

# 2. Backup Redis
echo "Backing up Redis..."
redis-cli SAVE
docker cp hippocampai_redis_1:/data/dump.rdb \
  ${BACKUP_DIR}/redis/dump.rdb

# 3. Backup Memories (JSON export)
echo "Exporting memories..."
python3 export_memories.py ${BACKUP_DIR}/memories

# 4. Backup Configuration
echo "Backing up configuration..."
tar czf ${BACKUP_DIR}/config/config.tar.gz \
  .env docker-compose.yml prometheus.yml monitoring/

# 5. Backup Logs
echo "Backing up logs..."
tar czf ${BACKUP_DIR}/logs/logs.tar.gz logs/ --exclude='*.gz'

# 6. Create backup manifest
cat > ${BACKUP_DIR}/manifest.txt <<EOF
HippocampAI Backup Manifest
Timestamp: ${TIMESTAMP}
Hostname: $(hostname)
Version: v0.2.5

Components:
- Qdrant Vector DB: $(du -sh ${BACKUP_DIR}/qdrant | awk '{print $1}')
- Redis Cache: $(du -sh ${BACKUP_DIR}/redis | awk '{print $1}')
- Memory Export: $(du -sh ${BACKUP_DIR}/memories | awk '{print $1}')
- Configuration: $(du -sh ${BACKUP_DIR}/config | awk '{print $1}')
- Logs: $(du -sh ${BACKUP_DIR}/logs | awk '{print $1}')

Total Size: $(du -sh ${BACKUP_DIR} | awk '{print $1}')
EOF

echo "=== Backup Completed ==="
echo "Location: ${BACKUP_DIR}"
echo "Total Size: $(du -sh ${BACKUP_DIR} | awk '{print $1}')"
```

Make executable and run:
```bash
chmod +x complete_backup.sh
./complete_backup.sh
```

---

## Recovery Procedures

### Scenario 1: Complete System Recovery

**Prerequisites:**
- Fresh HippocampAI installation
- Access to backup files
- Docker and dependencies installed

**Steps:**

```bash
#!/bin/bash
# restore_complete.sh

BACKUP_DIR="/backups/20241102_120000"  # Your backup directory

echo "=== Starting Complete System Recovery ==="

# 1. Stop all services
echo "Stopping services..."
docker-compose down

# 2. Restore Qdrant
echo "Restoring Qdrant..."
rm -rf qdrant_storage/*
tar xzf ${BACKUP_DIR}/qdrant/qdrant.tar.gz -C qdrant_storage/ --strip-components=2

# 3. Restore Redis
echo "Restoring Redis..."
docker run --rm \
  -v hippocampai_redis_data:/data \
  -v ${BACKUP_DIR}/redis:/backup \
  ubuntu:22.04 \
  cp /backup/dump.rdb /data/dump.rdb

# 4. Restore Configuration
echo "Restoring configuration..."
tar xzf ${BACKUP_DIR}/config/config.tar.gz

# 5. Start services
echo "Starting services..."
docker-compose up -d

# 6. Wait for services to be ready
echo "Waiting for services..."
sleep 10

# 7. Verify restoration
echo "Verifying restoration..."
python3 -c "
from hippocampai import MemoryClient
client = MemoryClient()
stats = client.get_memory_statistics(user_id='test')
print(f'✅ Restoration successful! Found {stats.get(\"total_memories\", 0)} memories')
"

echo "=== Recovery Completed ==="
```

---

### Scenario 2: Selective Memory Recovery

**Use case:** Restore specific user's memories

```python
#!/usr/bin/env python3
# restore_user_memories.py

from hippocampai import MemoryClient
import json

def restore_user_memories(backup_file: str, user_id: str):
    """Restore memories for a specific user"""
    client = MemoryClient()

    # Load backup
    with open(backup_file, 'r') as f:
        backup_data = json.load(f)

    # Filter user's memories
    user_memories = [
        m for m in backup_data['memories']
        if m['user_id'] == user_id
    ]

    # Restore
    restored = 0
    for memory_data in user_memories:
        try:
            client.remember(
                text=memory_data['text'],
                user_id=memory_data['user_id'],
                type=memory_data['type'],
                importance=memory_data['importance'],
                tags=memory_data.get('tags', []),
                metadata=memory_data.get('metadata', {})
            )
            restored += 1
        except Exception as e:
            print(f"Failed to restore memory: {e}")

    print(f"✅ Restored {restored}/{len(user_memories)} memories for user {user_id}")

if __name__ == "__main__":
    restore_user_memories(
        backup_file="/backups/memories_backup_20241102.json",
        user_id="alice"
    )
```

---

### Scenario 3: Point-in-Time Recovery

**Use case:** Restore to specific timestamp (before data corruption)

```bash
#!/bin/bash
# restore_point_in_time.sh

# List available backups
echo "Available backups:"
ls -lh /backups/ | grep -E '^d'

# Choose backup
read -p "Enter backup timestamp (YYYYMMDD_HHMMSS): " TIMESTAMP
BACKUP_DIR="/backups/${TIMESTAMP}"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "❌ Backup not found: $BACKUP_DIR"
    exit 1
fi

# Display backup info
cat ${BACKUP_DIR}/manifest.txt

# Confirm restoration
read -p "Restore from this backup? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Restoration cancelled"
    exit 0
fi

# Proceed with restoration
./restore_complete.sh $BACKUP_DIR
```

---

### Scenario 4: Incremental Recovery

**Use case:** Apply incremental backups in sequence

```bash
#!/bin/bash
# restore_incremental.sh

FULL_BACKUP="/backups/20241101_000000"
INCREMENTAL_BACKUPS=(
    "/backups/20241101_060000"
    "/backups/20241101_120000"
    "/backups/20241101_180000"
)

# Restore full backup
echo "Restoring full backup..."
./restore_complete.sh $FULL_BACKUP

# Apply incremental backups in order
for INCREMENTAL in "${INCREMENTAL_BACKUPS[@]}"; do
    echo "Applying incremental: $INCREMENTAL"
    python3 -c "
from hippocampai import MemoryClient
client = MemoryClient()
client.import_graph_from_json('${INCREMENTAL}/memories/backup.json', merge=True)
"
done

echo "✅ Incremental restoration completed"
```

---

## Disaster Recovery

### Disaster Recovery Plan

#### Phase 1: Assessment (0-5 minutes)

1. **Identify the incident**
   - Data corruption
   - Hardware failure
   - Ransomware/security breach
   - Human error

2. **Determine scope**
   - Affected systems
   - Data loss extent
   - Service impact

3. **Activate DR team**

---

#### Phase 2: Containment (5-15 minutes)

1. **Stop the bleeding**
```bash
# Stop all services
docker-compose down

# Disconnect from network if security incident
# Isolate affected systems
```

2. **Preserve evidence**
```bash
# Create forensic backup
tar czf /forensics/incident_$(date +%Y%m%d_%H%M%S).tar.gz \
  qdrant_storage/ logs/ data/
```

3. **Notify stakeholders**

---

#### Phase 3: Recovery (15-60 minutes)

1. **Prepare clean environment**
```bash
# Fresh Docker containers
docker-compose pull
docker system prune -af
```

2. **Restore from backup**
```bash
# Use latest good backup
./restore_complete.sh /backups/20241102_000000
```

3. **Verify integrity**
```python
from hippocampai import MemoryClient

client = MemoryClient()

# Test basic operations
memory = client.remember("DR test", user_id="admin")
results = client.recall("DR test", user_id="admin")

assert len(results) > 0, "Recovery failed"
print("✅ System recovered successfully")
```

---

#### Phase 4: Validation (60-120 minutes)

1. **Data integrity checks**
```python
# Verify memory counts
stats = client.get_memory_statistics(user_id="all")
# Compare with pre-incident metrics
```

2. **Functional testing**
```bash
# Run integration tests
pytest tests/test_all_features_integration.py
```

3. **Performance validation**
```bash
# Run load tests
locust -f tests/load_test.py --headless -u 100 -r 10
```

---

#### Phase 5: Post-Incident (1-7 days)

1. **Root cause analysis**
2. **Update DR procedures**
3. **Implement preventive measures**
4. **Post-incident review meeting**

---

### Geographic Redundancy

For mission-critical deployments:

```yaml
# Multi-region setup with replication

# Primary Region (US-WEST)
primary:
  qdrant: us-west.qdrant.example.com
  redis: us-west.redis.example.com
  backup: s3://backups-us-west/

# Secondary Region (US-EAST)
secondary:
  qdrant: us-east.qdrant.example.com
  redis: us-east.redis.example.com
  backup: s3://backups-us-east/

# Replication script
#!/bin/bash
# Sync from primary to secondary every 5 minutes
*/5 * * * * /usr/local/bin/replicate_to_secondary.sh
```

---

## Automated Backups

### Cron-based Automation

```bash
# /etc/cron.d/hippocampai-backup

# Full backup daily at 2 AM
0 2 * * * /opt/hippocampai/scripts/complete_backup.sh >> /var/log/hippocampai/backup.log 2>&1

# Incremental backup every 6 hours
0 */6 * * * /opt/hippocampai/scripts/incremental_backup.sh >> /var/log/hippocampai/backup.log 2>&1

# Memory export hourly
0 * * * * python3 /opt/hippocampai/scripts/export_memories.py >> /var/log/hippocampai/export.log 2>&1

# Cleanup old backups (keep 30 days)
0 3 * * * find /backups -type d -mtime +30 -exec rm -rf {} \; 2>/dev/null
```

---

### systemd Timer (Modern Linux)

```ini
# /etc/systemd/system/hippocampai-backup.timer
[Unit]
Description=HippocampAI Daily Backup Timer

[Timer]
OnCalendar=daily
OnCalendar=02:00
Persistent=true

[Install]
WantedBy=timers.target
```

```ini
# /etc/systemd/system/hippocampai-backup.service
[Unit]
Description=HippocampAI Backup Service

[Service]
Type=oneshot
ExecStart=/opt/hippocampai/scripts/complete_backup.sh
User=hippocampai
StandardOutput=journal
StandardError=journal
```

Enable:
```bash
sudo systemctl enable hippocampai-backup.timer
sudo systemctl start hippocampai-backup.timer
```

---

### Docker-based Backup Container

```yaml
# docker-compose.yml
services:
  backup:
    image: offen/docker-volume-backup:latest
    environment:
      BACKUP_CRON_EXPRESSION: "0 2 * * *"
      BACKUP_FILENAME: "backup-%Y%m%d-%H%M%S.tar.gz"
      BACKUP_RETENTION_DAYS: "30"
      BACKUP_PRUNING_PREFIX: "backup-"
    volumes:
      - qdrant_data:/backup/qdrant:ro
      - redis_data:/backup/redis:ro
      - ./backups:/archive
```

---

## Cloud Storage Integration

### AWS S3

```bash
#!/bin/bash
# backup_to_s3.sh

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
S3_BUCKET="s3://my-hippocampai-backups"

# Create backup
./complete_backup.sh

# Upload to S3
aws s3 sync ${BACKUP_DIR} ${S3_BUCKET}/$(basename ${BACKUP_DIR}) \
  --storage-class STANDARD_IA \
  --sse AES256

# Enable lifecycle policy for old backups
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-hippocampai-backups \
  --lifecycle-configuration file://lifecycle.json
```

lifecycle.json:
```json
{
  "Rules": [
    {
      "Id": "DeleteOldBackups",
      "Status": "Enabled",
      "Prefix": "",
      "Expiration": {
        "Days": 90
      },
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
```

---

### Google Cloud Storage

```bash
#!/bin/bash
# backup_to_gcs.sh

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
GCS_BUCKET="gs://my-hippocampai-backups"

# Create backup
./complete_backup.sh

# Upload to GCS
gsutil -m rsync -r ${BACKUP_DIR} ${GCS_BUCKET}/$(basename ${BACKUP_DIR})

# Set lifecycle policy
gsutil lifecycle set lifecycle.json ${GCS_BUCKET}
```

---

### Azure Blob Storage

```bash
#!/bin/bash
# backup_to_azure.sh

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
CONTAINER="hippocampai-backups"

# Create backup
./complete_backup.sh

# Upload to Azure
az storage blob upload-batch \
  --destination ${CONTAINER} \
  --source ${BACKUP_DIR} \
  --account-name mystorageaccount
```

---

## Best Practices

### 1. Follow 3-2-1 Rule

- **3** copies of data
- **2** different storage types
- **1** offsite copy

Example:
- Primary: Production Qdrant
- Secondary: Local backup disk
- Tertiary: Cloud storage (S3/GCS)

---

### 2. Encrypt Backups

```bash
# Encrypt with GPG
gpg --symmetric --cipher-algo AES256 \
  --output backup_encrypted.tar.gz.gpg \
  backup.tar.gz

# Decrypt
gpg --decrypt backup_encrypted.tar.gz.gpg > backup.tar.gz
```

---

### 3. Verify Backups Regularly

```bash
#!/bin/bash
# verify_backup.sh

BACKUP_FILE=$1

# Test backup integrity
if tar tzf ${BACKUP_FILE} > /dev/null 2>&1; then
    echo "✅ Backup integrity verified"
else
    echo "❌ Backup is corrupted!"
    exit 1
fi

# Test restoration in isolated environment
# ...
```

---

### 4. Document Recovery Procedures

Maintain runbook with:
- Step-by-step recovery instructions
- Contact information
- Access credentials location
- Decision trees for different scenarios

---

### 5. Monitor Backup Success

```python
# send_backup_alert.py

import smtplib
from email.mime.text import MIMEText

def send_alert(status, message):
    msg = MIMEText(f"Backup Status: {status}\n\n{message}")
    msg['Subject'] = f'HippocampAI Backup {status}'
    msg['From'] = 'backup@example.com'
    msg['To'] = 'admin@example.com'

    with smtplib.SMTP('localhost') as server:
        server.send_message(msg)

# Call from backup script
send_alert("SUCCESS", "Daily backup completed successfully")
```

---

## Testing & Validation

### Backup Testing Checklist

- [ ] **Integrity Test**: Verify backup files are not corrupted
- [ ] **Restoration Test**: Restore to test environment
- [ ] **Performance Test**: Measure restoration time
- [ ] **Completeness Test**: Verify all data restored
- [ ] **Functional Test**: Test application after restoration

### Quarterly DR Drill

Schedule full disaster recovery exercise:

1. **Week 1**: Plan and notify team
2. **Week 2**: Execute DR in staging
3. **Week 3**: Document findings
4. **Week 4**: Update procedures

---

## Retention Policy

### Recommended Retention Schedule

| Backup Type | Retention | Storage Tier |
|-------------|-----------|--------------|
| Hourly | 7 days | Fast (SSD) |
| Daily | 30 days | Standard |
| Weekly | 12 weeks | Standard |
| Monthly | 12 months | Archive (Glacier) |
| Yearly | 7 years | Archive |
| Pre-upgrade | Until next upgrade | Standard |

---

## Monitoring Backup Health

### Metrics to Track

```python
# backup_metrics.py

import prometheus_client as prom

# Metrics
backup_success = prom.Counter('hippocampai_backup_success_total', 'Successful backups')
backup_failure = prom.Counter('hippocampai_backup_failure_total', 'Failed backups')
backup_duration = prom.Histogram('hippocampai_backup_duration_seconds', 'Backup duration')
backup_size = prom.Gauge('hippocampai_backup_size_bytes', 'Backup size')
last_backup_timestamp = prom.Gauge('hippocampai_last_backup_timestamp', 'Last backup time')

# In backup script
with backup_duration.time():
    try:
        perform_backup()
        backup_success.inc()
        last_backup_timestamp.set(time.time())
    except Exception:
        backup_failure.inc()
```

---

## Additional Resources

- [Qdrant Backup Documentation](https://qdrant.tech/documentation/concepts/snapshots/)
- [Redis Persistence Documentation](https://redis.io/docs/management/persistence/)
- [Docker Volume Backup](https://docs.docker.com/storage/volumes/#back-up-restore-or-migrate-data-volumes)

---

**Version:** v0.2.5
**Last Updated:** November 2025
