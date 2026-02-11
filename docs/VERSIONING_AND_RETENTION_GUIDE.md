# Versioning & Retention Guide

**Version**: 0.2.5
**Last Updated**: 2026-02-11

---

## Overview

HippocampAI provides comprehensive versioning and retention management features:

1. **Memory Version History** - Track all changes to memories
2. **Enhanced Diff Support** - Detailed text diffs with unified diff format
3. **Audit Logs** - Complete audit trail of all operations
4. **Rollback Support** - Restore previous versions of memories
5. **Retention Policies** - Automatic memory cleanup with smart preservation

---

## 1. Memory Version History

### Overview

Every change to a memory is tracked as a version, allowing you to:

- View the complete history of changes
- Compare different versions
- Rollback to previous versions
- Track who made changes and when

### Basic Usage

```python
from hippocampai.versioning import MemoryVersionControl

# Initialize version control
version_control = MemoryVersionControl(max_versions_per_memory=10)

# Create first version
memory_data_v1 = {
    "id": "mem_001",
    "text": "I work at Google as a Software Engineer.",
    "user_id": "user_001",
    "type": "fact",
    "importance": 7.0,
    "tags": ["employment"]
}

v1 = version_control.create_version(
    memory_id="mem_001",
    data=memory_data_v1,
    created_by="user_001",
    change_summary="Initial creation"
)

print(f"Created version {v1.version_number}")
```

### Creating New Versions

```python
# Update memory
memory_data_v2 = memory_data_v1.copy()
memory_data_v2["text"] = "I work at Google as a Senior Software Engineer."
memory_data_v2["importance"] = 8.0
memory_data_v2["tags"] = ["employment", "promotion"]

v2 = version_control.create_version(
    memory_id="mem_001",
    data=memory_data_v2,
    created_by="user_001",
    change_summary="Promotion to Senior Engineer"
)

print(f"Created version {v2.version_number}")
print(f"Version ID: {v2.version_id}")
print(f"Created at: {v2.created_at}")
```

### Retrieving Versions

```python
# Get latest version
latest = version_control.get_version("mem_001")
print(f"Latest version: {latest.version_number}")

# Get specific version
v1 = version_control.get_version("mem_001", version_number=1)
print(f"Version 1 text: {v1.data['text']}")

# Get all versions
history = version_control.get_version_history("mem_001")
print(f"Total versions: {len(history)}")

for version in history:
    print(f"  v{version.version_number}: {version.change_summary}")
    print(f"    Created: {version.created_at}")
    print(f"    By: {version.created_by}")
```

---

## 2. Enhanced Diff Support

### Overview

Compare any two versions to see exactly what changed, with detailed text diffs using unified diff format.

### Basic Diff

```python
# Compare two versions
diff = version_control.compare_versions("mem_001", version1=1, version2=2)

print("Changes detected:")
print(f"  Added fields: {list(diff['added'].keys())}")
print(f"  Removed fields: {list(diff['removed'].keys())}")
print(f"  Changed fields: {list(diff['changed'].keys())}")

# View changed fields
for field, change in diff['changed'].items():
    print(f"\n{field}:")
    print(f"  Old: {change['old']}")
    print(f"  New: {change['new']}")
```

### Text Diff

```python
# Get detailed text diff (if text changed)
if diff['text_diff']:
    text_diff = diff['text_diff']

    print("\nText Diff Statistics:")
    print(f"  Added lines: {text_diff['added_lines']}")
    print(f"  Removed lines: {text_diff['removed_lines']}")
    print(f"  Old length: {text_diff['old_length']} chars")
    print(f"  New length: {text_diff['new_length']} chars")
    print(f"  Size change: {text_diff['size_change']} chars")

    print("\nUnified Diff:")
    print(text_diff['unified_diff'])
```

### Example Output

```
Text Diff Statistics:
  Added lines: 1
  Removed lines: 1
  Old length: 45 chars
  New length: 52 chars
  Size change: +7 chars

Unified Diff:
--- old
+++ new
@@ -1 +1 @@
-I work at Google as a Software Engineer.
+I work at Google as a Senior Software Engineer.
```

---

## 3. Audit Logs

### Overview

Track all operations performed on memories for compliance, debugging, and analytics.

### Recording Operations

```python
from hippocampai.versioning import ChangeType

# Add audit entry for creation
audit = version_control.add_audit_entry(
    memory_id="mem_001",
    change_type=ChangeType.CREATED,
    user_id="user_001",
    changes={"text": "Initial memory created"},
    metadata={"source": "api", "ip": "192.168.1.1"}
)

print(f"Audit entry created: {audit.audit_id}")
```

### Change Types

```python
class ChangeType(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    ACCESSED = "accessed"
    RELATIONSHIP_ADDED = "relationship_added"
    RELATIONSHIP_REMOVED = "relationship_removed"
```

### Querying Audit Trail

```python
# Get all audit entries for a memory
entries = version_control.get_audit_trail(memory_id="mem_001", limit=100)

for entry in entries:
    print(f"{entry.timestamp}: {entry.change_type.value}")
    print(f"  User: {entry.user_id}")
    print(f"  Changes: {entry.changes}")

# Filter by change type
updates = version_control.get_audit_trail(
    memory_id="mem_001",
    change_type=ChangeType.UPDATED
)

# Filter by user
user_actions = version_control.get_audit_trail(
    user_id="user_001",
    limit=50
)

# Get all audit entries
all_entries = version_control.get_audit_trail(limit=1000)
```

### Audit Entry Data

```python
# Convert to dictionary
audit_dict = audit.to_dict()

print(audit_dict)
# {
#     "audit_id": "uuid",
#     "memory_id": "mem_001",
#     "change_type": "updated",
#     "timestamp": "2026-10-28T12:00:00Z",
#     "user_id": "user_001",
#     "changes": {...},
#     "metadata": {...}
# }
```

### Cleanup Old Entries

```python
# Clear entries older than 90 days
cleared = version_control.clear_old_audit_entries(days_old=90)
print(f"Cleared {cleared} old audit entries")
```

---

## 4. Rollback Support

### Overview

Restore a memory to any previous version.

### Basic Rollback

```python
# Rollback to version 1
data = version_control.rollback("mem_001", version_number=1)

if data:
    print("Rollback successful!")
    print(f"Text: {data['text']}")
    print(f"Importance: {data['importance']}")

    # Apply the rollback
    # (You would update the actual memory in your system)
    memory.text = data['text']
    memory.importance = data['importance']
    # ... update other fields
else:
    print("Rollback failed - version not found")
```

### Safe Rollback Pattern

```python
# 1. Get current version first
current = version_control.get_version("mem_001")
current_version = current.version_number

# 2. Create backup version
version_control.create_version(
    memory_id="mem_001",
    data=current.data,
    created_by="system",
    change_summary=f"Backup before rollback to v{target_version}"
)

# 3. Perform rollback
data = version_control.rollback("mem_001", version_number=target_version)

# 4. Apply changes and log
if data:
    # Apply to actual memory
    # ...

    # Log the rollback
    version_control.add_audit_entry(
        memory_id="mem_001",
        change_type=ChangeType.UPDATED,
        user_id="admin",
        changes={"rollback_from": current_version, "rollback_to": target_version}
    )
```

---

## 5. Retention Policies

### Overview

Automatically delete old memories based on configurable policies with smart preservation rules.

### Creating Policies

```python
from hippocampai.retention import RetentionPolicyManager
from hippocampai.vector.qdrant_store import QdrantStore

# Initialize
qdrant = QdrantStore(url="http://localhost:6333")
retention_mgr = RetentionPolicyManager(qdrant_store=qdrant)

# Create a retention policy
policy = retention_mgr.create_policy(
    name="Archive old events",
    retention_days=30,
    user_id="user_001",
    memory_type="event",
    min_importance=7.0,      # Preserve if importance >= 7.0
    min_access_count=5,      # Preserve if accessed >= 5 times
    tags_to_preserve=["important", "milestone"],
    enabled=True
)

print(f"Created policy: {policy.name}")
print(f"Policy ID: {policy.id}")
```

### Policy Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Policy name |
| `retention_days` | int | Days to retain memories |
| `user_id` | str | User ID (None = global) |
| `memory_type` | str | Memory type (None = all) |
| `min_importance` | float | Preserve if importance >= threshold |
| `min_access_count` | int | Preserve if accessed >= N times |
| `tags_to_preserve` | list[str] | Tags that prevent deletion |
| `enabled` | bool | Whether policy is active |

### Preservation Rules

A memory is **preserved** (not deleted) if **ANY** of these conditions are true:

1. Age < retention_days
2. Importance >= min_importance
3. Access count >= min_access_count
4. Has any tag in tags_to_preserve

```python
# Example: Memory evaluation
memory = {
    "created_at": "35 days ago",
    "importance": 8.0,
    "access_count": 3,
    "tags": []
}

# Policy: retention_days=30, min_importance=7.0
should_delete = policy.should_delete(memory)
# False - preserved by importance (8.0 >= 7.0)
```

### Applying Policies

```python
# Dry run - see what would be deleted
result = retention_mgr.apply_policies(dry_run=True)

print(f"Would delete: {result['deleted']} memories")
print(f"Policies applied: {result['policies_applied']}")
print("\nDeletions by policy:")
for policy_name, count in result['deleted_by_policy'].items():
    print(f"  {policy_name}: {count} memories")

# Actually apply policies
result = retention_mgr.apply_policies(dry_run=False)
print(f"Deleted: {result['deleted']} memories")
```

### Managing Policies

```python
# Get all policies
policies = retention_mgr.get_policies()

# Get policies for specific user
user_policies = retention_mgr.get_policies(user_id="user_001")

# Get only enabled policies
enabled = retention_mgr.get_policies(enabled_only=True)

# Update policy
retention_mgr.update_policy(
    policy_id=policy.id,
    retention_days=60,
    min_importance=8.0
)

# Disable policy
retention_mgr.update_policy(
    policy_id=policy.id,
    enabled=False
)

# Delete policy
retention_mgr.delete_policy(policy.id)
```

### Expiring Memories Warning

```python
# Get memories expiring soon
expiring = retention_mgr.get_expiring_memories(
    user_id="user_001",
    days_threshold=7  # Within 7 days
)

print(f"Memories expiring soon: {len(expiring)}")

for mem in expiring:
    print(f"  • {mem['text'][:50]}...")
    print(f"    Days until expiration: {mem['days_until_expiration']}")
    print(f"    Policy: {mem['policy_name']}")
```

### Statistics

```python
stats = retention_mgr.get_statistics()

print(f"Total policies: {stats['total_policies']}")
print(f"Enabled: {stats['enabled_policies']}")
print(f"Disabled: {stats['disabled_policies']}")
print(f"Total deleted: {stats['total_memories_deleted']}")
```

---

## Complete Example

```python
from hippocampai.versioning import MemoryVersionControl, ChangeType
from hippocampai.retention import RetentionPolicyManager
from hippocampai.vector.qdrant_store import QdrantStore

# 1. Initialize
version_control = MemoryVersionControl()
qdrant = QdrantStore(url="http://localhost:6333")
retention_mgr = RetentionPolicyManager(qdrant_store=qdrant)

# 2. Create memory with version tracking
memory_data = {
    "id": "mem_001",
    "text": "I started learning Python",
    "user_id": "user_001",
    "type": "event",
    "importance": 6.0,
    "tags": ["learning"]
}

v1 = version_control.create_version(
    memory_id="mem_001",
    data=memory_data,
    created_by="user_001",
    change_summary="Created"
)

# Log creation
version_control.add_audit_entry(
    memory_id="mem_001",
    change_type=ChangeType.CREATED,
    user_id="user_001"
)

# 3. Update memory
memory_data["text"] = "I completed Python basics course"
memory_data["importance"] = 7.0
memory_data["tags"] = ["learning", "achievement"]

v2 = version_control.create_version(
    memory_id="mem_001",
    data=memory_data,
    created_by="user_001",
    change_summary="Completed course"
)

# Log update
version_control.add_audit_entry(
    memory_id="mem_001",
    change_type=ChangeType.UPDATED,
    user_id="user_001",
    changes={"importance": {"old": 6.0, "new": 7.0}}
)

# 4. Compare versions
diff = version_control.compare_versions("mem_001", 1, 2)
if diff['text_diff']:
    print("Text changed:")
    print(f"  Added {diff['text_diff']['added_lines']} lines")
    print(f"  Removed {diff['text_diff']['removed_lines']} lines")

# 5. Create retention policy
policy = retention_mgr.create_policy(
    name="Clean old events",
    retention_days=90,
    memory_type="event",
    min_importance=8.0,
    tags_to_preserve=["achievement"],
    enabled=True
)

# 6. Check if memory would be deleted
# (This memory would be preserved by the "achievement" tag)
would_delete = policy.should_delete(memory_data)
print(f"Would delete: {would_delete}")

# 7. View audit trail
entries = version_control.get_audit_trail(memory_id="mem_001")
print(f"\nAudit trail ({len(entries)} entries):")
for entry in entries:
    print(f"  {entry.timestamp}: {entry.change_type.value}")
```

---

## Best Practices

### 1. Version Control

- **Create versions on every update** - Don't skip versions
- **Use descriptive change summaries** - Help future debugging
- **Set appropriate max_versions** - Balance history vs storage
- **Record created_by** - Track accountability

### 2. Audit Logging

- **Log all operations** - Not just updates
- **Include metadata** - Source, IP, device, etc.
- **Use appropriate change types** - Be specific
- **Clean old entries periodically** - Prevent bloat

### 3. Retention Policies

- **Start with dry runs** - Test before applying
- **Use multiple preservation rules** - Importance, access, tags
- **Monitor expiring memories** - Warn users before deletion
- **Review policies regularly** - Adjust based on usage

### 4. Rollback Safety

- **Always create backup before rollback**
- **Log all rollbacks** - Track who and why
- **Test rollback data** - Validate before applying
- **Communicate with users** - If rolling back their changes

---

## Performance Considerations

| Feature | Storage Impact | Performance Impact |
|---------|---------------|-------------------|
| **Version History** | +50-100% per version | Minimal |
| **Audit Logs** | +10-20% | Minimal |
| **Retention Policies** | Reduces storage | Moderate (during cleanup) |
| **Text Diffs** | None (computed) | Low (cached) |

---

## Support

For questions or issues:

- Main README: `README.md`
- API Reference: `docs/API_COMPLETE_REFERENCE.md`
- Search Guide: `docs/SEARCH_ENHANCEMENTS_GUIDE.md`

---

**Generated**: 2026-02-11
**Version**: 0.2.5
