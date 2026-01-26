# HippocampAI - New Features Guide

This guide covers the latest enhancements to HippocampAI: **Collaboration**, **Predictive Analytics**, **Auto-Healing**, and **Conversation UI**.

---

## 🤝 **Multi-Agent Collaboration**

### Overview
Enable multiple agents to collaborate in shared memory spaces with fine-grained permissions, real-time notifications, and conflict resolution.

### Key Features

#### 1. Shared Memory Spaces
Create collaborative environments where multiple agents can work together:

```python
from hippocampai.multiagent.collaboration import CollaborationManager
from hippocampai.models.agent import PermissionType

# Initialize collaboration manager
collab_manager = CollaborationManager()

# Create shared space
space = collab_manager.create_space(
    name="Research Project",
    owner_agent_id=agent1_id,
    description="Collaborative research space",
    tags=["research", "ai"]
)

# Add collaborators with specific permissions
collab_manager.add_collaborator(
    space_id=space.id,
    agent_id=agent2_id,
    permissions=[PermissionType.READ, PermissionType.WRITE],
    inviter_id=agent1_id
)
```

#### 2. Permission System
Fine-grained control over what agents can do:

- **READ**: View memories in the space
- **WRITE**: Create and update memories
- **SHARE**: Invite other agents
- **DELETE**: Remove memories

```python
# Check permissions
can_write = space.has_permission(agent_id, PermissionType.WRITE)

# Update permissions
collab_manager.update_permissions(
    space_id=space.id,
    agent_id=agent2_id,
    permissions=[PermissionType.READ, PermissionType.WRITE, PermissionType.DELETE],
    updater_id=owner_id
)
```

#### 3. Real-Time Events
Track all collaboration activity:

```python
# Get recent events
events = collab_manager.get_space_events(space_id, limit=50)

for event in events:
    print(f"{event.event_type}: {event.data}")
```

**Event Types:**
- `MEMORY_ADDED` - New memory added to space
- `MEMORY_UPDATED` - Memory modified
- `MEMORY_DELETED` - Memory removed
- `AGENT_JOINED` - New collaborator added
- `AGENT_LEFT` - Collaborator removed
- `PERMISSION_CHANGED` - Permissions updated

#### 4. Conflict Detection & Resolution
Automatically detect and resolve conflicts:

```python
# Detect conflict
conflict = collab_manager.detect_conflict(
    space_id=space.id,
    memory_id=memory_id,
    conflicting_versions=[version1, version2],
    conflict_type=ConflictType.CONCURRENT_UPDATE
)

# Resolve conflict
collab_manager.resolve_conflict(
    conflict_id=conflict.id,
    resolved_version=merged_version,
    resolved_by=agent_id,
    strategy=ResolutionStrategy.LATEST_WINS
)
```

**Resolution Strategies:**
- `LATEST_WINS` - Most recent change wins
- `MERGE_CHANGES` - Attempt automatic merge
- `MANUAL` - Requires manual resolution
- `OWNER_WINS` - Owner's version takes precedence
- `HIGHEST_IMPORTANCE` - Most important version wins

#### 5. Notifications
Stay informed about collaboration activity:

```python
# Get notifications
notifications = collab_manager.get_notifications(
    agent_id=agent_id,
    unread_only=True
)

# Mark as read
collab_manager.mark_notification_read(agent_id, notification_id)
```

**Notification Types:**
- `MEMORY_CHANGE` - Memory was modified
- `PERMISSION_GRANTED` - New permissions granted
- `CONFLICT_DETECTED` - Conflict needs resolution
- `SPACE_INVITATION` - Invited to new space

### Use Cases
- **Team Research**: Multiple researchers collaborating on findings
- **Content Creation**: Writers and editors working together
- **Data Analysis**: Analysts sharing insights
- **Customer Support**: Agents sharing customer context
- **Project Management**: Teams tracking project memories

---

## 🔮 **Predictive Analytics**

### Overview
Proactively predict future memory patterns, detect anomalies, generate recommendations, and forecast metrics.

### Key Features

#### 1. Pattern Prediction
Predict when patterns will occur next:

```python
from hippocampai.pipeline.predictive_analytics import PredictiveAnalyticsEngine
from hippocampai.pipeline.temporal_analytics import TemporalAnalytics

# Initialize engines
temporal = TemporalAnalytics()
predictive = PredictiveAnalyticsEngine(temporal)

# Detect patterns
patterns = temporal.detect_temporal_patterns(memories, min_occurrences=3)

# Predict next occurrence
for pattern in patterns:
    prediction = predictive.predict_next_occurrence(
        user_id=user_id,
        pattern=pattern
    )
    print(f"Next occurrence: {prediction.predicted_datetime}")
    print(f"Confidence: {prediction.confidence:.0%}")
```

**Pattern Types Detected:**
- **Daily**: Activity at same time each day
- **Weekly**: Activity on specific day of week
- **Interval**: Custom recurring intervals
- **Custom**: User-specific patterns

#### 2. Anomaly Detection
Detect unusual behavior in real-time:

```python
# Detect anomalies
anomalies = predictive.detect_anomalies(
    user_id=user_id,
    memories=memories,
    lookback_days=30
)

for anomaly in anomalies:
    print(f"{anomaly.title} ({anomaly.severity.value})")
    print(f"Expected: {anomaly.expected_behavior}")
    print(f"Actual: {anomaly.actual_behavior}")
    print(f"Suggestions: {anomaly.suggestions}")
```

**Anomaly Types:**
- `UNUSUAL_ACTIVITY` - Spike or drop in activity
- `MISSING_PATTERN` - Expected pattern didn't occur
- `UNEXPECTED_CONTENT` - Content deviates from norm
- `BEHAVIOR_CHANGE` - User behavior shifted
- `DATA_QUALITY` - Quality issues detected

**Severity Levels:**
- `LOW` - Minor deviation
- `MEDIUM` - Notable deviation
- `HIGH` - Significant deviation
- `CRITICAL` - Urgent attention needed

#### 3. Proactive Recommendations
Get AI-powered suggestions:

```python
# Generate recommendations
recommendations = predictive.generate_recommendations(
    user_id=user_id,
    memories=memories,
    max_recommendations=10
)

for rec in recommendations:
    print(f"[Priority {rec.priority}/10] {rec.title}")
    print(f"Reason: {rec.reason}")
    print(f"Action: {rec.action}")
```

**Recommendation Types:**
- `REMEMBER_THIS` - Suggest creating memory
- `REVIEW_MEMORY` - Suggest reviewing existing
- `CREATE_HABIT` - Pattern suggests habit
- `UPDATE_PREFERENCE` - Update preference
- `CONSOLIDATE_MEMORIES` - Merge similar memories
- `ARCHIVE_MEMORY` - Archive old memory
- `SET_GOAL` - Suggest creating goal

#### 4. Metric Forecasting
Forecast future memory metrics:

```python
from hippocampai.models.prediction import ForecastMetric, ForecastHorizon

# Forecast activity for next week
forecast = predictive.forecast_metric(
    user_id=user_id,
    memories=memories,
    metric=ForecastMetric.ACTIVITY_LEVEL,
    horizon=ForecastHorizon.NEXT_WEEK
)

print(f"Predicted: {forecast.predicted_value:.1f} memories/day")
print(f"Confidence interval: {forecast.confidence_interval}")
```

**Forecastable Metrics:**
- `ACTIVITY_LEVEL` - Memories per time period
- `IMPORTANCE_AVERAGE` - Average importance score
- `TYPE_DISTRIBUTION` - Distribution of types
- `ENGAGEMENT` - User engagement level
- `MEMORY_QUALITY` - Overall quality score

**Forecast Horizons:**
- `NEXT_DAY` - 24 hours ahead
- `NEXT_WEEK` - 7 days ahead
- `NEXT_MONTH` - 30 days ahead
- `NEXT_QUARTER` - 90 days ahead

#### 5. Predictive Insights
High-level insights about memory behavior:

```python
# Generate insights
insights = predictive.generate_predictive_insights(
    user_id=user_id,
    memories=memories
)

for insight in insights:
    print(f"📊 {insight.title}")
    print(f"Type: {insight.insight_type}")
    print(f"Impact: {insight.impact}")
    print(f"Evidence: {insight.evidence}")
```

**Insight Types:**
- `pattern_emerging` - New pattern forming
- `behavior_shift` - Behavior changing
- `opportunity` - Improvement opportunity
- `risk` - Potential issue
- `achievement` - Milestone reached
- `milestone` - Important point reached

### Use Cases
- **Personal Analytics**: Understand your memory patterns
- **Productivity Optimization**: Identify peak performance times
- **Habit Formation**: Turn patterns into habits
- **Anomaly Alerts**: Catch unusual behavior early
- **Proactive Assistance**: AI suggests next actions
- **Forecasting**: Plan based on predicted needs

---

## 🔧 **Auto-Healing**

### Overview
Automatically maintain memory health through cleanup, consolidation, tagging, and optimization.

### Key Features

#### 1. Automatic Cleanup
Remove stale and low-quality memories:

```python
from hippocampai.pipeline.auto_healing import AutoHealingEngine
from hippocampai.models.healing import AutoHealingConfig

# Configure auto-healing
config = AutoHealingConfig(
    user_id=user_id,
    enabled=True,
    auto_cleanup_enabled=True,
    cleanup_threshold_days=90,
    require_user_approval=True,
    max_actions_per_run=50
)

# Run cleanup
report = healing_engine.auto_cleanup(
    user_id=user_id,
    memories=memories,
    config=config,
    dry_run=True  # Test without applying
)

print(f"Actions recommended: {len(report.actions_recommended)}")
print(f"Health improvement: {report.health_after - report.health_before:.1f}")
```

**Cleanup Actions:**
- Archive stale memories (old, unused)
- Delete very low quality memories
- Flag low confidence memories for review
- Update expired memories

#### 2. Smart Consolidation
Merge duplicate and similar memories:

```python
from hippocampai.models.healing import ConsolidationStrategy

# Run consolidation
report = healing_engine.auto_consolidate(
    user_id=user_id,
    memories=memories,
    config=config,
    strategy=ConsolidationStrategy.SEMANTIC_MERGE,
    dry_run=True
)

print(f"Duplicate clusters: {len(report.actions_recommended)}")
print(f"Memories to merge: {report.memories_affected}")
```

**Consolidation Strategies:**
- `SEMANTIC_MERGE` - Merge semantically similar
- `TIME_BASED` - Consolidate by time period
- `TYPE_BASED` - Consolidate by type
- `TAG_BASED` - Consolidate by tags
- `IMPORTANCE_WEIGHTED` - Keep most important

#### 3. Automatic Tagging
AI-powered tag suggestions:

```python
# Generate tag suggestions
tag_actions = healing_engine.auto_tag(memories)

for action in tag_actions:
    memory_id = action.memory_ids[0]
    suggested_tags = action.metadata['suggested_tags']
    print(f"Memory {memory_id}: Add tags {suggested_tags}")
```

**Tag Sources:**
- Memory type (fact, preference, etc.)
- Content keywords
- Context analysis
- Entity recognition
- Category classification

#### 4. Importance Adjustment
Automatically adjust importance based on usage:

```python
# Adjust importance scores
importance_actions = healing_engine.auto_importance_adjustment(
    memories=memories,
    access_weight=0.5  # Balance original vs access-based
)

print(f"Adjustments recommended: {len(importance_actions)}")
```

**Adjustment Factors:**
- Access frequency
- Recency of access
- Time since creation
- User interactions
- Related memory importance

#### 5. Full Health Check
Comprehensive analysis and healing:

```python
# Run full health check
report = healing_engine.run_full_health_check(
    user_id=user_id,
    memories=memories,
    config=config,
    dry_run=True
)

print(f"Health score: {report.health_before:.1f} → {report.health_after:.1f}")
print(f"Total actions: {len(report.actions_recommended)}")
print(f"Summary: {report.summary}")
```

**What It Checks:**
- Memory freshness
- Duplicate detection
- Stale memory identification
- Tag completeness
- Importance accuracy
- Overall quality

#### 6. Scheduled Maintenance
Automate healing with Celery tasks:

```python
from hippocampai.maintenance_tasks import (
    run_health_check_task,
    run_cleanup_task,
    run_deduplication_task
)

# Schedule daily health check at 3 AM
config_dict = config.dict()
run_health_check_task.apply_async(
    args=[user_id, config_dict],
    eta=datetime.now().replace(hour=3, minute=0)
)

# Or use periodic tasks (configured in celery)
# - Daily health check at 3 AM
# - Weekly cleanup on Sunday at 2 AM
# - Daily anomaly detection at 4 AM
```

### Healing Actions

| Action Type | Description | Reversible |
|------------|-------------|------------|
| `DELETE` | Remove memory | No |
| `ARCHIVE` | Archive old memory | Yes |
| `MERGE` | Consolidate duplicates | Yes |
| `UPDATE_TAGS` | Add/modify tags | Yes |
| `ADJUST_IMPORTANCE` | Update importance | Yes |
| `UPDATE_CONFIDENCE` | Adjust confidence | Yes |
| `CONSOLIDATE` | Merge similar memories | Yes |
| `REFRESH` | Update stale memory | Yes |

### Use Cases
- **Data Quality**: Maintain high-quality memory store
- **Storage Optimization**: Reduce duplicates, archive old data
- **Automatic Curation**: AI-curated tag taxonomy
- **Proactive Maintenance**: Prevent quality degradation
- **Cost Optimization**: Reduce storage and compute costs

---

## 💬 **Conversation UI** (Coming Soon)

### Overview
Web-based interface for visualizing and managing sessions, agents, and memories.

### Planned Features

#### 1. Session Management UI
- **Session List**: Browse all conversations
- **Session Timeline**: Visual timeline of messages
- **Entity Sidebar**: See extracted entities in real-time
- **Facts Panel**: View session facts and confidence
- **Summary Card**: Auto-generated session summaries

#### 2. Agent Collaboration Dashboard
- **Space Browser**: Navigate shared memory spaces
- **Activity Feed**: Real-time collaboration events
- **Permission Matrix**: Visual permission management
- **Notification Center**: Centralized notifications
- **Conflict Resolution UI**: Visual conflict resolution

#### 3. Health Dashboard
- **Health Metrics**: Real-time health scores
- **Trend Visualization**: Charts for memory trends
- **Healing Actions**: Review and apply healing actions
- **Recommendation Feed**: Proactive suggestions
- **Anomaly Alerts**: Visual anomaly notifications

#### 4. WebSocket Support
Real-time updates across the UI:

```python
# Server-side (FastAPI)
@app.websocket("/ws/sessions/{session_id}")
async def session_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    # Stream session updates in real-time
```

```javascript
// Client-side (React)
const socket = new WebSocket(`ws://localhost:8000/ws/sessions/${sessionId}`);

socket.onmessage = (event) => {
    const update = JSON.parse(event.data);
    // Update UI in real-time
};
```

### Tech Stack
- **Frontend**: React + TypeScript + TailwindCSS
- **State**: Zustand (lightweight state management)
- **Real-time**: Socket.IO / native WebSocket
- **UI Components**: Headless UI + custom components
- **Charts**: Recharts for visualizations
- **API Client**: Axios with TypeScript types

---

## 📊 **Complete Feature Matrix**

| Feature | Status | Description |
|---------|--------|-------------|
| **Collaboration** | ✅ Ready | Shared spaces, permissions, notifications |
| **Predictive Analytics** | ✅ Ready | Patterns, anomalies, forecasting, recommendations |
| **Auto-Healing** | ✅ Ready | Cleanup, consolidation, tagging, maintenance |
| **Scheduled Tasks** | ✅ Ready | Celery tasks for automation |
| **Conversation UI** | 🚧 Planned | Web interface (backend ready, frontend pending) |
| **WebSocket Support** | 🚧 Planned | Real-time updates |

---

## 🚀 **Getting Started**

### Quick Start: Collaboration

```python
from hippocampai.client import MemoryClient
from hippocampai.multiagent.collaboration import CollaborationManager

# Create agents
client = MemoryClient(user_id="user_id")
agent1 = client.create_agent("Agent 1", role="assistant")
agent2 = client.create_agent("Agent 2", role="assistant")

# Setup collaboration
collab = CollaborationManager()
space = collab.create_space("Project X", owner_agent_id=agent1.id)
collab.add_collaborator(space.id, agent2.id, [PermissionType.READ, PermissionType.WRITE], agent1.id)

# Create shared memory
memory = client.remember("Shared knowledge", agent_id=agent1.id)
collab.add_memory_to_space(space.id, memory.id, agent1.id)
```

### Quick Start: Predictive Analytics

```python
from hippocampai.pipeline.predictive_analytics import PredictiveAnalyticsEngine
from hippocampai.pipeline.temporal_analytics import TemporalAnalytics

# Initialize
temporal = TemporalAnalytics()
predictive = PredictiveAnalyticsEngine(temporal)

# Get memories
memories = client.get_memories()

# Detect patterns
patterns = temporal.detect_temporal_patterns(memories)

# Generate recommendations
recommendations = predictive.generate_recommendations(client.user_id, memories)

# Detect anomalies
anomalies = predictive.detect_anomalies(client.user_id, memories)
```

### Quick Start: Auto-Healing

```python
from hippocampai.pipeline.auto_healing import AutoHealingEngine
from hippocampai.monitoring.memory_health import MemoryHealthMonitor
from hippocampai.models.healing import AutoHealingConfig

# Initialize
health_monitor = MemoryHealthMonitor(embedder)
healing_engine = AutoHealingEngine(health_monitor)

# Configure
config = AutoHealingConfig(
    user_id=client.user_id,
    auto_cleanup_enabled=True,
    auto_dedup_enabled=True,
    auto_tag_enabled=True
)

# Run healing
report = healing_engine.run_full_health_check(
    user_id=client.user_id,
    memories=memories,
    config=config,
    dry_run=True
)

print(f"Health: {report.health_before} → {report.health_after}")
```

---

## 📚 **Examples**

See the `/examples` directory for complete demonstrations:

- **`20_collaboration_demo.py`** - Multi-agent collaboration
- **`21_predictive_analytics_demo.py`** - Predictions and forecasting
- **`22_auto_healing_demo.py`** - Auto-healing and maintenance

---

## 🔗 **API Integration**

### REST API Endpoints (Ready)

All features are available via REST API:

```bash
# Collaboration
POST   /v1/collaboration/spaces              # Create space
GET    /v1/collaboration/spaces/{id}         # Get space
POST   /v1/collaboration/spaces/{id}/members # Add member
GET    /v1/collaboration/spaces/{id}/events  # Get events

# Predictions
POST   /v1/predictions/patterns             # Detect patterns
POST   /v1/predictions/anomalies            # Detect anomalies
POST   /v1/predictions/recommendations      # Get recommendations
POST   /v1/predictions/forecast             # Forecast metrics

# Auto-Healing
POST   /v1/healing/health-check             # Run health check
POST   /v1/healing/cleanup                  # Run cleanup
POST   /v1/healing/consolidate              # Run consolidation
GET    /v1/healing/config                   # Get config
```

---

## 🎯 **Best Practices**

### Collaboration
1. **Permission Design**: Start restrictive, expand as needed
2. **Notification Management**: Filter by priority to avoid overload
3. **Conflict Strategy**: Define resolution strategy upfront
4. **Space Organization**: Create focused spaces per project
5. **Event Monitoring**: Regularly review collaboration events

### Predictive Analytics
1. **Pattern Threshold**: Adjust `min_occurrences` based on data volume
2. **Anomaly Sensitivity**: Tune lookback days for your use case
3. **Recommendation Limits**: Start with top 5-10 recommendations
4. **Forecast Horizon**: Match horizon to planning needs
5. **Confidence Thresholds**: Filter by confidence for high-value insights

### Auto-Healing
1. **Dry Run First**: Always test with `dry_run=True` first
2. **Gradual Rollout**: Start with conservative thresholds
3. **User Approval**: Enable `require_user_approval` for critical actions
4. **Action Limits**: Set reasonable `max_actions_per_run`
5. **Monitor Impact**: Track health scores over time
6. **Schedule Wisely**: Run maintenance during low-activity periods

---

## 📈 **Performance Considerations**

### Collaboration
- Event storage grows linearly with activity
- Notification cleanup recommended for inactive agents
- Space caching improves performance

### Predictive Analytics
- Pattern detection: O(n log n) for n memories
- Anomaly detection: O(n) with baseline calculation
- Forecasting: O(n) for time series analysis
- Consider sampling for very large datasets (>100k memories)

### Auto-Healing
- Health scoring: O(n) for n memories
- Duplicate detection: O(n²) worst case (optimized with embeddings)
- Consolidation: O(k×m) for k clusters of size m
- Run scheduled maintenance during off-peak hours

---

## 🐛 **Troubleshooting**

### Common Issues

**Collaboration:**
```
Error: "Permission denied"
→ Check agent permissions in space
→ Verify agent_id is correct
```

**Predictions:**
```
Error: "Insufficient data"
→ Need minimum 10-20 memories for patterns
→ Need 30+ days of data for anomaly detection
```

**Auto-Healing:**
```
Warning: "Too many actions"
→ Increase max_actions_per_run
→ Or run cleanup separately from dedup
```

---

## 🔮 **Roadmap**

### Phase 1: Backend (Completed ✅)
- [x] Collaboration models and manager
- [x] Predictive analytics engine
- [x] Auto-healing engine
- [x] Scheduled maintenance tasks
- [x] Example demonstrations

### Phase 2: API & WebSocket (Next)
- [ ] REST API endpoints for all features
- [ ] WebSocket support for real-time updates
- [ ] Server-Sent Events (SSE) for notifications
- [ ] API documentation and OpenAPI specs

### Phase 3: Frontend (Upcoming)
- [ ] React-based conversation UI
- [ ] Agent collaboration dashboard
- [ ] Health monitoring dashboard
- [ ] Real-time visualization components

### Phase 4: Advanced Features (Future)
- [ ] Machine learning model training
- [ ] Custom prediction models
- [ ] Advanced conflict resolution strategies
- [ ] Multi-user collaboration spaces

---

## 📞 **Support**

- **Documentation**: See `/docs` directory
- **Examples**: See `/examples` directory
- **Issues**: https://github.com/anthropics/HippocampAI/issues
- **Discussions**: https://github.com/anthropics/HippocampAI/discussions

---

## 📝 **License**

Apache 2.0 License - See LICENSE file for details.
