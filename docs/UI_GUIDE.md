# HippocampAI Dashboard - Complete UI Guide

**Version**: v0.5.0
**Last Updated**: 2026-02-11

---

## Table of Contents

- [Getting Started](#getting-started)
- [Navigation](#navigation)
- [Primary Pages](#primary-pages)
  - [Dashboard](#1-dashboard)
  - [Memories](#2-memories)
  - [Sleep Phase](#3-sleep-phase-consolidation)
- [Visualize Pages](#visualize-pages)
  - [Timeline](#4-timeline)
  - [Graph View](#5-graph-view-knowledge-graph)
  - [Clusters](#6-clusters-memory-islands)
  - [Heatmap](#7-heatmap)
  - [Concept Growth](#8-concept-growth)
- [Analyze Pages](#analyze-pages)
  - [Usage Trace](#9-usage-trace-replay)
  - [Health](#10-health)
  - [Analytics](#11-analytics)
  - [Observability](#12-observability)
  - [Bi-Temporal Facts](#13-bi-temporal-facts)
  - [Context Assembly](#14-context-assembly)
  - [Custom Schema](#15-custom-schema)
  - [Memory Classifier](#16-memory-classifier)
  - [Feedback](#17-relevance-feedback)
  - [Procedural Memory](#18-procedural-memory)
- [Manage Pages](#manage-pages)
  - [Clean-Up](#19-clean-up-hygiene)
  - [Collaboration](#20-collaboration)
  - [Agents](#21-agent-memory)
  - [Policies](#22-storage-policies)
  - [Triggers](#23-memory-triggers)
  - [Embedding Migration](#24-embedding-migration)

---

## Getting Started

### Login

Navigate to the root URL of your HippocampAI instance. You will see the login page with:

- **Session ID**: Enter your user/session identifier
- **API Key** (optional): For authenticated deployments
- **Try Demo Mode**: Loads sample data to explore the UI without a backend
- **Create New Session**: Register a new user session

After authentication, you are redirected to the Dashboard.

### Requirements

- A running HippocampAI backend (API server at port 8000)
- The frontend dev server (`npm run dev`) or a production build served via Nginx/CDN
- A modern browser (Chrome, Firefox, Safari, Edge)

---

## Navigation

The top navigation bar has four sections:

| Section | Pages |
|---------|-------|
| **Primary** | Dashboard, Memories, Sleep Phase |
| **Visualize** (dropdown) | Timeline, Graph View, Clusters, Heatmap, Concept Growth |
| **Analyze** (dropdown) | Usage Trace, Health, Analytics, Observability, Bi-Temporal Facts, Context Assembly, Custom Schema, Memory Classifier, Feedback, Procedural Memory |
| **Manage** (dropdown) | Clean-Up, Collaboration, Agents, Policies, Triggers, Embedding Migration |

---

## Primary Pages

### 1. Dashboard

**Route**: `/dashboard`

The main overview hub showing a summary of your entire memory system.

**What you can do:**

- **View KPI cards**: Total Memories, Entities, Concepts, Tags, Sleep Cycles, Memory Health score
- **See recent activity**: A live feed of the last 10 memory operations (creates, updates, deletes, retrievals)
- **Check Sleep Phase status**: See if consolidation is active, when it last ran, and mini stats (reviewed, promoted, archived, synthesized)
- **Trigger consolidation**: Click "Run Now" to manually start a sleep phase cycle, or "Dry Run" to preview without changes
- **View Memory Islands**: Top 3 semantic clusters with coherence scores
- **Monitor Knowledge Graph**: Entity, concept, and connection counts at a glance
- **Check Memory Hygiene**: Potential duplicates, uncategorized memories, archived count
- **Add new memory**: Click the "New Memory" button to open the creation modal

### 2. Memories

**Route**: `/memories`

The master-detail memory management interface.

**What you can do:**

- **Browse memories**: Scrollable list on the left, detailed view on the right
- **Search**: Full-text search across memory content and tags
- **Filter**: Advanced filtering by:
  - Memory type (fact, preference, goal, habit, event, context)
  - Importance range (min/max)
  - Date range (start/end)
  - Tags (multi-select)
- **Change view density**: Comfortable, Compact, or Ultra-compact layouts
- **Create memories**: Click "Add Memory" to open the creation modal with fields for text, type, importance, tags, and metadata
- **Edit memories**: Select a memory and click edit to modify its content, type, importance, or tags
- **Delete memories**: Remove memories with a confirmation dialog
- **Share memories**: Generate shareable links for specific memories
- **Rate memories**: Click the thumbs up/down/partial buttons on any memory card to submit relevance feedback
- **Keyboard navigation**: Use arrow keys to navigate the memory list
- **Real-time updates**: WebSocket integration shows new/updated/deleted memories in real time

**Classic view** (`/memories/classic`): An alternative grid/card layout available via the route

### 3. Sleep Phase (Consolidation)

**Route**: `/sleep-phase`

Memory consolidation management with 4 tabs.

**Run Tab:**
- View consolidation stats (Total Runs, Promoted, Synthesized, Archived)
- See the last consolidation result with detailed metrics
- Click "Dry Run" to preview what consolidation would do
- Click "Run Now" to execute a full consolidation cycle
- Read "How It Works" for a 3-step process explanation

**History Tab:**
- Browse all past consolidation runs
- Click any run to see detailed results
- Each run shows: reviewed count, promoted count, synthesized count, archived count, duration

**Settings Tab:**
- View current consolidation configuration
- See session statistics (users, sessions, memories, deleted)
- Check environment variables

**Danger Zone Tab:**
- Wipe user data with a confirmation flow
- Enter an optional reason
- Soft-delete behavior (data can be recovered)

---

## Visualize Pages

### 4. Timeline

**Route**: `/timeline`

Chronological view of memory evolution over time.

**What you can do:**
- **View stats**: Memories learned today, reinforced memories (high access), decayed memories (low importance), active sessions
- **Browse timeline**: Memories grouped by date with visual indicators
  - Green dots: Reinforced (frequently accessed)
  - Blue dots: High importance
  - Red dots: Decayed (low importance)
  - Gray dots: Normal
- **Check retention curves**: Strong, Medium, and Weak retention percentages

### 5. Graph View (Knowledge Graph)

**Route**: `/graph`

Interactive force-directed graph of knowledge relationships.

**What you can do:**
- **Explore the graph**: Pan, zoom, and click nodes to see connections
- **Filter by type**: Show only Memories, Entities, Concepts, or Tags
- **Search nodes**: Find specific entities or concepts
- **Zoom controls**: Zoom in/out and center the view
- **Export**: Download the graph as JSON
- **View node details**: Click any node to see its type, connections, content, importance, and views
- **Node color legend**: Each type has a distinct color for easy identification

### 6. Clusters (Memory Islands)

**Route**: `/cluster`

Semantic clustering of memories into related groups.

**What you can do:**
- **View overview**: Total clusters, memories, outliers, and duplicates
- **Browse clusters**: Click a cluster on the left to see its memories on the right
- **Identify outliers**: Memories with no tags or very low importance are flagged
- **Find duplicates**: Pairs of memories with >70% text similarity are highlighted
- **Analyze cluster quality**: Each cluster shows average importance and memory count

### 7. Heatmap

**Route**: `/heatmap`

GitHub-style activity visualization.

**What you can do:**
- **Select time range**: Week, Month, or Year view
- **View creation heatmap**: Green intensity grid showing when memories were created
- **View retrieval heatmap**: Blue intensity grid showing memory access patterns
- **Check importance distribution**: Bar chart of average importance over recent days
- **Hover for details**: Tooltip shows exact count and date for any cell

### 8. Concept Growth

**Route**: `/concept-growth`

Track how knowledge evolves over time across topics.

**What you can do:**
- **View top growing concepts**: 6 cards showing the fastest-growing topics with growth percentages, memory counts, importance, and relationship links
- **Drill into details**: Click a concept to see growth timeline charts (memory count, importance, relationships over time)
- **Browse all concepts table**: Sortable table with concept name, total memories, growth %, average importance, and relationship count

---

## Analyze Pages

### 9. Usage Trace (Replay)

**Route**: `/replay`

Track how memories are retrieved and used in conversations.

**What you can do:**
- **Search memories**: Find specific memories to trace
- **View usage stats**: Total retrievals, retrieval score
- **See retrieval events**: When each retrieval happened, which session, and the score
- **Check agent usage**: Which AI agents used this memory and how often
- **Reranker contribution**: See how the reranker affected retrieval scores
- **Impact analysis**: Read a narrative explanation of the memory's influence on responses

### 10. Health

**Route**: `/health`

Memory system quality monitoring.

**What you can do:**
- **Check overall health score**: 0-100 score with status badge (Excellent/Good/Fair/Needs Attention)
- **View distribution charts**:
  - Type distribution (pie chart)
  - Importance distribution (bar chart, ranges 0-10)
  - Confidence distribution (bar chart, 20% increments)
  - Top 10 tags (horizontal bar chart)
- **See issues and recommendations**: If health score detects problems, they are listed with suggested fixes
- **Browse recently accessed memories**: Last 5 memories with details

### 11. Analytics

**Route**: `/analytics`

Advanced insights and usage patterns.

**What you can do:**
- **Select time range**: 7, 30, or 90 days
- **View key metrics**: Memory velocity (memories/week), retention rate, average lifespan, lifecycle breakdown
- **Read key insights**: AI-generated observations about your memory patterns (creation rate, stale %, importance correlations, tag anomalies)
- **Explore charts**:
  - Memory creation timeline (line chart)
  - Engagement trend (area chart)
  - Type evolution (stacked area chart)
  - Importance vs access correlation (scatter plot)
  - Top performing tags (horizontal bar chart)
- **Lifecycle breakdown**: See new, active, and stale memory percentages

### 12. Observability

**Route**: `/observability`

System performance and infrastructure monitoring.

**What you can do:**
- **Toggle auto-refresh**: 5-second automatic refresh
- **Monitor health metrics**: Qdrant status, vector count, query latency, memory usage, index status, replication
- **View performance charts**:
  - Requests per second (last 60 seconds)
  - Latency distribution histogram
  - Memory growth over 30 days
- **Check system resources**: Database details (collections, vector dimensions, distance metric), storage breakdown, P50/P95/P99 latencies, cache hit rate
- **Browse ingestion logs**: Recent CREATE/UPDATE/DELETE operations with timestamps, memory IDs, status, and latency

### 13. Bi-Temporal Facts

**Route**: `/bitemporal`

Track facts across two time dimensions: when something was true in the real world vs. when it was recorded.

### 14. Context Assembly

**Route**: `/context`

Compose and preview context windows by combining memories, rules, and metadata for prompt engineering.

### 15. Custom Schema

**Route**: `/schema`

Define and validate custom entity types and relationships.

**What you can do:**
- **View schema definition**: Entity types (person, organization, location) with their attributes and types
- **See relationship types**: How entities connect (works_at, located_in) with source/target mappings
- **Validate data**: Select an entity type, paste JSON, and validate against the schema
- **View validation results**: Errors and warnings with field-level details
- **Use example templates**: Pre-built valid and invalid examples for testing

### 16. Memory Classifier

**Route**: `/classifier`

Automatic memory classification and categorization interface.

### 17. Relevance Feedback

**Route**: `/feedback`

Track how relevant retrieved memories are to user queries.

**What you can do:**
- **View feedback stats**: 4 cards showing Total Feedback, Relevant count (green), Not Relevant count (red), Partially Relevant count (yellow)
- **Analyze distribution**: Pie chart showing the breakdown of all feedback types with percentages
- **Refresh data**: Manual refresh button to reload the latest stats

**How feedback is collected:** Feedback is submitted inline from the Memories page. When viewing any memory, click the thumbs up (relevant), circle dot (partially relevant), or thumbs down (not relevant) buttons. The stats on this page aggregate all that feedback.

### 18. Procedural Memory

**Route**: `/procedural`

Extract, manage, and inject behavioral rules from interaction patterns.

**What you can do:**
- **View rule stats**: Total Rules, Active Rules, Average Confidence, Average Success Rate
- **Browse rules**: Each rule shows its text, active/inactive badge, confidence bar, and success rate bar
- **Rate rules**: Click thumbs up/down on any rule to update its success rate
- **Extract rules**: Click "Extract" to open a modal where you paste interaction history (one per line). The system extracts behavioral patterns into rules.
- **Consolidate rules**: Click "Consolidate" to merge similar or redundant rules automatically
- **Inject rules**: Click "Inject" to open a modal where you enter a base prompt. The system injects relevant procedural rules and shows the resulting enhanced prompt with a count of injected rules.

---

## Manage Pages

### 19. Clean-Up (Hygiene)

**Route**: `/hygiene`

Memory quality maintenance and cleanup tools.

**What you can do:**
- **View overview**: Total Memories, Duplicates Found, Low Importance count, Old & Unused count
- **Deduplicate**: Set similarity threshold (0.5-1.0) and remove duplicate memories
- **Merge similar**: Automatically merge high-similarity memories while preserving unique information
- **Remove low importance**: Set importance threshold (1-5) and remove memories below it
- **Auto-delete old**: Set retention period (30-365 days) and delete old unused memories
- **Privacy protection**: Detect and remove PII, IP addresses, and sensitive logs
- **Set retention rules**: Configure rules to keep high importance, frequently accessed, starred, or recent memories
- **Preview duplicates**: See the top 3 duplicate pairs before taking action

### 20. Collaboration

**Route**: `/collaboration`

Shared memory spaces for team coordination. (Coming soon - placeholder UI)

### 21. Agent Memory

**Route**: `/agents`

Multi-agent memory namespaces and isolation.

**What you can do:**
- **Browse agent namespaces**: 6 pre-configured agent cards (Research, Coding, EdTech, DevOps, Personal Assistant, General)
- **View per-agent stats**: Total memories, average importance, total access count, 7-day activity
- **Filter memories by agent**: Click an agent card to see only its memories
- **Compare performance**: Bar charts comparing memory count and average importance across agents

### 22. Storage Policies

**Route**: `/policies`

Configure rules for automatic memory storage decisions.

**What you can do:**
- **View policy stats**: Active Policies, Total Memories, Auto-Stored count, Manual Overrides
- **Configure 5 built-in policies**:
  - Important Facts (importance >= 7)
  - User Preferences (preference keywords detected)
  - Events & Milestones (date/event patterns)
  - Knowledge Extraction (definitions, explanations)
  - Auto-Synthesis (merge related memories)
- **Enable/disable policies**: Toggle each policy on or off
- **Adjust priority**: Move policies up or down in priority order
- **Review recent decisions**: See the last 5 memory storage decisions with the policy that triggered them

### 23. Memory Triggers

**Route**: `/triggers`

Automate actions when memory events occur.

**What you can do:**
- **View trigger stats**: Total Triggers, Active count, Total Fires, Events Monitored
- **Browse triggers**: Cards showing trigger name, event badge, action badge, fire count, and delete button
- **Create triggers**: Click "New Trigger" to open the creation modal:
  - **Name**: Give the trigger a descriptive name
  - **Event**: Select from: On Remember, On Recall, On Update, On Delete, On Conflict, On Expire
  - **Conditions** (optional): Add field/operator/value conditions. Operators: =, >, <, contains, matches
  - **Action**: Choose Webhook (sends HTTP POST to a URL), Log (writes to system log), or WebSocket (pushes real-time notification)
  - **Webhook URL**: If webhook action is selected, enter the target URL
- **Delete triggers**: Click the trash icon on any trigger card (with confirmation)
- **View fire history**: Click a trigger to see its history panel showing success/failure badges, timestamps, event details, memory IDs, and error messages

### 24. Embedding Migration

**Route**: `/migrations`

Migrate memory embeddings to a new model. (Admin feature)

**What you can do:**
- **Track existing migration**: Enter a migration ID to monitor an in-progress migration
- **Start new migration**:
  - Enter the new model name (e.g., `text-embedding-3-large`)
  - Optionally set vector dimensions (default: 384)
  - Confirm before starting (warning about duration and irreversibility)
- **Monitor progress**: Live progress bar showing migrated/total memories, polling every 3 seconds
- **View model info**: Side-by-side display of old model and new model names
- **Cancel migration**: Stop an in-progress migration with confirmation
- **See completion summary**: Green success card with final counts, or red failure card with error details

---

## Tips & Best Practices

1. **Start with the Dashboard** to get an overview of your memory system health
2. **Use the Memories page** as your primary workspace for creating, editing, and rating memories
3. **Run Sleep Phase** periodically to consolidate and clean up memories automatically
4. **Set up Triggers** to automate workflows (e.g., webhook on new memory creation)
5. **Check Health and Analytics** regularly to identify stale memories and usage patterns
6. **Use Procedural Memory** to extract behavioral rules from your interaction history and inject them into prompts
7. **Rate memories** using the inline feedback buttons to improve retrieval quality over time
8. **Use the Graph View** to discover unexpected connections between concepts
9. **Clean up with Hygiene** to remove duplicates and low-value memories periodically
10. **Monitor Observability** for system performance issues and latency spikes

---

**HippocampAI** - Autonomous memory engine with hybrid retrieval, knowledge graphs, and intelligent consolidation.
