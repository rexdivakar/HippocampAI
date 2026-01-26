// Memory types
export interface Memory {
  id: string;
  text: string;
  user_id: string;
  type: 'fact' | 'preference' | 'goal' | 'habit' | 'event' | 'context';
  importance: number;
  confidence: number;
  tags: string[];
  embedding?: number[];
  created_at: string;
  updated_at: string;
  last_accessed_at?: string;
  access_count: number;
  expires_at?: string;
  metadata?: Record<string, any>;
  session_id?: string;
  agent_id?: string;
}

export interface RetrievalResult {
  memory: Memory;
  score: number;
  rank: number;
}

// Health monitoring types
export interface HealthScore {
  overall_score: number;
  status: 'excellent' | 'good' | 'fair' | 'poor' | 'critical';
  freshness_score: number;
  diversity_score: number;
  consistency_score: number;
  coverage_score: number;
  engagement_score: number;
  issues: HealthIssue[];
  recommendations: string[];
}

export interface HealthIssue {
  severity: 'low' | 'medium' | 'high' | 'critical';
  category: string;
  message: string;
  affected_memory_ids: string[];
  suggestions: string[];
}

// Collaboration types
export interface CollaborationSpace {
  id: string;
  name: string;
  description?: string;
  owner_agent_id: string;
  collaborators: string[];
  permissions: Record<string, string[]>;
  memory_ids: string[];
  tags: string[];
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface CollaborationEvent {
  id: string;
  space_id: string;
  agent_id: string;
  event_type: string;
  data: Record<string, any>;
  timestamp: string;
}

export interface Notification {
  id: string;
  type: string;
  priority: 'low' | 'medium' | 'high';
  title: string;
  message: string;
  data: Record<string, any>;
  is_read: boolean;
  created_at: string;
}

// Prediction types
export interface TemporalPattern {
  pattern_type: 'daily' | 'weekly' | 'interval';
  description: string;
  frequency: number;
  confidence: number;
  regularity_score: number;
  occurrences_count: number;
  next_predicted?: string;
}

export interface Anomaly {
  id: string;
  title: string;
  description: string;
  anomaly_type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  expected_behavior: string;
  actual_behavior: string;
  suggestions: string[];
  detected_at: string;
}

export interface Recommendation {
  id: string;
  type: string;
  priority: number;
  title: string;
  reason: string;
  action: string;
  confidence: number;
  created_at: string;
}

export interface Forecast {
  metric: string;
  horizon: string;
  forecast_date: string;
  predicted_value: number;
  confidence_interval: {
    lower: number;
    upper: number;
  };
  confidence: number;
  method: string;
  historical_data: Array<{
    date: string;
    value: number;
  }>;
}

// Agent types
export interface Agent {
  id: string;
  name: string;
  user_id: string;
  role: string;
  capabilities: string[];
  active: boolean;
  created_at: string;
  metadata?: Record<string, any>;
}

// Session types
export interface Session {
  id: string;
  user_id: string;
  agent_id?: string;
  title?: string;
  status: 'active' | 'completed' | 'archived';
  message_count: number;
  memory_count: number;
  started_at: string;
  ended_at?: string;
  metadata?: Record<string, any>;
}

// Filter types
export interface MemoryFilters {
  types?: string[];
  tags?: string[];
  minImportance?: number;
  maxImportance?: number;
  startDate?: string;
  endDate?: string;
  sessionId?: string;
  agentId?: string;
  searchText?: string;
}

// WebSocket event types
export type WebSocketEventType =
  | 'memory:created'
  | 'memory:updated'
  | 'memory:deleted'
  | 'collaboration:event'
  | 'agent:notification'
  | 'health:alert'
  | 'health:score_changed'
  | 'prediction:pattern'
  | 'prediction:anomaly'
  | 'healing:action';

export interface WebSocketMessage<T = any> {
  event: WebSocketEventType;
  data: T;
  timestamp: string;
}

// ============================================================================
// BI-TEMPORAL FACT TYPES
// ============================================================================

export type FactStatus = 'active' | 'superseded' | 'retracted';

export interface BiTemporalFact {
  id: string;
  fact_id: string;
  text: string;
  user_id: string;
  entity_id?: string;
  property_name?: string;
  event_time: string;
  valid_from: string;
  valid_to?: string;
  system_time: string;
  status: FactStatus;
  superseded_by?: string;
  supersedes?: string;
  confidence: number;
  source: string;
  metadata: Record<string, any>;
}

export interface BiTemporalQueryResult {
  facts: BiTemporalFact[];
  query_time: string;
  as_of_system_time?: string;
  as_of_valid_time?: string;
}

// ============================================================================
// CONTEXT ASSEMBLY TYPES
// ============================================================================

export interface ContextConstraints {
  token_budget: number;
  max_items: number;
  recency_bias: number;
  entity_focus?: string;
  type_filter?: string[];
  min_relevance: number;
  allow_summaries: boolean;
  include_citations: boolean;
  deduplicate: boolean;
  time_range_days?: number;
}

export interface SelectedItem {
  memory_id: string;
  text: string;
  memory_type: string;
  relevance_score: number;
  importance: number;
  created_at: string;
  token_count: number;
  tags: string[];
  metadata: Record<string, any>;
}

export interface DroppedItem {
  memory_id: string;
  text: string;
  reason: 'token_budget' | 'low_relevance' | 'duplicate' | 'max_items' | 'type_filter';
  relevance_score: number;
}

export interface ContextPack {
  final_context_text: string;
  citations: string[];
  selected_items: SelectedItem[];
  dropped_items: DroppedItem[];
  total_tokens: number;
  query: string;
  user_id: string;
  session_id?: string;
  constraints: ContextConstraints;
  assembled_at: string;
  metadata: Record<string, any>;
}

// ============================================================================
// CUSTOM SCHEMA TYPES
// ============================================================================

export interface AttributeDefinition {
  name: string;
  type: 'string' | 'integer' | 'float' | 'boolean' | 'datetime' | 'list' | 'dict';
  required: boolean;
  description?: string;
  default_value?: any;
  enum_values?: string[];
  min_length?: number;
  max_length?: number;
  min_value?: number;
  max_value?: number;
}

export interface EntityTypeDefinition {
  name: string;
  description?: string;
  attributes: AttributeDefinition[];
  parent_type?: string;
}

export interface RelationshipTypeDefinition {
  name: string;
  description?: string;
  source_types: string[];
  target_types: string[];
  attributes: AttributeDefinition[];
  bidirectional: boolean;
}

export interface SchemaDefinition {
  name: string;
  version: string;
  description?: string;
  entity_types: EntityTypeDefinition[];
  relationship_types: RelationshipTypeDefinition[];
}

export interface ValidationError {
  field: string;
  message: string;
  value?: any;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: string[];
}

// ============================================================================
// BENCHMARK TYPES
// ============================================================================

export interface BenchmarkResult {
  name: string;
  operations: number;
  total_time_ms: number;
  ops_per_second: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  latency_p99_ms: number;
  latency_min_ms: number;
  latency_max_ms: number;
  errors: number;
  metadata: Record<string, any>;
}

export interface BenchmarkSuite {
  name: string;
  timestamp: string;
  results: BenchmarkResult[];
  system_info: Record<string, any>;
}
