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
