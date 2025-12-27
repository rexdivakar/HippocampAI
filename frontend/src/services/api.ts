import axios, { AxiosInstance } from 'axios';
import type {
  Memory,
  RetrievalResult,
  HealthScore,
  CollaborationSpace,
  CollaborationEvent,
  Notification,
  TemporalPattern,
  Anomaly,
  Recommendation,
  Forecast,
  BiTemporalFact,
  BiTemporalQueryResult,
  ContextPack,
  ContextConstraints,
} from '../types';

class APIClient {
  private client: AxiosInstance;
  private v1Client: AxiosInstance;

  constructor(baseURL: string = '/api') {
    // Client for /api prefixed routes (consolidation, dashboard, etc.)
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Client for /v1 routes (memory operations - no /api prefix)
    this.v1Client = axios.create({
      baseURL: '',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add auth token interceptor to both clients
    const addAuthInterceptor = (client: AxiosInstance) => {
      client.interceptors.request.use((config) => {
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      });
    };

    addAuthInterceptor(this.client);
    addAuthInterceptor(this.v1Client);
  }

  // ============================================================================
  // MEMORY OPERATIONS (use v1Client - routes at /v1/*)
  // ============================================================================

  async createMemory(data: {
    text: string;
    user_id: string;
    type?: string;
    importance?: number;
    tags?: string[];
    session_id?: string;
  }): Promise<Memory> {
    const response = await this.v1Client.post<Memory>('/v1/memories:remember', data);
    return response.data;
  }

  async recallMemories(data: {
    query: string;
    user_id: string;
    k?: number;
    filters?: Record<string, any>;
  }): Promise<RetrievalResult[]> {
    const response = await this.v1Client.post<RetrievalResult[]>('/v1/memories:recall', data);
    return response.data;
  }

  async getMemories(data: {
    user_id: string;
    filters?: Record<string, any>;
    limit?: number;
  }): Promise<Memory[]> {
    const response = await this.v1Client.post<Memory[]>('/v1/memories:get', data);
    return response.data;
  }

  async updateMemory(memoryId: string, data: Partial<Memory>): Promise<Memory> {
    const response = await this.v1Client.patch<Memory>('/v1/memories:update', {
      memory_id: memoryId,
      ...data,
    });
    return response.data;
  }

  async deleteMemory(memoryId: string, userId?: string): Promise<void> {
    await this.v1Client.delete('/v1/memories:delete', {
      data: { memory_id: memoryId, user_id: userId },
    });
  }

  // ============================================================================
  // HEALTH MONITORING (use v1Client - routes at /v1/*)
  // ============================================================================

  async getHealthScore(userId: string, detailed: boolean = true): Promise<HealthScore> {
    const response = await this.v1Client.post<HealthScore>('/v1/healing/health', {
      user_id: userId,
      detailed,
    });
    return response.data;
  }

  async detectStaleMemories(userId: string, thresholdDays: number = 90) {
    const response = await this.v1Client.post(`/v1/healing/health/stale`, {
      user_id: userId,
      threshold_days: thresholdDays,
    });
    return response.data;
  }

  async detectDuplicates(userId: string, similarityThreshold: number = 0.9) {
    const response = await this.v1Client.post(`/v1/healing/health/duplicates`, {
      user_id: userId,
      similarity_threshold: similarityThreshold,
    });
    return response.data;
  }

  async runAutoCleanup(userId: string, dryRun: boolean = true) {
    const response = await this.v1Client.post('/v1/healing/cleanup', {
      user_id: userId,
      dry_run: dryRun,
    });
    return response.data;
  }

  async runFullHealthCheck(userId: string, dryRun: boolean = true) {
    const response = await this.v1Client.post('/v1/healing/full-check', {
      user_id: userId,
      dry_run: dryRun,
    });
    return response.data;
  }

  // ============================================================================
  // COLLABORATION (use v1Client - routes at /v1/*)
  // ============================================================================

  async createSpace(data: {
    name: string;
    owner_agent_id: string;
    description?: string;
    tags?: string[];
  }): Promise<CollaborationSpace> {
    const response = await this.v1Client.post<{ space: CollaborationSpace }>(
      '/v1/collaboration/spaces',
      data
    );
    return response.data.space;
  }

  async getSpace(spaceId: string): Promise<CollaborationSpace> {
    const response = await this.v1Client.get<CollaborationSpace>(
      `/v1/collaboration/spaces/${spaceId}`
    );
    return response.data;
  }

  async listSpaces(agentId?: string, includeInactive: boolean = false): Promise<CollaborationSpace[]> {
    const response = await this.v1Client.get<{ spaces: CollaborationSpace[] }>(
      '/v1/collaboration/spaces',
      {
        params: { agent_id: agentId, include_inactive: includeInactive },
      }
    );
    return response.data.spaces;
  }

  async getSpaceEvents(spaceId: string, limit: number = 100): Promise<CollaborationEvent[]> {
    const response = await this.v1Client.get<{ events: CollaborationEvent[] }>(
      `/v1/collaboration/spaces/${spaceId}/events`,
      { params: { limit } }
    );
    return response.data.events;
  }

  async getNotifications(
    agentId: string,
    unreadOnly: boolean = false,
    limit: number = 50
  ): Promise<Notification[]> {
    const response = await this.v1Client.get<{ notifications: Notification[] }>(
      `/v1/collaboration/notifications/${agentId}`,
      { params: { unread_only: unreadOnly, limit } }
    );
    return response.data.notifications;
  }

  async markNotificationRead(agentId: string, notificationId: string): Promise<void> {
    await this.v1Client.post(
      `/v1/collaboration/notifications/${agentId}/${notificationId}/read`
    );
  }

  // ============================================================================
  // PREDICTIONS & ANALYTICS (use v1Client - routes at /v1/*)
  // ============================================================================

  async detectPatterns(userId: string, minOccurrences: number = 3): Promise<TemporalPattern[]> {
    const response = await this.v1Client.post<{ patterns: TemporalPattern[] }>(
      '/v1/predictions/patterns',
      { user_id: userId, min_occurrences: minOccurrences }
    );
    return response.data.patterns;
  }

  async detectAnomalies(userId: string, lookbackDays: number = 30): Promise<Anomaly[]> {
    const response = await this.v1Client.post<{ anomalies: Anomaly[] }>(
      '/v1/predictions/anomalies',
      { user_id: userId, lookback_days: lookbackDays }
    );
    return response.data.anomalies;
  }

  async getRecommendations(userId: string, maxRecommendations: number = 10): Promise<Recommendation[]> {
    const response = await this.v1Client.post<{ recommendations: Recommendation[] }>(
      '/v1/predictions/recommendations',
      { user_id: userId, max_recommendations: maxRecommendations }
    );
    return response.data.recommendations;
  }

  async getForecast(userId: string, metric: string, horizon: string): Promise<Forecast> {
    const response = await this.v1Client.post<Forecast>('/v1/predictions/forecast', {
      user_id: userId,
      metric,
      horizon,
    });
    return response.data;
  }

  async analyzeTrends(userId: string, timeWindowDays: number = 30, metric: string = 'activity') {
    const response = await this.v1Client.post('/v1/predictions/trends', {
      user_id: userId,
      time_window_days: timeWindowDays,
      metric,
    });
    return response.data;
  }

  // ============================================================================
  // CONSOLIDATION (SLEEP PHASE)
  // ============================================================================

  async getConsolidationStatus(userId: string) {
    const response = await this.client.get('/consolidation/status', {
      params: { user_id: userId },
    });
    return response.data;
  }

  async getConsolidationStats(userId: string) {
    const response = await this.client.get('/consolidation/stats', {
      params: { user_id: userId },
    });
    return response.data;
  }

  async getConsolidationRuns(userId: string, limit: number = 50) {
    const response = await this.client.get('/consolidation/runs', {
      params: { user_id: userId, limit },
    });
    return response.data;
  }

  async getConsolidationRun(runId: string, userId: string) {
    const response = await this.client.get(`/consolidation/runs/${runId}`, {
      params: { user_id: userId },
    });
    return response.data;
  }

  async getConsolidationConfig(userId: string) {
    const response = await this.client.get('/consolidation/config', {
      params: { user_id: userId },
    });
    return response.data;
  }

  async getLatestConsolidationRun(userId: string) {
    const response = await this.client.get('/consolidation/latest', {
      params: { user_id: userId },
    });
    return response.data;
  }

  async triggerConsolidation(userId: string, dryRun: boolean = true, lookbackHours: number = 24) {
    const response = await this.client.post('/consolidation/trigger', {
      dry_run: dryRun,
      lookback_hours: lookbackHours,
      user_id: userId,
    });
    return response.data;
  }

  // ============================================================================
  // AUTH
  // ============================================================================

  setAuthToken(token: string): void {
    localStorage.setItem('auth_token', token);
  }

  clearAuthToken(): void {
    localStorage.removeItem('auth_token');
  }

  getAuthToken(): string | null {
    return localStorage.getItem('auth_token');
  }

  // ============================================================================
  // BI-TEMPORAL FACTS
  // ============================================================================

  async storeBiTemporalFact(data: {
    text: string;
    user_id: string;
    entity_id?: string;
    property_name?: string;
    valid_from?: string;
    valid_to?: string;
    confidence?: number;
    source?: string;
    metadata?: Record<string, any>;
  }): Promise<BiTemporalFact> {
    const response = await this.v1Client.post<BiTemporalFact>('/v1/bitemporal/facts:store', data);
    return response.data;
  }

  async queryBiTemporalFacts(data: {
    user_id: string;
    entity_id?: string;
    property_name?: string;
    as_of_system_time?: string;
    as_of_valid_time?: string;
    valid_time_start?: string;
    valid_time_end?: string;
    include_superseded?: boolean;
  }): Promise<BiTemporalQueryResult> {
    const response = await this.v1Client.post<BiTemporalQueryResult>('/v1/bitemporal/facts:query', data);
    return response.data;
  }

  async reviseBiTemporalFact(data: {
    original_fact_id: string;
    new_text: string;
    user_id: string;
    valid_from?: string;
    valid_to?: string;
    confidence?: number;
    metadata?: Record<string, any>;
  }): Promise<BiTemporalFact> {
    const response = await this.v1Client.post<BiTemporalFact>('/v1/bitemporal/facts:revise', data);
    return response.data;
  }

  async retractBiTemporalFact(data: {
    fact_id: string;
    user_id: string;
  }): Promise<BiTemporalFact> {
    const response = await this.v1Client.post<BiTemporalFact>('/v1/bitemporal/facts:retract', data);
    return response.data;
  }

  async getBiTemporalFactHistory(factId: string): Promise<BiTemporalFact[]> {
    const response = await this.v1Client.post<{ history: BiTemporalFact[] }>('/v1/bitemporal/facts:history', {
      fact_id: factId,
    });
    return response.data.history;
  }

  async getLatestValidFact(data: {
    entity_id: string;
    property_name: string;
    user_id: string;
  }): Promise<BiTemporalFact | null> {
    const response = await this.v1Client.post<BiTemporalFact | null>('/v1/bitemporal/facts:latest', data);
    return response.data;
  }

  // ============================================================================
  // CONTEXT ASSEMBLY
  // ============================================================================

  async assembleContext(data: {
    user_id: string;
    query: string;
    session_id?: string;
    token_budget?: number;
    max_items?: number;
    recency_bias?: number;
    type_filter?: string[];
    min_relevance?: number;
    include_citations?: boolean;
    deduplicate?: boolean;
  }): Promise<ContextPack> {
    const response = await this.v1Client.post<ContextPack>('/v1/context:assemble', data);
    return response.data;
  }

  async assembleContextText(data: {
    user_id: string;
    query: string;
    session_id?: string;
    token_budget?: number;
    max_items?: number;
  }): Promise<string> {
    const response = await this.v1Client.post<{ context: string }>('/v1/context:assemble/text', data);
    return response.data.context;
  }

  // ============================================================================
  // AGENTIC CLASSIFIER
  // ============================================================================

  async classifyMemory(data: {
    text: string;
    user_id: string;
  }): Promise<{
    memory_type: string;
    confidence: number;
    confidence_level: string;
    reasoning: string;
    alternative_type?: string;
    alternative_confidence?: number;
  }> {
    const response = await this.v1Client.post('/v1/classify', data);
    return response.data;
  }
}

export const apiClient = new APIClient();
