import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import type {
  Memory,
  RetrievalResult,
  HealthScore,
  CollaborationSpace,
  CollaborationEvent,
  CollaborationConflict,
  Notification,
  TemporalPattern,
  Anomaly,
  Recommendation,
  Forecast,
  PredictiveInsight,
  PeakActivity,
  BiTemporalFact,
  BiTemporalQueryResult,
  ContextPack,
  FeedbackType,
  FeedbackResponse,
  AggregatedFeedback,
  FeedbackStats,
  Trigger,
  TriggerEvent,
  TriggerCondition,
  TriggerAction,
  TriggerFireEntry,
  ProceduralRule,
  ProceduralInjectionResult,
  ProspectiveIntent,
  EmbeddingMigration,
  DashboardStats,
  DashboardActivity,
  SessionStats,
  SessionInfo,
  CompactionResult,
  CompactionHistoryEntry,
  ExtractedFact,
  ExtractedEntity,
  EntityRelationship,
  RelationshipNetwork,
  ClusterAnalysis,
  OptimalClusters,
  TemporalAnalysis,
  HealingConfig,
  HealingActionResult,
  StaleMemory,
  DuplicateCluster,
  KnowledgeGap,
  FullHealthCheckResult,
} from '../types';

// Retry configuration
interface RetryConfig {
  maxRetries: number;
  baseDelay: number;
  maxDelay: number;
}

const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxRetries: 3,
  baseDelay: 1000,
  maxDelay: 10000,
};

// Extended config to track retries
interface ExtendedAxiosRequestConfig extends InternalAxiosRequestConfig {
  _retryCount?: number;
  _retryConfig?: RetryConfig;
  signal?: AbortSignal;
}

// Sleep utility for retry delays
const sleep = (ms: number): Promise<void> => new Promise(resolve => setTimeout(resolve, ms));

// Calculate exponential backoff delay
const getRetryDelay = (retryCount: number, config: RetryConfig): number => {
  const delay = config.baseDelay * Math.pow(2, retryCount);
  return Math.min(delay, config.maxDelay);
};

// Check if error is retryable
const isRetryableError = (error: AxiosError): boolean => {
  // Retry on network errors
  if (!error.response) return true;
  
  // Retry on 5xx server errors
  const status = error.response.status;
  if (status >= 500 && status < 600) return true;
  
  // Retry on 429 (rate limited)
  if (status === 429) return true;
  
  // Don't retry on client errors (4xx except 429)
  return false;
};

class APIClient {
  private client: AxiosInstance;
  private v1Client: AxiosInstance;
  private activeRequests: Map<string, AbortController> = new Map();

  constructor(baseURL: string = '/api') {
    // Client for /api prefixed routes (consolidation, dashboard, etc.)
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000, // 30 second timeout
    });

    // Client for /v1 routes (memory operations - no /api prefix)
    this.v1Client = axios.create({
      baseURL: '',
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
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

    // Add retry interceptor to both clients
    const addRetryInterceptor = (client: AxiosInstance) => {
      client.interceptors.response.use(
        (response) => response,
        async (error: AxiosError) => {
          const config = error.config as ExtendedAxiosRequestConfig | undefined;
          
          if (!config) {
            return Promise.reject(error);
          }

          // Check if request was cancelled
          if (axios.isCancel(error)) {
            return Promise.reject(error);
          }

          const retryConfig = config._retryConfig || DEFAULT_RETRY_CONFIG;
          const retryCount = config._retryCount || 0;

          // Check if we should retry
          if (retryCount >= retryConfig.maxRetries || !isRetryableError(error)) {
            return Promise.reject(error);
          }

          // Calculate delay and wait
          const delay = getRetryDelay(retryCount, retryConfig);
          console.log(`Retrying request (attempt ${retryCount + 1}/${retryConfig.maxRetries}) after ${delay}ms`);
          await sleep(delay);

          // Update retry count and retry
          config._retryCount = retryCount + 1;
          return client.request(config);
        }
      );
    };

    addAuthInterceptor(this.client);
    addAuthInterceptor(this.v1Client);
    addRetryInterceptor(this.client);
    addRetryInterceptor(this.v1Client);
  }

  // Create an AbortController for a request, cancelling any previous request with the same key
  createAbortController(requestKey: string): AbortController {
    // Cancel any existing request with the same key
    const existing = this.activeRequests.get(requestKey);
    if (existing) {
      existing.abort();
    }

    // Create new controller
    const controller = new AbortController();
    this.activeRequests.set(requestKey, controller);
    return controller;
  }

  // Clean up an AbortController after request completes
  cleanupAbortController(requestKey: string): void {
    this.activeRequests.delete(requestKey);
  }

  // Cancel a specific request
  cancelRequest(requestKey: string): void {
    const controller = this.activeRequests.get(requestKey);
    if (controller) {
      controller.abort();
      this.activeRequests.delete(requestKey);
    }
  }

  // Cancel all active requests
  cancelAllRequests(): void {
    this.activeRequests.forEach((controller) => controller.abort());
    this.activeRequests.clear();
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

  async detectStaleMemories(userId: string, thresholdDays: number = 90): Promise<{ stale_memories: StaleMemory[]; count: number }> {
    const response = await this.v1Client.post<{ stale_memories: StaleMemory[]; count: number }>(`/v1/healing/health/stale`, {
      user_id: userId,
      threshold_days: thresholdDays,
    });
    return response.data;
  }

  async detectDuplicates(userId: string, similarityThreshold: number = 0.9): Promise<{ clusters: DuplicateCluster[]; total_duplicates: number }> {
    const response = await this.v1Client.post<{ clusters: DuplicateCluster[]; total_duplicates: number }>(`/v1/healing/health/duplicates`, {
      user_id: userId,
      similarity_threshold: similarityThreshold,
    });
    return response.data;
  }

  async detectKnowledgeGapsDetailed(userId: string): Promise<{ gaps: KnowledgeGap[]; count: number }> {
    const response = await this.v1Client.post<{ gaps: KnowledgeGap[]; count: number }>(
      '/v1/healing/health/gaps',
      { user_id: userId }
    );
    return response.data;
  }

  async runAutoCleanup(userId: string, dryRun: boolean = true) {
    const response = await this.v1Client.post('/v1/healing/cleanup', {
      user_id: userId,
      dry_run: dryRun,
    });
    return response.data;
  }

  async runFullHealthCheck(userId: string, dryRun: boolean = true): Promise<FullHealthCheckResult> {
    const response = await this.v1Client.post<FullHealthCheckResult>('/v1/healing/full-check', {
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

  async classifyMemory(
    data: {
      text: string;
      user_id?: string;
    },
    abortSignal?: AbortSignal
  ): Promise<{
    memory_type: string;
    confidence: number;
    confidence_level: string;
    reasoning: string;
    alternative_type?: string;
    alternative_confidence?: number;
  }> {
    const response = await this.v1Client.post('/v1/classify', data, {
      signal: abortSignal,
    });
    return response.data;
  }

  // ============================================================================
  // FEEDBACK
  // ============================================================================

  async submitFeedback(data: {
    memory_id: string;
    user_id: string;
    query?: string;
    feedback_type: FeedbackType;
  }): Promise<FeedbackResponse> {
    const response = await this.v1Client.post<FeedbackResponse>(
      `/v1/memories/${data.memory_id}/feedback`,
      { user_id: data.user_id, query: data.query || '', feedback_type: data.feedback_type }
    );
    return response.data;
  }

  async getMemoryFeedback(memoryId: string): Promise<AggregatedFeedback> {
    const response = await this.v1Client.get<AggregatedFeedback>(
      `/v1/memories/${memoryId}/feedback`
    );
    return response.data;
  }

  async getFeedbackStats(userId: string): Promise<FeedbackStats> {
    const response = await this.v1Client.get<FeedbackStats>('/v1/feedback/stats', {
      params: { user_id: userId },
    });
    return response.data;
  }

  // ============================================================================
  // TRIGGERS
  // ============================================================================

  async createTrigger(data: {
    name: string;
    user_id: string;
    event: TriggerEvent;
    conditions?: TriggerCondition[];
    action?: TriggerAction;
    action_config?: Record<string, any>;
  }): Promise<Trigger> {
    const response = await this.v1Client.post<Trigger>('/v1/triggers', data);
    return response.data;
  }

  async listTriggers(userId: string): Promise<Trigger[]> {
    const response = await this.v1Client.get<Trigger[]>('/v1/triggers', {
      params: { user_id: userId },
    });
    return response.data;
  }

  async deleteTrigger(triggerId: string, userId: string): Promise<void> {
    await this.v1Client.delete(`/v1/triggers/${triggerId}`, {
      params: { user_id: userId },
    });
  }

  async getTriggerHistory(triggerId: string): Promise<TriggerFireEntry[]> {
    const response = await this.v1Client.get<TriggerFireEntry[]>(
      `/v1/triggers/${triggerId}/history`
    );
    return response.data;
  }

  // ============================================================================
  // PROCEDURAL MEMORY
  // ============================================================================

  async getProceduralRules(userId: string): Promise<ProceduralRule[]> {
    const response = await this.v1Client.get<ProceduralRule[]>('/v1/procedural/rules', {
      params: { user_id: userId },
    });
    return response.data;
  }

  async extractProceduralRules(userId: string, interactions: string[]): Promise<ProceduralRule[]> {
    const response = await this.v1Client.post<ProceduralRule[]>(
      '/v1/procedural/extract',
      { user_id: userId, interactions }
    );
    return response.data;
  }

  async injectProceduralRules(userId: string, basePrompt: string): Promise<ProceduralInjectionResult> {
    const response = await this.v1Client.post<ProceduralInjectionResult>(
      '/v1/procedural/inject',
      { user_id: userId, base_prompt: basePrompt }
    );
    return response.data;
  }

  async updateRuleFeedback(ruleId: string, wasSuccessful: boolean): Promise<{ id: string; success_rate: number; updated_at: string }> {
    const response = await this.v1Client.put<{ id: string; success_rate: number; updated_at: string }>(
      `/v1/procedural/rules/${ruleId}/feedback`,
      { was_successful: wasSuccessful }
    );
    return response.data;
  }

  async consolidateRules(userId: string): Promise<{ user_id: string; consolidated_count: number }> {
    const response = await this.v1Client.post<{ user_id: string; consolidated_count: number }>(
      '/v1/procedural/consolidate',
      null,
      { params: { user_id: userId } }
    );
    return response.data;
  }

  // ============================================================================
  // PROSPECTIVE MEMORY
  // ============================================================================

  async createProspectiveIntent(data: {
    user_id: string;
    intent_text: string;
    trigger_type?: string;
    action_description?: string;
    context_keywords?: string[];
    context_pattern?: string;
    similarity_threshold?: number;
    recurrence?: string;
    priority?: number;
    expires_at?: string;
    tags?: string[];
    metadata?: Record<string, any>;
  }): Promise<ProspectiveIntent> {
    const response = await this.v1Client.post<ProspectiveIntent>(
      '/v1/prospective/intents',
      data
    );
    return response.data;
  }

  async parseProspectiveIntent(userId: string, text: string): Promise<ProspectiveIntent> {
    const response = await this.v1Client.post<ProspectiveIntent>(
      '/v1/prospective/intents:parse',
      { user_id: userId, text }
    );
    return response.data;
  }

  async listProspectiveIntents(userId: string, status?: string): Promise<ProspectiveIntent[]> {
    const response = await this.v1Client.get<ProspectiveIntent[]>(
      '/v1/prospective/intents',
      { params: { user_id: userId, ...(status ? { status } : {}) } }
    );
    return response.data;
  }

  async getProspectiveIntent(intentId: string): Promise<ProspectiveIntent> {
    const response = await this.v1Client.get<ProspectiveIntent>(
      `/v1/prospective/intents/${intentId}`
    );
    return response.data;
  }

  async cancelProspectiveIntent(intentId: string, userId: string): Promise<ProspectiveIntent> {
    const response = await this.v1Client.put<ProspectiveIntent>(
      `/v1/prospective/intents/${intentId}/cancel`,
      null,
      { params: { user_id: userId } }
    );
    return response.data;
  }

  async completeProspectiveIntent(intentId: string, userId: string): Promise<ProspectiveIntent> {
    const response = await this.v1Client.put<ProspectiveIntent>(
      `/v1/prospective/intents/${intentId}/complete`,
      null,
      { params: { user_id: userId } }
    );
    return response.data;
  }

  async evaluateProspectiveContext(
    userId: string,
    contextText: string,
    contextEmbedding?: number[]
  ): Promise<ProspectiveIntent[]> {
    const response = await this.v1Client.post<ProspectiveIntent[]>(
      '/v1/prospective/evaluate',
      {
        user_id: userId,
        context_text: contextText,
        ...(contextEmbedding ? { context_embedding: contextEmbedding } : {}),
      }
    );
    return response.data;
  }

  async consolidateProspectiveIntents(
    userId: string
  ): Promise<{ user_id: string; consolidated_count: number }> {
    const response = await this.v1Client.post<{
      user_id: string;
      consolidated_count: number;
    }>('/v1/prospective/consolidate', { user_id: userId });
    return response.data;
  }

  async expireProspectiveIntents(
    userId?: string
  ): Promise<{ expired_count: number }> {
    const response = await this.v1Client.post<{ expired_count: number }>(
      '/v1/prospective/expire',
      { user_id: userId || null }
    );
    return response.data;
  }

  // ============================================================================
  // EMBEDDING MIGRATION
  // ============================================================================

  async startEmbeddingMigration(data: {
    new_model: string;
    new_dimension?: number;
  }): Promise<EmbeddingMigration> {
    const response = await this.v1Client.post<EmbeddingMigration>(
      '/v1/admin/embeddings/migrate',
      data
    );
    return response.data;
  }

  async getMigrationStatus(migrationId: string): Promise<EmbeddingMigration> {
    const response = await this.v1Client.get<EmbeddingMigration>(
      `/v1/admin/embeddings/migration/${migrationId}`
    );
    return response.data;
  }

  async cancelMigration(migrationId: string): Promise<void> {
    await this.v1Client.post(`/v1/admin/embeddings/migration/${migrationId}/cancel`);
  }

  // ============================================================================
  // DASHBOARD
  // ============================================================================

  async getDashboardStats(userId: string): Promise<DashboardStats> {
    const response = await this.client.get<DashboardStats>('/dashboard/stats', {
      params: { user_id: userId },
    });
    return response.data;
  }

  async getRecentActivity(userId: string): Promise<DashboardActivity[]> {
    const response = await this.client.get<{ activities: DashboardActivity[] }>(
      '/dashboard/recent-activity',
      { params: { user_id: userId } }
    );
    return response.data.activities || [];
  }

  // ============================================================================
  // SESSION MANAGEMENT
  // ============================================================================

  async listSessions(userId: string): Promise<SessionInfo[]> {
    const response = await this.client.get<SessionInfo[]>('/sessions/list', {
      params: { user_id: userId },
    });
    return response.data;
  }

  async getSessionStats(): Promise<SessionStats> {
    const response = await this.client.get<SessionStats>('/sessions/stats');
    return response.data;
  }

  async getSession(sessionId: string, userId: string): Promise<SessionInfo> {
    const response = await this.client.get<SessionInfo>(`/sessions/${sessionId}`, {
      params: { user_id: userId },
    });
    return response.data;
  }

  async softDeleteSession(sessionId: string, userId: string): Promise<void> {
    await this.client.post('/sessions/soft-delete', {
      session_id: sessionId,
      user_id: userId,
    });
  }

  async wipeUserData(data: {
    user_id: string;
    admin_user_id: string;
    reason?: string;
    confirm: boolean;
  }): Promise<{ sessions_deleted: number }> {
    const response = await this.client.post<{ sessions_deleted: number }>(
      '/sessions/wipe-user-data',
      data
    );
    return response.data;
  }

  // ============================================================================
  // COMPACTION
  // ============================================================================

  async compactConversations(data: {
    user_id: string;
    session_id?: string;
    target_ratio?: number;
  }): Promise<CompactionResult> {
    const response = await this.client.post<CompactionResult>('/compaction/compact', data);
    return response.data;
  }

  async getCompactionHistory(userId: string): Promise<CompactionHistoryEntry[]> {
    const response = await this.client.get<CompactionHistoryEntry[]>('/compaction/history', {
      params: { user_id: userId },
    });
    return response.data;
  }

  async estimateCompaction(data: {
    user_id: string;
    session_id?: string;
  }): Promise<{ estimated_savings_percent: number; original_tokens: number }> {
    const response = await this.client.post<{
      estimated_savings_percent: number;
      original_tokens: number;
    }>('/compaction/estimate', data);
    return response.data;
  }

  // ============================================================================
  // INTELLIGENCE (Entity, Relationship, Clustering, Temporal)
  // ============================================================================

  async extractFacts(
    text: string,
    userId: string
  ): Promise<ExtractedFact[]> {
    const response = await this.v1Client.post<{ facts: ExtractedFact[] }>(
      '/v1/intelligence/facts:extract',
      { text, user_id: userId }
    );
    return response.data.facts;
  }

  async extractEntities(
    text: string,
    userId: string
  ): Promise<ExtractedEntity[]> {
    const response = await this.v1Client.post<{ entities: ExtractedEntity[] }>(
      '/v1/intelligence/entities:extract',
      { text, user_id: userId }
    );
    return response.data.entities;
  }

  async searchEntities(
    query: string,
    userId: string
  ): Promise<ExtractedEntity[]> {
    const response = await this.v1Client.post<{ entities: ExtractedEntity[] }>(
      '/v1/intelligence/entities:search',
      { query, user_id: userId }
    );
    return response.data.entities;
  }

  async getEntityProfile(entityId: string): Promise<ExtractedEntity> {
    const response = await this.v1Client.get<ExtractedEntity>(
      `/v1/intelligence/entities/${entityId}`
    );
    return response.data;
  }

  async analyzeRelationships(
    text: string,
    userId: string
  ): Promise<EntityRelationship[]> {
    const response = await this.v1Client.post<{ relationships: EntityRelationship[] }>(
      '/v1/intelligence/relationships:analyze',
      { text, user_id: userId }
    );
    return response.data.relationships;
  }

  async getEntityRelationships(entityId: string): Promise<EntityRelationship[]> {
    const response = await this.v1Client.get<{ relationships: EntityRelationship[] }>(
      `/v1/intelligence/relationships/${entityId}`
    );
    return response.data.relationships;
  }

  async getRelationshipNetwork(userId: string): Promise<RelationshipNetwork> {
    const response = await this.v1Client.get<RelationshipNetwork>(
      '/v1/intelligence/relationships:network',
      { params: { user_id: userId } }
    );
    return response.data;
  }

  async analyzeClusters(
    userId: string,
    numClusters?: number
  ): Promise<ClusterAnalysis> {
    const response = await this.v1Client.post<ClusterAnalysis>(
      '/v1/intelligence/clustering:analyze',
      { user_id: userId, num_clusters: numClusters }
    );
    return response.data;
  }

  async optimizeClusters(userId: string): Promise<OptimalClusters> {
    const response = await this.v1Client.post<OptimalClusters>(
      '/v1/intelligence/clustering:optimize',
      { user_id: userId }
    );
    return response.data;
  }

  async analyzeTemporalPatterns(userId: string): Promise<TemporalAnalysis> {
    const response = await this.v1Client.post<TemporalAnalysis>(
      '/v1/intelligence/temporal:analyze',
      { user_id: userId }
    );
    return response.data;
  }

  async getTemporalPeakTimes(userId: string): Promise<PeakActivity[]> {
    const response = await this.v1Client.post<{ peak_times: PeakActivity[] }>(
      '/v1/intelligence/temporal:peak-times',
      { user_id: userId }
    );
    return response.data.peak_times;
  }

  // ============================================================================
  // ADDITIONAL HEALING METHODS
  // ============================================================================

  async detectKnowledgeGaps(userId: string): Promise<{ gaps: string[] }> {
    const response = await this.v1Client.post<{ gaps: string[] }>(
      '/v1/healing/health/gaps',
      { user_id: userId }
    );
    return response.data;
  }

  async runDeduplication(userId: string, dryRun: boolean = true): Promise<HealingActionResult> {
    const response = await this.v1Client.post<HealingActionResult>(
      '/v1/healing/deduplication',
      { user_id: userId, dry_run: dryRun }
    );
    return response.data;
  }

  async consolidateMemories(userId: string, dryRun: boolean = true): Promise<HealingActionResult> {
    const response = await this.v1Client.post<HealingActionResult>(
      '/v1/healing/consolidate',
      { user_id: userId, dry_run: dryRun }
    );
    return response.data;
  }

  async autoTagMemories(userId: string, dryRun: boolean = true): Promise<HealingActionResult> {
    const response = await this.v1Client.post<HealingActionResult>(
      '/v1/healing/tagging',
      { user_id: userId, dry_run: dryRun }
    );
    return response.data;
  }

  async adjustImportance(userId: string, dryRun: boolean = true): Promise<HealingActionResult> {
    const response = await this.v1Client.post<HealingActionResult>(
      '/v1/healing/importance',
      { user_id: userId, dry_run: dryRun }
    );
    return response.data;
  }

  async getHealingConfig(userId: string): Promise<HealingConfig> {
    const response = await this.v1Client.get<{ config: HealingConfig }>(
      `/v1/healing/config/${userId}`
    );
    return response.data.config;
  }

  async updateHealingConfig(userId: string, config: Partial<HealingConfig>): Promise<HealingConfig> {
    const response = await this.v1Client.post<{ success: boolean; config: HealingConfig }>(
      '/v1/healing/config',
      { user_id: userId, ...config }
    );
    return response.data.config;
  }

  // ============================================================================
  // ADDITIONAL COLLABORATION METHODS
  // ============================================================================

  async deleteSpace(spaceId: string): Promise<void> {
    await this.v1Client.delete(`/v1/collaboration/spaces/${spaceId}`);
  }

  async addCollaborator(
    spaceId: string,
    agentId: string,
    permissions: string[] = ['read']
  ): Promise<void> {
    await this.v1Client.post(`/v1/collaboration/spaces/${spaceId}/collaborators`, {
      agent_id: agentId,
      permissions,
    });
  }

  async removeCollaborator(spaceId: string, agentId: string): Promise<void> {
    await this.v1Client.delete(
      `/v1/collaboration/spaces/${spaceId}/collaborators/${agentId}`
    );
  }

  async updateCollaboratorPermissions(
    spaceId: string,
    agentId: string,
    permissions: string[]
  ): Promise<void> {
    await this.v1Client.put(
      `/v1/collaboration/spaces/${spaceId}/collaborators/${agentId}/permissions`,
      { permissions }
    );
  }

  async addMemoryToSpace(spaceId: string, memoryId: string): Promise<void> {
    await this.v1Client.post(`/v1/collaboration/spaces/${spaceId}/memories`, {
      memory_id: memoryId,
    });
  }

  async removeMemoryFromSpace(spaceId: string, memoryId: string): Promise<void> {
    await this.v1Client.delete(
      `/v1/collaboration/spaces/${spaceId}/memories/${memoryId}`
    );
  }

  async getConflicts(): Promise<CollaborationConflict[]> {
    const response = await this.v1Client.get<{ conflicts: CollaborationConflict[] }>(
      '/v1/collaboration/conflicts'
    );
    return response.data.conflicts;
  }

  async resolveConflict(
    conflictId: string,
    resolution: string
  ): Promise<void> {
    await this.v1Client.post(`/v1/collaboration/conflicts/${conflictId}/resolve`, {
      resolution,
    });
  }

  // ============================================================================
  // ADDITIONAL PREDICTION METHODS
  // ============================================================================

  async predictNextOccurrence(
    userId: string,
    patternType: string
  ): Promise<{ next_predicted: string; confidence: number }> {
    const response = await this.v1Client.post<{
      next_predicted: string;
      confidence: number;
    }>('/v1/predictions/patterns/predict', {
      user_id: userId,
      pattern_type: patternType,
    });
    return response.data;
  }

  async getInsights(userId: string): Promise<PredictiveInsight[]> {
    const response = await this.v1Client.post<{ insights: PredictiveInsight[] }>(
      '/v1/predictions/insights',
      { user_id: userId }
    );
    return response.data.insights;
  }

  async getPeakActivityTimes(userId: string): Promise<PeakActivity[]> {
    const response = await this.v1Client.post<{ peaks: PeakActivity[] }>(
      '/v1/predictions/activity/peaks',
      { user_id: userId }
    );
    return response.data.peaks;
  }
}

export const apiClient = new APIClient();
