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
} from '../types';

class APIClient {
  private client: AxiosInstance;

  constructor(baseURL: string = '/api') {
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add auth token interceptor
    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem('auth_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });
  }

  // ============================================================================
  // MEMORY OPERATIONS
  // ============================================================================

  async createMemory(data: {
    text: string;
    user_id: string;
    type?: string;
    importance?: number;
    tags?: string[];
    session_id?: string;
  }): Promise<Memory> {
    const response = await this.client.post<Memory>('/v1/memories:remember', data);
    return response.data;
  }

  async recallMemories(data: {
    query: string;
    user_id: string;
    k?: number;
    filters?: Record<string, any>;
  }): Promise<RetrievalResult[]> {
    const response = await this.client.post<RetrievalResult[]>('/v1/memories:recall', data);
    return response.data;
  }

  async getMemories(data: {
    user_id: string;
    filters?: Record<string, any>;
    limit?: number;
  }): Promise<Memory[]> {
    const response = await this.client.post<Memory[]>('/v1/memories:get', data);
    return response.data;
  }

  async updateMemory(memoryId: string, data: Partial<Memory>): Promise<Memory> {
    const response = await this.client.patch<Memory>('/v1/memories:update', {
      memory_id: memoryId,
      ...data,
    });
    return response.data;
  }

  async deleteMemory(memoryId: string, userId?: string): Promise<void> {
    await this.client.delete('/v1/memories:delete', {
      data: { memory_id: memoryId, user_id: userId },
    });
  }

  // ============================================================================
  // HEALTH MONITORING
  // ============================================================================

  async getHealthScore(userId: string, detailed: boolean = true): Promise<HealthScore> {
    const response = await this.client.post<HealthScore>('/v1/healing/health', {
      user_id: userId,
      detailed,
    });
    return response.data;
  }

  async detectStaleMemories(userId: string, thresholdDays: number = 90) {
    const response = await this.client.post(`/v1/healing/health/stale`, {
      user_id: userId,
      threshold_days: thresholdDays,
    });
    return response.data;
  }

  async detectDuplicates(userId: string, similarityThreshold: number = 0.9) {
    const response = await this.client.post(`/v1/healing/health/duplicates`, {
      user_id: userId,
      similarity_threshold: similarityThreshold,
    });
    return response.data;
  }

  async runAutoCleanup(userId: string, dryRun: boolean = true) {
    const response = await this.client.post('/v1/healing/cleanup', {
      user_id: userId,
      dry_run: dryRun,
    });
    return response.data;
  }

  async runFullHealthCheck(userId: string, dryRun: boolean = true) {
    const response = await this.client.post('/v1/healing/full-check', {
      user_id: userId,
      dry_run: dryRun,
    });
    return response.data;
  }

  // ============================================================================
  // COLLABORATION
  // ============================================================================

  async createSpace(data: {
    name: string;
    owner_agent_id: string;
    description?: string;
    tags?: string[];
  }): Promise<CollaborationSpace> {
    const response = await this.client.post<{ space: CollaborationSpace }>(
      '/v1/collaboration/spaces',
      data
    );
    return response.data.space;
  }

  async getSpace(spaceId: string): Promise<CollaborationSpace> {
    const response = await this.client.get<CollaborationSpace>(
      `/v1/collaboration/spaces/${spaceId}`
    );
    return response.data;
  }

  async listSpaces(agentId?: string, includeInactive: boolean = false): Promise<CollaborationSpace[]> {
    const response = await this.client.get<{ spaces: CollaborationSpace[] }>(
      '/v1/collaboration/spaces',
      {
        params: { agent_id: agentId, include_inactive: includeInactive },
      }
    );
    return response.data.spaces;
  }

  async getSpaceEvents(spaceId: string, limit: number = 100): Promise<CollaborationEvent[]> {
    const response = await this.client.get<{ events: CollaborationEvent[] }>(
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
    const response = await this.client.get<{ notifications: Notification[] }>(
      `/v1/collaboration/notifications/${agentId}`,
      { params: { unread_only: unreadOnly, limit } }
    );
    return response.data.notifications;
  }

  async markNotificationRead(agentId: string, notificationId: string): Promise<void> {
    await this.client.post(
      `/v1/collaboration/notifications/${agentId}/${notificationId}/read`
    );
  }

  // ============================================================================
  // PREDICTIONS & ANALYTICS
  // ============================================================================

  async detectPatterns(userId: string, minOccurrences: number = 3): Promise<TemporalPattern[]> {
    const response = await this.client.post<{ patterns: TemporalPattern[] }>(
      '/v1/predictions/patterns',
      { user_id: userId, min_occurrences: minOccurrences }
    );
    return response.data.patterns;
  }

  async detectAnomalies(userId: string, lookbackDays: number = 30): Promise<Anomaly[]> {
    const response = await this.client.post<{ anomalies: Anomaly[] }>(
      '/v1/predictions/anomalies',
      { user_id: userId, lookback_days: lookbackDays }
    );
    return response.data.anomalies;
  }

  async getRecommendations(userId: string, maxRecommendations: number = 10): Promise<Recommendation[]> {
    const response = await this.client.post<{ recommendations: Recommendation[] }>(
      '/v1/predictions/recommendations',
      { user_id: userId, max_recommendations: maxRecommendations }
    );
    return response.data.recommendations;
  }

  async getForecast(userId: string, metric: string, horizon: string): Promise<Forecast> {
    const response = await this.client.post<Forecast>('/v1/predictions/forecast', {
      user_id: userId,
      metric,
      horizon,
    });
    return response.data;
  }

  async analyzeTrends(userId: string, timeWindowDays: number = 30, metric: string = 'activity') {
    const response = await this.client.post('/v1/predictions/trends', {
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
    }, {
      params: { user_id: userId },
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
}

export const apiClient = new APIClient();
