import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Lightbulb,
  RefreshCw,
  Plus,
  Layers,
  Clock,
  Zap,
  X,
  CheckCircle,
  AlertCircle,
  XCircle,
  Timer,
  Search,
  Ban,
  Trash2,
} from 'lucide-react';
import { apiClient } from '../services/api';
import type { ProspectiveIntent } from '../types';
import clsx from 'clsx';

interface ProspectiveMemoryPageProps {
  userId: string;
}

const statusColors: Record<string, string> = {
  pending: 'badge-blue',
  triggered: 'badge-warning',
  completed: 'badge-success',
  expired: 'badge-gray',
  cancelled: 'badge-gray',
};

const statusIcons: Record<string, typeof Clock> = {
  pending: Timer,
  triggered: Zap,
  completed: CheckCircle,
  expired: AlertCircle,
  cancelled: Ban,
};

const triggerTypeLabels: Record<string, string> = {
  time_based: 'Time-Based',
  event_based: 'Event-Based',
  hybrid: 'Hybrid',
};

export function ProspectiveMemoryPage({ userId }: ProspectiveMemoryPageProps) {
  const queryClient = useQueryClient();
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showParseModal, setShowParseModal] = useState(false);
  const [showEvaluateModal, setShowEvaluateModal] = useState(false);
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Create form state
  const [createForm, setCreateForm] = useState({
    intent_text: '',
    action_description: '',
    trigger_type: 'event_based',
    context_keywords: '',
    context_pattern: '',
    priority: 5,
    recurrence: 'none',
  });

  // Parse form state
  const [parseText, setParseText] = useState('');

  // Evaluate form state
  const [evaluateText, setEvaluateText] = useState('');
  const [evaluateResults, setEvaluateResults] = useState<ProspectiveIntent[] | null>(null);

  // Auto-dismiss error after 5 seconds
  useEffect(() => {
    if (errorMessage) {
      const timer = setTimeout(() => setErrorMessage(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [errorMessage]);

  const handleMutationError = useCallback((err: unknown, action: string) => {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    setErrorMessage(`Failed to ${action}: ${msg}`);
  }, []);

  // Fetch intents
  const { data: intents = [], isLoading, isError, refetch } = useQuery({
    queryKey: ['prospective-intents', userId, statusFilter],
    queryFn: () => apiClient.listProspectiveIntents(userId, statusFilter || undefined),
  });

  // Create intent mutation
  const createMutation = useMutation({
    mutationFn: () => {
      const keywords = createForm.context_keywords
        .split(',')
        .map((k) => k.trim())
        .filter(Boolean);
      return apiClient.createProspectiveIntent({
        user_id: userId,
        intent_text: createForm.intent_text,
        action_description: createForm.action_description,
        trigger_type: createForm.trigger_type,
        context_keywords: keywords.length > 0 ? keywords : undefined,
        context_pattern: createForm.context_pattern || undefined,
        priority: createForm.priority,
        recurrence: createForm.recurrence,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prospective-intents', userId] });
      setShowCreateModal(false);
      setCreateForm({
        intent_text: '',
        action_description: '',
        trigger_type: 'event_based',
        context_keywords: '',
        context_pattern: '',
        priority: 5,
        recurrence: 'none',
      });
    },
    onError: (err: unknown) => handleMutationError(err, 'create intent'),
  });

  // Parse natural language mutation
  const parseMutation = useMutation({
    mutationFn: () => apiClient.parseProspectiveIntent(userId, parseText),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prospective-intents', userId] });
      setShowParseModal(false);
      setParseText('');
    },
    onError: (err: unknown) => handleMutationError(err, 'parse intent'),
  });

  // Evaluate context mutation
  const evaluateMutation = useMutation({
    mutationFn: () => apiClient.evaluateProspectiveContext(userId, evaluateText),
    onSuccess: (results) => {
      setEvaluateResults(results);
      queryClient.invalidateQueries({ queryKey: ['prospective-intents', userId] });
    },
    onError: (err: unknown) => handleMutationError(err, 'evaluate context'),
  });

  // Cancel intent mutation
  const cancelMutation = useMutation({
    mutationFn: (intentId: string) => apiClient.cancelProspectiveIntent(intentId, userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prospective-intents', userId] });
    },
    onError: (err: unknown) => handleMutationError(err, 'cancel intent'),
  });

  // Complete intent mutation
  const completeMutation = useMutation({
    mutationFn: (intentId: string) => apiClient.completeProspectiveIntent(intentId, userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prospective-intents', userId] });
    },
    onError: (err: unknown) => handleMutationError(err, 'complete intent'),
  });

  // Consolidate mutation
  const consolidateMutation = useMutation({
    mutationFn: () => apiClient.consolidateProspectiveIntents(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prospective-intents', userId] });
    },
    onError: (err: unknown) => handleMutationError(err, 'consolidate intents'),
  });

  // Expire mutation
  const expireMutation = useMutation({
    mutationFn: () => apiClient.expireProspectiveIntents(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prospective-intents', userId] });
    },
    onError: (err: unknown) => handleMutationError(err, 'expire intents'),
  });

  // Stats
  const pendingCount = intents.filter((i) => i.status === 'pending').length;
  const triggeredCount = intents.filter((i) => i.status === 'triggered').length;
  const completedCount = intents.filter((i) => i.status === 'completed').length;
  const avgPriority =
    intents.length > 0
      ? intents.reduce((sum, i) => sum + i.priority, 0) / intents.length
      : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-amber-100 rounded-lg">
            <Lightbulb className="w-6 h-6 text-amber-600" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Prospective Memory</h1>
            <p className="text-sm text-gray-500">
              Remembering to remember — manage future intentions and contextual triggers
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => refetch()}
            disabled={isLoading}
            className="btn-secondary flex items-center space-x-2"
          >
            <RefreshCw className={clsx('w-4 h-4', isLoading && 'animate-spin')} />
            <span>Refresh</span>
          </button>
          <button
            onClick={() => setShowEvaluateModal(true)}
            className="btn-secondary flex items-center space-x-2"
          >
            <Search className="w-4 h-4" />
            <span>Evaluate</span>
          </button>
          <button
            onClick={() => consolidateMutation.mutate()}
            disabled={consolidateMutation.isPending}
            className="btn-secondary flex items-center space-x-2"
          >
            {consolidateMutation.isPending ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Layers className="w-4 h-4" />
            )}
            <span>Consolidate</span>
          </button>
          <button
            onClick={() => expireMutation.mutate()}
            disabled={expireMutation.isPending}
            className="btn-secondary flex items-center space-x-2"
          >
            <Trash2 className="w-4 h-4" />
            <span>Expire Stale</span>
          </button>
          <button
            onClick={() => setShowParseModal(true)}
            className="btn-secondary flex items-center space-x-2"
          >
            <Zap className="w-4 h-4" />
            <span>Natural Language</span>
          </button>
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn-primary flex items-center space-x-2"
          >
            <Plus className="w-4 h-4" />
            <span>Create Intent</span>
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {(isError || errorMessage) && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center space-x-3">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
          <p className="text-sm font-medium text-red-800 flex-1">
            {errorMessage || 'Failed to load prospective intents'}
          </p>
          <button
            onClick={() => {
              setErrorMessage(null);
              refetch();
            }}
            className="text-sm text-red-700 hover:text-red-900 font-medium"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Lightbulb className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Total Intents</p>
              <p className="text-2xl font-bold text-gray-900">{intents.length}</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-amber-100 rounded-lg">
              <Timer className="w-5 h-5 text-amber-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Pending</p>
              <p className="text-2xl font-bold text-amber-600">{pendingCount}</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-orange-100 rounded-lg">
              <Zap className="w-5 h-5 text-orange-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Triggered</p>
              <p className="text-2xl font-bold text-orange-600">{triggeredCount}</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-green-100 rounded-lg">
              <CheckCircle className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Completed</p>
              <p className="text-2xl font-bold text-green-600">{completedCount}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Status Filter Tabs */}
      <div className="flex items-center space-x-2">
        {[null, 'pending', 'triggered', 'completed', 'expired', 'cancelled'].map((status) => (
          <button
            key={status || 'all'}
            onClick={() => setStatusFilter(status)}
            className={clsx(
              'px-3 py-1.5 text-sm rounded-lg transition-colors',
              statusFilter === status
                ? 'bg-primary-100 text-primary-700 font-medium'
                : 'text-gray-600 hover:bg-gray-100'
            )}
          >
            {status ? status.charAt(0).toUpperCase() + status.slice(1) : 'All'}
          </button>
        ))}
      </div>

      {/* Intents List */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Intents</h3>
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-6 h-6 text-primary-600 animate-spin" />
          </div>
        ) : intents.length > 0 ? (
          <div className="space-y-4">
            {intents.map((intent) => {
              const StatusIcon = statusIcons[intent.status] || AlertCircle;
              return (
                <div key={intent.id} className="card-hover">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 space-y-3">
                      <div className="flex items-center space-x-2 flex-wrap gap-y-1">
                        <span
                          className={clsx(
                            'badge',
                            statusColors[intent.status] || 'badge-gray'
                          )}
                        >
                          <StatusIcon className="w-3 h-3 mr-1" />
                          {intent.status}
                        </span>
                        <span className="badge badge-blue">
                          {triggerTypeLabels[intent.trigger_type] || intent.trigger_type}
                        </span>
                        <span className="badge badge-purple">Priority: {intent.priority}</span>
                        {intent.recurrence !== 'none' && (
                          <span className="badge badge-green">
                            <RefreshCw className="w-3 h-3 mr-1" />
                            {intent.recurrence}
                          </span>
                        )}
                        <span className="text-xs text-gray-400 font-mono">
                          {intent.id.slice(0, 12)}
                        </span>
                      </div>

                      <p className="text-gray-800 font-medium">{intent.intent_text}</p>

                      {intent.action_description &&
                        intent.action_description !== intent.intent_text && (
                          <p className="text-sm text-gray-600">{intent.action_description}</p>
                        )}

                      <div className="flex items-center space-x-4 text-xs text-gray-500">
                        {intent.context_keywords.length > 0 && (
                          <span>
                            Keywords:{' '}
                            {intent.context_keywords.map((kw, i) => (
                              <span
                                key={i}
                                className="inline-block bg-gray-100 px-1.5 py-0.5 rounded text-gray-700 mr-1"
                              >
                                {kw}
                              </span>
                            ))}
                          </span>
                        )}
                        <span>
                          Created: {new Date(intent.created_at).toLocaleDateString()}
                        </span>
                        {intent.triggered_at && (
                          <span>
                            Triggered: {new Date(intent.triggered_at).toLocaleString()}
                          </span>
                        )}
                        {intent.trigger_count > 0 && (
                          <span>Fired: {intent.trigger_count}x</span>
                        )}
                        {intent.tags.length > 0 && (
                          <span>Tags: {intent.tags.join(', ')}</span>
                        )}
                      </div>
                    </div>

                    {/* Action buttons */}
                    <div className="flex items-center space-x-1 ml-4">
                      {(intent.status === 'pending' || intent.status === 'triggered') && (
                        <button
                          onClick={() => completeMutation.mutate(intent.id)}
                          className="p-1.5 text-gray-400 hover:text-green-600 hover:bg-green-50 rounded transition-colors"
                          title="Mark as completed"
                        >
                          <CheckCircle className="w-4 h-4" />
                        </button>
                      )}
                      {intent.status === 'pending' && (
                        <button
                          onClick={() => cancelMutation.mutate(intent.id)}
                          className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                          title="Cancel intent"
                        >
                          <XCircle className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="card text-center py-12 text-gray-400">
            <Lightbulb className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>No prospective intents yet</p>
            <p className="text-sm mt-1">
              Create intents to remember future actions based on time or context
            </p>
          </div>
        )}
      </div>

      {/* Create Intent Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setShowCreateModal(false)}
          />
          <div className="relative bg-white rounded-xl shadow-2xl w-full max-w-lg m-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between p-6 border-b">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-amber-100 rounded-lg">
                  <Plus className="w-5 h-5 text-amber-600" />
                </div>
                <h2 className="text-lg font-semibold text-gray-900">Create Intent</h2>
              </div>
              <button
                onClick={() => setShowCreateModal(false)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-gray-500" />
              </button>
            </div>
            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Intent Text *
                </label>
                <input
                  type="text"
                  value={createForm.intent_text}
                  onChange={(e) =>
                    setCreateForm({ ...createForm, intent_text: e.target.value })
                  }
                  placeholder="e.g., Remind me to follow up with John"
                  className="input"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Action Description
                </label>
                <input
                  type="text"
                  value={createForm.action_description}
                  onChange={(e) =>
                    setCreateForm({ ...createForm, action_description: e.target.value })
                  }
                  placeholder="What to do when triggered"
                  className="input"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Trigger Type
                  </label>
                  <select
                    value={createForm.trigger_type}
                    onChange={(e) =>
                      setCreateForm({ ...createForm, trigger_type: e.target.value })
                    }
                    className="input"
                  >
                    <option value="event_based">Event-Based</option>
                    <option value="time_based">Time-Based</option>
                    <option value="hybrid">Hybrid</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Priority (0-10)
                  </label>
                  <input
                    type="number"
                    min={0}
                    max={10}
                    value={createForm.priority}
                    onChange={(e) =>
                      setCreateForm({
                        ...createForm,
                        priority: parseInt(e.target.value) || 5,
                      })
                    }
                    className="input"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Context Keywords (comma-separated)
                </label>
                <input
                  type="text"
                  value={createForm.context_keywords}
                  onChange={(e) =>
                    setCreateForm({ ...createForm, context_keywords: e.target.value })
                  }
                  placeholder="e.g., budget, cost, spending"
                  className="input"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Context Pattern (regex)
                </label>
                <input
                  type="text"
                  value={createForm.context_pattern}
                  onChange={(e) =>
                    setCreateForm({ ...createForm, context_pattern: e.target.value })
                  }
                  placeholder="e.g., budget|cost|spending"
                  className="input"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Recurrence
                </label>
                <select
                  value={createForm.recurrence}
                  onChange={(e) =>
                    setCreateForm({ ...createForm, recurrence: e.target.value })
                  }
                  className="input"
                >
                  <option value="none">None</option>
                  <option value="daily">Daily</option>
                  <option value="weekly">Weekly</option>
                  <option value="monthly">Monthly</option>
                </select>
              </div>
            </div>
            <div className="p-6 border-t flex justify-end space-x-3">
              <button
                onClick={() => setShowCreateModal(false)}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={() => createMutation.mutate()}
                disabled={createMutation.isPending || !createForm.intent_text.trim()}
                className="btn-primary flex items-center space-x-2"
              >
                {createMutation.isPending && (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                )}
                <span>Create Intent</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Parse Natural Language Modal */}
      {showParseModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setShowParseModal(false)}
          />
          <div className="relative bg-white rounded-xl shadow-2xl w-full max-w-lg m-4">
            <div className="flex items-center justify-between p-6 border-b">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-amber-100 rounded-lg">
                  <Zap className="w-5 h-5 text-amber-600" />
                </div>
                <h2 className="text-lg font-semibold text-gray-900">
                  Create from Natural Language
                </h2>
              </div>
              <button
                onClick={() => setShowParseModal(false)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-gray-500" />
              </button>
            </div>
            <div className="p-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Describe what you want to remember
              </label>
              <textarea
                value={parseText}
                onChange={(e) => setParseText(e.target.value)}
                placeholder="e.g., When the user mentions budget, bring up the Q3 cost overrun&#10;Remind me to follow up with John after the meeting&#10;Every Monday, summarize the week's key decisions"
                className="input h-32 resize-none"
              />
            </div>
            <div className="p-6 border-t flex justify-end space-x-3">
              <button onClick={() => setShowParseModal(false)} className="btn-secondary">
                Cancel
              </button>
              <button
                onClick={() => parseMutation.mutate()}
                disabled={parseMutation.isPending || !parseText.trim()}
                className="btn-primary flex items-center space-x-2"
              >
                {parseMutation.isPending && (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                )}
                <span>Create Intent</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Evaluate Context Modal */}
      {showEvaluateModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => {
              setShowEvaluateModal(false);
              setEvaluateResults(null);
            }}
          />
          <div className="relative bg-white rounded-xl shadow-2xl w-full max-w-2xl m-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between p-6 border-b">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-amber-100 rounded-lg">
                  <Search className="w-5 h-5 text-amber-600" />
                </div>
                <h2 className="text-lg font-semibold text-gray-900">
                  Evaluate Context
                </h2>
              </div>
              <button
                onClick={() => {
                  setShowEvaluateModal(false);
                  setEvaluateResults(null);
                }}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-gray-500" />
              </button>
            </div>
            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Context text to evaluate against pending intents
                </label>
                <textarea
                  value={evaluateText}
                  onChange={(e) => setEvaluateText(e.target.value)}
                  placeholder="e.g., What's our budget status for this quarter?"
                  className="input h-24 resize-none"
                />
              </div>
              <button
                onClick={() => evaluateMutation.mutate()}
                disabled={evaluateMutation.isPending || !evaluateText.trim()}
                className="btn-primary flex items-center space-x-2"
              >
                {evaluateMutation.isPending && (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                )}
                <span>Evaluate</span>
              </button>

              {evaluateResults !== null && (
                <div className="space-y-3">
                  {evaluateResults.length > 0 ? (
                    <>
                      <div className="flex items-center space-x-2">
                        <Zap className="w-5 h-5 text-orange-600" />
                        <span className="text-sm font-medium text-orange-700">
                          {evaluateResults.length} intent(s) triggered!
                        </span>
                      </div>
                      {evaluateResults.map((intent) => (
                        <div
                          key={intent.id}
                          className="bg-orange-50 border border-orange-200 rounded-lg p-3"
                        >
                          <p className="font-medium text-gray-800">{intent.intent_text}</p>
                          <p className="text-sm text-gray-600 mt-1">
                            {intent.action_description}
                          </p>
                          <div className="flex items-center space-x-3 mt-2 text-xs text-gray-500">
                            <span>Priority: {intent.priority}</span>
                            <span>
                              {triggerTypeLabels[intent.trigger_type]}
                            </span>
                          </div>
                        </div>
                      ))}
                    </>
                  ) : (
                    <div className="flex items-center space-x-2 text-gray-500">
                      <CheckCircle className="w-5 h-5" />
                      <span className="text-sm">
                        No intents matched this context
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
