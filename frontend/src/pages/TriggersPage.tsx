import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Bell,
  RefreshCw,
  Plus,
  Trash2,
  Zap,
  Activity,
  Clock,
  X,
  ChevronRight,
  AlertTriangle,
} from 'lucide-react';
import { apiClient } from '../services/api';
import type {
  Trigger,
  TriggerEvent,
  TriggerConditionOp,
  TriggerAction,
} from '../types';
import clsx from 'clsx';
import { formatDistanceToNow } from 'date-fns';

interface TriggersPageProps {
  userId: string;
}

const EVENT_OPTIONS: { value: TriggerEvent; label: string }[] = [
  { value: 'on_remember', label: 'On Remember' },
  { value: 'on_recall', label: 'On Recall' },
  { value: 'on_update', label: 'On Update' },
  { value: 'on_delete', label: 'On Delete' },
  { value: 'on_conflict', label: 'On Conflict' },
  { value: 'on_expire', label: 'On Expire' },
];

const OP_OPTIONS: { value: TriggerConditionOp; label: string }[] = [
  { value: 'eq', label: '=' },
  { value: 'gt', label: '>' },
  { value: 'lt', label: '<' },
  { value: 'contains', label: 'contains' },
  { value: 'matches', label: 'matches' },
];

const ACTION_OPTIONS: { value: TriggerAction; label: string }[] = [
  { value: 'webhook', label: 'Webhook' },
  { value: 'log', label: 'Log' },
  { value: 'websocket', label: 'WebSocket' },
];

const eventBadgeColor = (event: string): string => {
  switch (event) {
    case 'on_remember': return 'bg-green-100 text-green-700';
    case 'on_recall': return 'bg-blue-100 text-blue-700';
    case 'on_update': return 'bg-yellow-100 text-yellow-700';
    case 'on_delete': return 'bg-red-100 text-red-700';
    case 'on_conflict': return 'bg-orange-100 text-orange-700';
    case 'on_expire': return 'bg-purple-100 text-purple-700';
    default: return 'bg-gray-100 text-gray-700';
  }
};

const actionBadgeColor = (action: string): string => {
  switch (action) {
    case 'webhook': return 'bg-indigo-100 text-indigo-700';
    case 'log': return 'bg-gray-100 text-gray-700';
    case 'websocket': return 'bg-yellow-100 text-yellow-700';
    default: return 'bg-gray-100 text-gray-700';
  }
};

export function TriggersPage({ userId }: TriggersPageProps) {
  const queryClient = useQueryClient();
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedTrigger, setSelectedTrigger] = useState<Trigger | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

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

  // Form state
  const [newName, setNewName] = useState('');
  const [newEvent, setNewEvent] = useState<TriggerEvent>('on_remember');
  const [newConditions, setNewConditions] = useState<{ field: string; operator: TriggerConditionOp; value: string }[]>([]);
  const [newAction, setNewAction] = useState<TriggerAction>('log');
  const [newActionConfig, setNewActionConfig] = useState<Record<string, string>>({});

  const { data: triggers = [], isLoading, isError, refetch } = useQuery({
    queryKey: ['triggers', userId],
    queryFn: () => apiClient.listTriggers(userId),
  });

  const { data: history = [] } = useQuery({
    queryKey: ['trigger-history', selectedTrigger?.id],
    queryFn: () => apiClient.getTriggerHistory(selectedTrigger!.id),
    enabled: !!selectedTrigger,
  });

  const createMutation = useMutation({
    mutationFn: () =>
      apiClient.createTrigger({
        name: newName,
        user_id: userId,
        event: newEvent,
        conditions: newConditions,
        action: newAction,
        action_config: newActionConfig,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['triggers', userId] });
      resetForm();
      setShowCreateModal(false);
    },
    onError: (err: unknown) => handleMutationError(err, 'create trigger'),
  });

  const deleteMutation = useMutation({
    mutationFn: (triggerId: string) => apiClient.deleteTrigger(triggerId, userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['triggers', userId] });
      setSelectedTrigger(null);
    },
    onError: (err: unknown) => handleMutationError(err, 'delete trigger'),
  });

  const resetForm = () => {
    setNewName('');
    setNewEvent('on_remember');
    setNewConditions([]);
    setNewAction('log');
    setNewActionConfig({});
  };

  const addCondition = () => {
    setNewConditions([...newConditions, { field: '', operator: 'eq', value: '' }]);
  };

  const updateCondition = (index: number, field: string, value: string) => {
    const updated = [...newConditions];
    updated[index] = { ...updated[index], [field]: value };
    setNewConditions(updated);
  };

  const removeCondition = (index: number) => {
    setNewConditions(newConditions.filter((_, i) => i !== index));
  };

  const activeTriggers = triggers.filter((t) => t.enabled);
  const totalFires = triggers.reduce((sum, t) => sum + t.fired_count, 0);
  const uniqueEvents = new Set(triggers.map((t) => t.event)).size;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-yellow-100 rounded-lg">
            <Bell className="w-6 h-6 text-yellow-600" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Memory Triggers</h1>
            <p className="text-sm text-gray-500">Automate actions when memory events occur</p>
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
            onClick={() => setShowCreateModal(true)}
            className="btn-primary flex items-center space-x-2"
          >
            <Plus className="w-4 h-4" />
            <span>New Trigger</span>
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {(isError || errorMessage) && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center space-x-3">
          <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0" />
          <p className="text-sm font-medium text-red-800 flex-1">
            {errorMessage || 'Failed to load triggers'}
          </p>
          <button
            onClick={() => { setErrorMessage(null); refetch(); }}
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
              <Bell className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Total Triggers</p>
              <p className="text-2xl font-bold text-gray-900">{triggers.length}</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-green-100 rounded-lg">
              <Activity className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Active</p>
              <p className="text-2xl font-bold text-green-600">{activeTriggers.length}</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-orange-100 rounded-lg">
              <Zap className="w-5 h-5 text-orange-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Total Fires</p>
              <p className="text-2xl font-bold text-orange-600">{totalFires}</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Activity className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Events Monitored</p>
              <p className="text-2xl font-bold text-purple-600">{uniqueEvents}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Trigger List */}
        <div className="lg:col-span-2 space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">Triggers</h3>
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-6 h-6 text-primary-600 animate-spin" />
            </div>
          ) : triggers.length > 0 ? (
            triggers.map((trigger) => (
              <div
                key={trigger.id}
                className={clsx(
                  'card-hover cursor-pointer',
                  selectedTrigger?.id === trigger.id && 'ring-2 ring-primary-500'
                )}
                onClick={() => setSelectedTrigger(trigger)}
              >
                <div className="flex items-start justify-between">
                  <div className="space-y-2 flex-1">
                    <div className="flex items-center space-x-2">
                      <span className="font-medium text-gray-900">{trigger.name}</span>
                      {!trigger.enabled && (
                        <span className="badge badge-gray">Disabled</span>
                      )}
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className={clsx('badge', eventBadgeColor(trigger.event))}>
                        {trigger.event.replace(/_/g, ' ')}
                      </span>
                      <ChevronRight className="w-4 h-4 text-gray-400" />
                      <span className={clsx('badge', actionBadgeColor(trigger.action))}>
                        {trigger.action}
                      </span>
                    </div>
                    <div className="flex items-center space-x-4 text-xs text-gray-500">
                      <span className="flex items-center space-x-1">
                        <Zap className="w-3 h-3" />
                        <span>{trigger.fired_count} fires</span>
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (window.confirm('Delete this trigger?')) {
                        deleteMutation.mutate(trigger.id);
                      }
                    }}
                    className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))
          ) : (
            <div className="card text-center py-12 text-gray-400">
              <Bell className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>No triggers configured yet</p>
              <p className="text-sm mt-1">Create a trigger to automate memory actions</p>
            </div>
          )}
        </div>

        {/* History Panel */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">Fire History</h3>
          {selectedTrigger ? (
            history.length > 0 ? (
              <div className="space-y-3">
                {history.map((entry, idx) => (
                  <div key={idx} className="card text-sm">
                    <div className="flex items-center space-x-2 mb-2">
                      <Zap className={clsx('w-4 h-4', entry.success ? 'text-green-500' : 'text-red-500')} />
                      <span className="text-gray-500">
                        {formatDistanceToNow(new Date(entry.fired_at), { addSuffix: true })}
                      </span>
                      <span className={clsx('badge', entry.success ? 'badge-success' : 'badge-danger')}>
                        {entry.success ? 'Success' : 'Failed'}
                      </span>
                    </div>
                    <div className="text-xs text-gray-600">
                      <span>Event: {entry.event}</span>
                      {entry.memory_id && (
                        <span className="ml-2 font-mono">Memory: {entry.memory_id.slice(0, 12)}...</span>
                      )}
                    </div>
                    {entry.error && (
                      <div className="bg-red-50 rounded p-2 mt-2">
                        <p className="text-xs text-red-600">{entry.error}</p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="card text-center py-8 text-gray-400">
                <Clock className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No fire history for this trigger</p>
              </div>
            )
          ) : (
            <div className="card text-center py-8 text-gray-400">
              <Bell className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">Select a trigger to view its history</p>
            </div>
          )}
        </div>
      </div>

      {/* Create Trigger Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/50" onClick={() => setShowCreateModal(false)} />
          <div className="relative bg-white rounded-xl shadow-2xl w-full max-w-lg m-4">
            <div className="flex items-center justify-between p-6 border-b">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-yellow-100 rounded-lg">
                  <Bell className="w-5 h-5 text-yellow-600" />
                </div>
                <h2 className="text-lg font-semibold text-gray-900">New Trigger</h2>
              </div>
              <button
                onClick={() => setShowCreateModal(false)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-gray-500" />
              </button>
            </div>

            <div className="p-6 space-y-4">
              {/* Name */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="My trigger"
                  className="input"
                />
              </div>

              {/* Event */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Event</label>
                <select
                  value={newEvent}
                  onChange={(e) => setNewEvent(e.target.value as TriggerEvent)}
                  className="input"
                >
                  {EVENT_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>

              {/* Conditions */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-700">Conditions</label>
                  <button onClick={addCondition} className="text-sm text-primary-600 hover:text-primary-700">
                    + Add Condition
                  </button>
                </div>
                {newConditions.map((cond, idx) => (
                  <div key={idx} className="flex items-center space-x-2 mb-2">
                    <input
                      type="text"
                      value={cond.field}
                      onChange={(e) => updateCondition(idx, 'field', e.target.value)}
                      placeholder="field"
                      className="input flex-1"
                    />
                    <select
                      value={cond.operator}
                      onChange={(e) => updateCondition(idx, 'operator', e.target.value)}
                      className="input w-24"
                    >
                      {OP_OPTIONS.map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                    <input
                      type="text"
                      value={cond.value}
                      onChange={(e) => updateCondition(idx, 'value', e.target.value)}
                      placeholder="value"
                      className="input flex-1"
                    />
                    <button
                      onClick={() => removeCondition(idx)}
                      className="p-1.5 text-red-400 hover:text-red-600"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>

              {/* Action */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Action</label>
                <select
                  value={newAction}
                  onChange={(e) => setNewAction(e.target.value as TriggerAction)}
                  className="input"
                >
                  {ACTION_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>

              {/* Webhook URL (if webhook action) */}
              {newAction === 'webhook' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Webhook URL</label>
                  <input
                    type="url"
                    value={newActionConfig.url || ''}
                    onChange={(e) => setNewActionConfig({ ...newActionConfig, url: e.target.value })}
                    placeholder="https://example.com/webhook"
                    className="input"
                  />
                </div>
              )}
            </div>

            <div className="p-6 border-t flex justify-end space-x-3">
              <button onClick={() => setShowCreateModal(false)} className="btn-secondary">
                Cancel
              </button>
              <button
                onClick={() => createMutation.mutate()}
                disabled={createMutation.isPending || !newName.trim()}
                className="btn-primary flex items-center space-x-2"
              >
                {createMutation.isPending && <RefreshCw className="w-4 h-4 animate-spin" />}
                <span>Create Trigger</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
