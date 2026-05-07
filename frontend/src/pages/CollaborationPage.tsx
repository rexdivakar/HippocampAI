import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Users,
  RefreshCw,
  Plus,
  Trash2,
  Bell,
  AlertTriangle,
  CheckCircle,
  ChevronDown,
  ChevronRight,
  X,
  Activity,
  Database,
} from 'lucide-react';
import { apiClient } from '../services/api';
import type {
  CollaborationSpace,
  CollaborationEvent,
  Notification,
  CollaborationConflict,
} from '../types';
import clsx from 'clsx';
import { formatDistanceToNow } from 'date-fns';

interface CollaborationPageProps {
  userId: string;
}

const priorityBadgeColor = (priority: Notification['priority']): string => {
  switch (priority) {
    case 'high': return 'bg-red-100 text-red-700';
    case 'medium': return 'bg-yellow-100 text-yellow-700';
    case 'low': return 'bg-blue-100 text-blue-700';
    default: return 'bg-gray-100 text-gray-700';
  }
};

const conflictStatusColor = (status: CollaborationConflict['status']): string => {
  switch (status) {
    case 'open': return 'bg-orange-100 text-orange-700';
    case 'resolved': return 'bg-green-100 text-green-700';
    default: return 'bg-gray-100 text-gray-700';
  }
};

// Sub-component to render expanded space events to keep query isolation clean
function SpaceEvents({ spaceId }: { spaceId: string }) {
  const { data: events = [], isLoading } = useQuery<CollaborationEvent[]>({
    queryKey: ['space-events', spaceId],
    queryFn: () => apiClient.getSpaceEvents(spaceId, 20),
  });

  if (isLoading) {
    return (
      <div className="flex items-center space-x-2 py-4 text-gray-400 text-sm">
        <RefreshCw className="w-4 h-4 animate-spin" />
        <span>Loading events...</span>
      </div>
    );
  }

  if (events.length === 0) {
    return (
      <p className="text-sm text-gray-400 py-2">No events recorded for this space.</p>
    );
  }

  return (
    <ul className="space-y-2">
      {events.map((event) => (
        <li key={event.id} className="flex items-start space-x-3 text-sm">
          <Activity className="w-4 h-4 text-indigo-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2">
              <span className="font-medium text-gray-700">{event.event_type}</span>
              <span className="text-gray-400 text-xs">
                {formatDistanceToNow(new Date(event.timestamp), { addSuffix: true })}
              </span>
            </div>
            <p className="text-gray-500 text-xs font-mono truncate">agent: {event.agent_id}</p>
          </div>
        </li>
      ))}
    </ul>
  );
}

export function CollaborationPage({ userId }: CollaborationPageProps) {
  const queryClient = useQueryClient();

  // UI state
  const [expandedSpaceId, setExpandedSpaceId] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Create form state
  const [newName, setNewName] = useState('');
  const [newDescription, setNewDescription] = useState('');
  const [newTags, setNewTags] = useState('');

  // Auto-dismiss error banner after 5 seconds
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

  // ── Queries ────────────────────────────────────────────────────────────────

  const {
    data: spaces = [],
    isLoading: spacesLoading,
    isError: spacesError,
    refetch: refetchSpaces,
  } = useQuery<CollaborationSpace[]>({
    queryKey: ['collaboration-spaces'],
    queryFn: () => apiClient.listSpaces(userId),
  });

  const {
    data: notifications = [],
    isLoading: notificationsLoading,
    refetch: refetchNotifications,
  } = useQuery<Notification[]>({
    queryKey: ['collaboration-notifications', userId],
    queryFn: () => apiClient.getNotifications(userId, true, 50),
  });

  const {
    data: conflicts = [],
    isLoading: conflictsLoading,
    refetch: refetchConflicts,
  } = useQuery<CollaborationConflict[]>({
    queryKey: ['collaboration-conflicts'],
    queryFn: () => apiClient.getConflicts(),
  });

  // ── Mutations ──────────────────────────────────────────────────────────────

  const createSpaceMutation = useMutation({
    mutationFn: () =>
      apiClient.createSpace({
        name: newName.trim(),
        owner_agent_id: userId,
        description: newDescription.trim() || undefined,
        tags: newTags.trim()
          ? newTags.split(',').map((t) => t.trim()).filter(Boolean)
          : undefined,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['collaboration-spaces'] });
      resetCreateForm();
      setShowCreateModal(false);
    },
    onError: (err: unknown) => handleMutationError(err, 'create space'),
  });

  const deleteSpaceMutation = useMutation({
    mutationFn: (spaceId: string) => apiClient.deleteSpace(spaceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['collaboration-spaces'] });
      setExpandedSpaceId(null);
    },
    onError: (err: unknown) => handleMutationError(err, 'delete space'),
  });

  const markReadMutation = useMutation({
    mutationFn: (notificationId: string) =>
      apiClient.markNotificationRead(userId, notificationId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['collaboration-notifications', userId] });
    },
    onError: (err: unknown) => handleMutationError(err, 'mark notification as read'),
  });

  const resolveConflictMutation = useMutation({
    mutationFn: ({ conflictId, resolution }: { conflictId: string; resolution: string }) =>
      apiClient.resolveConflict(conflictId, resolution),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['collaboration-conflicts'] });
    },
    onError: (err: unknown) => handleMutationError(err, 'resolve conflict'),
  });

  // ── Derived stats ──────────────────────────────────────────────────────────

  const totalCollaborators = spaces.reduce((sum, s) => sum + s.collaborators.length, 0);
  const unreadCount = notifications.filter((n) => !n.is_read).length;
  const openConflicts = conflicts.filter((c) => c.status === 'open').length;

  const resetCreateForm = () => {
    setNewName('');
    setNewDescription('');
    setNewTags('');
  };

  const handleRefreshAll = () => {
    void refetchSpaces();
    void refetchNotifications();
    void refetchConflicts();
  };

  const isAnyLoading = spacesLoading || notificationsLoading || conflictsLoading;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-indigo-100 rounded-lg">
            <Users className="w-6 h-6 text-indigo-600" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Collaboration</h1>
            <p className="text-sm text-gray-500">Shared memory spaces and team coordination</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleRefreshAll}
            disabled={isAnyLoading}
            className="btn-secondary flex items-center space-x-2"
          >
            <RefreshCw className={clsx('w-4 h-4', isAnyLoading && 'animate-spin')} />
            <span>Refresh</span>
          </button>
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn-primary flex items-center space-x-2"
          >
            <Plus className="w-4 h-4" />
            <span>New Space</span>
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {(spacesError || errorMessage) && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center space-x-3">
          <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0" />
          <p className="text-sm font-medium text-red-800 flex-1">
            {errorMessage || 'Failed to load collaboration data'}
          </p>
          <button
            onClick={() => { setErrorMessage(null); void refetchSpaces(); }}
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
            <div className="p-2 bg-indigo-100 rounded-lg">
              <Users className="w-5 h-5 text-indigo-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Total Spaces</p>
              <p className="text-2xl font-bold text-gray-900">{spaces.length}</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Activity className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Collaborators</p>
              <p className="text-2xl font-bold text-blue-600">{totalCollaborators}</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-yellow-100 rounded-lg">
              <Bell className="w-5 h-5 text-yellow-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Unread Alerts</p>
              <p className="text-2xl font-bold text-yellow-600">{unreadCount}</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-orange-100 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-orange-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Open Conflicts</p>
              <p className="text-2xl font-bold text-orange-600">{openConflicts}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main grid: spaces list (2 cols) + notifications panel (1 col) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Spaces List */}
        <div className="lg:col-span-2 space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">Spaces</h3>
          {spacesLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-6 h-6 text-primary-600 animate-spin" />
            </div>
          ) : spaces.length > 0 ? (
            spaces.map((space) => {
              const isExpanded = expandedSpaceId === space.id;
              return (
                <div key={space.id} className="card">
                  {/* Space header row */}
                  <div
                    className="flex items-start justify-between cursor-pointer"
                    onClick={() => setExpandedSpaceId(isExpanded ? null : space.id)}
                  >
                    <div className="flex items-start space-x-3 flex-1 min-w-0">
                      <button className="mt-0.5 text-gray-400 flex-shrink-0">
                        {isExpanded ? (
                          <ChevronDown className="w-5 h-5" />
                        ) : (
                          <ChevronRight className="w-5 h-5" />
                        )}
                      </button>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2 flex-wrap gap-y-1">
                          <span className="font-medium text-gray-900">{space.name}</span>
                          <span
                            className={clsx(
                              'badge text-xs',
                              space.is_active ? 'badge-success' : 'badge-gray'
                            )}
                          >
                            {space.is_active ? 'Active' : 'Inactive'}
                          </span>
                          {space.tags.map((tag) => (
                            <span key={tag} className="badge bg-gray-100 text-gray-600 text-xs">
                              {tag}
                            </span>
                          ))}
                        </div>
                        {space.description && (
                          <p className="text-sm text-gray-500 mt-0.5 truncate">{space.description}</p>
                        )}
                        <div className="flex items-center space-x-4 mt-1 text-xs text-gray-400">
                          <span className="flex items-center space-x-1">
                            <Users className="w-3 h-3" />
                            <span>{space.collaborators.length} collaborators</span>
                          </span>
                          <span className="flex items-center space-x-1">
                            <Database className="w-3 h-3" />
                            <span>{space.memory_ids.length} memories</span>
                          </span>
                          <span>
                            {formatDistanceToNow(new Date(space.created_at), { addSuffix: true })}
                          </span>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (window.confirm(`Delete space "${space.name}"? This cannot be undone.`)) {
                          deleteSpaceMutation.mutate(space.id);
                        }
                      }}
                      className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors ml-2 flex-shrink-0"
                      title="Delete space"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Expanded detail */}
                  {isExpanded && (
                    <div className="mt-4 pt-4 border-t space-y-4">
                      {/* Collaborators */}
                      <div>
                        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                          Collaborators
                        </h4>
                        {space.collaborators.length > 0 ? (
                          <div className="flex flex-wrap gap-2">
                            {space.collaborators.map((agentId) => (
                              <span
                                key={agentId}
                                className="inline-flex items-center space-x-1 bg-indigo-50 text-indigo-700 text-xs font-mono px-2 py-1 rounded"
                              >
                                <Users className="w-3 h-3" />
                                <span>{agentId}</span>
                              </span>
                            ))}
                          </div>
                        ) : (
                          <p className="text-sm text-gray-400">No collaborators yet.</p>
                        )}
                      </div>

                      {/* Memory count */}
                      <div>
                        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                          Shared Memories
                        </h4>
                        <p className="text-sm text-gray-600">
                          {space.memory_ids.length === 0
                            ? 'No memories linked to this space.'
                            : `${space.memory_ids.length} memor${space.memory_ids.length === 1 ? 'y' : 'ies'} linked`}
                        </p>
                      </div>

                      {/* Recent events */}
                      <div>
                        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                          Recent Events
                        </h4>
                        <SpaceEvents spaceId={space.id} />
                      </div>
                    </div>
                  )}
                </div>
              );
            })
          ) : (
            <div className="card text-center py-12 text-gray-400">
              <Users className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>No collaboration spaces yet</p>
              <p className="text-sm mt-1">Create a space to start sharing memories with other agents</p>
            </div>
          )}
        </div>

        {/* Notifications Panel */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">Notifications</h3>
          {notificationsLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-6 h-6 text-primary-600 animate-spin" />
            </div>
          ) : notifications.length > 0 ? (
            <div className="space-y-3">
              {notifications.map((notification) => (
                <div
                  key={notification.id}
                  className={clsx(
                    'card text-sm',
                    !notification.is_read && 'border-l-4 border-l-indigo-400'
                  )}
                >
                  <div className="flex items-start justify-between space-x-2">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2 mb-1 flex-wrap gap-y-1">
                        <span className="font-medium text-gray-900 text-sm">{notification.title}</span>
                        <span className={clsx('badge text-xs', priorityBadgeColor(notification.priority))}>
                          {notification.priority}
                        </span>
                      </div>
                      <p className="text-gray-600 text-xs leading-relaxed">{notification.message}</p>
                      <p className="text-gray-400 text-xs mt-1">
                        {formatDistanceToNow(new Date(notification.created_at), { addSuffix: true })}
                      </p>
                    </div>
                    {!notification.is_read && (
                      <button
                        onClick={() => markReadMutation.mutate(notification.id)}
                        disabled={markReadMutation.isPending}
                        className="p-1 text-gray-400 hover:text-green-600 hover:bg-green-50 rounded transition-colors flex-shrink-0"
                        title="Mark as read"
                      >
                        <CheckCircle className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="card text-center py-8 text-gray-400">
              <Bell className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No unread notifications</p>
            </div>
          )}
        </div>
      </div>

      {/* Conflicts Section */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-900">Conflicts</h3>
        {conflictsLoading ? (
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="w-6 h-6 text-primary-600 animate-spin" />
          </div>
        ) : conflicts.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {conflicts.map((conflict) => (
              <div key={conflict.id} className="card">
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0 space-y-2">
                    <div className="flex items-center space-x-2 flex-wrap gap-y-1">
                      <span className={clsx('badge text-xs', conflictStatusColor(conflict.status))}>
                        {conflict.status}
                      </span>
                      <span className="badge bg-gray-100 text-gray-600 text-xs">
                        {conflict.conflict_type}
                      </span>
                    </div>
                    <p className="text-sm text-gray-700">{conflict.description}</p>
                    <div className="text-xs text-gray-400 space-y-0.5">
                      <p>Space: <span className="font-mono">{conflict.space_id.slice(0, 12)}...</span></p>
                      <p>Memory: <span className="font-mono">{conflict.memory_id.slice(0, 12)}...</span></p>
                      <p>
                        {formatDistanceToNow(new Date(conflict.created_at), { addSuffix: true })}
                      </p>
                    </div>
                  </div>
                  {conflict.status === 'open' && (
                    <button
                      onClick={() => {
                        const resolution = window.prompt('Enter resolution description:');
                        if (resolution?.trim()) {
                          resolveConflictMutation.mutate({
                            conflictId: conflict.id,
                            resolution: resolution.trim(),
                          });
                        }
                      }}
                      disabled={resolveConflictMutation.isPending}
                      className="ml-3 flex-shrink-0 p-1.5 text-gray-400 hover:text-green-600 hover:bg-green-50 rounded transition-colors"
                      title="Resolve conflict"
                    >
                      <CheckCircle className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="card text-center py-8 text-gray-400">
            <CheckCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No conflicts detected</p>
          </div>
        )}
      </div>

      {/* Create Space Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => { setShowCreateModal(false); resetCreateForm(); }}
          />
          <div className="relative bg-white rounded-xl shadow-2xl w-full max-w-lg m-4">
            <div className="flex items-center justify-between p-6 border-b">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-indigo-100 rounded-lg">
                  <Users className="w-5 h-5 text-indigo-600" />
                </div>
                <h2 className="text-lg font-semibold text-gray-900">New Collaboration Space</h2>
              </div>
              <button
                onClick={() => { setShowCreateModal(false); resetCreateForm(); }}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-gray-500" />
              </button>
            </div>

            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Name <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="My collaboration space"
                  className="input"
                  autoFocus
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                <textarea
                  value={newDescription}
                  onChange={(e) => setNewDescription(e.target.value)}
                  placeholder="What is this space for?"
                  className="input h-24 resize-none"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Tags
                  <span className="text-gray-400 font-normal ml-1">(comma-separated)</span>
                </label>
                <input
                  type="text"
                  value={newTags}
                  onChange={(e) => setNewTags(e.target.value)}
                  placeholder="research, shared, project-x"
                  className="input"
                />
              </div>
            </div>

            <div className="p-6 border-t flex justify-end space-x-3">
              <button
                onClick={() => { setShowCreateModal(false); resetCreateForm(); }}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={() => createSpaceMutation.mutate()}
                disabled={createSpaceMutation.isPending || !newName.trim()}
                className="btn-primary flex items-center space-x-2"
              >
                {createSpaceMutation.isPending && <RefreshCw className="w-4 h-4 animate-spin" />}
                <span>Create Space</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
