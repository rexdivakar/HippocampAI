import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  ArrowRightLeft,
  RefreshCw,
  Play,
  XCircle,
  CheckCircle,
  AlertTriangle,
  Shield,
} from 'lucide-react';
import { apiClient } from '../services/api';
import type { MigrationStatus, EmbeddingMigration } from '../types';
import clsx from 'clsx';

interface EmbeddingMigrationPageProps {
  userId: string;
}

const statusColors: Record<MigrationStatus, string> = {
  pending: 'bg-yellow-100 text-yellow-700',
  in_progress: 'bg-blue-100 text-blue-700',
  completed: 'bg-green-100 text-green-700',
  failed: 'bg-red-100 text-red-700',
  cancelled: 'bg-gray-100 text-gray-700',
};

const statusIcons: Record<MigrationStatus, typeof CheckCircle> = {
  pending: RefreshCw,
  in_progress: RefreshCw,
  completed: CheckCircle,
  failed: AlertTriangle,
  cancelled: XCircle,
};

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export function EmbeddingMigrationPage({ userId: _userId }: EmbeddingMigrationPageProps) {
  const queryClient = useQueryClient();
  const [newModel, setNewModel] = useState('');
  const [newDimensions, setNewDimensions] = useState('');
  const [confirmStart, setConfirmStart] = useState(false);
  const [migrationId, setMigrationId] = useState<string | null>(null);
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

  const { data: migration, isLoading } = useQuery({
    queryKey: ['migration-status', migrationId],
    queryFn: () => apiClient.getMigrationStatus(migrationId!),
    enabled: !!migrationId,
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.status === 'in_progress' || data?.status === 'pending' ? 3000 : false;
    },
  });

  const startMutation = useMutation({
    mutationFn: () =>
      apiClient.startEmbeddingMigration({
        new_model: newModel,
        new_dimension: newDimensions ? parseInt(newDimensions, 10) : undefined,
      }),
    onSuccess: (data: EmbeddingMigration) => {
      setMigrationId(data.id);
      queryClient.invalidateQueries({ queryKey: ['migration-status'] });
      setNewModel('');
      setNewDimensions('');
      setConfirmStart(false);
    },
    onError: (err: unknown) => handleMutationError(err, 'start migration'),
  });

  const cancelMutation = useMutation({
    mutationFn: () => apiClient.cancelMigration(migrationId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['migration-status', migrationId] });
    },
    onError: (err: unknown) => handleMutationError(err, 'cancel migration'),
  });

  const isActive = migration && (migration.status === 'in_progress' || migration.status === 'pending');
  const progressPercent = migration && migration.total_memories > 0
    ? (migration.migrated_count / migration.total_memories) * 100
    : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-teal-100 rounded-lg">
            <ArrowRightLeft className="w-6 h-6 text-teal-600" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Embedding Migration</h1>
            <p className="text-sm text-gray-500">Migrate memory embeddings to a new model</p>
          </div>
          <span className="badge bg-orange-100 text-orange-700 flex items-center space-x-1">
            <Shield className="w-3 h-3" />
            <span>Admin</span>
          </span>
        </div>
      </div>

      {/* Error Banner */}
      {errorMessage && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center space-x-3">
          <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0" />
          <p className="text-sm font-medium text-red-800 flex-1">{errorMessage}</p>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-sm text-red-700 hover:text-red-900 font-medium"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Migration ID Input (to resume tracking) */}
      {!migration && !isLoading && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Track Existing Migration</h3>
          <div className="flex items-center space-x-3">
            <input
              type="text"
              value={migrationId || ''}
              onChange={(e) => setMigrationId(e.target.value || null)}
              placeholder="Enter migration ID to track progress..."
              className="input flex-1"
            />
          </div>
        </div>
      )}

      {/* Current Migration Status */}
      {migration && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Migration Status</h3>
          <div className="space-y-4">
            {/* Status Badge */}
            <div className="flex items-center space-x-3">
              {(() => {
                const StatusIcon = statusIcons[migration.status];
                return (
                  <StatusIcon className={clsx(
                    'w-5 h-5',
                    migration.status === 'in_progress' && 'animate-spin text-blue-600',
                    migration.status === 'completed' && 'text-green-600',
                    migration.status === 'failed' && 'text-red-600',
                    migration.status === 'cancelled' && 'text-gray-600',
                    migration.status === 'pending' && 'text-yellow-600',
                  )} />
                );
              })()}
              <span className={clsx('badge', statusColors[migration.status])}>
                {migration.status.replace(/_/g, ' ').toUpperCase()}
              </span>
              <span className="text-xs text-gray-400 font-mono">ID: {migration.id}</span>
            </div>

            {/* Model Info */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                <p className="text-xs text-gray-500 mb-1">Old Model</p>
                <p className="text-sm font-medium text-gray-900 font-mono">
                  {migration.old_model || 'N/A'}
                </p>
              </div>
              <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                <p className="text-xs text-blue-600 mb-1">New Model</p>
                <p className="text-sm font-medium text-blue-900 font-mono">
                  {migration.new_model}
                </p>
              </div>
            </div>

            {/* Progress Bar */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">Progress</span>
                <span className="text-sm font-medium text-gray-900">
                  {migration.migrated_count} / {migration.total_memories}
                  {migration.failed_count > 0 && (
                    <span className="text-red-600 ml-2">({migration.failed_count} failed)</span>
                  )}
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className={clsx(
                    'h-3 rounded-full transition-all duration-500',
                    migration.status === 'completed' ? 'bg-green-500' :
                    migration.status === 'failed' ? 'bg-red-500' :
                    'bg-blue-500'
                  )}
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">{progressPercent.toFixed(1)}% complete</p>
            </div>

            {/* Cancel button (only when active) */}
            {isActive && (
              <button
                onClick={() => {
                  if (window.confirm('Are you sure you want to cancel this migration?')) {
                    cancelMutation.mutate();
                  }
                }}
                disabled={cancelMutation.isPending}
                className="btn-danger flex items-center space-x-2"
              >
                {cancelMutation.isPending ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <XCircle className="w-4 h-4" />
                )}
                <span>Cancel Migration</span>
              </button>
            )}

            {/* Completed Summary */}
            {migration.status === 'completed' && (
              <div className="bg-green-50 rounded-lg p-4 border border-green-200 flex items-start space-x-3">
                <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-green-800">Migration Completed</p>
                  <p className="text-sm text-green-700">
                    Successfully migrated {migration.migrated_count} memories from{' '}
                    <span className="font-mono">{migration.old_model}</span> to{' '}
                    <span className="font-mono">{migration.new_model}</span>.
                    {migration.failed_count > 0 && ` ${migration.failed_count} memories failed.`}
                  </p>
                </div>
              </div>
            )}

            {/* Failed Summary */}
            {migration.status === 'failed' && (
              <div className="bg-red-50 rounded-lg p-4 border border-red-200 flex items-start space-x-3">
                <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-red-800">Migration Failed</p>
                  <p className="text-sm text-red-700">
                    {migration.migrated_count} of {migration.total_memories} memories were migrated before failure.
                    {migration.failed_count > 0 && ` ${migration.failed_count} memories failed.`}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Start New Migration */}
      {!isActive && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Start New Migration</h3>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  New Model Name
                </label>
                <input
                  type="text"
                  value={newModel}
                  onChange={(e) => setNewModel(e.target.value)}
                  placeholder="e.g., text-embedding-3-large"
                  className="input"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Dimension (optional, default 384)
                </label>
                <input
                  type="number"
                  value={newDimensions}
                  onChange={(e) => setNewDimensions(e.target.value)}
                  placeholder="e.g., 1536"
                  className="input"
                />
              </div>
            </div>

            {!confirmStart ? (
              <button
                onClick={() => setConfirmStart(true)}
                disabled={!newModel.trim()}
                className="btn-primary flex items-center space-x-2"
              >
                <Play className="w-4 h-4" />
                <span>Start Migration</span>
              </button>
            ) : (
              <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
                <div className="flex items-start space-x-3">
                  <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm font-medium text-yellow-800">
                      Are you sure you want to start this migration?
                    </p>
                    <p className="text-sm text-yellow-700 mt-1">
                      This will re-embed all memories using <span className="font-mono">{newModel}</span>.
                      This operation may take a while and cannot be undone (only cancelled).
                    </p>
                    <div className="flex items-center space-x-3 mt-3">
                      <button
                        onClick={() => startMutation.mutate()}
                        disabled={startMutation.isPending}
                        className="btn-primary flex items-center space-x-2"
                      >
                        {startMutation.isPending ? (
                          <RefreshCw className="w-4 h-4 animate-spin" />
                        ) : (
                          <Play className="w-4 h-4" />
                        )}
                        <span>Confirm & Start</span>
                      </button>
                      <button
                        onClick={() => setConfirmStart(false)}
                        className="btn-secondary"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="w-6 h-6 text-primary-600 animate-spin" />
        </div>
      )}
    </div>
  );
}
