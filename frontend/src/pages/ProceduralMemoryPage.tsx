import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  BookOpen,
  RefreshCw,
  Play,
  Layers,
  Zap,
  ThumbsUp,
  ThumbsDown,
  X,
  CheckCircle,
  AlertCircle,
} from 'lucide-react';
import { apiClient } from '../services/api';
import clsx from 'clsx';

interface ProceduralMemoryPageProps {
  userId: string;
}

export function ProceduralMemoryPage({ userId }: ProceduralMemoryPageProps) {
  const queryClient = useQueryClient();
  const [showExtractModal, setShowExtractModal] = useState(false);
  const [showInjectModal, setShowInjectModal] = useState(false);
  const [extractText, setExtractText] = useState('');
  const [injectPrompt, setInjectPrompt] = useState('');
  const [injectionResult, setInjectionResult] = useState<{ prompt: string; rules_injected: number } | null>(null);
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

  const { data: rules = [], isLoading, isError, refetch } = useQuery({
    queryKey: ['procedural-rules', userId],
    queryFn: () => apiClient.getProceduralRules(userId),
  });

  const extractMutation = useMutation({
    mutationFn: () => {
      const interactions = extractText.split('\n').filter((l) => l.trim());
      return apiClient.extractProceduralRules(userId, interactions);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['procedural-rules', userId] });
      setShowExtractModal(false);
      setExtractText('');
    },
    onError: (err: unknown) => handleMutationError(err, 'extract rules'),
  });

  const injectMutation = useMutation({
    mutationFn: () => apiClient.injectProceduralRules(userId, injectPrompt),
    onSuccess: (result) => {
      setInjectionResult(result);
    },
    onError: (err: unknown) => handleMutationError(err, 'inject rules'),
  });

  const consolidateMutation = useMutation({
    mutationFn: () => apiClient.consolidateRules(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['procedural-rules', userId] });
    },
    onError: (err: unknown) => handleMutationError(err, 'consolidate rules'),
  });

  const feedbackMutation = useMutation({
    mutationFn: ({ ruleId, success }: { ruleId: string; success: boolean }) =>
      apiClient.updateRuleFeedback(ruleId, success),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['procedural-rules', userId] });
    },
    onError: (err: unknown) => handleMutationError(err, 'update rule feedback'),
  });

  const activeRules = rules.filter((r) => r.active);
  const avgConfidence = rules.length > 0
    ? rules.reduce((sum, r) => sum + r.confidence, 0) / rules.length
    : 0;
  const avgSuccessRate = rules.length > 0
    ? rules.reduce((sum, r) => sum + r.success_rate, 0) / rules.length
    : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-indigo-100 rounded-lg">
            <BookOpen className="w-6 h-6 text-indigo-600" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Procedural Memory</h1>
            <p className="text-sm text-gray-500">Extract and manage behavioral rules from interactions</p>
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
            onClick={() => setShowExtractModal(true)}
            className="btn-secondary flex items-center space-x-2"
          >
            <Play className="w-4 h-4" />
            <span>Extract</span>
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
            onClick={() => setShowInjectModal(true)}
            className="btn-primary flex items-center space-x-2"
          >
            <Zap className="w-4 h-4" />
            <span>Inject</span>
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {(isError || errorMessage) && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center space-x-3">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
          <p className="text-sm font-medium text-red-800 flex-1">
            {errorMessage || 'Failed to load procedural rules'}
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
              <BookOpen className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Total Rules</p>
              <p className="text-2xl font-bold text-gray-900">{rules.length}</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-green-100 rounded-lg">
              <CheckCircle className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Active Rules</p>
              <p className="text-2xl font-bold text-green-600">{activeRules.length}</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-purple-100 rounded-lg">
              <AlertCircle className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Avg Confidence</p>
              <p className="text-2xl font-bold text-purple-600">{(avgConfidence * 100).toFixed(0)}%</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-orange-100 rounded-lg">
              <Zap className="w-5 h-5 text-orange-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Avg Success Rate</p>
              <p className="text-2xl font-bold text-orange-600">{(avgSuccessRate * 100).toFixed(0)}%</p>
            </div>
          </div>
        </div>
      </div>

      {/* Rules List */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Rules</h3>
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-6 h-6 text-primary-600 animate-spin" />
          </div>
        ) : rules.length > 0 ? (
          <div className="space-y-4">
            {rules.map((rule) => (
              <div key={rule.id} className="card-hover">
                <div className="flex items-start justify-between">
                  <div className="flex-1 space-y-3">
                    <div className="flex items-center space-x-2">
                      <span className={clsx(
                        'badge',
                        rule.active ? 'badge-success' : 'badge-gray'
                      )}>
                        {rule.active ? 'Active' : 'Inactive'}
                      </span>
                      <span className="text-xs text-gray-400 font-mono">{rule.id.slice(0, 12)}</span>
                    </div>
                    <p className="text-gray-800">{rule.rule_text}</p>
                    <div className="flex items-center space-x-6">
                      {/* Confidence bar */}
                      <div className="flex items-center space-x-2 flex-1">
                        <span className="text-xs text-gray-500 w-20">Confidence</span>
                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-purple-500 h-2 rounded-full"
                            style={{ width: `${rule.confidence * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-600 w-10 text-right">
                          {(rule.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                      {/* Success rate bar */}
                      <div className="flex items-center space-x-2 flex-1">
                        <span className="text-xs text-gray-500 w-20">Success</span>
                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-green-500 h-2 rounded-full"
                            style={{ width: `${rule.success_rate * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-600 w-10 text-right">
                          {(rule.success_rate * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-1 ml-4">
                    <button
                      onClick={() => feedbackMutation.mutate({ ruleId: rule.id, success: true })}
                      className="p-1.5 text-gray-400 hover:text-green-600 hover:bg-green-50 rounded transition-colors"
                      title="Mark as successful"
                    >
                      <ThumbsUp className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => feedbackMutation.mutate({ ruleId: rule.id, success: false })}
                      className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                      title="Mark as unsuccessful"
                    >
                      <ThumbsDown className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="card text-center py-12 text-gray-400">
            <BookOpen className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>No procedural rules yet</p>
            <p className="text-sm mt-1">Extract rules from your interaction history</p>
          </div>
        )}
      </div>

      {/* Extract Modal */}
      {showExtractModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/50" onClick={() => setShowExtractModal(false)} />
          <div className="relative bg-white rounded-xl shadow-2xl w-full max-w-lg m-4">
            <div className="flex items-center justify-between p-6 border-b">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-indigo-100 rounded-lg">
                  <Play className="w-5 h-5 text-indigo-600" />
                </div>
                <h2 className="text-lg font-semibold text-gray-900">Extract Rules</h2>
              </div>
              <button
                onClick={() => setShowExtractModal(false)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-gray-500" />
              </button>
            </div>
            <div className="p-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Interactions (one per line)
              </label>
              <textarea
                value={extractText}
                onChange={(e) => setExtractText(e.target.value)}
                placeholder="Enter past interactions, one per line...&#10;User asked for X and I responded with Y&#10;When context was Z, the best approach was W"
                className="input h-40 resize-none"
              />
            </div>
            <div className="p-6 border-t flex justify-end space-x-3">
              <button onClick={() => setShowExtractModal(false)} className="btn-secondary">
                Cancel
              </button>
              <button
                onClick={() => extractMutation.mutate()}
                disabled={extractMutation.isPending || !extractText.trim()}
                className="btn-primary flex items-center space-x-2"
              >
                {extractMutation.isPending && <RefreshCw className="w-4 h-4 animate-spin" />}
                <span>Extract Rules</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Inject Modal */}
      {showInjectModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/50" onClick={() => { setShowInjectModal(false); setInjectionResult(null); }} />
          <div className="relative bg-white rounded-xl shadow-2xl w-full max-w-2xl m-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between p-6 border-b">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-indigo-100 rounded-lg">
                  <Zap className="w-5 h-5 text-indigo-600" />
                </div>
                <h2 className="text-lg font-semibold text-gray-900">Inject Rules into Prompt</h2>
              </div>
              <button
                onClick={() => { setShowInjectModal(false); setInjectionResult(null); }}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-gray-500" />
              </button>
            </div>
            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Base Prompt</label>
                <textarea
                  value={injectPrompt}
                  onChange={(e) => setInjectPrompt(e.target.value)}
                  placeholder="Enter the base prompt to inject rules into..."
                  className="input h-32 resize-none"
                />
              </div>
              <button
                onClick={() => injectMutation.mutate()}
                disabled={injectMutation.isPending || !injectPrompt.trim()}
                className="btn-primary flex items-center space-x-2"
              >
                {injectMutation.isPending && <RefreshCw className="w-4 h-4 animate-spin" />}
                <span>Inject Rules</span>
              </button>

              {injectionResult && (
                <div className="space-y-3">
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="w-5 h-5 text-green-600" />
                    <span className="text-sm font-medium text-green-700">
                      {injectionResult.rules_injected} rules injected
                    </span>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Resulting Prompt
                    </label>
                    <div className="bg-gray-900 rounded-lg p-4">
                      <pre className="text-xs text-green-400 font-mono whitespace-pre-wrap">
                        {injectionResult.prompt}
                      </pre>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
