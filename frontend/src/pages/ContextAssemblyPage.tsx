import { useState, useEffect, useRef, useCallback } from 'react';
import { apiClient } from '../services/api';
import type { ContextPack } from '../types';
import {
  Layers,
  Search,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  FileText,
  AlertTriangle,
  Settings,
  RefreshCw,
  XCircle,
} from 'lucide-react';

interface ContextAssemblyPageProps {
  userId: string;
}

export function ContextAssemblyPage({ userId }: ContextAssemblyPageProps) {
  const [query, setQuery] = useState('');
  const [contextPack, setContextPack] = useState<ContextPack | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showDropped, setShowDropped] = useState(false);
  const [retryCount, setRetryCount] = useState(0);

  // Settings
  const [tokenBudget, setTokenBudget] = useState(4000);
  const [maxItems, setMaxItems] = useState(20);
  const [recencyBias, setRecencyBias] = useState(0.3);
  const [minRelevance, setMinRelevance] = useState(0.1);
  const [includeCitations, setIncludeCitations] = useState(true);
  const [deduplicate, setDeduplicate] = useState(true);
  const [typeFilter, setTypeFilter] = useState<string[]>([]);

  // AbortController for cancellable requests
  const abortControllerRef = useRef<AbortController | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const handleAssemble = useCallback(async (retrying = false) => {
    if (!query.trim()) return;

    // Cancel previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    setLoading(true);
    if (!retrying) {
      setError(null);
      setRetryCount(0);
    }

    try {
      const result = await apiClient.assembleContext({
        user_id: userId,
        query: query,
        token_budget: tokenBudget,
        max_items: maxItems,
        recency_bias: recencyBias,
        min_relevance: minRelevance,
        include_citations: includeCitations,
        deduplicate: deduplicate,
        type_filter: typeFilter.length > 0 ? typeFilter : undefined,
      });
      setContextPack(result);
      setError(null);
    } catch (err: any) {
      // Ignore abort errors
      if (err.name === 'AbortError' || err.name === 'CanceledError') return;
      
      // User-friendly error messages
      let errorMessage = 'Failed to assemble context';
      if (err.response?.status === 429) {
        errorMessage = 'Too many requests. Please wait a moment and try again.';
      } else if (err.response?.status >= 500) {
        errorMessage = 'Server error. Please try again later.';
      } else if (err.code === 'ECONNABORTED') {
        errorMessage = 'Request timed out. Try reducing the token budget or max items.';
      } else if (!navigator.onLine) {
        errorMessage = 'No internet connection.';
      } else {
        errorMessage = err.message || 'Failed to assemble context';
      }
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [query, userId, tokenBudget, maxItems, recencyBias, minRelevance, includeCitations, deduplicate, typeFilter]);

  const handleRetry = useCallback(() => {
    setRetryCount(prev => prev + 1);
    handleAssemble(true);
  }, [handleAssemble]);

  const handleCancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setLoading(false);
  }, []);

  const handleCopy = async () => {
    if (!contextPack) return;
    await navigator.clipboard.writeText(contextPack.final_context_text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getReasonLabel = (reason: string) => {
    switch (reason) {
      case 'token_budget':
        return 'Token limit exceeded';
      case 'low_relevance':
        return 'Low relevance';
      case 'duplicate':
        return 'Duplicate content';
      case 'max_items':
        return 'Max items reached';
      case 'type_filter':
        return 'Filtered by type';
      default:
        return reason;
    }
  };

  const memoryTypes = ['fact', 'preference', 'goal', 'habit', 'event', 'context'];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <Layers className="w-7 h-7 text-purple-500" />
          Context Assembly
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Automatically assemble relevant context from memories for LLM prompts
        </p>
      </div>

      {/* Query Input */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <div className="flex gap-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Query / Prompt
            </label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your query or prompt to find relevant context..."
              rows={3}
              className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white resize-none"
            />
          </div>
        </div>

        <div className="flex items-center justify-between mt-4">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
          >
            <Settings className="w-4 h-4" />
            {showSettings ? 'Hide Settings' : 'Show Settings'}
            {showSettings ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          <div className="flex gap-2">
            {loading && (
              <button
                onClick={handleCancel}
                className="flex items-center gap-2 px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
              >
                <XCircle className="w-4 h-4" />
                Cancel
              </button>
            )}
            <button
              onClick={() => handleAssemble(false)}
              disabled={!query.trim() || loading}
              className="flex items-center gap-2 px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Search className="w-4 h-4" />
              )}
              {loading ? 'Assembling...' : 'Assemble Context'}
            </button>
          </div>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Token Budget
                </label>
                <input
                  type="number"
                  value={tokenBudget}
                  onChange={(e) => setTokenBudget(parseInt(e.target.value) || 4000)}
                  min={100}
                  max={32000}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Max Items
                </label>
                <input
                  type="number"
                  value={maxItems}
                  onChange={(e) => setMaxItems(parseInt(e.target.value) || 20)}
                  min={1}
                  max={100}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Recency Bias
                </label>
                <input
                  type="number"
                  value={recencyBias}
                  onChange={(e) => setRecencyBias(parseFloat(e.target.value) || 0.3)}
                  min={0}
                  max={1}
                  step={0.1}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Min Relevance
                </label>
                <input
                  type="number"
                  value={minRelevance}
                  onChange={(e) => setMinRelevance(parseFloat(e.target.value) || 0.1)}
                  min={0}
                  max={1}
                  step={0.1}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              </div>
            </div>

            <div className="flex flex-wrap gap-4 mt-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeCitations}
                  onChange={(e) => setIncludeCitations(e.target.checked)}
                  className="w-4 h-4 text-purple-600 rounded"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">Include Citations</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={deduplicate}
                  onChange={(e) => setDeduplicate(e.target.checked)}
                  className="w-4 h-4 text-purple-600 rounded"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">Deduplicate</span>
              </label>
            </div>

            <div className="mt-4">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Type Filter (select to include)
              </label>
              <div className="flex flex-wrap gap-2">
                {memoryTypes.map((type) => (
                  <button
                    key={type}
                    onClick={() => {
                      if (typeFilter.includes(type)) {
                        setTypeFilter(typeFilter.filter((t) => t !== type));
                      } else {
                        setTypeFilter([...typeFilter, type]);
                      }
                    }}
                    className={`px-3 py-1 text-sm rounded-full transition-colors ${
                      typeFilter.includes(type)
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                    }`}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Error Message with Retry */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-red-700 dark:text-red-400">{error}</p>
                {retryCount > 0 && (
                  <p className="text-sm text-red-600 dark:text-red-500 mt-1">
                    Retry attempt {retryCount} failed
                  </p>
                )}
              </div>
            </div>
            <button
              onClick={handleRetry}
              disabled={loading}
              className="flex items-center gap-1 px-3 py-1.5 text-sm bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 rounded hover:bg-red-200 dark:hover:bg-red-900/50 disabled:opacity-50"
            >
              <RefreshCw className="w-4 h-4" />
              Retry
            </button>
          </div>
        </div>
      )}

      {/* Results */}
      {contextPack && (
        <div className="space-y-6">
          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm">
              <div className="text-2xl font-bold text-purple-600">{contextPack.total_tokens}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Total Tokens</div>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm">
              <div className="text-2xl font-bold text-green-600">{contextPack.selected_items.length}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Selected Items</div>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm">
              <div className="text-2xl font-bold text-yellow-600">{contextPack.dropped_items.length}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Dropped Items</div>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm">
              <div className="text-2xl font-bold text-blue-600">{contextPack.citations.length}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Citations</div>
            </div>
          </div>

          {/* Assembled Context */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm">
            <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                <FileText className="w-5 h-5" />
                Assembled Context
              </h2>
              <button
                onClick={handleCopy}
                className="flex items-center gap-2 px-3 py-1.5 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
              >
                {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                {copied ? 'Copied!' : 'Copy'}
              </button>
            </div>
            <div className="p-4">
              {contextPack.final_context_text ? (
                <pre className="whitespace-pre-wrap text-sm text-gray-900 dark:text-white font-mono bg-gray-50 dark:bg-gray-900/50 p-4 rounded-lg overflow-x-auto">
                  {contextPack.final_context_text}
                </pre>
              ) : (
                <p className="text-gray-500 dark:text-gray-400 italic">No context assembled (no relevant memories found)</p>
              )}
            </div>
          </div>

          {/* Selected Items */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Selected Items ({contextPack.selected_items.length})
              </h2>
            </div>
            <div className="divide-y divide-gray-200 dark:divide-gray-700">
              {contextPack.selected_items.map((item, index) => (
                <div key={item.memory_id} className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-medium text-gray-500 dark:text-gray-400">#{index + 1}</span>
                        <span className={`text-xs px-2 py-0.5 rounded-full ${
                          item.memory_type === 'fact' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' :
                          item.memory_type === 'preference' ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400' :
                          'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
                        }`}>
                          {item.memory_type}
                        </span>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {item.token_count} tokens
                        </span>
                      </div>
                      <p className="text-gray-900 dark:text-white">{item.text}</p>
                      <div className="flex items-center gap-4 mt-2 text-xs text-gray-500 dark:text-gray-400">
                        <span>Relevance: {(item.relevance_score * 100).toFixed(1)}%</span>
                        <span>Importance: {item.importance.toFixed(1)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
              {contextPack.selected_items.length === 0 && (
                <div className="p-4 text-center text-gray-500 dark:text-gray-400">
                  No items selected
                </div>
              )}
            </div>
          </div>

          {/* Dropped Items */}
          {contextPack.dropped_items.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm">
              <button
                onClick={() => setShowDropped(!showDropped)}
                className="w-full p-4 flex items-center justify-between border-b border-gray-200 dark:border-gray-700"
              >
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-yellow-500" />
                  Dropped Items ({contextPack.dropped_items.length})
                </h2>
                {showDropped ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </button>
              {showDropped && (
                <div className="divide-y divide-gray-200 dark:divide-gray-700">
                  {contextPack.dropped_items.map((item) => (
                    <div key={item.memory_id} className="p-4 bg-gray-50 dark:bg-gray-900/50">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-xs px-2 py-0.5 bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400 rounded-full">
                              {getReasonLabel(item.reason)}
                            </span>
                            <span className="text-xs text-gray-500 dark:text-gray-400">
                              Relevance: {(item.relevance_score * 100).toFixed(1)}%
                            </span>
                          </div>
                          <p className="text-gray-600 dark:text-gray-400 text-sm">{item.text}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
