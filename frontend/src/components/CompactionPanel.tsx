import { useState, useEffect } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import {
  Layers,
  Eye,
  Loader2,
  CheckCircle,
  XCircle,
  TrendingDown,
  Coins,
  Clock,
  FileText,
  Zap,
  Brain,
  Database,
  Lightbulb,
  Tag,
  ChevronDown,
  ChevronUp,
  Info,
  Sparkles,
} from 'lucide-react';
import clsx from 'clsx';

interface CompactionPanelProps {
  userId: string;
  sessionId?: string;
}

interface CompactionMetrics {
  input_memories: number;
  input_tokens: number;
  input_characters: number;
  output_memories: number;
  output_tokens: number;
  output_characters: number;
  compression_ratio: number;
  tokens_saved: number;
  memories_merged: number;
  clusters_found: number;
  llm_calls: number;
  duration_seconds: number;
  estimated_input_cost: number;
  estimated_output_cost: number;
  estimated_storage_saved_bytes: number;
  avg_memory_size_before: number;
  avg_memory_size_after: number;
  types_compacted: Record<string, number>;
  key_facts_preserved: number;
  entities_preserved: number;
  context_retention_score: number;
}

interface CompactionResult {
  id: string;
  user_id: string;
  session_id: string | null;
  status: string;
  started_at: string;
  completed_at: string | null;
  metrics: CompactionMetrics;
  actions: Array<{ action: string; [key: string]: any }>;
  summary: string | null;
  insights: string[];
  preserved_facts: string[];
  preserved_entities: string[];
  config: Record<string, any>;
  dry_run: boolean;
  error: string | null;
}

interface TypeInfo {
  types: string[];
  descriptions: Record<string, string>;
}

const LOOKBACK_OPTIONS = [
  { value: 24, label: 'Last 24 hours' },
  { value: 72, label: 'Last 3 days' },
  { value: 168, label: 'Last week' },
  { value: 720, label: 'Last month' },
  { value: 2160, label: 'Last 3 months' },
];

export function CompactionPanel({ userId, sessionId }: CompactionPanelProps) {
  const [lastResult, setLastResult] = useState<CompactionResult | null>(null);
  const [lookbackHours, setLookbackHours] = useState(168);
  const [selectedTypes, setSelectedTypes] = useState<string[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [minMemories, setMinMemories] = useState(5);

  // Fetch available types
  const { data: typeInfo } = useQuery<TypeInfo>({
    queryKey: ['compaction-types'],
    queryFn: async () => {
      const res = await fetch('/api/compaction/types');
      return res.json();
    },
  });

  // Initialize selected types when typeInfo loads
  useEffect(() => {
    if (typeInfo?.types && selectedTypes.length === 0) {
      setSelectedTypes(typeInfo.types);
    }
  }, [typeInfo, selectedTypes.length]);

  // Preview query
  const { data: preview, isLoading: previewLoading, refetch: refetchPreview } = useQuery({
    queryKey: ['compaction-preview', userId, sessionId, lookbackHours, selectedTypes],
    queryFn: async () => {
      const params = new URLSearchParams({
        user_id: userId,
        lookback_hours: lookbackHours.toString(),
      });
      if (sessionId) params.append('session_id', sessionId);
      if (selectedTypes.length > 0 && selectedTypes.length < (typeInfo?.types.length || 0)) {
        params.append('memory_types', selectedTypes.join(','));
      }
      const res = await fetch(`/api/compaction/preview?${params}`);
      return res.json();
    },
    enabled: selectedTypes.length > 0,
  });

  // Compact mutation
  const compactMutation = useMutation({
    mutationFn: async (dryRun: boolean) => {
      const res = await fetch('/api/compaction/compact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          session_id: sessionId,
          lookback_hours: lookbackHours,
          min_memories: minMemories,
          dry_run: dryRun,
          memory_types: selectedTypes.length > 0 && selectedTypes.length < (typeInfo?.types.length || 0)
            ? selectedTypes
            : null,
        }),
      });
      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Compaction failed');
      }
      return res.json();
    },
    onSuccess: (data) => {
      setLastResult(data);
      refetchPreview();
    },
  });

  const toggleType = (type: string) => {
    setSelectedTypes(prev =>
      prev.includes(type)
        ? prev.filter(t => t !== type)
        : [...prev, type]
    );
  };

  const selectAllTypes = () => {
    if (typeInfo?.types) {
      setSelectedTypes(typeInfo.types);
    }
  };

  const clearAllTypes = () => {
    setSelectedTypes([]);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-100 rounded-lg">
            <Layers className="w-5 h-5 text-purple-600" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">Memory Compaction</h3>
            <p className="text-sm text-gray-500">Consolidate memories into concise summaries to save tokens</p>
          </div>
        </div>
      </div>

      {/* Info Banner */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-start gap-3">
        <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
        <div className="text-sm text-blue-800">
          <p className="font-medium mb-1">How Compaction Helps</p>
          <ul className="list-disc list-inside space-y-1 text-blue-700">
            <li>Reduces token usage by consolidating similar memories</li>
            <li>Preserves key facts and entities while removing redundancy</li>
            <li>Improves retrieval speed with fewer, more relevant memories</li>
            <li>Saves storage space and reduces API costs</li>
          </ul>
        </div>
      </div>

      {/* Memory Type Selection */}
      <div className="bg-white rounded-xl border p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-medium text-gray-700 flex items-center gap-2">
            <Tag className="w-4 h-4" />
            Memory Types to Compact
          </h4>
          <div className="flex gap-2">
            <button
              onClick={selectAllTypes}
              className="text-xs text-purple-600 hover:text-purple-800"
            >
              Select All
            </button>
            <span className="text-gray-300">|</span>
            <button
              onClick={clearAllTypes}
              className="text-xs text-gray-500 hover:text-gray-700"
            >
              Clear
            </button>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {typeInfo?.types.map((type) => (
            <button
              key={type}
              onClick={() => toggleType(type)}
              className={clsx(
                'px-3 py-1.5 rounded-lg text-sm font-medium transition-all flex items-center gap-1.5',
                selectedTypes.includes(type)
                  ? 'bg-purple-100 text-purple-700 border-2 border-purple-300'
                  : 'bg-gray-100 text-gray-600 border-2 border-transparent hover:bg-gray-200'
              )}
              title={typeInfo.descriptions[type]}
            >
              {selectedTypes.includes(type) && <CheckCircle className="w-3.5 h-3.5" />}
              {type}
              {preview?.by_type?.[type] && (
                <span className="text-xs opacity-70">({preview.by_type[type]})</span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Preview Stats */}
      {preview && !previewLoading && (
        <div className="bg-gray-50 rounded-xl border p-4">
          <h4 className="text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
            <Database className="w-4 h-4" />
            Current Memory Stats
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <MetricBox label="Memories" value={preview.total_memories} icon="📝" />
            <MetricBox label="Tokens" value={preview.total_tokens?.toLocaleString()} icon="🔤" />
            <MetricBox label="Clusters" value={preview.compactable_clusters} icon="📦" />
            <MetricBox label="Est. Compression" value={preview.estimated_compression} icon="📉" highlight />
          </div>
          {preview.estimated_tokens_saved > 0 && (
            <div className="mt-3 pt-3 border-t text-sm text-green-700 flex items-center gap-2">
              <Sparkles className="w-4 h-4" />
              Estimated savings: <strong>{preview.estimated_tokens_saved.toLocaleString()}</strong> tokens
            </div>
          )}
        </div>
      )}

      {/* Settings */}
      <div className="bg-white rounded-xl border p-4">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full flex items-center justify-between text-sm font-medium text-gray-700"
        >
          <span className="flex items-center gap-2">
            <Clock className="w-4 h-4" />
            Settings
          </span>
          {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
        
        <div className={clsx('mt-4 space-y-4', !showAdvanced && 'hidden')}>
          <div className="flex items-center gap-4">
            <label className="text-sm text-gray-600 w-32">Lookback Period:</label>
            <select
              value={lookbackHours}
              onChange={(e) => setLookbackHours(parseInt(e.target.value))}
              className="flex-1 px-3 py-1.5 border rounded-lg text-sm"
            >
              {LOOKBACK_OPTIONS.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-4">
            <label className="text-sm text-gray-600 w-32">Min Memories:</label>
            <input
              type="number"
              value={minMemories}
              onChange={(e) => setMinMemories(parseInt(e.target.value) || 5)}
              min={2}
              max={50}
              className="w-20 px-3 py-1.5 border rounded-lg text-sm"
            />
            <span className="text-xs text-gray-500">Minimum memories needed to trigger compaction</span>
          </div>
        </div>
        
        {!showAdvanced && (
          <div className="mt-2 text-xs text-gray-500">
            Lookback: {LOOKBACK_OPTIONS.find(o => o.value === lookbackHours)?.label}
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3">
        <button
          onClick={() => compactMutation.mutate(true)}
          disabled={compactMutation.isPending || selectedTypes.length === 0}
          className={clsx(
            'flex-1 px-4 py-3 rounded-lg font-medium flex items-center justify-center gap-2 transition-all',
            compactMutation.isPending || selectedTypes.length === 0
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
              : 'bg-gray-100 hover:bg-gray-200 text-gray-700 border border-gray-300'
          )}
        >
          <Eye className="w-4 h-4" />
          Preview
        </button>
        <button
          onClick={() => compactMutation.mutate(false)}
          disabled={compactMutation.isPending || selectedTypes.length === 0}
          className={clsx(
            'flex-1 px-4 py-3 rounded-lg font-medium flex items-center justify-center gap-2 transition-all',
            compactMutation.isPending || selectedTypes.length === 0
              ? 'bg-purple-400 text-white cursor-not-allowed'
              : 'bg-purple-600 hover:bg-purple-700 text-white'
          )}
        >
          {compactMutation.isPending ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Zap className="w-4 h-4" />
          )}
          {compactMutation.isPending ? 'Compacting...' : 'Compact Now'}
        </button>
      </div>

      {/* Result */}
      {lastResult && (
        <CompactionResultCard result={lastResult} />
      )}

      {/* Error */}
      {compactMutation.isError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
          <XCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
          <div>
            <h4 className="font-medium text-red-800">Compaction Failed</h4>
            <p className="text-sm text-red-700">{(compactMutation.error as Error).message}</p>
          </div>
        </div>
      )}
    </div>
  );
}

function MetricBox({ label, value, icon, highlight = false }: { 
  label: string; 
  value: string | number; 
  icon: string;
  highlight?: boolean;
}) {
  return (
    <div className={clsx(
      'bg-white rounded-lg p-3 text-center border',
      highlight && 'ring-2 ring-green-400'
    )}>
      <div className="text-lg mb-1">{icon}</div>
      <div className={clsx('text-xl font-bold', highlight ? 'text-green-600' : 'text-gray-900')}>
        {value}
      </div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  );
}


function CompactionResultCard({ result }: { result: CompactionResult }) {
  const m = result.metrics;
  const isDryRun = result.dry_run;
  const totalCost = m.estimated_input_cost + m.estimated_output_cost;
  const [showDetails, setShowDetails] = useState(false);

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className={clsx(
      'rounded-xl border p-6',
      isDryRun ? 'bg-yellow-50 border-yellow-200' : 'bg-green-50 border-green-200'
    )}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          {result.status === 'completed' ? (
            <CheckCircle className="w-6 h-6 text-green-600" />
          ) : result.status === 'skipped' ? (
            <Info className="w-6 h-6 text-yellow-600" />
          ) : (
            <XCircle className="w-6 h-6 text-red-600" />
          )}
          <div>
            <h3 className="font-semibold text-gray-900">
              {isDryRun ? 'Compaction Preview' : 'Compaction Complete'}
            </h3>
            <p className="text-sm text-gray-600">
              {isDryRun ? 'No changes made - preview only' : 'Memories have been consolidated'}
            </p>
          </div>
        </div>
        <span className={clsx(
          'px-3 py-1 rounded-full text-xs font-semibold',
          isDryRun ? 'bg-yellow-100 text-yellow-700' : 'bg-green-100 text-green-700'
        )}>
          {isDryRun ? 'DRY RUN' : 'LIVE'}
        </span>
      </div>

      {/* Main Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        <MetricCard
          icon={<FileText className="w-4 h-4 text-blue-600" />}
          label="Input"
          value={`${m.input_memories} memories`}
          subvalue={`${m.input_tokens.toLocaleString()} tokens`}
        />
        <MetricCard
          icon={<Layers className="w-4 h-4 text-purple-600" />}
          label="Output"
          value={`${m.output_memories} summaries`}
          subvalue={`${m.output_tokens.toLocaleString()} tokens`}
        />
        <MetricCard
          icon={<TrendingDown className="w-4 h-4 text-green-600" />}
          label="Compression"
          value={`${(m.compression_ratio * 100).toFixed(1)}%`}
          subvalue={`${m.tokens_saved.toLocaleString()} saved`}
          highlight={m.compression_ratio > 0.5}
        />
        <MetricCard
          icon={<Database className="w-4 h-4 text-indigo-600" />}
          label="Storage Saved"
          value={formatBytes(m.estimated_storage_saved_bytes)}
          subvalue={`${m.memories_merged} merged`}
        />
      </div>

      {/* Insights */}
      {result.insights && result.insights.length > 0 && (
        <div className="bg-white rounded-lg p-4 border mb-4">
          <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
            <Lightbulb className="w-4 h-4 text-yellow-500" />
            Insights
          </h4>
          <ul className="space-y-1">
            {result.insights.map((insight, i) => (
              <li key={i} className="text-sm text-gray-600">{insight}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Preserved Information */}
      {(result.preserved_facts.length > 0 || result.preserved_entities.length > 0) && (
        <div className="bg-white rounded-lg p-4 border mb-4">
          <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
            <Brain className="w-4 h-4 text-purple-500" />
            Preserved Information
          </h4>
          <div className="grid md:grid-cols-2 gap-4">
            {result.preserved_facts.length > 0 && (
              <div>
                <p className="text-xs text-gray-500 mb-1">Key Facts ({m.key_facts_preserved})</p>
                <div className="flex flex-wrap gap-1">
                  {result.preserved_facts.slice(0, 5).map((fact, i) => (
                    <span key={i} className="px-2 py-0.5 bg-blue-50 text-blue-700 rounded text-xs">
                      {fact.length > 50 ? fact.slice(0, 50) + '...' : fact}
                    </span>
                  ))}
                  {result.preserved_facts.length > 5 && (
                    <span className="text-xs text-gray-400">+{result.preserved_facts.length - 5} more</span>
                  )}
                </div>
              </div>
            )}
            {result.preserved_entities.length > 0 && (
              <div>
                <p className="text-xs text-gray-500 mb-1">Entities ({m.entities_preserved})</p>
                <div className="flex flex-wrap gap-1">
                  {result.preserved_entities.slice(0, 8).map((entity, i) => (
                    <span key={i} className="px-2 py-0.5 bg-purple-50 text-purple-700 rounded text-xs">
                      {entity}
                    </span>
                  ))}
                  {result.preserved_entities.length > 8 && (
                    <span className="text-xs text-gray-400">+{result.preserved_entities.length - 8} more</span>
                  )}
                </div>
              </div>
            )}
          </div>
          {m.context_retention_score > 0 && (
            <div className="mt-3 pt-3 border-t">
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500">Context Retention:</span>
                <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className={clsx(
                      'h-full rounded-full',
                      m.context_retention_score > 0.7 ? 'bg-green-500' : 
                      m.context_retention_score > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                    )}
                    style={{ width: `${m.context_retention_score * 100}%` }}
                  />
                </div>
                <span className="text-xs font-medium">{(m.context_retention_score * 100).toFixed(0)}%</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Types Compacted */}
      {Object.keys(m.types_compacted).length > 0 && (
        <div className="bg-white rounded-lg p-3 border mb-4">
          <p className="text-xs text-gray-500 mb-2">Types Compacted</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(m.types_compacted).map(([type, count]) => (
              <span key={type} className="px-2 py-1 bg-gray-100 rounded text-xs">
                {type}: <strong>{count}</strong>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Processing Stats */}
      <div className="bg-white rounded-lg p-3 border mb-4">
        <div className="flex items-center justify-between text-sm flex-wrap gap-2">
          <span className="text-gray-600 flex items-center gap-1">
            <Clock className="w-4 h-4" />
            {m.duration_seconds.toFixed(2)}s
          </span>
          <span className="text-gray-600">
            {m.clusters_found} clusters
          </span>
          <span className="text-gray-600">
            {m.llm_calls} LLM calls
          </span>
          <span className="text-gray-600 flex items-center gap-1">
            <Coins className="w-4 h-4" />
            ${totalCost.toFixed(4)}
          </span>
        </div>
      </div>

      {/* Summary */}
      {result.summary && (
        <div className="bg-white rounded-lg p-4 border mb-4">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Summary</h4>
          <pre className="text-sm text-gray-600 whitespace-pre-wrap font-sans">{result.summary}</pre>
        </div>
      )}

      {/* Expandable Actions */}
      {result.actions && result.actions.length > 0 && (
        <div className="border-t pt-4">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900"
          >
            {showDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            {showDetails ? 'Hide' : 'Show'} Actions ({result.actions.length})
          </button>
          
          {showDetails && (
            <div className="mt-3 space-y-2">
              {result.actions.map((action, i) => (
                <div key={i} className="text-sm bg-gray-50 rounded p-2 border">
                  <span className="font-medium text-gray-700">{action.action}</span>
                  {action.input_count && (
                    <span className="text-gray-500 ml-2">
                      ({action.input_count} inputs → {action.output_tokens} tokens)
                    </span>
                  )}
                  {action.key_facts_found !== undefined && (
                    <span className="text-green-600 ml-2">
                      {action.key_facts_found} facts preserved
                    </span>
                  )}
                  {action.preview && (
                    <p className="text-gray-500 mt-1 text-xs truncate">{action.preview}</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Footer */}
      <div className="mt-4 pt-3 border-t flex items-center justify-between text-xs text-gray-500">
        <span>ID: {result.id.slice(0, 8)}</span>
        <span>{new Date(result.started_at).toLocaleString()}</span>
      </div>
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value,
  subvalue,
  highlight = false,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  subvalue: string;
  highlight?: boolean;
}) {
  return (
    <div className={clsx(
      'bg-white rounded-lg p-3 border',
      highlight && 'ring-2 ring-green-400 ring-offset-1'
    )}>
      <div className="flex items-center gap-2 mb-1">
        {icon}
        <span className="text-xs text-gray-500">{label}</span>
      </div>
      <div className={clsx('text-lg font-bold', highlight ? 'text-green-600' : 'text-gray-900')}>
        {value}
      </div>
      <div className="text-xs text-gray-500">{subvalue}</div>
    </div>
  );
}
