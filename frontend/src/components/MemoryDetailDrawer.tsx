import { useState } from 'react';
import { X, Eye, Code, Database, Zap, TrendingDown, Link as LinkIcon, FileJson, ThumbsUp, ThumbsDown, CircleDot, AlertTriangle } from 'lucide-react';
import type { Memory, FeedbackType } from '../types';
import clsx from 'clsx';

interface MemoryDetailDrawerProps {
  memory: Memory | null;
  isOpen: boolean;
  onClose: () => void;
  onFeedback?: (memoryId: string, feedbackType: FeedbackType) => void;
}

export function MemoryDetailDrawer({ memory, isOpen, onClose, onFeedback }: MemoryDetailDrawerProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'technical' | 'json'>('overview');

  if (!isOpen || !memory) return null;

  // Mock data - in production these would come from backend
  const mockData = {
    cleanedText: memory.text.replace(/[^\w\s]/gi, ''),
    embeddingPreview: memory.embedding
      ? `[${memory.embedding.slice(0, 5).map((v) => v.toFixed(4)).join(', ')}... (${memory.embedding.length} dims)]`
      : 'Not available',
    bm25Tokens: memory.text
      .toLowerCase()
      .split(/\s+/)
      .filter((w) => w.length > 3)
      .slice(0, 10),
    rerankerScore: (memory.confidence * 10).toFixed(2),
    decayRate: (10 - memory.importance) / 10,
    linkedMemories: [
      { id: 'mem-001', text: 'Related memory 1', score: 0.87 },
      { id: 'mem-002', text: 'Related memory 2', score: 0.72 },
    ],
  };

  return (
    <>
      {/* Overlay */}
      <div
        className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40"
        onClick={onClose}
      />

      {/* Drawer */}
      <div className="fixed right-0 top-0 h-full w-full md:w-[600px] bg-white shadow-2xl z-50 overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 bg-gradient-to-r from-primary-500 to-blue-500 text-white">
          <div className="flex items-center space-x-3">
            <Eye className="w-6 h-6" />
            <div>
              <h2 className="text-lg font-bold">Memory Inspector</h2>
              <p className="text-xs text-blue-100">ID: {memory.id.slice(0, 16)}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/20 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-200 bg-gray-50">
          <button
            onClick={() => setActiveTab('overview')}
            className={clsx(
              'flex-1 px-6 py-3 text-sm font-medium transition-colors',
              activeTab === 'overview'
                ? 'bg-white text-primary-700 border-b-2 border-primary-600'
                : 'text-gray-600 hover:text-gray-900'
            )}
          >
            Overview
          </button>
          <button
            onClick={() => setActiveTab('technical')}
            className={clsx(
              'flex-1 px-6 py-3 text-sm font-medium transition-colors',
              activeTab === 'technical'
                ? 'bg-white text-primary-700 border-b-2 border-primary-600'
                : 'text-gray-600 hover:text-gray-900'
            )}
          >
            Technical
          </button>
          <button
            onClick={() => setActiveTab('json')}
            className={clsx(
              'flex-1 px-6 py-3 text-sm font-medium transition-colors',
              activeTab === 'json'
                ? 'bg-white text-primary-700 border-b-2 border-primary-600'
                : 'text-gray-600 hover:text-gray-900'
            )}
          >
            JSON
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {activeTab === 'overview' && (
            <>
              {/* Raw Text */}
              <div>
                <div className="flex items-center space-x-2 mb-3">
                  <Eye className="w-5 h-5 text-primary-600" />
                  <h3 className="text-sm font-semibold text-gray-900">Raw Text</h3>
                </div>
                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  <p className="text-sm text-gray-900">{memory.text}</p>
                </div>
              </div>

              {/* Cleaned Text */}
              <div>
                <div className="flex items-center space-x-2 mb-3">
                  <Code className="w-5 h-5 text-green-600" />
                  <h3 className="text-sm font-semibold text-gray-900">Cleaned Text</h3>
                </div>
                <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                  <p className="text-sm text-gray-900">{mockData.cleanedText}</p>
                </div>
              </div>

              {/* Key Metrics */}
              <div>
                <h3 className="text-sm font-semibold text-gray-900 mb-3">Key Metrics</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                    <p className="text-xs text-blue-600 mb-1">Importance</p>
                    <p className="text-2xl font-bold text-blue-900">
                      {memory.importance.toFixed(1)}/10
                    </p>
                  </div>
                  <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                    <p className="text-xs text-green-600 mb-1">Confidence</p>
                    <p className="text-2xl font-bold text-green-900">
                      {(memory.confidence * 100).toFixed(0)}%
                    </p>
                  </div>
                  <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                    <p className="text-xs text-purple-600 mb-1">Access Count</p>
                    <p className="text-2xl font-bold text-purple-900">{memory.access_count}</p>
                  </div>
                  <div className="bg-red-50 rounded-lg p-4 border border-red-200">
                    <p className="text-xs text-red-600 mb-1">Decay Rate</p>
                    <p className="text-2xl font-bold text-red-900">
                      {(mockData.decayRate * 100).toFixed(0)}%
                    </p>
                  </div>
                </div>
              </div>

              {/* Metadata */}
              <div>
                <div className="flex items-center space-x-2 mb-3">
                  <Database className="w-5 h-5 text-purple-600" />
                  <h3 className="text-sm font-semibold text-gray-900">Metadata</h3>
                </div>
                <div className="bg-purple-50 rounded-lg p-4 border border-purple-200 space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Type:</span>
                    <span className="font-medium text-gray-900">{memory.type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">User ID:</span>
                    <span className="font-mono text-xs text-gray-900">
                      {memory.user_id.slice(0, 16)}...
                    </span>
                  </div>
                  {memory.session_id && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Session:</span>
                      <span className="font-mono text-xs text-gray-900">
                        {memory.session_id.slice(0, 16)}...
                      </span>
                    </div>
                  )}
                  {memory.agent_id && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Agent:</span>
                      <span className="font-mono text-xs text-gray-900">
                        {memory.agent_id}
                      </span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span className="text-gray-600">Created:</span>
                    <span className="text-gray-900">
                      {new Date(memory.created_at).toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Updated:</span>
                    <span className="text-gray-900">
                      {new Date(memory.updated_at).toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>

              {/* Tags */}
              {memory.tags.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold text-gray-900 mb-3">Tags</h3>
                  <div className="flex flex-wrap gap-2">
                    {memory.tags.map((tag) => (
                      <span
                        key={tag}
                        className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Feedback */}
              {onFeedback && (
                <div>
                  <h3 className="text-sm font-semibold text-gray-900 mb-3">Rate This Memory</h3>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => onFeedback(memory.id, 'relevant')}
                      className="flex items-center space-x-1.5 px-3 py-2 text-sm bg-green-50 text-green-700 hover:bg-green-100 rounded-lg transition-colors border border-green-200"
                    >
                      <ThumbsUp className="w-4 h-4" />
                      <span>Relevant</span>
                    </button>
                    <button
                      onClick={() => onFeedback(memory.id, 'partially_relevant')}
                      className="flex items-center space-x-1.5 px-3 py-2 text-sm bg-yellow-50 text-yellow-700 hover:bg-yellow-100 rounded-lg transition-colors border border-yellow-200"
                    >
                      <CircleDot className="w-4 h-4" />
                      <span>Partial</span>
                    </button>
                    <button
                      onClick={() => onFeedback(memory.id, 'not_relevant')}
                      className="flex items-center space-x-1.5 px-3 py-2 text-sm bg-red-50 text-red-700 hover:bg-red-100 rounded-lg transition-colors border border-red-200"
                    >
                      <ThumbsDown className="w-4 h-4" />
                      <span>Not Relevant</span>
                    </button>
                    <button
                      onClick={() => onFeedback(memory.id, 'outdated')}
                      className="flex items-center space-x-1.5 px-3 py-2 text-sm bg-gray-50 text-gray-700 hover:bg-gray-100 rounded-lg transition-colors border border-gray-200"
                    >
                      <AlertTriangle className="w-4 h-4" />
                      <span>Outdated</span>
                    </button>
                  </div>
                </div>
              )}
            </>
          )}

          {activeTab === 'technical' && (
            <>
              {/* Embedding Vector */}
              <div>
                <div className="flex items-center space-x-2 mb-3">
                  <Zap className="w-5 h-5 text-yellow-600" />
                  <h3 className="text-sm font-semibold text-gray-900">Embedding Vector</h3>
                </div>
                <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
                  <p className="text-xs font-mono text-gray-900">{mockData.embeddingPreview}</p>
                </div>
              </div>

              {/* BM25 Tokens */}
              <div>
                <div className="flex items-center space-x-2 mb-3">
                  <Code className="w-5 h-5 text-indigo-600" />
                  <h3 className="text-sm font-semibold text-gray-900">BM25 Token List</h3>
                </div>
                <div className="bg-indigo-50 rounded-lg p-4 border border-indigo-200">
                  <div className="flex flex-wrap gap-2">
                    {mockData.bm25Tokens.map((token, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-1 bg-white text-indigo-700 rounded text-xs font-mono"
                      >
                        {token}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Reranker Score */}
              <div>
                <div className="flex items-center space-x-2 mb-3">
                  <TrendingDown className="w-5 h-5 text-orange-600" />
                  <h3 className="text-sm font-semibold text-gray-900">
                    Reranker (Cross-Encoder) Score
                  </h3>
                </div>
                <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
                  <div className="flex items-center justify-between">
                    <span className="text-3xl font-bold text-orange-900">
                      {mockData.rerankerScore}
                    </span>
                    <div className="flex-1 ml-4">
                      <div className="bg-orange-200 rounded-full h-3">
                        <div
                          className="bg-orange-600 h-3 rounded-full"
                          style={{ width: `${(parseFloat(mockData.rerankerScore) / 10) * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Decay Information */}
              <div>
                <div className="flex items-center space-x-2 mb-3">
                  <TrendingDown className="w-5 h-5 text-red-600" />
                  <h3 className="text-sm font-semibold text-gray-900">Decay Information</h3>
                </div>
                <div className="bg-red-50 rounded-lg p-4 border border-red-200 space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Current Importance:</span>
                    <span className="font-bold text-gray-900">{memory.importance.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Decay Rate:</span>
                    <span className="font-bold text-red-700">
                      {(mockData.decayRate * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Last Accessed:</span>
                    <span className="text-gray-900">
                      {memory.last_accessed_at
                        ? new Date(memory.last_accessed_at).toLocaleDateString()
                        : 'Never'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Linked Memories */}
              <div>
                <div className="flex items-center space-x-2 mb-3">
                  <LinkIcon className="w-5 h-5 text-teal-600" />
                  <h3 className="text-sm font-semibold text-gray-900">Linked Memories</h3>
                </div>
                <div className="space-y-2">
                  {mockData.linkedMemories.map((linked) => (
                    <div
                      key={linked.id}
                      className="bg-teal-50 rounded-lg p-3 border border-teal-200"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-mono text-gray-500">{linked.id}</span>
                        <span className="text-xs font-semibold text-teal-700">
                          Score: {linked.score}
                        </span>
                      </div>
                      <p className="text-sm text-gray-900">{linked.text}</p>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {activeTab === 'json' && (
            <div>
              <div className="flex items-center space-x-2 mb-3">
                <FileJson className="w-5 h-5 text-gray-600" />
                <h3 className="text-sm font-semibold text-gray-900">Raw JSON Object</h3>
              </div>
              <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
                <pre className="text-xs text-green-400 font-mono">
                  {JSON.stringify(memory, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="border-t border-gray-200 p-4 bg-gray-50">
          <button
            onClick={onClose}
            className="w-full btn-secondary"
          >
            Close Inspector
          </button>
        </div>
      </div>
    </>
  );
}
