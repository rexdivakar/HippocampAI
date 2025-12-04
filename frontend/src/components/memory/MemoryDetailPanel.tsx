import { format } from 'date-fns';
import {
  Edit2,
  Trash2,
  Share2,
  Download,
  Calendar,
  TrendingUp,
  Tag,
  Clock,
  User,
  Eye,
  Hash
} from 'lucide-react';
import type { Memory } from '../../types';
import clsx from 'clsx';

interface MemoryDetailPanelProps {
  memory: Memory;
  onEdit: (memory: Memory) => void;
  onDelete: (memory: Memory) => void;
}

const typeColors: Record<string, string> = {
  fact: 'text-blue-600 bg-blue-50 border-blue-200',
  preference: 'text-purple-600 bg-purple-50 border-purple-200',
  goal: 'text-green-600 bg-green-50 border-green-200',
  habit: 'text-orange-600 bg-orange-50 border-orange-200',
  event: 'text-pink-600 bg-pink-50 border-pink-200',
  context: 'text-gray-600 bg-gray-50 border-gray-200',
};

const typeIcons: Record<string, string> = {
  fact: '📊',
  preference: '❤️',
  goal: '🎯',
  habit: '🔄',
  event: '📅',
  context: '💬',
};

export function MemoryDetailPanel({ memory, onEdit, onDelete }: MemoryDetailPanelProps) {
  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="border-b border-gray-200 p-6">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center space-x-3">
            <span className="text-3xl">{typeIcons[memory.type] || '💭'}</span>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">
                {memory.text.split('\n')[0].substring(0, 100)}
              </h2>
              <div className="flex items-center space-x-3 mt-2">
                <span
                  className={clsx(
                    'px-3 py-1 rounded-full text-sm font-medium border',
                    typeColors[memory.type]
                  )}
                >
                  {memory.type}
                </span>
                <div className="flex items-center space-x-1 text-sm text-gray-600">
                  <TrendingUp className="w-4 h-4" />
                  <span>Importance: {memory.importance.toFixed(1)}</span>
                </div>
                <div className="flex items-center space-x-1 text-sm text-gray-500">
                  <Calendar className="w-4 h-4" />
                  <span>{format(new Date(memory.created_at), 'MMM d, yyyy')}</span>
                </div>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => onEdit(memory)}
              className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              title="Edit memory"
            >
              <Edit2 className="w-4 h-4" />
            </button>
            <button
              onClick={() => onDelete(memory)}
              className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
              title="Delete memory"
            >
              <Trash2 className="w-4 h-4" />
            </button>
            <button
              className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              title="Share memory"
            >
              <Share2 className="w-4 h-4" />
            </button>
            <button
              className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              title="Export memory"
            >
              <Download className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {/* Memory Text */}
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-gray-700 mb-2 uppercase tracking-wide">
            Content
          </h3>
          <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
            <p className="text-gray-800 whitespace-pre-wrap leading-relaxed">{memory.text}</p>
          </div>
        </div>

        {/* Tags */}
        {memory.tags.length > 0 && (
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-gray-700 mb-2 uppercase tracking-wide flex items-center">
              <Tag className="w-4 h-4 mr-1" />
              Tags
            </h3>
            <div className="flex flex-wrap gap-2">
              {memory.tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-flex items-center px-3 py-1.5 bg-primary-50 text-primary-700 rounded-lg text-sm font-medium border border-primary-200"
                >
                  <Hash className="w-3 h-3 mr-1" />
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Metadata Grid */}
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-gray-700 mb-3 uppercase tracking-wide">
            Metadata
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 text-gray-600 mb-1">
                <Hash className="w-4 h-4" />
                <span className="text-xs font-medium uppercase">Memory ID</span>
              </div>
              <p className="text-sm text-gray-900 font-mono truncate">{memory.id}</p>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 text-gray-600 mb-1">
                <User className="w-4 h-4" />
                <span className="text-xs font-medium uppercase">User ID</span>
              </div>
              <p className="text-sm text-gray-900 font-mono truncate">{memory.user_id}</p>
            </div>

            {memory.session_id && (
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <div className="flex items-center space-x-2 text-gray-600 mb-1">
                  <Hash className="w-4 h-4" />
                  <span className="text-xs font-medium uppercase">Session ID</span>
                </div>
                <p className="text-sm text-gray-900 font-mono truncate">{memory.session_id}</p>
              </div>
            )}

            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 text-gray-600 mb-1">
                <TrendingUp className="w-4 h-4" />
                <span className="text-xs font-medium uppercase">Importance</span>
              </div>
              <p className="text-sm text-gray-900">{memory.importance.toFixed(2)}</p>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 text-gray-600 mb-1">
                <TrendingUp className="w-4 h-4" />
                <span className="text-xs font-medium uppercase">Confidence</span>
              </div>
              <p className="text-sm text-gray-900">{(memory.confidence ?? 0).toFixed(2)}</p>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 text-gray-600 mb-1">
                <Eye className="w-4 h-4" />
                <span className="text-xs font-medium uppercase">Access Count</span>
              </div>
              <p className="text-sm text-gray-900">{memory.access_count || 0}</p>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 text-gray-600 mb-1">
                <Calendar className="w-4 h-4" />
                <span className="text-xs font-medium uppercase">Created</span>
              </div>
              <p className="text-sm text-gray-900">
                {format(new Date(memory.created_at), 'MMM d, yyyy h:mm a')}
              </p>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 text-gray-600 mb-1">
                <Clock className="w-4 h-4" />
                <span className="text-xs font-medium uppercase">Updated</span>
              </div>
              <p className="text-sm text-gray-900">
                {format(new Date(memory.updated_at), 'MMM d, yyyy h:mm a')}
              </p>
            </div>

            {memory.expires_at && (
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <div className="flex items-center space-x-2 text-gray-600 mb-1">
                  <Clock className="w-4 h-4" />
                  <span className="text-xs font-medium uppercase">Expires</span>
                </div>
                <p className="text-sm text-gray-900">
                  {format(new Date(memory.expires_at), 'MMM d, yyyy h:mm a')}
                </p>
              </div>
            )}

            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 text-gray-600 mb-1">
                <Hash className="w-4 h-4" />
                <span className="text-xs font-medium uppercase">Text Length</span>
              </div>
              <p className="text-sm text-gray-900">{memory.text.length} chars</p>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 text-gray-600 mb-1">
                <Hash className="w-4 h-4" />
                <span className="text-xs font-medium uppercase">Estimated Tokens</span>
              </div>
              <p className="text-sm text-gray-900">~{Math.ceil(memory.text.length / 4)} tokens</p>
            </div>
          </div>
        </div>

        {/* Additional Metadata */}
        {memory.metadata && Object.keys(memory.metadata).length > 0 && (
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-gray-700 mb-2 uppercase tracking-wide">
              Additional Data
            </h3>
            <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
              <pre className="text-xs text-gray-800 overflow-x-auto">
                {JSON.stringify(memory.metadata, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
