import { Memory } from '../types';
import { format } from 'date-fns';
import {
  X,
  Brain,
  Star,
  Tag,
  Clock,
  TrendingUp,
  Calendar,
  User,
  Hash,
  Edit,
  Trash2,
} from 'lucide-react';

interface MemoryDetailModalProps {
  memory: Memory | null;
  onClose: () => void;
  onEdit?: (memory: Memory) => void;
  onDelete?: (memory: Memory) => void;
}

export function MemoryDetailModal({
  memory,
  onClose,
  onEdit,
  onDelete,
}: MemoryDetailModalProps) {
  if (!memory) return null;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-start justify-center p-4 pt-20 overflow-y-auto">
      <div className="bg-white rounded-2xl shadow-2xl max-w-3xl w-full my-8 overflow-hidden border border-gray-100">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-gray-100">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/30">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Memory Details</h2>
              <p className="text-sm text-gray-500">ID: {memory.id.slice(0, 8)}...</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors text-gray-400 hover:text-gray-600"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {/* Memory Text */}
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide mb-2">
              Content
            </h3>
            <p className="text-lg text-gray-900 leading-relaxed">{memory.text}</p>
          </div>

          {/* Key Metrics Grid */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="card">
              <div className="flex items-center space-x-2 mb-2">
                <Star className="w-5 h-5 text-yellow-500" />
                <span className="text-sm font-medium text-gray-600">Importance</span>
              </div>
              <p className="text-2xl font-bold text-gray-900">{memory.importance.toFixed(1)}</p>
              <div className="mt-2 h-2 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-yellow-400 to-yellow-600"
                  style={{ width: `${(memory.importance / 10) * 100}%` }}
                />
              </div>
            </div>

            <div className="card">
              <div className="flex items-center space-x-2 mb-2">
                <TrendingUp className="w-5 h-5 text-green-500" />
                <span className="text-sm font-medium text-gray-600">Confidence</span>
              </div>
              <p className="text-2xl font-bold text-gray-900">
                {(memory.confidence * 100).toFixed(0)}%
              </p>
              <div className="mt-2 h-2 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-green-400 to-green-600"
                  style={{ width: `${memory.confidence * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* Metadata */}
          <div className="space-y-3 mb-6">
            <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide">
              Metadata
            </h3>

            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center space-x-2 text-gray-700">
                <Brain className="w-4 h-4 text-gray-400" />
                <span className="text-sm">Type:</span>
                <span className="font-medium">{memory.type}</span>
              </div>

              <div className="flex items-center space-x-2 text-gray-700">
                <TrendingUp className="w-4 h-4 text-gray-400" />
                <span className="text-sm">Access Count:</span>
                <span className="font-medium">{memory.access_count}</span>
              </div>

              <div className="flex items-center space-x-2 text-gray-700">
                <Calendar className="w-4 h-4 text-gray-400" />
                <span className="text-sm">Created:</span>
                <span className="font-medium">
                  {format(new Date(memory.created_at), 'MMM d, yyyy HH:mm')}
                </span>
              </div>

              <div className="flex items-center space-x-2 text-gray-700">
                <Clock className="w-4 h-4 text-gray-400" />
                <span className="text-sm">Updated:</span>
                <span className="font-medium">
                  {format(new Date(memory.updated_at), 'MMM d, yyyy HH:mm')}
                </span>
              </div>

              {memory.last_accessed_at && (
                <div className="flex items-center space-x-2 text-gray-700">
                  <Clock className="w-4 h-4 text-gray-400" />
                  <span className="text-sm">Last Access:</span>
                  <span className="font-medium">
                    {format(new Date(memory.last_accessed_at), 'MMM d, yyyy HH:mm')}
                  </span>
                </div>
              )}

              {memory.user_id && (
                <div className="flex items-center space-x-2 text-gray-700">
                  <User className="w-4 h-4 text-gray-400" />
                  <span className="text-sm">User ID:</span>
                  <span className="font-medium">{memory.user_id.slice(0, 8)}...</span>
                </div>
              )}

              {memory.session_id && (
                <div className="flex items-center space-x-2 text-gray-700">
                  <Hash className="w-4 h-4 text-gray-400" />
                  <span className="text-sm">Session:</span>
                  <span className="font-medium">{memory.session_id.slice(0, 8)}...</span>
                </div>
              )}

              {memory.agent_id && (
                <div className="flex items-center space-x-2 text-gray-700">
                  <Brain className="w-4 h-4 text-gray-400" />
                  <span className="text-sm">Agent:</span>
                  <span className="font-medium">{memory.agent_id.slice(0, 8)}...</span>
                </div>
              )}
            </div>
          </div>

          {/* Tags */}
          {memory.tags.length > 0 && (
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide mb-3">
                Tags
              </h3>
              <div className="flex flex-wrap gap-2">
                {memory.tags.map((tag) => (
                  <span
                    key={tag}
                    className="inline-flex items-center space-x-1 px-3 py-1.5 bg-primary-100 text-primary-700 rounded-lg text-sm font-medium"
                  >
                    <Tag className="w-3 h-3" />
                    <span>{tag}</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Additional Metadata */}
          {memory.metadata && Object.keys(memory.metadata).length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide mb-3">
                Additional Data
              </h3>
              <div className="card bg-gray-50">
                <pre className="text-sm text-gray-700 overflow-x-auto">
                  {JSON.stringify(memory.metadata, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-end space-x-3 px-6 py-4 border-t border-gray-100 bg-gray-50/50">
          {onEdit && (
            <button
              onClick={() => onEdit(memory)}
              className="px-4 py-2.5 text-sm font-medium text-gray-700 hover:bg-gray-100 rounded-lg transition-colors flex items-center space-x-2"
            >
              <Edit className="w-4 h-4" />
              <span>Edit</span>
            </button>
          )}
          {onDelete && (
            <button
              onClick={() => onDelete(memory)}
              className="px-4 py-2.5 text-sm font-medium bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors flex items-center space-x-2 shadow-sm"
            >
              <Trash2 className="w-4 h-4" />
              <span>Delete</span>
            </button>
          )}
          <button
            onClick={onClose}
            className="px-4 py-2.5 text-sm font-medium bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition-colors shadow-sm"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
