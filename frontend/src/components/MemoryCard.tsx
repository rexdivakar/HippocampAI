import { Memory } from '../types';
import { formatDistanceToNow } from 'date-fns';
import {
  Brain,
  Star,
  Tag,
  Clock,
  TrendingUp,
  Edit,
  Trash2,
  Eye,
  Share2,
} from 'lucide-react';
import clsx from 'clsx';

interface MemoryCardProps {
  memory: Memory;
  onView?: (memory: Memory) => void;
  onEdit?: (memory: Memory) => void;
  onShare?: (memory: Memory) => void;
  onDelete?: (memory: Memory) => void;
  selected?: boolean;
}

const typeColors: Record<string, string> = {
  fact: 'bg-blue-100 text-blue-800',
  preference: 'bg-purple-100 text-purple-800',
  goal: 'bg-green-100 text-green-800',
  habit: 'bg-yellow-100 text-yellow-800',
  event: 'bg-red-100 text-red-800',
  context: 'bg-gray-100 text-gray-800',
};

const importanceColor = (importance: number): string => {
  if (importance >= 8) return 'text-red-500';
  if (importance >= 6) return 'text-orange-500';
  if (importance >= 4) return 'text-yellow-500';
  return 'text-gray-400';
};

export function MemoryCard({
  memory,
  onView,
  onEdit,
  onShare,
  onDelete,
  selected = false,
}: MemoryCardProps) {
  return (
    <div
      className={clsx(
        'card-hover relative overflow-hidden',
        selected && 'ring-2 ring-primary-500'
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center space-x-2">
          <Brain className="w-5 h-5 text-primary-600" />
          <span className={clsx('badge', typeColors[memory.type] || typeColors.context)}>
            {memory.type}
          </span>
        </div>

        <div className="flex items-center space-x-1">
          <Star className={clsx('w-4 h-4', importanceColor(memory.importance))} />
          <span className="text-sm font-medium text-gray-600">{memory.importance.toFixed(1)}</span>
        </div>
      </div>

      {/* Content */}
      <p className="text-gray-700 mb-4 line-clamp-3">{memory.text}</p>

      {/* Tags */}
      {memory.tags.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-3">
          {memory.tags.slice(0, 3).map((tag) => (
            <span key={tag} className="inline-flex items-center space-x-1 text-xs px-2 py-1 bg-gray-100 text-gray-600 rounded-md">
              <Tag className="w-3 h-3" />
              <span>{tag}</span>
            </span>
          ))}
          {memory.tags.length > 3 && (
            <span className="text-xs text-gray-400">+{memory.tags.length - 3} more</span>
          )}
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between text-sm text-gray-500 pt-3 border-t border-gray-100">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1">
            <Clock className="w-4 h-4" />
            <span>{formatDistanceToNow(new Date(memory.created_at), { addSuffix: true })}</span>
          </div>
          <div className="flex items-center space-x-1">
            <TrendingUp className="w-4 h-4" />
            <span>{memory.access_count} views</span>
          </div>
        </div>

        <div className="flex items-center space-x-1">
          {onView && (
            <button
              onClick={() => onView(memory)}
              className="p-1.5 text-gray-400 hover:text-primary-600 hover:bg-primary-50 rounded transition-colors"
              title="View details"
            >
              <Eye className="w-4 h-4" />
            </button>
          )}
          {onEdit && (
            <button
              onClick={() => onEdit(memory)}
              className="p-1.5 text-gray-400 hover:text-amber-600 hover:bg-amber-50 rounded transition-colors"
              title="Edit"
            >
              <Edit className="w-4 h-4" />
            </button>
          )}
          {onShare && (
            <button
              onClick={() => onShare(memory)}
              className="p-1.5 text-gray-400 hover:text-green-600 hover:bg-green-50 rounded transition-colors"
              title="Share"
            >
              <Share2 className="w-4 h-4" />
            </button>
          )}
          {onDelete && (
            <button
              onClick={() => onDelete(memory)}
              className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
              title="Delete"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Confidence indicator */}
      <div className="absolute bottom-0 left-0 right-0 h-1 bg-gray-100">
        <div
          className="h-full bg-gradient-to-r from-green-500 to-green-600"
          style={{ width: `${memory.confidence * 100}%` }}
        />
      </div>
    </div>
  );
}
