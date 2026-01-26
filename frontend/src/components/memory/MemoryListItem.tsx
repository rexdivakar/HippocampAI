import { formatDistanceToNow } from 'date-fns';
import { CircleDot, Calendar, TrendingUp } from 'lucide-react';
import type { Memory } from '../../types';
import clsx from 'clsx';

interface MemoryListItemProps {
  memory: Memory;
  isSelected: boolean;
  density: 'comfortable' | 'compact' | 'ultra';
  onClick: () => void;
}

const typeColors: Record<string, string> = {
  fact: 'text-blue-600 bg-blue-50',
  preference: 'text-purple-600 bg-purple-50',
  goal: 'text-green-600 bg-green-50',
  habit: 'text-orange-600 bg-orange-50',
  event: 'text-pink-600 bg-pink-50',
  context: 'text-gray-600 bg-gray-50',
};

const typeIcons: Record<string, string> = {
  fact: '📊',
  preference: '❤️',
  goal: '🎯',
  habit: '🔄',
  event: '📅',
  context: '💬',
};

export function MemoryListItem({ memory, isSelected, density, onClick }: MemoryListItemProps) {
  const timeAgo = formatDistanceToNow(new Date(memory.created_at), { addSuffix: true });

  // Comfortable mode - card-like with full details
  if (density === 'comfortable') {
    return (
      <button
        onClick={onClick}
        className={clsx(
          'w-full text-left p-4 border-b border-gray-100 hover:bg-gray-50 transition-colors',
          isSelected && 'bg-primary-50 border-l-4 border-l-primary-600'
        )}
      >
        <div className="flex items-start justify-between mb-2">
          <div className="flex items-center space-x-2 flex-1 min-w-0">
            <span className="text-lg flex-shrink-0">{typeIcons[memory.type] || '💭'}</span>
            <h3 className="font-medium text-gray-900 truncate">
              {memory.text.split('\n')[0].substring(0, 60)}
              {memory.text.length > 60 ? '...' : ''}
            </h3>
          </div>
          {isSelected && <CircleDot className="w-4 h-4 text-primary-600 flex-shrink-0 ml-2" />}
        </div>

        <div className="flex items-center space-x-3 text-xs text-gray-500">
          <span className={clsx('px-2 py-1 rounded-full font-medium', typeColors[memory.type])}>
            {memory.type}
          </span>
          <div className="flex items-center space-x-1">
            <TrendingUp className="w-3 h-3" />
            <span>{memory.importance.toFixed(1)}</span>
          </div>
          <div className="flex items-center space-x-1">
            <Calendar className="w-3 h-3" />
            <span>{timeAgo}</span>
          </div>
        </div>

        {memory.tags.length > 0 && (
          <div className="flex items-center gap-1 mt-2 flex-wrap">
            {memory.tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                className="inline-flex items-center px-2 py-0.5 text-xs bg-gray-100 text-gray-700 rounded"
              >
                #{tag}
              </span>
            ))}
            {memory.tags.length > 3 && (
              <span className="text-xs text-gray-400">+{memory.tags.length - 3}</span>
            )}
          </div>
        )}
      </button>
    );
  }

  // Compact mode - reduced padding, single line
  if (density === 'compact') {
    return (
      <button
        onClick={onClick}
        className={clsx(
          'w-full text-left px-4 py-2.5 border-b border-gray-100 hover:bg-gray-50 transition-colors',
          isSelected && 'bg-primary-50 border-l-4 border-l-primary-600'
        )}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2 flex-1 min-w-0">
            <span className="text-sm flex-shrink-0">{typeIcons[memory.type] || '💭'}</span>
            <span className="font-medium text-sm text-gray-900 truncate">
              {memory.text.split('\n')[0].substring(0, 50)}
            </span>
          </div>
          <div className="flex items-center space-x-2 text-xs text-gray-500 flex-shrink-0 ml-3">
            <span className={clsx('px-1.5 py-0.5 rounded text-xs', typeColors[memory.type])}>
              {memory.type}
            </span>
            <span>{memory.importance.toFixed(1)}</span>
            <span className="text-gray-400">{timeAgo.replace(' ago', '')}</span>
          </div>
        </div>
      </button>
    );
  }

  // Ultra mode - minimal single line, data table style
  return (
    <button
      onClick={onClick}
      className={clsx(
        'w-full text-left px-3 py-1.5 border-b border-gray-100 hover:bg-gray-50 transition-colors',
        isSelected && 'bg-primary-50 border-l-2 border-l-primary-600'
      )}
    >
      <div className="flex items-center space-x-2 text-xs">
        <span className="flex-shrink-0">{typeIcons[memory.type] || '💭'}</span>
        <span className="text-gray-900 truncate flex-1 min-w-0">
          {memory.text.split('\n')[0].substring(0, 40)}
        </span>
        <span className={clsx('px-1 py-0.5 rounded text-xs flex-shrink-0', typeColors[memory.type])}>
          {memory.type.substring(0, 4)}
        </span>
        <span className="text-gray-500 flex-shrink-0 w-8 text-right">{memory.importance.toFixed(1)}</span>
        <span className="text-gray-400 flex-shrink-0 w-12 text-right">
          {timeAgo.replace(' ago', '').replace('about ', '')}
        </span>
      </div>
    </button>
  );
}
