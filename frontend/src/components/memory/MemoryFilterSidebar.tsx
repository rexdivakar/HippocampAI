import { X, Filter, Calendar, TrendingUp, Tag } from 'lucide-react';
import type { Memory, MemoryFilters } from '../../types';
import { useState } from 'react';
import clsx from 'clsx';

interface MemoryFilterSidebarProps {
  filters: MemoryFilters;
  onFiltersChange: (filters: MemoryFilters) => void;
  memories: Memory[];
  onClose: () => void;
}

export function MemoryFilterSidebar({
  filters,
  onFiltersChange,
  memories,
  onClose,
}: MemoryFilterSidebarProps) {
  const [localFilters, setLocalFilters] = useState<MemoryFilters>(filters);

  // Get unique memory types from data
  const memoryTypes = Array.from(new Set(memories.map((m) => m.type)));

  // Get popular tags
  const allTags = memories.flatMap((m) => m.tags);
  const tagCounts = allTags.reduce(
    (acc, tag) => {
      acc[tag] = (acc[tag] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );
  const popularTags = Object.entries(tagCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20)
    .map(([tag]) => tag);

  const handleTypeToggle = (type: string) => {
    const currentTypes = localFilters.types || [];
    const newTypes = currentTypes.includes(type)
      ? currentTypes.filter((t) => t !== type)
      : [...currentTypes, type];

    const updated = { ...localFilters, types: newTypes.length > 0 ? newTypes : undefined };
    setLocalFilters(updated);
    onFiltersChange(updated);
  };

  const handleTagToggle = (tag: string) => {
    const currentTags = localFilters.tags || [];
    const newTags = currentTags.includes(tag)
      ? currentTags.filter((t) => t !== tag)
      : [...currentTags, tag];

    const updated = { ...localFilters, tags: newTags.length > 0 ? newTags : undefined };
    setLocalFilters(updated);
    onFiltersChange(updated);
  };

  const handleImportanceChange = (min: number, max: number) => {
    const updated = {
      ...localFilters,
      minImportance: min > 0 ? min : undefined,
      maxImportance: max < 10 ? max : undefined,
    };
    setLocalFilters(updated);
    onFiltersChange(updated);
  };

  const handleDateRangeChange = (range: 'today' | 'week' | 'month' | 'all') => {
    const now = new Date();
    let startDate: Date | undefined;

    switch (range) {
      case 'today':
        startDate = new Date(now.setHours(0, 0, 0, 0));
        break;
      case 'week':
        startDate = new Date(now.setDate(now.getDate() - 7));
        break;
      case 'month':
        startDate = new Date(now.setDate(now.getDate() - 30));
        break;
      case 'all':
        startDate = undefined;
        break;
    }

    const updated = {
      ...localFilters,
      startDate: startDate?.toISOString(),
      endDate: undefined,
    };
    setLocalFilters(updated);
    onFiltersChange(updated);
  };

  const handleReset = () => {
    setLocalFilters({});
    onFiltersChange({});
  };

  const activeFiltersCount =
    (localFilters.types?.length || 0) +
    (localFilters.tags?.length || 0) +
    (localFilters.minImportance !== undefined ? 1 : 0) +
    (localFilters.startDate !== undefined ? 1 : 0);

  return (
    <div className="w-full lg:w-56 xl:w-64 2xl:w-72 min-w-[224px] max-w-[288px] bg-white border-r border-gray-200 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <Filter className="w-5 h-5 text-gray-700" />
            <h2 className="font-semibold text-gray-900">Filters</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-gray-400 hover:text-gray-600 rounded transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        {activeFiltersCount > 0 && (
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-500">{activeFiltersCount} active</span>
            <button
              onClick={handleReset}
              className="text-xs text-primary-600 hover:text-primary-700 font-medium"
            >
              Reset all
            </button>
          </div>
        )}
      </div>

      {/* Filters */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Memory Types */}
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-3 uppercase tracking-wide">
            Type
          </h3>
          <div className="space-y-2">
            {memoryTypes.map((type) => {
              const isSelected = localFilters.types?.includes(type);
              return (
                <label
                  key={type}
                  className="flex items-center space-x-2 cursor-pointer group"
                >
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => handleTypeToggle(type)}
                    className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                  />
                  <span className="text-sm text-gray-700 group-hover:text-gray-900 capitalize">
                    {type}
                  </span>
                  <span className="text-xs text-gray-400">
                    ({memories.filter((m) => m.type === type).length})
                  </span>
                </label>
              );
            })}
          </div>
        </div>

        {/* Importance Range */}
        <div>
          <div className="flex items-center space-x-2 mb-3">
            <TrendingUp className="w-4 h-4 text-gray-600" />
            <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
              Importance
            </h3>
          </div>
          <div className="space-y-3">
            <div>
              <label className="text-xs text-gray-600 mb-1 block">Minimum</label>
              <input
                type="range"
                min="0"
                max="10"
                step="0.5"
                value={localFilters.minImportance || 0}
                onChange={(e) =>
                  handleImportanceChange(
                    parseFloat(e.target.value),
                    localFilters.maxImportance || 10
                  )
                }
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>0</span>
                <span className="font-medium text-gray-700">
                  {localFilters.minImportance || 0}
                </span>
                <span>10</span>
              </div>
            </div>
            <div>
              <label className="text-xs text-gray-600 mb-1 block">Maximum</label>
              <input
                type="range"
                min="0"
                max="10"
                step="0.5"
                value={localFilters.maxImportance || 10}
                onChange={(e) =>
                  handleImportanceChange(
                    localFilters.minImportance || 0,
                    parseFloat(e.target.value)
                  )
                }
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>0</span>
                <span className="font-medium text-gray-700">
                  {localFilters.maxImportance || 10}
                </span>
                <span>10</span>
              </div>
            </div>
          </div>
        </div>

        {/* Date Range */}
        <div>
          <div className="flex items-center space-x-2 mb-3">
            <Calendar className="w-4 h-4 text-gray-600" />
            <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
              Date Range
            </h3>
          </div>
          <div className="space-y-2">
            {[
              { value: 'today', label: 'Today' },
              { value: 'week', label: 'Last 7 days' },
              { value: 'month', label: 'Last 30 days' },
              { value: 'all', label: 'All time' },
            ].map((option) => (
              <button
                key={option.value}
                onClick={() => handleDateRangeChange(option.value as any)}
                className={clsx(
                  'w-full text-left px-3 py-2 text-sm rounded-lg transition-colors',
                  !localFilters.startDate && option.value === 'all'
                    ? 'bg-primary-50 text-primary-700 font-medium'
                    : 'text-gray-700 hover:bg-gray-50'
                )}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        {/* Popular Tags */}
        {popularTags.length > 0 && (
          <div>
            <div className="flex items-center space-x-2 mb-3">
              <Tag className="w-4 h-4 text-gray-600" />
              <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
                Popular Tags
              </h3>
            </div>
            <div className="flex flex-wrap gap-2">
              {popularTags.map((tag) => {
                const isSelected = localFilters.tags?.includes(tag);
                return (
                  <button
                    key={tag}
                    onClick={() => handleTagToggle(tag)}
                    className={clsx(
                      'px-2 py-1 text-xs rounded-lg border transition-colors',
                      isSelected
                        ? 'bg-primary-50 text-primary-700 border-primary-200 font-medium'
                        : 'bg-gray-50 text-gray-700 border-gray-200 hover:bg-gray-100'
                    )}
                  >
                    #{tag}
                    <span className="ml-1 text-gray-500">({tagCounts[tag]})</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
