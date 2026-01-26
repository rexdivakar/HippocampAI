import { useState } from 'react';
import { Filter, X, Search } from 'lucide-react';
import type { MemoryFilters } from '../types';
import clsx from 'clsx';

interface MemoryFiltersProps {
  filters: MemoryFilters;
  onFiltersChange: (filters: MemoryFilters) => void;
  onReset: () => void;
}

const memoryTypes = ['fact', 'preference', 'goal', 'habit', 'event', 'context'];

export function MemoryFiltersComponent({
  filters,
  onFiltersChange,
  onReset,
}: MemoryFiltersProps) {
  const [showFilters, setShowFilters] = useState(false);

  const updateFilters = (updates: Partial<MemoryFilters>) => {
    onFiltersChange({ ...filters, ...updates });
  };

  const toggleType = (type: string) => {
    const currentTypes = filters.types || [];
    const newTypes = currentTypes.includes(type)
      ? currentTypes.filter((t) => t !== type)
      : [...currentTypes, type];
    updateFilters({ types: newTypes.length > 0 ? newTypes : undefined });
  };

  const hasActiveFilters =
    filters.types?.length ||
    filters.tags?.length ||
    filters.minImportance ||
    filters.maxImportance ||
    filters.searchText;

  return (
    <div className="space-y-4">
      {/* Search bar and filter toggle */}
      <div className="flex items-center space-x-3">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search memories..."
            value={filters.searchText || ''}
            onChange={(e) => updateFilters({ searchText: e.target.value || undefined })}
            className="input pl-10"
          />
        </div>

        <button
          onClick={() => setShowFilters(!showFilters)}
          className={clsx(
            'btn flex items-center space-x-2',
            hasActiveFilters ? 'btn-primary' : 'btn-secondary'
          )}
        >
          <Filter className="w-4 h-4" />
          <span>Filters</span>
          {hasActiveFilters && (
            <span className="px-2 py-0.5 bg-white/30 rounded-full text-xs">
              {[
                filters.types?.length,
                filters.tags?.length,
                filters.minImportance ? 1 : 0,
                filters.searchText ? 1 : 0,
              ]
                .filter(Boolean)
                .reduce((a, b) => (a || 0) + (b || 0), 0)}
            </span>
          )}
        </button>

        {hasActiveFilters && (
          <button onClick={onReset} className="btn-secondary flex items-center space-x-2">
            <X className="w-4 h-4" />
            <span>Reset</span>
          </button>
        )}
      </div>

      {/* Expanded filters */}
      {showFilters && (
        <div className="card space-y-4">
          {/* Memory Types */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Memory Types
            </label>
            <div className="flex flex-wrap gap-2">
              {memoryTypes.map((type) => (
                <button
                  key={type}
                  onClick={() => toggleType(type)}
                  className={clsx(
                    'px-3 py-1.5 rounded-lg text-sm font-medium transition-all',
                    filters.types?.includes(type)
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  )}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>

          {/* Importance Range */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Importance Range: {filters.minImportance || 0} - {filters.maxImportance || 10}
            </label>
            <div className="flex items-center space-x-4">
              <input
                type="range"
                min="0"
                max="10"
                step="0.5"
                value={filters.minImportance || 0}
                onChange={(e) =>
                  updateFilters({ minImportance: parseFloat(e.target.value) || undefined })
                }
                className="flex-1"
              />
              <input
                type="range"
                min="0"
                max="10"
                step="0.5"
                value={filters.maxImportance || 10}
                onChange={(e) =>
                  updateFilters({ maxImportance: parseFloat(e.target.value) || undefined })
                }
                className="flex-1"
              />
            </div>
          </div>

          {/* Date Range */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Start Date</label>
              <input
                type="date"
                value={filters.startDate || ''}
                onChange={(e) => updateFilters({ startDate: e.target.value || undefined })}
                className="input"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">End Date</label>
              <input
                type="date"
                value={filters.endDate || ''}
                onChange={(e) => updateFilters({ endDate: e.target.value || undefined })}
                className="input"
              />
            </div>
          </div>

          {/* Tags */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Filter by Tags (comma-separated)
            </label>
            <input
              type="text"
              placeholder="work, personal, project"
              value={filters.tags?.join(', ') || ''}
              onChange={(e) =>
                updateFilters({
                  tags: e.target.value
                    ? e.target.value.split(',').map((t) => t.trim())
                    : undefined,
                })
              }
              className="input"
            />
          </div>
        </div>
      )}
    </div>
  );
}
