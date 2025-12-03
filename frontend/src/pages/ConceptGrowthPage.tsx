import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  TrendingUp,
  Network,
  Star,
  RefreshCw,
  ArrowRight,
  Sparkles,
  Calendar,
  Link as LinkIcon,
} from 'lucide-react';
import { apiClient } from '../services/api';
import type { Memory } from '../types';
import clsx from 'clsx';
import { format, subMonths, eachMonthOfInterval, differenceInMonths } from 'date-fns';

interface ConceptGrowthPageProps {
  userId: string;
}

interface Concept {
  name: string;
  memories: Memory[];
  timelineData: {
    month: Date;
    count: number;
    avgImportance: number;
    relationships: number;
  }[];
  totalGrowth: number;
  importanceGrowth: number;
  relationshipGrowth: number;
}

export function ConceptGrowthPage({ userId }: ConceptGrowthPageProps) {
  const [refreshKey, setRefreshKey] = useState(0);
  const [selectedConcept, setSelectedConcept] = useState<string | null>(null);

  // Fetch memories
  const { data: memories = [], isLoading } = useQuery({
    queryKey: ['memories', userId, refreshKey],
    queryFn: async () => {
      const result = await apiClient.getMemories({
        user_id: userId,
        limit: 10000,
      });
      return result;
    },
  });

  // Extract concepts from tags
  const concepts = useMemo<Concept[]>(() => {
    const conceptMap = new Map<string, Memory[]>();

    // Group by tags
    memories.forEach((memory: Memory) => {
      memory.tags.forEach((tag: string) => {
        if (!conceptMap.has(tag)) {
          conceptMap.set(tag, []);
        }
        conceptMap.get(tag)!.push(memory);
      });
    });

    // Calculate growth for each concept
    const now = new Date();
    const months = eachMonthOfInterval({
      start: subMonths(now, 6),
      end: now,
    });

    return Array.from(conceptMap.entries())
      .filter(([_, mems]) => mems.length >= 3) // Only concepts with 3+ memories
      .map(([name, mems]) => {
        // Sort memories by date
        const sortedMems = [...mems].sort(
          (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
        );

        // Build timeline data
        const timelineData = months.map((month) => {
          const monthMems = sortedMems.filter(
            (m) => new Date(m.created_at) <= month
          );

          const count = monthMems.length;
          const avgImportance =
            count > 0
              ? monthMems.reduce((sum, m) => sum + m.importance, 0) / count
              : 0;
          const relationships = monthMems.reduce((sum, m) => sum + m.access_count, 0);

          return {
            month,
            count,
            avgImportance,
            relationships,
          };
        });

        // Calculate growth rates
        const firstMonth = timelineData[0];
        const lastMonth = timelineData[timelineData.length - 1];

        const totalGrowth =
          firstMonth.count > 0
            ? ((lastMonth.count - firstMonth.count) / firstMonth.count) * 100
            : 0;

        const importanceGrowth =
          firstMonth.avgImportance > 0
            ? ((lastMonth.avgImportance - firstMonth.avgImportance) /
                firstMonth.avgImportance) *
              100
            : 0;

        const relationshipGrowth =
          firstMonth.relationships > 0
            ? ((lastMonth.relationships - firstMonth.relationships) /
                firstMonth.relationships) *
              100
            : 0;

        return {
          name,
          memories: sortedMems,
          timelineData,
          totalGrowth,
          importanceGrowth,
          relationshipGrowth,
        };
      })
      .sort((a, b) => b.totalGrowth - a.totalGrowth);
  }, [memories]);

  const selectedConceptData = useMemo(() => {
    return concepts.find((c) => c.name === selectedConcept);
  }, [concepts, selectedConcept]);

  const handleRefresh = () => {
    setRefreshKey((prev) => prev + 1);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Sparkles className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Concept Growth Map</h1>
            <p className="text-gray-600">Track how your knowledge evolves over time</p>
          </div>
        </div>

        <button
          onClick={handleRefresh}
          disabled={isLoading}
          className="btn-secondary flex items-center space-x-2"
        >
          <RefreshCw className={clsx('w-4 h-4', isLoading && 'animate-spin')} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Top Growing Concepts */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {concepts.slice(0, 6).map((concept) => {
          const firstMonth = concept.timelineData[0];
          const lastMonth = concept.timelineData[concept.timelineData.length - 1];
          const isSelected = selectedConcept === concept.name;

          return (
            <button
              key={concept.name}
              onClick={() =>
                setSelectedConcept(isSelected ? null : concept.name)
              }
              className={clsx(
                'card text-left transition-all duration-200 cursor-pointer',
                isSelected && 'ring-2 ring-primary-600'
              )}
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h3 className="text-lg font-bold text-gray-900 mb-1">{concept.name}</h3>
                  <div className="flex items-center space-x-2 text-sm text-gray-600">
                    <Calendar className="w-4 h-4" />
                    <span>
                      {differenceInMonths(lastMonth.month, firstMonth.month)} months
                    </span>
                  </div>
                </div>

                <div
                  className={clsx(
                    'px-3 py-1 rounded-full text-sm font-semibold',
                    concept.totalGrowth > 100
                      ? 'bg-green-100 text-green-700'
                      : concept.totalGrowth > 50
                      ? 'bg-blue-100 text-blue-700'
                      : 'bg-gray-100 text-gray-700'
                  )}
                >
                  +{concept.totalGrowth.toFixed(0)}%
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2 mb-3">
                <div className="bg-gray-50 rounded-lg p-2">
                  <p className="text-xs text-gray-600 mb-1">Memories</p>
                  <div className="flex items-baseline space-x-1">
                    <p className="text-lg font-bold text-gray-900">{lastMonth.count}</p>
                    <ArrowRight className="w-3 h-3 text-gray-400" />
                    <p className="text-sm text-gray-500">{firstMonth.count}</p>
                  </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-2">
                  <p className="text-xs text-gray-600 mb-1">Importance</p>
                  <div className="flex items-baseline space-x-1">
                    <p className="text-lg font-bold text-gray-900">
                      {lastMonth.avgImportance.toFixed(1)}
                    </p>
                    {concept.importanceGrowth > 0 && (
                      <TrendingUp className="w-3 h-3 text-green-600" />
                    )}
                  </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-2">
                  <p className="text-xs text-gray-600 mb-1">Links</p>
                  <div className="flex items-baseline space-x-1">
                    <p className="text-lg font-bold text-gray-900">
                      {lastMonth.relationships}
                    </p>
                    {concept.relationshipGrowth > 0 && (
                      <TrendingUp className="w-3 h-3 text-blue-600" />
                    )}
                  </div>
                </div>
              </div>

              {/* Mini Timeline */}
              <div className="h-12 flex items-end space-x-0.5">
                {concept.timelineData.map((point, idx) => {
                  const max = Math.max(...concept.timelineData.map((p) => p.count));
                  const height = max > 0 ? (point.count / max) * 100 : 0;

                  return (
                    <div
                      key={idx}
                      className="flex-1 bg-primary-500 rounded-t transition-all hover:bg-primary-600"
                      style={{ height: `${height}%` }}
                    />
                  );
                })}
              </div>
            </button>
          );
        })}
      </div>

      {/* Detailed View */}
      {selectedConceptData && (
        <div className="card">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <Sparkles className="w-6 h-6 text-purple-600" />
              <h2 className="text-2xl font-bold text-gray-900">{selectedConceptData.name}</h2>
              <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm font-medium">
                {selectedConceptData.memories.length} memories
              </span>
            </div>

            <button
              onClick={() => setSelectedConcept(null)}
              className="text-sm text-gray-600 hover:text-gray-900"
            >
              Close
            </button>
          </div>

          {/* Growth Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <TrendingUp className="w-5 h-5 text-green-600" />
                <h3 className="text-sm font-semibold text-green-900">Memory Growth</h3>
              </div>
              <p className="text-3xl font-bold text-green-900">
                +{selectedConceptData.totalGrowth.toFixed(0)}%
              </p>
              <p className="text-sm text-gray-600 mt-1">
                {selectedConceptData.timelineData[0].count} →{' '}
                {
                  selectedConceptData.timelineData[
                    selectedConceptData.timelineData.length - 1
                  ].count
                }{' '}
                memories
              </p>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <Star className="w-5 h-5 text-blue-600" />
                <h3 className="text-sm font-semibold text-blue-900">Importance Growth</h3>
              </div>
              <p className="text-3xl font-bold text-blue-900">
                {selectedConceptData.importanceGrowth > 0 ? '+' : ''}
                {selectedConceptData.importanceGrowth.toFixed(0)}%
              </p>
              <p className="text-sm text-gray-600 mt-1">
                {selectedConceptData.timelineData[0].avgImportance.toFixed(1)} →{' '}
                {selectedConceptData.timelineData[
                  selectedConceptData.timelineData.length - 1
                ].avgImportance.toFixed(1)}
              </p>
            </div>

            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <Network className="w-5 h-5 text-purple-600" />
                <h3 className="text-sm font-semibold text-purple-900">Relationship Growth</h3>
              </div>
              <p className="text-3xl font-bold text-purple-900">
                {selectedConceptData.relationshipGrowth > 0 ? '+' : ''}
                {selectedConceptData.relationshipGrowth.toFixed(0)}%
              </p>
              <p className="text-sm text-gray-600 mt-1">
                {selectedConceptData.timelineData[0].relationships} →{' '}
                {
                  selectedConceptData.timelineData[
                    selectedConceptData.timelineData.length - 1
                  ].relationships
                }{' '}
                links
              </p>
            </div>
          </div>

          {/* Timeline Chart */}
          <div>
            <h3 className="text-lg font-bold text-gray-900 mb-4">Growth Timeline</h3>

            {/* Memory Count Over Time */}
            <div className="mb-6">
              <p className="text-sm text-gray-600 mb-3">Memory Count</p>
              <div className="h-32 flex items-end space-x-2">
                {selectedConceptData.timelineData.map((point, idx) => {
                  const max = Math.max(
                    ...selectedConceptData.timelineData.map((p) => p.count)
                  );
                  const height = max > 0 ? (point.count / max) * 100 : 0;

                  return (
                    <div key={idx} className="flex-1 flex flex-col items-center">
                      <div
                        className="w-full bg-green-500 rounded-t hover:bg-green-600 transition-colors cursor-pointer"
                        style={{ height: `${height}%` }}
                        title={`${format(point.month, 'MMM yyyy')}: ${point.count} memories`}
                      />
                      <span className="text-xs text-gray-500 mt-2">
                        {format(point.month, 'MMM')}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Importance Over Time */}
            <div className="mb-6">
              <p className="text-sm text-gray-600 mb-3">Average Importance</p>
              <div className="h-32 flex items-end space-x-2">
                {selectedConceptData.timelineData.map((point, idx) => {
                  const height = (point.avgImportance / 10) * 100;

                  return (
                    <div key={idx} className="flex-1 flex flex-col items-center">
                      <div
                        className="w-full bg-blue-500 rounded-t hover:bg-blue-600 transition-colors cursor-pointer"
                        style={{ height: `${height}%` }}
                        title={`${format(point.month, 'MMM yyyy')}: ${point.avgImportance.toFixed(
                          1
                        )} importance`}
                      />
                      <span className="text-xs text-gray-500 mt-2">
                        {format(point.month, 'MMM')}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Relationships Over Time */}
            <div>
              <p className="text-sm text-gray-600 mb-3">Relationship Links</p>
              <div className="h-32 flex items-end space-x-2">
                {selectedConceptData.timelineData.map((point, idx) => {
                  const max = Math.max(
                    ...selectedConceptData.timelineData.map((p) => p.relationships)
                  );
                  const height = max > 0 ? (point.relationships / max) * 100 : 0;

                  return (
                    <div key={idx} className="flex-1 flex flex-col items-center">
                      <div
                        className="w-full bg-purple-500 rounded-t hover:bg-purple-600 transition-colors cursor-pointer"
                        style={{ height: `${height}%` }}
                        title={`${format(point.month, 'MMM yyyy')}: ${
                          point.relationships
                        } relationships`}
                      />
                      <span className="text-xs text-gray-500 mt-2">
                        {format(point.month, 'MMM')}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* All Concepts List */}
      <div className="card">
        <div className="flex items-center space-x-3 mb-6">
          <LinkIcon className="w-6 h-6 text-gray-600" />
          <h2 className="text-xl font-bold text-gray-900">All Concepts</h2>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                  Concept
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                  Memories
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                  Growth
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                  Avg Importance
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                  Relationships
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {concepts.map((concept) => {
                const lastMonth = concept.timelineData[concept.timelineData.length - 1];

                return (
                  <tr
                    key={concept.name}
                    className="hover:bg-gray-50 cursor-pointer"
                    onClick={() => setSelectedConcept(concept.name)}
                  >
                    <td className="px-4 py-3 text-gray-900 font-medium">{concept.name}</td>
                    <td className="px-4 py-3 text-gray-900">{lastMonth.count}</td>
                    <td className="px-4 py-3">
                      <span
                        className={clsx(
                          'px-2 py-1 rounded-full text-xs font-medium',
                          concept.totalGrowth > 100
                            ? 'bg-green-100 text-green-700'
                            : concept.totalGrowth > 50
                            ? 'bg-blue-100 text-blue-700'
                            : 'bg-gray-100 text-gray-700'
                        )}
                      >
                        +{concept.totalGrowth.toFixed(0)}%
                      </span>
                    </td>
                    <td className="px-4 py-3 text-gray-900">
                      {lastMonth.avgImportance.toFixed(1)}
                    </td>
                    <td className="px-4 py-3 text-gray-900">{lastMonth.relationships}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
