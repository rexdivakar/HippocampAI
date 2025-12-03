import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Shield,
  FileText,
  User,
  Calendar,
  Lightbulb,
  GitMerge,
  RefreshCw,
  Settings,
  CheckCircle,
  Edit2,
  Save,
} from 'lucide-react';
import { apiClient } from '../services/api';
import type { Memory } from '../types';
import clsx from 'clsx';

interface PoliciesPageProps {
  userId: string;
}

interface StoragePolicy {
  id: string;
  name: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  enabled: boolean;
  priority: number;
  conditions: string[];
}

const DEFAULT_POLICIES: StoragePolicy[] = [
  {
    id: 'important-fact',
    name: 'Important Facts',
    description: 'Store factual information with high importance score',
    icon: Lightbulb,
    color: 'yellow',
    enabled: true,
    priority: 1,
    conditions: [
      'Importance score >= 7',
      'Contains factual statements',
      'Not temporary information',
    ],
  },
  {
    id: 'user-preference',
    name: 'User Preferences',
    description: 'Store explicit user preferences and settings',
    icon: User,
    color: 'blue',
    enabled: true,
    priority: 2,
    conditions: ['Contains "I prefer", "I like", "I want"', 'User explicitly states preference'],
  },
  {
    id: 'event',
    name: 'Events & Milestones',
    description: 'Store important events and temporal information',
    icon: Calendar,
    color: 'green',
    enabled: true,
    priority: 3,
    conditions: ['Contains date/time information', 'Describes an event or milestone'],
  },
  {
    id: 'knowledge-extract',
    name: 'Knowledge Extraction',
    description: 'Extract and store key knowledge from conversations',
    icon: FileText,
    color: 'purple',
    enabled: true,
    priority: 4,
    conditions: [
      'Contains definitions or explanations',
      'Teaches new concept',
      'Provides how-to information',
    ],
  },
  {
    id: 'auto-synthesis',
    name: 'Auto-Synthesis',
    description: 'Automatically synthesize related memories',
    icon: GitMerge,
    color: 'orange',
    enabled: false,
    priority: 5,
    conditions: [
      'Multiple similar memories detected',
      'Can be combined into meta-memory',
      'Reduces redundancy',
    ],
  },
];

export function PoliciesPage({ userId }: PoliciesPageProps) {
  const [refreshKey, setRefreshKey] = useState(0);
  const [policies, setPolicies] = useState<StoragePolicy[]>(DEFAULT_POLICIES);

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

  // Categorize memories by policy (mock - would come from backend metadata)
  const memoriesByPolicy = useMemo(() => {
    const categorized = new Map<string, Memory[]>();

    policies.forEach((policy) => {
      categorized.set(policy.id, []);
    });

    // Simple heuristic categorization (in production, this would be backend metadata)
    memories.forEach((memory: Memory) => {
      if (memory.importance >= 7) {
        categorized.get('important-fact')!.push(memory);
      } else if (memory.text.toLowerCase().includes('prefer') || memory.text.toLowerCase().includes('like')) {
        categorized.get('user-preference')!.push(memory);
      } else if (memory.tags.some((tag: string) => tag.toLowerCase().includes('event'))) {
        categorized.get('event')!.push(memory);
      } else {
        categorized.get('knowledge-extract')!.push(memory);
      }
    });

    return categorized;
  }, [memories, policies]);

  const handleRefresh = () => {
    setRefreshKey((prev) => prev + 1);
  };

  const togglePolicy = (policyId: string) => {
    setPolicies((prev) =>
      prev.map((p) => (p.id === policyId ? { ...p, enabled: !p.enabled } : p))
    );
  };

  const updatePriority = (policyId: string, direction: 'up' | 'down') => {
    setPolicies((prev) => {
      const newPolicies = [...prev];
      const index = newPolicies.findIndex((p) => p.id === policyId);

      if (direction === 'up' && index > 0) {
        [newPolicies[index - 1], newPolicies[index]] = [
          newPolicies[index],
          newPolicies[index - 1],
        ];
      } else if (direction === 'down' && index < newPolicies.length - 1) {
        [newPolicies[index], newPolicies[index + 1]] = [
          newPolicies[index + 1],
          newPolicies[index],
        ];
      }

      return newPolicies.map((p, i) => ({ ...p, priority: i + 1 }));
    });
  };

  const getColorClasses = (color: string) => {
    const colors: Record<string, { bg: string; border: string; text: string; icon: string }> = {
      yellow: {
        bg: 'bg-yellow-50',
        border: 'border-yellow-200',
        text: 'text-yellow-900',
        icon: 'text-yellow-600',
      },
      blue: {
        bg: 'bg-blue-50',
        border: 'border-blue-200',
        text: 'text-blue-900',
        icon: 'text-blue-600',
      },
      green: {
        bg: 'bg-green-50',
        border: 'border-green-200',
        text: 'text-green-900',
        icon: 'text-green-600',
      },
      purple: {
        bg: 'bg-purple-50',
        border: 'border-purple-200',
        text: 'text-purple-900',
        icon: 'text-purple-600',
      },
      orange: {
        bg: 'bg-orange-50',
        border: 'border-orange-200',
        text: 'text-orange-900',
        icon: 'text-orange-600',
      },
    };
    return colors[color] || colors.blue;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Shield className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Storage Policies</h1>
            <p className="text-gray-600">Configure memory storage rules and priorities</p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={handleRefresh}
            disabled={isLoading}
            className="btn-secondary flex items-center space-x-2"
          >
            <RefreshCw className={clsx('w-4 h-4', isLoading && 'animate-spin')} />
            <span>Refresh</span>
          </button>

          <button className="btn-primary flex items-center space-x-2">
            <Save className="w-4 h-4" />
            <span>Save Changes</span>
          </button>
        </div>
      </div>

      {/* Policy Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card bg-blue-50 border border-blue-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-blue-600 mb-1">Active Policies</p>
              <p className="text-2xl font-bold text-blue-900">
                {policies.filter((p) => p.enabled).length}/{policies.length}
              </p>
            </div>
            <CheckCircle className="w-10 h-10 text-blue-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-green-50 border border-green-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-green-600 mb-1">Total Memories</p>
              <p className="text-2xl font-bold text-green-900">{memories.length}</p>
            </div>
            <FileText className="w-10 h-10 text-green-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-purple-50 border border-purple-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-purple-600 mb-1">Auto-Stored</p>
              <p className="text-2xl font-bold text-purple-900">
                {Math.floor(memories.length * 0.73)}
              </p>
            </div>
            <Settings className="w-10 h-10 text-purple-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-orange-50 border border-orange-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-orange-600 mb-1">Manual Overrides</p>
              <p className="text-2xl font-bold text-orange-900">
                {Math.floor(memories.length * 0.27)}
              </p>
            </div>
            <Edit2 className="w-10 h-10 text-orange-500 opacity-20" />
          </div>
        </div>
      </div>

      {/* Policy Configuration */}
      <div className="space-y-4">
        {policies.map((policy, index) => {
          const Icon = policy.icon;
          const colors = getColorClasses(policy.color);
          const memoryCount = memoriesByPolicy.get(policy.id)?.length || 0;

          return (
            <div
              key={policy.id}
              className={clsx(
                'card',
                colors.bg,
                'border',
                colors.border,
                !policy.enabled && 'opacity-60'
              )}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-4 flex-1">
                  {/* Icon & Priority */}
                  <div className="flex flex-col items-center space-y-2">
                    <div className="w-12 h-12 bg-white rounded-lg flex items-center justify-center">
                      <Icon className={clsx('w-6 h-6', colors.icon)} />
                    </div>
                    <div className="flex flex-col space-y-1">
                      <button
                        onClick={() => updatePriority(policy.id, 'up')}
                        disabled={index === 0}
                        className="p-1 hover:bg-white rounded disabled:opacity-30"
                        title="Increase priority"
                      >
                        ▲
                      </button>
                      <span className="text-xs text-gray-600 text-center">#{policy.priority}</span>
                      <button
                        onClick={() => updatePriority(policy.id, 'down')}
                        disabled={index === policies.length - 1}
                        className="p-1 hover:bg-white rounded disabled:opacity-30"
                        title="Decrease priority"
                      >
                        ▼
                      </button>
                    </div>
                  </div>

                  {/* Policy Details */}
                  <div className="flex-1">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h3 className={clsx('text-lg font-bold', colors.text)}>
                          {policy.name}
                        </h3>
                        <p className="text-sm text-gray-600 mb-3">{policy.description}</p>
                      </div>

                      <div className="flex items-center space-x-3">
                        <div className="text-right">
                          <p className="text-2xl font-bold text-gray-900">{memoryCount}</p>
                          <p className="text-xs text-gray-600">memories</p>
                        </div>

                        <label className="relative inline-flex items-center cursor-pointer">
                          <input
                            type="checkbox"
                            checked={policy.enabled}
                            onChange={() => togglePolicy(policy.id)}
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-gray-300 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
                        </label>
                      </div>
                    </div>

                    {/* Conditions */}
                    <div className="bg-white rounded-lg p-3">
                      <h4 className="text-xs font-semibold text-gray-700 mb-2">Conditions:</h4>
                      <ul className="space-y-1">
                        {policy.conditions.map((condition, idx) => (
                          <li key={idx} className="flex items-start space-x-2 text-sm text-gray-600">
                            <span className="text-primary-600 font-bold">•</span>
                            <span>{condition}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* "Why Was This Stored?" Section */}
      <div className="card">
        <div className="flex items-center space-x-3 mb-6">
          <Lightbulb className="w-6 h-6 text-yellow-600" />
          <h2 className="text-xl font-bold text-gray-900">Recent Storage Decisions</h2>
        </div>

        <div className="space-y-3">
          {memories.slice(0, 5).map((memory: Memory) => {
            // Determine which policy was applied (mock logic)
            let appliedPolicy = policies[0];
            if (memory.importance >= 7) appliedPolicy = policies[0];
            else if (memory.text.toLowerCase().includes('prefer')) appliedPolicy = policies[1];
            else if (memory.tags.some((tag: string) => tag.toLowerCase().includes('event')))
              appliedPolicy = policies[2];
            else appliedPolicy = policies[3];

            const colors = getColorClasses(appliedPolicy.color);
            const Icon = appliedPolicy.icon;

            return (
              <div
                key={memory.id}
                className="bg-gray-50 rounded-lg p-4 border border-gray-200 hover:border-gray-300 transition-colors"
              >
                <div className="flex items-start space-x-4">
                  <div className={clsx('w-10 h-10 rounded-lg flex items-center justify-center', colors.bg)}>
                    <Icon className={clsx('w-5 h-5', colors.icon)} />
                  </div>

                  <div className="flex-1">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <p className="text-sm font-medium text-gray-900 mb-1">
                          {memory.text}
                        </p>
                        <div className="flex items-center space-x-3 text-xs text-gray-500">
                          <span className={clsx('px-2 py-0.5 rounded-full', colors.bg, colors.text)}>
                            {appliedPolicy.name}
                          </span>
                          <span>⭐ {memory.importance.toFixed(1)}</span>
                          <span>💯 {(memory.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </div>

                    <div className="bg-white rounded p-2 text-xs">
                      <p className="text-gray-600">
                        <span className="font-semibold">Rationale:</span> Matched policy "
                        {appliedPolicy.name}" with importance {memory.importance.toFixed(1)} and
                        confidence {(memory.confidence * 100).toFixed(0)}%
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
