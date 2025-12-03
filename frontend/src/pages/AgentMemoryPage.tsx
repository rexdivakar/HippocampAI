import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Bot,
  Brain,
  Code,
  GraduationCap,
  Server,
  User,
  RefreshCw,
  TrendingUp,
  Eye,
  Star,
  Filter,
} from 'lucide-react';
import { apiClient } from '../services/api';
import type { Memory } from '../types';
import { MemoryCard } from '../components/MemoryCard';
import clsx from 'clsx';

interface AgentMemoryPageProps {
  userId: string;
}

interface AgentNamespace {
  id: string;
  name: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  description: string;
}

const AGENT_NAMESPACES: AgentNamespace[] = [
  {
    id: 'research-agent',
    name: 'Research Agent',
    icon: GraduationCap,
    color: 'blue',
    description: 'Academic research and information gathering',
  },
  {
    id: 'coding-agent',
    name: 'Coding Agent',
    icon: Code,
    color: 'green',
    description: 'Code generation and software development',
  },
  {
    id: 'edtech-agent',
    name: 'EdTech Agent',
    icon: GraduationCap,
    color: 'purple',
    description: 'Educational content and tutoring',
  },
  {
    id: 'devops-agent',
    name: 'DevOps Agent',
    icon: Server,
    color: 'orange',
    description: 'Infrastructure and deployment automation',
  },
  {
    id: 'personal-assistant',
    name: 'Personal Assistant',
    icon: User,
    color: 'pink',
    description: 'Personal tasks and schedule management',
  },
  {
    id: 'general-agent',
    name: 'General Agent',
    icon: Bot,
    color: 'gray',
    description: 'General purpose AI assistant',
  },
];

export function AgentMemoryPage({ userId }: AgentMemoryPageProps) {
  const [refreshKey, setRefreshKey] = useState(0);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);

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

  // Group memories by agent
  const agentMemories = useMemo(() => {
    const grouped = new Map<string, Memory[]>();

    AGENT_NAMESPACES.forEach((agent) => {
      grouped.set(agent.id, []);
    });

    memories.forEach((memory: Memory) => {
      const agentId = memory.agent_id || 'general-agent';
      if (!grouped.has(agentId)) {
        grouped.set(agentId, []);
      }
      grouped.get(agentId)!.push(memory);
    });

    return grouped;
  }, [memories]);

  // Calculate agent stats
  const agentStats = useMemo(() => {
    return AGENT_NAMESPACES.map((agent) => {
      const agentMems = agentMemories.get(agent.id) || [];
      const totalMemories = agentMems.length;
      const avgImportance =
        totalMemories > 0
          ? agentMems.reduce((sum: number, m: Memory) => sum + m.importance, 0) / totalMemories
          : 0;
      const totalAccess = agentMems.reduce((sum: number, m: Memory) => sum + m.access_count, 0);
      const recentActivity = agentMems.filter(
        (m: Memory) =>
          new Date(m.created_at) > new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
      ).length;

      return {
        ...agent,
        totalMemories,
        avgImportance,
        totalAccess,
        recentActivity,
      };
    });
  }, [agentMemories]);

  const filteredMemories = useMemo(() => {
    if (!selectedAgent) return [];
    return agentMemories.get(selectedAgent) || [];
  }, [selectedAgent, agentMemories]);

  const handleRefresh = () => {
    setRefreshKey((prev) => prev + 1);
  };

  const getColorClasses = (color: string) => {
    const colors: Record<string, { bg: string; border: string; text: string; icon: string }> = {
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
      pink: {
        bg: 'bg-pink-50',
        border: 'border-pink-200',
        text: 'text-pink-900',
        icon: 'text-pink-600',
      },
      gray: {
        bg: 'bg-gray-50',
        border: 'border-gray-200',
        text: 'text-gray-900',
        icon: 'text-gray-600',
      },
    };
    return colors[color] || colors.gray;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Bot className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Agent Memory Namespaces</h1>
            <p className="text-gray-600">Multi-agent memory orchestration and isolation</p>
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

      {/* Agent Namespace Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {agentStats.map((agent) => {
          const Icon = agent.icon;
          const colors = getColorClasses(agent.color);
          const isSelected = selectedAgent === agent.id;

          return (
            <button
              key={agent.id}
              onClick={() =>
                setSelectedAgent(isSelected ? null : agent.id)
              }
              className={clsx(
                'card text-left transition-all duration-200 cursor-pointer',
                colors.bg,
                'border',
                colors.border,
                isSelected && 'ring-2 ring-primary-600'
              )}
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div
                    className={clsx(
                      'w-12 h-12 rounded-lg flex items-center justify-center',
                      isSelected ? 'bg-primary-100' : 'bg-white'
                    )}
                  >
                    <Icon
                      className={clsx(
                        'w-6 h-6',
                        isSelected ? 'text-primary-600' : colors.icon
                      )}
                    />
                  </div>
                  <div>
                    <h3 className={clsx('text-lg font-bold', colors.text)}>
                      {agent.name}
                    </h3>
                    <p className="text-xs text-gray-600">{agent.description}</p>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="bg-white rounded-lg p-3">
                  <div className="flex items-center space-x-2 mb-1">
                    <Brain className="w-4 h-4 text-gray-500" />
                    <p className="text-xs text-gray-600">Total</p>
                  </div>
                  <p className={clsx('text-xl font-bold', colors.text)}>
                    {agent.totalMemories}
                  </p>
                </div>

                <div className="bg-white rounded-lg p-3">
                  <div className="flex items-center space-x-2 mb-1">
                    <Star className="w-4 h-4 text-gray-500" />
                    <p className="text-xs text-gray-600">Avg Import</p>
                  </div>
                  <p className={clsx('text-xl font-bold', colors.text)}>
                    {agent.avgImportance.toFixed(1)}
                  </p>
                </div>

                <div className="bg-white rounded-lg p-3">
                  <div className="flex items-center space-x-2 mb-1">
                    <Eye className="w-4 h-4 text-gray-500" />
                    <p className="text-xs text-gray-600">Access</p>
                  </div>
                  <p className={clsx('text-xl font-bold', colors.text)}>
                    {agent.totalAccess}
                  </p>
                </div>

                <div className="bg-white rounded-lg p-3">
                  <div className="flex items-center space-x-2 mb-1">
                    <TrendingUp className="w-4 h-4 text-gray-500" />
                    <p className="text-xs text-gray-600">7d Activity</p>
                  </div>
                  <p className={clsx('text-xl font-bold', colors.text)}>
                    {agent.recentActivity}
                  </p>
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Agent Memories */}
      {selectedAgent && (
        <div className="card">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <Filter className="w-6 h-6 text-primary-600" />
              <h2 className="text-xl font-bold text-gray-900">
                {AGENT_NAMESPACES.find((a) => a.id === selectedAgent)?.name} Memories
              </h2>
              <span className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm font-medium">
                {filteredMemories.length} memories
              </span>
            </div>

            <button
              onClick={() => setSelectedAgent(null)}
              className="text-sm text-gray-600 hover:text-gray-900"
            >
              Clear Filter
            </button>
          </div>

          {filteredMemories.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <Brain className="w-16 h-16 mx-auto mb-4 text-gray-400" />
              <p>No memories found for this agent</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredMemories.map((memory: Memory) => (
                <MemoryCard
                  key={memory.id}
                  memory={memory}
                  onView={() => {}}
                  onEdit={() => {}}
                  onShare={() => {}}
                  onDelete={() => {}}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Agent Comparison */}
      {!selectedAgent && (
        <div className="card">
          <div className="flex items-center space-x-3 mb-6">
            <TrendingUp className="w-6 h-6 text-primary-600" />
            <h2 className="text-xl font-bold text-gray-900">Agent Performance Comparison</h2>
          </div>

          <div className="space-y-4">
            {/* Memory Count */}
            <div>
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Memory Count</h3>
              <div className="space-y-2">
                {agentStats
                  .sort((a, b) => b.totalMemories - a.totalMemories)
                  .map((agent) => {
                    const maxMemories = Math.max(...agentStats.map((a) => a.totalMemories));
                    const percentage =
                      maxMemories > 0 ? (agent.totalMemories / maxMemories) * 100 : 0;
                    const colors = getColorClasses(agent.color);

                    return (
                      <div key={agent.id} className="flex items-center space-x-3">
                        <span className="text-sm text-gray-600 w-40 truncate">
                          {agent.name}
                        </span>
                        <div className="flex-1 bg-gray-200 rounded-full h-6 relative overflow-hidden">
                          <div
                            className={clsx('h-6 rounded-full flex items-center justify-end pr-2')}
                            style={{
                              width: `${percentage}%`,
                              background: `linear-gradient(to right, ${colors.bg}, ${colors.icon})`,
                            }}
                          >
                            <span className="text-xs font-semibold text-gray-900">
                              {agent.totalMemories}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>

            {/* Average Importance */}
            <div>
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Average Importance</h3>
              <div className="space-y-2">
                {agentStats
                  .sort((a, b) => b.avgImportance - a.avgImportance)
                  .map((agent) => {
                    const colors = getColorClasses(agent.color);

                    return (
                      <div key={agent.id} className="flex items-center space-x-3">
                        <span className="text-sm text-gray-600 w-40 truncate">
                          {agent.name}
                        </span>
                        <div className="flex-1 bg-gray-200 rounded-full h-6 relative overflow-hidden">
                          <div
                            className={clsx('h-6 rounded-full flex items-center justify-end pr-2')}
                            style={{
                              width: `${(agent.avgImportance / 10) * 100}%`,
                              background: `linear-gradient(to right, ${colors.bg}, ${colors.icon})`,
                            }}
                          >
                            <span className="text-xs font-semibold text-gray-900">
                              {agent.avgImportance.toFixed(1)}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
