import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Sparkles,
  Trash2,
  Copy,
  GitMerge,
  Shield,
  Calendar,
  Star,
  RefreshCw,
  CheckCircle,
  AlertTriangle,
} from 'lucide-react';
import { apiClient } from '../services/api';
import clsx from 'clsx';

interface HygienePageProps {
  userId: string;
}

export function HygienePage({ userId }: HygienePageProps) {
  const [refreshKey, setRefreshKey] = useState(0);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.85);
  const [importanceThreshold, setImportanceThreshold] = useState(3);
  const [retentionDays, setRetentionDays] = useState(90);

  // Fetch memories
  const { data: memories = [], isLoading } = useQuery({
    queryKey: ['memories', userId, refreshKey],
    queryFn: async () => {
      const result = await apiClient.getMemories({
        user_id: userId,
        filters: {
          session_id: userId, // Pass userId as session_id to match by either field
        },
        limit: 1000,
      });
      return result;
    },
  });

  // Find duplicates
  const duplicates = findDuplicates(memories, similarityThreshold);
  const lowImportance = memories.filter((m) => m.importance < importanceThreshold);
  const oldMemories = memories.filter((m) => {
    const daysSince = (Date.now() - new Date(m.created_at).getTime()) / (1000 * 60 * 60 * 24);
    return daysSince > retentionDays && m.access_count === 0;
  });

  const handleRefresh = () => {
    setRefreshKey((prev) => prev + 1);
  };

  const handleDeduplication = async () => {
    // In production, this would call the backend API
    console.log('Deduplicating', duplicates.length, 'pairs');
    alert(`Would deduplicate ${duplicates.length} duplicate pairs`);
  };

  const handleCleanup = async (type: string) => {
    console.log('Cleaning up:', type);
    alert(`Would clean up ${type} memories`);
  };

  return (
    <div className="w-full space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="flex items-center space-x-3">
          <Sparkles className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Memory Hygiene Tools</h1>
            <p className="text-gray-600">Maintain a healthy and clean knowledge base</p>
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

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card bg-blue-50 border border-blue-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-blue-600 mb-1">Total Memories</p>
              <p className="text-2xl font-bold text-blue-900">{memories.length}</p>
            </div>
            <CheckCircle className="w-10 h-10 text-blue-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-yellow-50 border border-yellow-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-yellow-600 mb-1">Duplicates Found</p>
              <p className="text-2xl font-bold text-yellow-900">{duplicates.length}</p>
            </div>
            <Copy className="w-10 h-10 text-yellow-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-orange-50 border border-orange-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-orange-600 mb-1">Low Importance</p>
              <p className="text-2xl font-bold text-orange-900">{lowImportance.length}</p>
            </div>
            <AlertTriangle className="w-10 h-10 text-orange-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-red-50 border border-red-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-red-600 mb-1">Old & Unused</p>
              <p className="text-2xl font-bold text-red-900">{oldMemories.length}</p>
            </div>
            <Trash2 className="w-10 h-10 text-red-500 opacity-20" />
          </div>
        </div>
      </div>

      {/* Clean-Up Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Deduplicate Memories */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center">
              <Copy className="w-6 h-6 text-yellow-600" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900">Deduplicate Memories</h2>
              <p className="text-sm text-gray-600">Remove similar or duplicate memories</p>
            </div>
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Similarity Threshold: {(similarityThreshold * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.5"
              max="1"
              step="0.05"
              value={similarityThreshold}
              onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          <div className="bg-yellow-50 rounded-lg p-4 mb-4">
            <p className="text-sm text-gray-700">
              Found <span className="font-bold text-yellow-700">{duplicates.length}</span> potential
              duplicate pairs
            </p>
          </div>

          <button
            onClick={handleDeduplication}
            disabled={duplicates.length === 0}
            className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Deduplicate Now
          </button>
        </div>

        {/* Merge Similar Memories */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
              <GitMerge className="w-6 h-6 text-purple-600" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900">Merge Similar Memories</h2>
              <p className="text-sm text-gray-600">Combine related memories intelligently</p>
            </div>
          </div>

          <div className="bg-purple-50 rounded-lg p-4 mb-4">
            <p className="text-sm text-gray-700">
              This will merge memories with high semantic similarity while preserving unique
              information.
            </p>
          </div>

          <button
            onClick={() => handleCleanup('merge')}
            className="btn-primary w-full bg-purple-600 hover:bg-purple-700"
          >
            Merge Memories
          </button>
        </div>

        {/* Remove Low Importance */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center">
              <AlertTriangle className="w-6 h-6 text-orange-600" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900">Remove Low Importance</h2>
              <p className="text-sm text-gray-600">Clean up trivial or noise memories</p>
            </div>
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Importance Threshold: {importanceThreshold}/10
            </label>
            <input
              type="range"
              min="1"
              max="5"
              step="0.5"
              value={importanceThreshold}
              onChange={(e) => setImportanceThreshold(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          <div className="bg-orange-50 rounded-lg p-4 mb-4">
            <p className="text-sm text-gray-700">
              Found <span className="font-bold text-orange-700">{lowImportance.length}</span>{' '}
              memories below threshold
            </p>
          </div>

          <button
            onClick={() => handleCleanup('low-importance')}
            disabled={lowImportance.length === 0}
            className="btn-primary w-full bg-orange-600 hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Remove Low Importance
          </button>
        </div>

        {/* Auto-Delete Old Memories */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center">
              <Calendar className="w-6 h-6 text-red-600" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900">Auto-Delete Old Memories</h2>
              <p className="text-sm text-gray-600">Remove unused memories based on age</p>
            </div>
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Retention Period: {retentionDays} days
            </label>
            <input
              type="range"
              min="30"
              max="365"
              step="30"
              value={retentionDays}
              onChange={(e) => setRetentionDays(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <div className="bg-red-50 rounded-lg p-4 mb-4">
            <p className="text-sm text-gray-700">
              Found <span className="font-bold text-red-700">{oldMemories.length}</span> old unused
              memories
            </p>
          </div>

          <button
            onClick={() => handleCleanup('old')}
            disabled={oldMemories.length === 0}
            className="btn-primary w-full bg-red-600 hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Delete Old Memories
          </button>
        </div>

        {/* Privacy Protection */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-indigo-100 rounded-lg flex items-center justify-center">
              <Shield className="w-6 h-6 text-indigo-600" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900">Privacy Protection</h2>
              <p className="text-sm text-gray-600">Remove sensitive information</p>
            </div>
          </div>

          <div className="bg-indigo-50 rounded-lg p-4 mb-4">
            <p className="text-sm text-gray-700">
              Automatically detect and remove personally identifiable information (PII), IP addresses,
              and random logs.
            </p>
          </div>

          <button
            onClick={() => handleCleanup('pii')}
            className="btn-primary w-full bg-indigo-600 hover:bg-indigo-700"
          >
            Scan & Remove PII
          </button>
        </div>

        {/* Retention Rules */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
              <Star className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900">Set Retention Rules</h2>
              <p className="text-sm text-gray-600">Preserve important memories</p>
            </div>
          </div>

          <div className="bg-green-50 rounded-lg p-4 mb-4">
            <div className="space-y-2 text-sm text-gray-700">
              <label className="flex items-center space-x-2">
                <input type="checkbox" className="rounded text-green-600" defaultChecked />
                <span>Keep high importance memories (≥ 7)</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="checkbox" className="rounded text-green-600" defaultChecked />
                <span>Keep frequently accessed memories (≥ 5 uses)</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="checkbox" className="rounded text-green-600" />
                <span>Keep starred/favorited memories</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="checkbox" className="rounded text-green-600" />
                <span>Keep recent memories (&lt; 30 days)</span>
              </label>
            </div>
          </div>

          <button className="btn-primary w-full bg-green-600 hover:bg-green-700">
            Apply Retention Rules
          </button>
        </div>
      </div>

      {/* Preview Section */}
      {duplicates.length > 0 && (
        <div className="card">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Duplicate Preview (Top 3)</h2>
          <div className="space-y-4">
            {duplicates.slice(0, 3).map((pair, idx) => (
              <div key={idx} className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {pair.map((memory) => (
                    <div key={memory.id} className="bg-white rounded-lg p-3">
                      <p className="text-sm text-gray-900 mb-2">{memory.text}</p>
                      <div className="flex items-center space-x-2 text-xs text-gray-500">
                        <span>ID: {memory.id.slice(0, 8)}</span>
                        <span>⭐ {memory.importance.toFixed(1)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Simple duplicate detection using Jaccard similarity
function findDuplicates(memories: any[], threshold: number): Array<any[]> {
  const duplicates: Array<any[]> = [];
  for (let i = 0; i < memories.length; i++) {
    for (let j = i + 1; j < memories.length; j++) {
      const similarity = calculateSimilarity(memories[i].text, memories[j].text);
      if (similarity >= threshold) {
        duplicates.push([memories[i], memories[j]]);
      }
    }
  }
  return duplicates;
}

function calculateSimilarity(text1: string, text2: string): number {
  const words1 = new Set(text1.toLowerCase().split(/\s+/));
  const words2 = new Set(text2.toLowerCase().split(/\s+/));
  const intersection = new Set([...words1].filter((x) => words2.has(x)));
  const union = new Set([...words1, ...words2]);
  return intersection.size / union.size;
}
