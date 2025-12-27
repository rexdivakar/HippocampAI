import { useState, useEffect } from 'react';
import { apiClient } from '../services/api';
import type { BiTemporalFact, BiTemporalQueryResult } from '../types';
import {
  Clock,
  History,
  Plus,
  Search,
  Edit,
  Trash2,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  CheckCircle,
  XCircle,
  RefreshCw,
} from 'lucide-react';

interface BiTemporalPageProps {
  userId: string;
}

export function BiTemporalPage({ userId }: BiTemporalPageProps) {
  const [facts, setFacts] = useState<BiTemporalFact[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  const [selectedFact, setSelectedFact] = useState<BiTemporalFact | null>(null);
  const [factHistory, setFactHistory] = useState<BiTemporalFact[]>([]);
  const [expandedFacts, setExpandedFacts] = useState<Set<string>>(new Set());

  // Query filters
  const [entityFilter, setEntityFilter] = useState('');
  const [propertyFilter, setPropertyFilter] = useState('');
  const [includeSuperseded, setIncludeSuperseded] = useState(false);

  // Add fact form
  const [newFactText, setNewFactText] = useState('');
  const [newFactEntity, setNewFactEntity] = useState('');
  const [newFactProperty, setNewFactProperty] = useState('');
  const [newFactValidFrom, setNewFactValidFrom] = useState('');
  const [newFactValidTo, setNewFactValidTo] = useState('');

  useEffect(() => {
    loadFacts();
  }, [userId, includeSuperseded]);

  const loadFacts = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiClient.queryBiTemporalFacts({
        user_id: userId,
        entity_id: entityFilter || undefined,
        property_name: propertyFilter || undefined,
        include_superseded: includeSuperseded,
      });
      setFacts(result.facts);
    } catch (err: any) {
      setError(err.message || 'Failed to load facts');
    } finally {
      setLoading(false);
    }
  };

  const handleAddFact = async () => {
    if (!newFactText.trim()) return;

    try {
      await apiClient.storeBiTemporalFact({
        text: newFactText,
        user_id: userId,
        entity_id: newFactEntity || undefined,
        property_name: newFactProperty || undefined,
        valid_from: newFactValidFrom || undefined,
        valid_to: newFactValidTo || undefined,
      });
      setShowAddModal(false);
      resetAddForm();
      loadFacts();
    } catch (err: any) {
      setError(err.message || 'Failed to add fact');
    }
  };

  const handleReviseFact = async (fact: BiTemporalFact, newText: string) => {
    try {
      await apiClient.reviseBiTemporalFact({
        original_fact_id: fact.id,
        new_text: newText,
        user_id: userId,
      });
      loadFacts();
    } catch (err: any) {
      setError(err.message || 'Failed to revise fact');
    }
  };

  const handleRetractFact = async (fact: BiTemporalFact) => {
    if (!confirm('Are you sure you want to retract this fact?')) return;

    try {
      await apiClient.retractBiTemporalFact({
        fact_id: fact.id,
        user_id: userId,
      });
      loadFacts();
    } catch (err: any) {
      setError(err.message || 'Failed to retract fact');
    }
  };

  const handleViewHistory = async (fact: BiTemporalFact) => {
    try {
      const history = await apiClient.getBiTemporalFactHistory(fact.fact_id);
      setFactHistory(history);
      setSelectedFact(fact);
      setShowHistoryModal(true);
    } catch (err: any) {
      setError(err.message || 'Failed to load history');
    }
  };

  const resetAddForm = () => {
    setNewFactText('');
    setNewFactEntity('');
    setNewFactProperty('');
    setNewFactValidFrom('');
    setNewFactValidTo('');
  };

  const toggleExpanded = (factId: string) => {
    const newExpanded = new Set(expandedFacts);
    if (newExpanded.has(factId)) {
      newExpanded.delete(factId);
    } else {
      newExpanded.add(factId);
    }
    setExpandedFacts(newExpanded);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'superseded':
        return <AlertCircle className="w-4 h-4 text-yellow-500" />;
      case 'retracted':
        return <XCircle className="w-4 h-4 text-red-500" />;
      default:
        return null;
    }
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString();
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <Clock className="w-7 h-7 text-indigo-500" />
            Bi-Temporal Facts
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Track facts with validity periods and revision history
          </p>
        </div>
        <button
          onClick={() => setShowAddModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
        >
          <Plus className="w-4 h-4" />
          Add Fact
        </button>
      </div>

      {/* Filters */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm">
        <div className="flex flex-wrap gap-4 items-end">
          <div className="flex-1 min-w-[200px]">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Entity ID
            </label>
            <input
              type="text"
              value={entityFilter}
              onChange={(e) => setEntityFilter(e.target.value)}
              placeholder="Filter by entity..."
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>
          <div className="flex-1 min-w-[200px]">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Property Name
            </label>
            <input
              type="text"
              value={propertyFilter}
              onChange={(e) => setPropertyFilter(e.target.value)}
              placeholder="Filter by property..."
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="includeSuperseded"
              checked={includeSuperseded}
              onChange={(e) => setIncludeSuperseded(e.target.checked)}
              className="w-4 h-4 text-indigo-600 rounded"
            />
            <label htmlFor="includeSuperseded" className="text-sm text-gray-700 dark:text-gray-300">
              Include superseded
            </label>
          </div>
          <button
            onClick={loadFacts}
            className="flex items-center gap-2 px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
          >
            <Search className="w-4 h-4" />
            Search
          </button>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 text-red-700 dark:text-red-400">
          {error}
        </div>
      )}

      {/* Facts List */}
      <div className="space-y-4">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-8 h-8 text-indigo-500 animate-spin" />
          </div>
        ) : facts.length === 0 ? (
          <div className="text-center py-12 text-gray-500 dark:text-gray-400">
            No facts found. Add your first bi-temporal fact!
          </div>
        ) : (
          facts.map((fact) => (
            <div
              key={fact.id}
              className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700"
            >
              <div
                className="p-4 cursor-pointer"
                onClick={() => toggleExpanded(fact.id)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      {getStatusIcon(fact.status)}
                      <span className={`text-xs px-2 py-0.5 rounded-full ${
                        fact.status === 'active' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                        fact.status === 'superseded' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' :
                        'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                      }`}>
                        {fact.status}
                      </span>
                      {fact.entity_id && (
                        <span className="text-xs px-2 py-0.5 bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400 rounded-full">
                          {fact.entity_id}
                        </span>
                      )}
                      {fact.property_name && (
                        <span className="text-xs px-2 py-0.5 bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400 rounded-full">
                          {fact.property_name}
                        </span>
                      )}
                    </div>
                    <p className="text-gray-900 dark:text-white font-medium">{fact.text}</p>
                    <div className="flex items-center gap-4 mt-2 text-xs text-gray-500 dark:text-gray-400">
                      <span>Valid: {formatDate(fact.valid_from)}</span>
                      {fact.valid_to && <span>→ {formatDate(fact.valid_to)}</span>}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {expandedFacts.has(fact.id) ? (
                      <ChevronUp className="w-5 h-5 text-gray-400" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-gray-400" />
                    )}
                  </div>
                </div>
              </div>

              {expandedFacts.has(fact.id) && (
                <div className="border-t border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-900/50">
                  <div className="grid grid-cols-2 gap-4 text-sm mb-4">
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">Fact ID:</span>
                      <span className="ml-2 text-gray-900 dark:text-white font-mono text-xs">{fact.fact_id}</span>
                    </div>
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">Confidence:</span>
                      <span className="ml-2 text-gray-900 dark:text-white">{(fact.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">System Time:</span>
                      <span className="ml-2 text-gray-900 dark:text-white">{formatDate(fact.system_time)}</span>
                    </div>
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">Source:</span>
                      <span className="ml-2 text-gray-900 dark:text-white">{fact.source}</span>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleViewHistory(fact);
                      }}
                      className="flex items-center gap-1 px-3 py-1.5 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
                    >
                      <History className="w-4 h-4" />
                      History
                    </button>
                    {fact.status === 'active' && (
                      <>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            const newText = prompt('Enter revised text:', fact.text);
                            if (newText && newText !== fact.text) {
                              handleReviseFact(fact, newText);
                            }
                          }}
                          className="flex items-center gap-1 px-3 py-1.5 text-sm bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded hover:bg-blue-200 dark:hover:bg-blue-900/50"
                        >
                          <Edit className="w-4 h-4" />
                          Revise
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRetractFact(fact);
                          }}
                          className="flex items-center gap-1 px-3 py-1.5 text-sm bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 rounded hover:bg-red-200 dark:hover:bg-red-900/50"
                        >
                          <Trash2 className="w-4 h-4" />
                          Retract
                        </button>
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* Add Fact Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-md">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">Add Bi-Temporal Fact</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Fact Text *
                </label>
                <textarea
                  value={newFactText}
                  onChange={(e) => setNewFactText(e.target.value)}
                  placeholder="Enter the fact..."
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Entity ID
                  </label>
                  <input
                    type="text"
                    value={newFactEntity}
                    onChange={(e) => setNewFactEntity(e.target.value)}
                    placeholder="e.g., alice"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Property Name
                  </label>
                  <input
                    type="text"
                    value={newFactProperty}
                    onChange={(e) => setNewFactProperty(e.target.value)}
                    placeholder="e.g., employer"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Valid From
                  </label>
                  <input
                    type="datetime-local"
                    value={newFactValidFrom}
                    onChange={(e) => setNewFactValidFrom(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Valid To
                  </label>
                  <input
                    type="datetime-local"
                    value={newFactValidTo}
                    onChange={(e) => setNewFactValidTo(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>
              </div>
            </div>
            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => {
                  setShowAddModal(false);
                  resetAddForm();
                }}
                className="px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
              >
                Cancel
              </button>
              <button
                onClick={handleAddFact}
                disabled={!newFactText.trim()}
                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Add Fact
              </button>
            </div>
          </div>
        </div>
      )}

      {/* History Modal */}
      {showHistoryModal && selectedFact && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-2xl max-h-[80vh] overflow-y-auto">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
              <History className="w-5 h-5" />
              Fact History
            </h2>
            <div className="space-y-4">
              {factHistory.map((version, index) => (
                <div
                  key={version.id}
                  className={`p-4 rounded-lg border ${
                    index === 0
                      ? 'border-indigo-300 dark:border-indigo-700 bg-indigo-50 dark:bg-indigo-900/20'
                      : 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-2">
                    {getStatusIcon(version.status)}
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      Version {factHistory.length - index}
                    </span>
                    {index === 0 && (
                      <span className="text-xs px-2 py-0.5 bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400 rounded-full">
                        Current
                      </span>
                    )}
                  </div>
                  <p className="text-gray-900 dark:text-white">{version.text}</p>
                  <div className="flex gap-4 mt-2 text-xs text-gray-500 dark:text-gray-400">
                    <span>System: {formatDate(version.system_time)}</span>
                    <span>Valid: {formatDate(version.valid_from)}</span>
                  </div>
                </div>
              ))}
            </div>
            <div className="flex justify-end mt-6">
              <button
                onClick={() => {
                  setShowHistoryModal(false);
                  setSelectedFact(null);
                  setFactHistory([]);
                }}
                className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
