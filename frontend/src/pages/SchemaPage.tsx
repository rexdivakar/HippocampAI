import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Database,
  Check,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  CheckCircle,
  FileJson,
  Copy,
  RefreshCw,
} from 'lucide-react';
import { apiClient } from '../services/api';
import type { SchemaDefinition, ValidationResult, Memory } from '../types';

interface SchemaPageProps {
  userId: string;
}

// Build schema definition dynamically from live memory data
function buildSchemaFromMemories(memories: Memory[]): SchemaDefinition {
  const typeCounts: Record<string, number> = {};
  let withExpiry = 0;
  let withTags = 0;
  let withMetadata = 0;

  for (const m of memories) {
    typeCounts[m.type] = (typeCounts[m.type] || 0) + 1;
    if (m.expires_at) withExpiry++;
    if (m.tags && m.tags.length > 0) withTags++;
    if (m.metadata && Object.keys(m.metadata).length > 0) withMetadata++;
  }

  const total = memories.length || 1;
  const typeList = Object.keys(typeCounts).sort();

  return {
    name: 'memory',
    version: '1.0.0',
    description: `HippocampAI memory model — ${memories.length} memories loaded`,
    entity_types: [
      {
        name: 'memory',
        description: 'Core memory record stored in the vector database',
        attributes: [
          { name: 'id', type: 'string', required: true, description: 'Unique memory identifier (UUID)' },
          { name: 'text', type: 'string', required: true, description: 'Memory content' },
          { name: 'user_id', type: 'string', required: true, description: 'Owning user identifier' },
          {
            name: 'type',
            type: 'string',
            required: true,
            description: `Memory category (${typeList.length > 0 ? typeList.join(', ') : 'fact, preference, goal, habit, event, context, summary, procedural, prospective'})`,
            enum_values: typeList.length > 0 ? typeList : ['fact', 'preference', 'goal', 'habit', 'event', 'context', 'summary', 'procedural', 'prospective'],
          },
          { name: 'importance', type: 'float', required: true, description: 'Importance score (0–10)', min_value: 0, max_value: 10 },
          { name: 'confidence', type: 'float', required: true, description: 'Confidence score (0–1)', min_value: 0, max_value: 1 },
          {
            name: 'tags',
            type: 'list',
            required: false,
            description: `Searchable labels — ${withTags} of ${total} memories tagged (${Math.round((withTags / total) * 100)}%)`,
          },
          {
            name: 'expires_at',
            type: 'datetime',
            required: false,
            description: `Expiry timestamp — set on ${withExpiry} of ${total} memories (${Math.round((withExpiry / total) * 100)}%)`,
          },
          {
            name: 'metadata',
            type: 'dict',
            required: false,
            description: `Arbitrary key-value payload — present on ${withMetadata} of ${total} memories (${Math.round((withMetadata / total) * 100)}%)`,
          },
          { name: 'session_id', type: 'string', required: false, description: 'Session this memory belongs to' },
          { name: 'embedding', type: 'list', required: false, description: 'Vector embedding (omitted in API responses)' },
          { name: 'created_at', type: 'datetime', required: true, description: 'Creation timestamp' },
          { name: 'updated_at', type: 'datetime', required: true, description: 'Last update timestamp' },
          { name: 'last_accessed_at', type: 'datetime', required: false, description: 'Last retrieval timestamp' },
          { name: 'access_count', type: 'integer', required: true, description: 'Number of times recalled', min_value: 0 },
        ],
      },
      {
        name: 'memory_type_distribution',
        description: 'Live count of each memory type in the collection',
        attributes: Object.entries(typeCounts).sort(([, a], [, b]) => b - a).map(([type, count]) => ({
          name: type,
          type: 'integer' as const,
          required: false,
          description: `${count} memories (${Math.round((count / total) * 100)}%)`,
          min_value: 0,
        })),
      },
    ],
    relationship_types: [
      {
        name: 'belongs_to_session',
        description: 'Memory is part of a conversation session',
        source_types: ['memory'],
        target_types: ['session'],
        attributes: [],
        bidirectional: false,
      },
      {
        name: 'similar_to',
        description: 'Vector similarity relationship used for deduplication and recall',
        source_types: ['memory'],
        target_types: ['memory'],
        attributes: [],
        bidirectional: true,
      },
    ],
  };
}

export function SchemaPage({ userId }: SchemaPageProps) {
  const [expandedTypes, setExpandedTypes] = useState<Set<string>>(new Set(['memory']));
  const [validationInput, setValidationInput] = useState('');
  const [selectedEntityType, setSelectedEntityType] = useState('memory');
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [copied, setCopied] = useState(false);

  const {
    data: memories = [],
    isLoading,
    isError,
    refetch,
  } = useQuery({
    queryKey: ['schema-memories', userId],
    queryFn: () => apiClient.getMemories({ user_id: userId, limit: 10000 }),
  });

  const schema = useMemo(() => buildSchemaFromMemories(memories), [memories]);

  const toggleExpanded = (typeName: string) => {
    const newExpanded = new Set(expandedTypes);
    if (newExpanded.has(typeName)) {
      newExpanded.delete(typeName);
    } else {
      newExpanded.add(typeName);
    }
    setExpandedTypes(newExpanded);
  };

  const handleValidate = () => {
    try {
      const attributes = JSON.parse(validationInput);
      const entityType = schema.entity_types.find(t => t.name === selectedEntityType);

      if (!entityType) {
        setValidationResult({
          valid: false,
          errors: [{ field: 'entity_type', message: 'Unknown entity type' }],
          warnings: [],
        });
        return;
      }

      const errors: { field: string; message: string; value?: unknown }[] = [];
      const warnings: string[] = [];

      // Check required attributes
      for (const attr of entityType.attributes) {
        if (attr.required && !(attr.name in attributes)) {
          errors.push({
            field: attr.name,
            message: `Missing required attribute: ${attr.name}`,
          });
        }
      }

      // Check attribute types and constraints
      for (const [key, value] of Object.entries(attributes)) {
        const attrDef = entityType.attributes.find(a => a.name === key);

        if (!attrDef) {
          warnings.push(`Unknown attribute: ${key}`);
          continue;
        }

        // Type checking
        if (attrDef.type === 'string' && typeof value !== 'string') {
          errors.push({ field: key, message: `Expected string, got ${typeof value}`, value });
        }
        if ((attrDef.type === 'integer' || attrDef.type === 'float') && typeof value !== 'number') {
          errors.push({ field: key, message: `Expected number, got ${typeof value}`, value });
        }

        // Enum check
        if (attrDef.enum_values && !attrDef.enum_values.includes(value as string)) {
          errors.push({
            field: key,
            message: `Value must be one of: ${attrDef.enum_values.join(', ')}`,
            value,
          });
        }

        // Range check
        if (typeof value === 'number') {
          if (attrDef.min_value !== undefined && value < attrDef.min_value) {
            errors.push({ field: key, message: `Value must be at least ${attrDef.min_value}`, value });
          }
          if (attrDef.max_value !== undefined && value > attrDef.max_value) {
            errors.push({ field: key, message: `Value must be at most ${attrDef.max_value}`, value });
          }
        }
      }

      setValidationResult({
        valid: errors.length === 0,
        errors,
        warnings,
      });
    } catch {
      setValidationResult({
        valid: false,
        errors: [{ field: 'json', message: 'Invalid JSON format' }],
        warnings: [],
      });
    }
  };

  const handleCopySchema = async () => {
    await navigator.clipboard.writeText(JSON.stringify(schema, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      string: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
      integer: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
      float: 'bg-teal-100 text-teal-700 dark:bg-teal-900/30 dark:text-teal-400',
      boolean: 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400',
      datetime: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400',
      list: 'bg-pink-100 text-pink-700 dark:bg-pink-900/30 dark:text-pink-400',
      dict: 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400',
    };
    return colors[type] || 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300';
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <Database className="w-7 h-7 text-emerald-500" />
            Memory Schema
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Live memory model fields and collection statistics
          </p>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isLoading}
          className="flex items-center gap-2 px-3 py-1.5 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {isError && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 flex items-center gap-2 text-red-700 dark:text-red-400">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>Failed to load memory data. Schema reflects the model definition only.</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Schema Definition */}
        <div className="space-y-4">
          {/* Schema Info */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  {schema.name}
                </h2>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Version {schema.version}
                </p>
              </div>
              <button
                onClick={handleCopySchema}
                className="flex items-center gap-2 px-3 py-1.5 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
              >
                {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                {copied ? 'Copied!' : 'Copy JSON'}
              </button>
            </div>
            {schema.description && (
              <p className="text-gray-600 dark:text-gray-400 text-sm">{schema.description}</p>
            )}
          </div>

          {/* Entity Types */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Entity Types ({schema.entity_types.length})
              </h3>
            </div>
            <div className="divide-y divide-gray-200 dark:divide-gray-700">
              {schema.entity_types.map((entityType) => (
                <div key={entityType.name}>
                  <button
                    onClick={() => toggleExpanded(entityType.name)}
                    className="w-full p-4 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700/50"
                  >
                    <div className="flex items-center gap-3">
                      <span className="font-medium text-gray-900 dark:text-white">
                        {entityType.name}
                      </span>
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {entityType.attributes.length} attributes
                      </span>
                    </div>
                    {expandedTypes.has(entityType.name) ? (
                      <ChevronUp className="w-5 h-5 text-gray-400" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-gray-400" />
                    )}
                  </button>
                  {expandedTypes.has(entityType.name) && (
                    <div className="px-4 pb-4 bg-gray-50 dark:bg-gray-900/50">
                      {entityType.description && (
                        <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                          {entityType.description}
                        </p>
                      )}
                      <div className="space-y-2">
                        {entityType.attributes.map((attr) => (
                          <div
                            key={attr.name}
                            className="flex items-center justify-between p-2 bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700"
                          >
                            <div className="flex items-start gap-2 flex-1 min-w-0">
                              <span className="font-mono text-sm text-gray-900 dark:text-white whitespace-nowrap">
                                {attr.name}
                              </span>
                              {attr.required && (
                                <span className="text-xs text-red-500">*</span>
                              )}
                              {attr.description && (
                                <span className="text-xs text-gray-500 dark:text-gray-400 truncate">
                                  {attr.description}
                                </span>
                              )}
                            </div>
                            <div className="flex items-center gap-2 flex-shrink-0 ml-2">
                              <span className={`text-xs px-2 py-0.5 rounded ${getTypeColor(attr.type)}`}>
                                {attr.type}
                              </span>
                              {attr.enum_values && attr.enum_values.length <= 5 && (
                                <span className="text-xs text-gray-500 dark:text-gray-400">
                                  [{attr.enum_values.join(', ')}]
                                </span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Relationship Types */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Relationship Types ({schema.relationship_types.length})
              </h3>
            </div>
            <div className="divide-y divide-gray-200 dark:divide-gray-700">
              {schema.relationship_types.map((relType) => (
                <div key={relType.name} className="p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="font-medium text-gray-900 dark:text-white">
                      {relType.name}
                    </span>
                    {relType.bidirectional && (
                      <span className="text-xs px-2 py-0.5 bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400 rounded">
                        bidirectional
                      </span>
                    )}
                  </div>
                  {relType.description && (
                    <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">{relType.description}</p>
                  )}
                  <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                    <span className="px-2 py-0.5 bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400 rounded">
                      {relType.source_types.join(', ')}
                    </span>
                    <span>→</span>
                    <span className="px-2 py-0.5 bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 rounded">
                      {relType.target_types.join(', ')}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Validation Panel */}
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
              <FileJson className="w-5 h-5" />
              Validate Memory JSON
            </h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Entity Type
                </label>
                <select
                  value={selectedEntityType}
                  onChange={(e) => setSelectedEntityType(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  {schema.entity_types.map((type) => (
                    <option key={type.name} value={type.name}>
                      {type.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Attributes (JSON)
                </label>
                <textarea
                  value={validationInput}
                  onChange={(e) => setValidationInput(e.target.value)}
                  placeholder='{"text": "User prefers dark mode", "type": "preference", "importance": 7.5, "confidence": 0.9}'
                  rows={6}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white font-mono text-sm"
                />
              </div>

              <button
                onClick={handleValidate}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors"
              >
                <Check className="w-4 h-4" />
                Validate
              </button>
            </div>

            {/* Validation Result */}
            {validationResult && (
              <div className={`mt-4 p-4 rounded-lg ${
                validationResult.valid
                  ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800'
                  : 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800'
              }`}>
                <div className="flex items-center gap-2 mb-2">
                  {validationResult.valid ? (
                    <>
                      <CheckCircle className="w-5 h-5 text-green-500" />
                      <span className="font-medium text-green-700 dark:text-green-400">
                        Valid
                      </span>
                    </>
                  ) : (
                    <>
                      <AlertCircle className="w-5 h-5 text-red-500" />
                      <span className="font-medium text-red-700 dark:text-red-400">
                        Invalid
                      </span>
                    </>
                  )}
                </div>

                {validationResult.errors.length > 0 && (
                  <div className="space-y-1">
                    {validationResult.errors.map((error, i) => (
                      <p key={i} className="text-sm text-red-600 dark:text-red-400">
                        • {error.field}: {error.message}
                      </p>
                    ))}
                  </div>
                )}

                {validationResult.warnings.length > 0 && (
                  <div className="mt-2 space-y-1">
                    {validationResult.warnings.map((warning, i) => (
                      <p key={i} className="text-sm text-yellow-600 dark:text-yellow-400">
                        ⚠ {warning}
                      </p>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Example Templates */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Example Templates
            </h3>
            <div className="space-y-2">
              <button
                onClick={() => {
                  setSelectedEntityType('memory');
                  setValidationInput('{\n  "text": "User prefers dark mode",\n  "type": "preference",\n  "importance": 7.5,\n  "confidence": 0.9\n}');
                }}
                className="w-full text-left px-3 py-2 text-sm bg-gray-50 dark:bg-gray-700 rounded hover:bg-gray-100 dark:hover:bg-gray-600"
              >
                Preference (valid)
              </button>
              <button
                onClick={() => {
                  setSelectedEntityType('memory');
                  setValidationInput('{\n  "type": "fact",\n  "importance": 8\n}');
                }}
                className="w-full text-left px-3 py-2 text-sm bg-gray-50 dark:bg-gray-700 rounded hover:bg-gray-100 dark:hover:bg-gray-600"
              >
                Fact (missing required text)
              </button>
              <button
                onClick={() => {
                  setSelectedEntityType('memory');
                  setValidationInput('{\n  "text": "User joined the team on Monday",\n  "type": "event",\n  "importance": 6.0,\n  "confidence": 0.95,\n  "tags": ["onboarding", "team"]\n}');
                }}
                className="w-full text-left px-3 py-2 text-sm bg-gray-50 dark:bg-gray-700 rounded hover:bg-gray-100 dark:hover:bg-gray-600"
              >
                Event (valid with tags)
              </button>
              <button
                onClick={() => {
                  setSelectedEntityType('memory');
                  setValidationInput('{\n  "text": "User likes Python",\n  "type": "unknown_type",\n  "importance": 5,\n  "confidence": 0.8\n}');
                }}
                className="w-full text-left px-3 py-2 text-sm bg-gray-50 dark:bg-gray-700 rounded hover:bg-gray-100 dark:hover:bg-gray-600"
              >
                Unknown type (invalid enum)
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
