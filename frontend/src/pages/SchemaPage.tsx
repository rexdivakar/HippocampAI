import { useState } from 'react';
import {
  Database,
  Check,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  CheckCircle,
  FileJson,
  Copy,
} from 'lucide-react';
import type { SchemaDefinition, ValidationResult } from '../types';

interface SchemaPageProps {
  userId: string;
}

// Default schema for display
const DEFAULT_SCHEMA: SchemaDefinition = {
  name: 'default',
  version: '1.0.0',
  description: 'Default HippocampAI schema',
  entity_types: [
    {
      name: 'person',
      description: 'A person entity',
      attributes: [
        { name: 'name', type: 'string', required: true, description: 'Person name' },
        { name: 'email', type: 'string', required: false, description: 'Email address' },
        { name: 'age', type: 'integer', required: false, min_value: 0, max_value: 150 },
      ],
    },
    {
      name: 'organization',
      description: 'A company or organization',
      attributes: [
        { name: 'name', type: 'string', required: true },
        { name: 'industry', type: 'string', required: false },
      ],
    },
    {
      name: 'location',
      description: 'A place or location',
      attributes: [
        { name: 'name', type: 'string', required: true },
        { name: 'type', type: 'string', required: false, enum_values: ['city', 'country', 'address'] },
      ],
    },
  ],
  relationship_types: [
    {
      name: 'works_at',
      description: 'Employment relationship',
      source_types: ['person'],
      target_types: ['organization'],
      attributes: [],
      bidirectional: false,
    },
    {
      name: 'located_in',
      description: 'Location relationship',
      source_types: ['person', 'organization'],
      target_types: ['location'],
      attributes: [],
      bidirectional: false,
    },
  ],
};

export function SchemaPage({ userId: _userId }: SchemaPageProps) {
  const [schema] = useState<SchemaDefinition>(DEFAULT_SCHEMA);
  const [expandedTypes, setExpandedTypes] = useState<Set<string>>(new Set());
  const [validationInput, setValidationInput] = useState('');
  const [selectedEntityType, setSelectedEntityType] = useState('person');
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [copied, setCopied] = useState(false);

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

      const errors: { field: string; message: string; value?: any }[] = [];
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
        if (attrDef.type === 'integer' && typeof value !== 'number') {
          errors.push({ field: key, message: `Expected integer, got ${typeof value}`, value });
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
    } catch (e) {
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
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <Database className="w-7 h-7 text-emerald-500" />
          Custom Schema
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Define and validate custom entity types and relationships
        </p>
      </div>

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
                            <div className="flex items-center gap-2">
                              <span className="font-mono text-sm text-gray-900 dark:text-white">
                                {attr.name}
                              </span>
                              {attr.required && (
                                <span className="text-xs text-red-500">*</span>
                              )}
                            </div>
                            <div className="flex items-center gap-2">
                              <span className={`text-xs px-2 py-0.5 rounded ${getTypeColor(attr.type)}`}>
                                {attr.type}
                              </span>
                              {attr.enum_values && (
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
              Validate Entity
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
                  placeholder='{"name": "John Doe", "email": "john@example.com"}'
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
                  setSelectedEntityType('person');
                  setValidationInput('{\n  "name": "John Doe",\n  "email": "john@example.com",\n  "age": 30\n}');
                }}
                className="w-full text-left px-3 py-2 text-sm bg-gray-50 dark:bg-gray-700 rounded hover:bg-gray-100 dark:hover:bg-gray-600"
              >
                Person (valid)
              </button>
              <button
                onClick={() => {
                  setSelectedEntityType('person');
                  setValidationInput('{\n  "email": "john@example.com"\n}');
                }}
                className="w-full text-left px-3 py-2 text-sm bg-gray-50 dark:bg-gray-700 rounded hover:bg-gray-100 dark:hover:bg-gray-600"
              >
                Person (missing required)
              </button>
              <button
                onClick={() => {
                  setSelectedEntityType('location');
                  setValidationInput('{\n  "name": "New York",\n  "type": "city"\n}');
                }}
                className="w-full text-left px-3 py-2 text-sm bg-gray-50 dark:bg-gray-700 rounded hover:bg-gray-100 dark:hover:bg-gray-600"
              >
                Location (valid)
              </button>
              <button
                onClick={() => {
                  setSelectedEntityType('location');
                  setValidationInput('{\n  "name": "NYC",\n  "type": "invalid_type"\n}');
                }}
                className="w-full text-left px-3 py-2 text-sm bg-gray-50 dark:bg-gray-700 rounded hover:bg-gray-100 dark:hover:bg-gray-600"
              >
                Location (invalid enum)
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
