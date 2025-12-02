import { useState, useEffect, FormEvent } from 'react';
import { X, Edit, Save } from 'lucide-react';
import type { Memory } from '../types';

interface EditMemoryModalProps {
  memory: Memory | null;
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (memoryId: string, data: {
    text?: string;
    type?: 'fact' | 'preference' | 'goal' | 'habit' | 'event' | 'context';
    importance?: number;
    tags?: string[];
    metadata?: Record<string, any>;
  }) => Promise<void>;
}

const memoryTypes = ['fact', 'preference', 'goal', 'habit', 'event', 'context'];

export function EditMemoryModal({ memory, isOpen, onClose, onSubmit }: EditMemoryModalProps) {
  const [text, setText] = useState('');
  const [type, setType] = useState<'fact' | 'preference' | 'goal' | 'habit' | 'event' | 'context'>('fact');
  const [importance, setImportance] = useState(5);
  const [tags, setTags] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Initialize form with memory data when modal opens
  useEffect(() => {
    if (memory && isOpen) {
      setText(memory.text);
      setType(memory.type);
      setImportance(memory.importance);
      setTags(memory.tags?.join(', ') || '');
      setError('');
    }
  }, [memory, isOpen]);

  if (!isOpen || !memory) return null;

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');

    if (!text.trim()) {
      setError('Memory text is required');
      return;
    }

    setLoading(true);

    try {
      await onSubmit(memory.id, {
        text: text.trim(),
        type,
        importance,
        tags: tags.split(',').map(t => t.trim()).filter(Boolean),
      });

      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to update memory');
      console.error('Update memory error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    if (!loading) {
      setError('');
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 bg-gradient-to-r from-amber-500 to-orange-500 text-white">
          <div className="flex items-center space-x-3">
            <Edit className="w-6 h-6" />
            <div>
              <h2 className="text-xl font-bold">Edit Memory</h2>
              <p className="text-sm text-amber-100">Update memory details and metadata</p>
            </div>
          </div>

          <button
            onClick={handleClose}
            disabled={loading}
            className="p-2 hover:bg-white/20 rounded-lg transition-colors disabled:opacity-50"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 overflow-y-auto max-h-[calc(90vh-180px)]">
          <div className="space-y-6">
            {/* Memory ID Display */}
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500 font-mono">
                ID: {memory.id}
              </p>
            </div>

            {/* Memory Text */}
            <div>
              <label htmlFor="edit-text" className="block text-sm font-medium text-gray-700 mb-2">
                Memory Text <span className="text-red-500">*</span>
              </label>
              <textarea
                id="edit-text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={5}
                className="input resize-none"
                placeholder="Enter the memory content..."
                disabled={loading}
                required
              />
              <p className="mt-1 text-xs text-gray-500">
                Update the main content of your memory.
              </p>
            </div>

            {/* Memory Type */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Memory Type
              </label>
              <div className="flex flex-wrap gap-2">
                {memoryTypes.map((t) => (
                  <button
                    key={t}
                    type="button"
                    onClick={() => setType(t as 'fact' | 'preference' | 'goal' | 'habit' | 'event' | 'context')}
                    disabled={loading}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      type === t
                        ? 'bg-amber-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>

            {/* Importance */}
            <div>
              <label htmlFor="edit-importance" className="block text-sm font-medium text-gray-700 mb-2">
                Importance: {importance}/10
              </label>
              <input
                type="range"
                id="edit-importance"
                min="0"
                max="10"
                step="0.5"
                value={importance}
                onChange={(e) => setImportance(parseFloat(e.target.value))}
                disabled={loading}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Low</span>
                <span>Medium</span>
                <span>High</span>
              </div>
            </div>

            {/* Tags */}
            <div>
              <label htmlFor="edit-tags" className="block text-sm font-medium text-gray-700 mb-2">
                Tags
              </label>
              <input
                type="text"
                id="edit-tags"
                value={tags}
                onChange={(e) => setTags(e.target.value)}
                className="input"
                placeholder="work, personal, project (comma-separated)"
                disabled={loading}
              />
              <p className="mt-1 text-xs text-gray-500">
                Comma-separated tags to help organize and find this memory.
              </p>
            </div>

            {/* Memory Stats */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-gray-50 rounded-lg p-3">
                <p className="text-xs text-gray-500">Access Count</p>
                <p className="text-lg font-semibold text-gray-900">{memory.access_count}</p>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <p className="text-xs text-gray-500">Confidence</p>
                <p className="text-lg font-semibold text-gray-900">{(memory.confidence * 100).toFixed(0)}%</p>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <p className="text-xs text-gray-500">Created</p>
                <p className="text-sm font-medium text-gray-900">
                  {new Date(memory.created_at).toLocaleDateString()}
                </p>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                <p className="text-sm text-red-600">{error}</p>
              </div>
            )}
          </div>
        </form>

        {/* Footer Actions */}
        <div className="flex items-center justify-end space-x-3 p-6 border-t border-gray-200 bg-gray-50">
          <button
            type="button"
            onClick={handleClose}
            disabled={loading}
            className="btn-secondary disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={loading || !text.trim()}
            className="btn-primary flex items-center space-x-2 disabled:opacity-50 bg-amber-600 hover:bg-amber-700"
          >
            <Save className="w-4 h-4" />
            <span>{loading ? 'Saving...' : 'Save Changes'}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
