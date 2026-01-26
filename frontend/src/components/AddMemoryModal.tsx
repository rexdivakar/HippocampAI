import { useState, FormEvent } from 'react';
import { X, Brain, Plus } from 'lucide-react';

interface AddMemoryModalProps {
  userId: string;
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: {
    text: string;
    type: string;
    importance: number;
    tags: string[];
    sessionId?: string;
  }) => Promise<void>;
}

const memoryTypes = ['fact', 'preference', 'goal', 'habit', 'event', 'context'];

export function AddMemoryModal({ userId, isOpen, onClose, onSubmit }: AddMemoryModalProps) {
  const [text, setText] = useState('');
  const [type, setType] = useState('fact');
  const [importance, setImportance] = useState(5);
  const [tags, setTags] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  if (!isOpen) return null;

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');

    if (!text.trim()) {
      setError('Memory text is required');
      return;
    }

    setLoading(true);

    try {
      await onSubmit({
        text: text.trim(),
        type,
        importance,
        tags: tags.split(',').map(t => t.trim()).filter(Boolean),
        sessionId: sessionId.trim() || undefined,
      });

      // Reset form
      setText('');
      setType('fact');
      setImportance(5);
      setTags('');
      setSessionId('');
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to create memory');
      console.error('Create memory error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    if (!loading) {
      setText('');
      setType('fact');
      setImportance(5);
      setTags('');
      setSessionId('');
      setError('');
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-start justify-center p-4 pt-20 overflow-y-auto">
      <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full my-8 overflow-hidden border border-gray-100">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-gray-100">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-lg shadow-primary-500/30">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Create New Memory</h2>
              <p className="text-sm text-gray-500">Add a new memory to your store</p>
            </div>
          </div>

          <button
            onClick={handleClose}
            disabled={loading}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50 text-gray-400 hover:text-gray-600"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6">
          <div className="space-y-6">
            {/* Memory Text */}
            <div>
              <label htmlFor="text" className="block text-sm font-medium text-gray-700 mb-2">
                Memory Text <span className="text-red-500">*</span>
              </label>
              <textarea
                id="text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={4}
                className="input resize-none"
                placeholder="Enter the memory content..."
                disabled={loading}
                required
              />
              <p className="mt-1 text-xs text-gray-500">
                The main content of your memory. Be specific and clear.
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
                    onClick={() => setType(t)}
                    disabled={loading}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      type === t
                        ? 'bg-primary-600 text-white'
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
              <label htmlFor="importance" className="block text-sm font-medium text-gray-700 mb-2">
                Importance: {importance}/10
              </label>
              <input
                type="range"
                id="importance"
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
              <label htmlFor="tags" className="block text-sm font-medium text-gray-700 mb-2">
                Tags
              </label>
              <input
                type="text"
                id="tags"
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

            {/* Session ID (Optional) */}
            <div>
              <label htmlFor="sessionId" className="block text-sm font-medium text-gray-700 mb-2">
                Session ID <span className="text-gray-400 text-xs">(Optional)</span>
              </label>
              <input
                type="text"
                id="sessionId"
                value={sessionId}
                onChange={(e) => setSessionId(e.target.value)}
                className="input"
                placeholder="Leave blank to auto-generate"
                disabled={loading}
              />
              <p className="mt-1 text-xs text-gray-500">
                Link this memory to a specific session.
              </p>
            </div>

            {/* User ID Display */}
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm text-gray-600">
                <span className="font-medium">User ID:</span> {userId}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                This memory will be stored in Qdrant with vector embeddings and all metadata.
              </p>
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
        <div className="flex items-center justify-end space-x-3 px-6 py-4 border-t border-gray-100 bg-gray-50/50">
          <button
            type="button"
            onClick={handleClose}
            disabled={loading}
            className="px-4 py-2.5 text-sm font-medium text-gray-700 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={loading || !text.trim()}
            className="px-4 py-2.5 text-sm font-medium bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2 shadow-sm"
          >
            <Plus className="w-4 h-4" />
            <span>{loading ? 'Creating...' : 'Create Memory'}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
