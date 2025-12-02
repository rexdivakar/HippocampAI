import { useState } from 'react';
import { X, Share2, Copy, Download, CheckCircle } from 'lucide-react';
import type { Memory } from '../types';

interface ShareMemoryModalProps {
  memory: Memory | null;
  isOpen: boolean;
  onClose: () => void;
}

export function ShareMemoryModal({ memory, isOpen, onClose }: ShareMemoryModalProps) {
  const [copied, setCopied] = useState(false);
  const [exportFormat, setExportFormat] = useState<'json' | 'text' | 'markdown'>('json');

  if (!isOpen || !memory) return null;

  const handleCopyToClipboard = async () => {
    const shareableData = {
      id: memory.id,
      text: memory.text,
      type: memory.type,
      importance: memory.importance,
      tags: memory.tags,
      created_at: memory.created_at,
    };

    try {
      await navigator.clipboard.writeText(JSON.stringify(shareableData, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleExport = () => {
    let content = '';
    let filename = `memory-${memory.id.substring(0, 8)}`;
    let mimeType = '';

    switch (exportFormat) {
      case 'json':
        content = JSON.stringify(memory, null, 2);
        filename += '.json';
        mimeType = 'application/json';
        break;
      case 'text':
        content = `Memory: ${memory.text}\n\n`;
        content += `Type: ${memory.type}\n`;
        content += `Importance: ${memory.importance}/10\n`;
        content += `Confidence: ${(memory.confidence * 100).toFixed(0)}%\n`;
        content += `Tags: ${memory.tags?.join(', ') || 'None'}\n`;
        content += `Created: ${new Date(memory.created_at).toLocaleString()}\n`;
        content += `Access Count: ${memory.access_count}\n`;
        filename += '.txt';
        mimeType = 'text/plain';
        break;
      case 'markdown':
        content = `# Memory\n\n`;
        content += `**${memory.text}**\n\n`;
        content += `- **Type:** ${memory.type}\n`;
        content += `- **Importance:** ${memory.importance}/10\n`;
        content += `- **Confidence:** ${(memory.confidence * 100).toFixed(0)}%\n`;
        content += `- **Tags:** ${memory.tags?.join(', ') || 'None'}\n`;
        content += `- **Created:** ${new Date(memory.created_at).toLocaleString()}\n`;
        content += `- **Access Count:** ${memory.access_count}\n\n`;
        content += `**ID:** \`${memory.id}\`\n`;
        filename += '.md';
        mimeType = 'text/markdown';
        break;
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const shareableLink = `${window.location.origin}/memory/${memory.id}`;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-lg w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 bg-gradient-to-r from-green-500 to-emerald-500 text-white">
          <div className="flex items-center space-x-3">
            <Share2 className="w-6 h-6" />
            <div>
              <h2 className="text-xl font-bold">Share Memory</h2>
              <p className="text-sm text-green-100">Export or copy memory data</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 hover:bg-white/20 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Memory Preview */}
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm font-medium text-gray-900 line-clamp-3">
              {memory.text}
            </p>
            <div className="flex items-center space-x-4 mt-3 text-xs text-gray-500">
              <span className="px-2 py-1 bg-white rounded">{memory.type}</span>
              <span>⭐ {memory.importance}/10</span>
              <span>{memory.tags?.length || 0} tags</span>
            </div>
          </div>

          {/* Shareable Link */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Shareable Link
            </label>
            <div className="flex items-center space-x-2">
              <input
                type="text"
                value={shareableLink}
                readOnly
                className="input flex-1 text-sm font-mono bg-gray-50"
              />
              <button
                onClick={handleCopyToClipboard}
                className="btn-secondary flex items-center space-x-2 whitespace-nowrap"
              >
                {copied ? (
                  <>
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span>Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-4 h-4" />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
            <p className="mt-1 text-xs text-gray-500">
              Share this link to allow others to view this memory.
            </p>
          </div>

          {/* Export Options */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Export Format
            </label>
            <div className="flex gap-2">
              {['json', 'text', 'markdown'].map((format) => (
                <button
                  key={format}
                  onClick={() => setExportFormat(format as any)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all flex-1 ${
                    exportFormat === format
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {format.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Export Button */}
          <button
            onClick={handleExport}
            className="btn-primary w-full flex items-center justify-center space-x-2 bg-green-600 hover:bg-green-700"
          >
            <Download className="w-4 h-4" />
            <span>Download as {exportFormat.toUpperCase()}</span>
          </button>

          {/* Info */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <p className="text-xs text-blue-700">
              <strong>Note:</strong> Exported files contain all memory metadata including embeddings.
              The shareable link will work only if the recipient has access to this HippocampAI instance.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
