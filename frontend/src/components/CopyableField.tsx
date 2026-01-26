import { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import clsx from 'clsx';

interface CopyableFieldProps {
  value: string;
  label?: string;
  icon?: React.ReactNode;
  className?: string;
  mono?: boolean;
  inline?: boolean;
}

export function CopyableField({
  value,
  label,
  icon,
  className,
  mono = false,
  inline = false,
}: CopyableFieldProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  if (inline) {
    // Inline variant for navbar
    return (
      <div className={clsx('flex items-center space-x-2 flex-shrink-0', className)}>
        <span className={clsx('text-sm whitespace-nowrap overflow-hidden text-ellipsis max-w-xs', mono && 'font-mono')} title={value}>
          {value}
        </span>
        <button
          onClick={handleCopy}
          className="flex-shrink-0 p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded transition-colors relative group"
          title={copied ? 'Copied!' : 'Copy to clipboard'}
        >
          {copied ? (
            <Check className="w-4 h-4 text-green-600" />
          ) : (
            <Copy className="w-4 h-4" />
          )}
          {copied && (
            <span className="absolute -top-8 left-1/2 transform -translate-x-1/2 px-2 py-1 bg-gray-900 text-white text-xs rounded whitespace-nowrap z-10">
              Copied!
            </span>
          )}
        </button>
      </div>
    );
  }

  // Card variant for metadata
  return (
    <div className={clsx('bg-white border border-gray-200 rounded-lg p-4', className)}>
      {label && (
        <div className="flex items-center space-x-2 text-gray-600 mb-1">
          {icon}
          <span className="text-xs font-medium uppercase">{label}</span>
        </div>
      )}
      <div className="flex items-start justify-between space-x-2">
        <p
          className={clsx(
            'text-sm text-gray-900 break-all',
            mono && 'font-mono'
          )}
        >
          {value}
        </p>
        <button
          onClick={handleCopy}
          className="flex-shrink-0 p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded transition-colors relative group"
          title={copied ? 'Copied!' : 'Copy to clipboard'}
        >
          {copied ? (
            <Check className="w-4 h-4 text-green-600" />
          ) : (
            <Copy className="w-4 h-4" />
          )}
          {copied && (
            <span className="absolute -top-8 right-0 px-2 py-1 bg-gray-900 text-white text-xs rounded whitespace-nowrap z-10">
              Copied!
            </span>
          )}
        </button>
      </div>
    </div>
  );
}
