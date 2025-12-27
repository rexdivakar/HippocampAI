import { useState } from 'react';
import {
  Brain,
  Sparkles,
  Zap,
  AlertCircle,
  CheckCircle,
  HelpCircle,
  RefreshCw,
  Copy,
  Check,
} from 'lucide-react';

interface ClassificationResult {
  memory_type: string;
  confidence: number;
  confidence_level: 'HIGH' | 'MEDIUM' | 'LOW' | 'UNCERTAIN';
  reasoning: string;
  alternative_type?: string;
  alternative_confidence?: number;
}

interface AgenticClassifierPageProps {
  userId: string;
}

// Memory type definitions for display
const MEMORY_TYPES = {
  fact: {
    label: 'Fact',
    description: 'Personal info, identity, biographical data',
    examples: ['My name is Alex', 'I work at Google', 'I live in SF'],
    color: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
  },
  preference: {
    label: 'Preference',
    description: 'Likes, dislikes, opinions, favorites',
    examples: ['I love pizza', 'I prefer dark mode', 'Python is my favorite'],
    color: 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400',
  },
  goal: {
    label: 'Goal',
    description: 'Intentions, aspirations, plans',
    examples: ['I want to learn Python', 'My goal is to run a marathon'],
    color: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
  },
  habit: {
    label: 'Habit',
    description: 'Routines, regular activities, patterns',
    examples: ['I usually wake up at 7am', 'I exercise every day'],
    color: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400',
  },
  event: {
    label: 'Event',
    description: 'Specific occurrences, meetings, time-bound activities',
    examples: ['I met John yesterday', 'The meeting is at 3pm'],
    color: 'bg-pink-100 text-pink-700 dark:bg-pink-900/30 dark:text-pink-400',
  },
  context: {
    label: 'Context',
    description: 'General conversation, observations, acknowledgments',
    examples: ['The weather is nice', "That's interesting", 'I understand'],
    color: 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300',
  },
};

const CONFIDENCE_COLORS = {
  HIGH: 'text-green-600 dark:text-green-400',
  MEDIUM: 'text-yellow-600 dark:text-yellow-400',
  LOW: 'text-orange-600 dark:text-orange-400',
  UNCERTAIN: 'text-red-600 dark:text-red-400',
};

export function AgenticClassifierPage({ userId: _userId }: AgenticClassifierPageProps) {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<Array<{ text: string; result: ClassificationResult }>>([]);
  const [copied, setCopied] = useState(false);

  // Simulated classification (in production, this would call the API)
  const handleClassify = async () => {
    if (!inputText.trim()) return;

    setLoading(true);
    setError(null);

    try {
      // Simulate API call with pattern-based classification
      await new Promise(resolve => setTimeout(resolve, 500));

      const text = inputText.toLowerCase();
      let memoryType = 'context';
      let confidence = 0.7;
      let reasoning = 'Pattern-based classification';

      // Simple pattern matching for demo
      if (text.includes('my name is') || text.includes('i am') || text.includes('i work at') || text.includes('i live in')) {
        memoryType = 'fact';
        confidence = 0.92;
        reasoning = 'Contains identity or biographical information patterns';
      } else if (text.includes('i love') || text.includes('i prefer') || text.includes('i like') || text.includes('favorite')) {
        memoryType = 'preference';
        confidence = 0.89;
        reasoning = 'Contains preference or opinion indicators';
      } else if (text.includes('i want to') || text.includes('my goal') || text.includes('i plan to') || text.includes('i hope to')) {
        memoryType = 'goal';
        confidence = 0.88;
        reasoning = 'Contains future intention or aspiration patterns';
      } else if (text.includes('usually') || text.includes('every day') || text.includes('always') || text.includes('routine')) {
        memoryType = 'habit';
        confidence = 0.85;
        reasoning = 'Contains routine or regular activity patterns';
      } else if (text.includes('yesterday') || text.includes('tomorrow') || text.includes('meeting') || text.includes('last week')) {
        memoryType = 'event';
        confidence = 0.86;
        reasoning = 'Contains temporal markers indicating specific occurrence';
      }

      const confidenceLevel = confidence >= 0.9 ? 'HIGH' : confidence >= 0.7 ? 'MEDIUM' : confidence >= 0.5 ? 'LOW' : 'UNCERTAIN';

      const classificationResult: ClassificationResult = {
        memory_type: memoryType,
        confidence,
        confidence_level: confidenceLevel,
        reasoning,
        alternative_type: memoryType === 'preference' ? 'fact' : undefined,
        alternative_confidence: memoryType === 'preference' ? 0.45 : undefined,
      };

      setResult(classificationResult);
      setHistory(prev => [{ text: inputText, result: classificationResult }, ...prev.slice(0, 9)]);
    } catch (err: any) {
      setError(err.message || 'Classification failed');
    } finally {
      setLoading(false);
    }
  };

  const handleCopyResult = async () => {
    if (!result) return;
    await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getConfidenceIcon = (level: string) => {
    switch (level) {
      case 'HIGH':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'MEDIUM':
        return <AlertCircle className="w-5 h-5 text-yellow-500" />;
      case 'LOW':
        return <HelpCircle className="w-5 h-5 text-orange-500" />;
      default:
        return <HelpCircle className="w-5 h-5 text-red-500" />;
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <Brain className="w-7 h-7 text-violet-500" />
          Agentic Memory Classifier
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          LLM-powered memory classification with confidence scoring and reasoning
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Classification Input */}
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-violet-500" />
              Classify Memory
            </h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Memory Text
                </label>
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Enter a memory to classify... e.g., 'I love coffee' or 'My name is Alex'"
                  rows={4}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white resize-none"
                />
              </div>

              <button
                onClick={handleClassify}
                disabled={!inputText.trim() || loading}
                className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-violet-600 text-white rounded-lg hover:bg-violet-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? (
                  <RefreshCw className="w-5 h-5 animate-spin" />
                ) : (
                  <Zap className="w-5 h-5" />
                )}
                {loading ? 'Classifying...' : 'Classify'}
              </button>
            </div>

            {/* Error */}
            {error && (
              <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-400">
                {error}
              </div>
            )}

            {/* Result */}
            {result && (
              <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-gray-900 dark:text-white">Classification Result</h3>
                  <button
                    onClick={handleCopyResult}
                    className="flex items-center gap-1 px-2 py-1 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
                  >
                    {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                    {copied ? 'Copied!' : 'Copy'}
                  </button>
                </div>

                <div className="space-y-4">
                  {/* Primary Type */}
                  <div className="flex items-center gap-4">
                    <div className="flex-1">
                      <div className="text-sm text-gray-500 dark:text-gray-400 mb-1">Type</div>
                      <span className={`inline-block px-3 py-1.5 rounded-lg text-sm font-medium ${MEMORY_TYPES[result.memory_type as keyof typeof MEMORY_TYPES]?.color || 'bg-gray-100 text-gray-700'}`}>
                        {MEMORY_TYPES[result.memory_type as keyof typeof MEMORY_TYPES]?.label || result.memory_type}
                      </span>
                    </div>
                    <div className="flex-1">
                      <div className="text-sm text-gray-500 dark:text-gray-400 mb-1">Confidence</div>
                      <div className="flex items-center gap-2">
                        {getConfidenceIcon(result.confidence_level)}
                        <span className={`font-semibold ${CONFIDENCE_COLORS[result.confidence_level]}`}>
                          {(result.confidence * 100).toFixed(1)}%
                        </span>
                        <span className="text-sm text-gray-500 dark:text-gray-400">
                          ({result.confidence_level})
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Reasoning */}
                  <div>
                    <div className="text-sm text-gray-500 dark:text-gray-400 mb-1">Reasoning</div>
                    <p className="text-gray-900 dark:text-white">{result.reasoning}</p>
                  </div>

                  {/* Alternative */}
                  {result.alternative_type && (
                    <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                      <div className="text-sm text-gray-500 dark:text-gray-400 mb-1">Alternative Classification</div>
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-1 rounded text-xs ${MEMORY_TYPES[result.alternative_type as keyof typeof MEMORY_TYPES]?.color || 'bg-gray-100 text-gray-700'}`}>
                          {MEMORY_TYPES[result.alternative_type as keyof typeof MEMORY_TYPES]?.label || result.alternative_type}
                        </span>
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                          {((result.alternative_confidence || 0) * 100).toFixed(1)}% confidence
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* History */}
          {history.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm">
              <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                <h3 className="font-semibold text-gray-900 dark:text-white">Recent Classifications</h3>
              </div>
              <div className="divide-y divide-gray-200 dark:divide-gray-700">
                {history.map((item, index) => (
                  <div key={index} className="p-4">
                    <p className="text-gray-900 dark:text-white text-sm mb-2 truncate">
                      "{item.text}"
                    </p>
                    <div className="flex items-center gap-3">
                      <span className={`px-2 py-0.5 rounded text-xs ${MEMORY_TYPES[item.result.memory_type as keyof typeof MEMORY_TYPES]?.color || 'bg-gray-100 text-gray-700'}`}>
                        {item.result.memory_type}
                      </span>
                      <span className={`text-xs ${CONFIDENCE_COLORS[item.result.confidence_level]}`}>
                        {(item.result.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Memory Types Reference */}
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-4">Memory Types</h3>
            <div className="space-y-4">
              {Object.entries(MEMORY_TYPES).map(([key, type]) => (
                <div key={key} className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${type.color}`}>
                      {type.label}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{type.description}</p>
                  <div className="flex flex-wrap gap-1">
                    {type.examples.map((example, i) => (
                      <button
                        key={i}
                        onClick={() => setInputText(example)}
                        className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded hover:bg-gray-200 dark:hover:bg-gray-600 cursor-pointer"
                      >
                        {example}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Confidence Levels */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-4">Confidence Levels</h3>
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <CheckCircle className="w-5 h-5 text-green-500" />
                <div>
                  <div className="font-medium text-green-600 dark:text-green-400">HIGH (90%+)</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Very confident</div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-yellow-500" />
                <div>
                  <div className="font-medium text-yellow-600 dark:text-yellow-400">MEDIUM (70-90%)</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Reasonably confident</div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <HelpCircle className="w-5 h-5 text-orange-500" />
                <div>
                  <div className="font-medium text-orange-600 dark:text-orange-400">LOW (50-70%)</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Uncertain</div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <HelpCircle className="w-5 h-5 text-red-500" />
                <div>
                  <div className="font-medium text-red-600 dark:text-red-400">UNCERTAIN (&lt;50%)</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Very uncertain</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
