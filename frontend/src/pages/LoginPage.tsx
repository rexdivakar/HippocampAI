import { useState, FormEvent } from 'react';
import { Brain, KeyRound, User, UserPlus } from 'lucide-react';
import { apiClient } from '../services/api';

interface LoginPageProps {
  onLogin: (token: string, userId: string) => void;
}

export function LoginPage({ onLogin }: LoginPageProps) {
  const [userId, setUserId] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');

    if (!userId.trim()) {
      setError('User ID is required');
      return;
    }

    setLoading(true);

    try {
      // Validate session with backend
      const response = await fetch('/auth/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ unique_id: userId, api_key: apiKey || null }),
      });

      const data = await response.json();

      if (!data.valid) {
        setError('Session ID not available');
        setLoading(false);
        return;
      }

      // Session is valid - proceed with login
      const token = apiKey || 'session_' + Date.now();
      apiClient.setAuthToken(token);
      onLogin(token, userId);
    } catch (err) {
      setError('Session ID not available');
      console.error('Login error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSignup = async () => {
    setError('');
    setLoading(true);

    try {
      const response = await fetch('/auth/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: null, metadata: null }),
      });

      if (!response.ok) {
        throw new Error('Signup failed');
      }

      const data = await response.json();

      // Automatically log in with the new session
      const token = 'session_' + Date.now();
      apiClient.setAuthToken(token);
      onLogin(token, data.user_id);

    } catch (err) {
      setError('Signup failed. Please try again.');
      console.error('Signup error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDemoLogin = () => {
    // Use an existing user_id that has memories in Qdrant
    const demoUserId = '0177aa41-efb9-4b24-b423-ffd78f61521c';
    const demoToken = 'demo_token_' + Date.now();
    apiClient.setAuthToken(demoToken);
    onLogin(demoToken, demoUserId);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-500 via-primary-600 to-secondary-600 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo & Title */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-white rounded-2xl shadow-lg mb-4">
            <Brain className="w-12 h-12 text-primary-600" />
          </div>
          <h1 className="text-4xl font-bold text-white mb-2">HippocampAI</h1>
          <p className="text-primary-100 text-lg">Memory Visualization Platform</p>
        </div>

        {/* Login Card */}
        <div className="bg-white rounded-2xl shadow-2xl p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Lookup</h2>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* User ID / Session ID */}
            <div>
              <label htmlFor="userId" className="block text-sm font-medium text-gray-700 mb-2">
                Session ID
              </label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  id="userId"
                  type="text"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  className="input pl-10"
                  placeholder="Enter your session ID or user ID"
                  disabled={loading}
                />
              </div>
            </div>

            {/* API Key (Optional) */}
            <div>
              <label htmlFor="apiKey" className="block text-sm font-medium text-gray-700 mb-2">
                API Key <span className="text-gray-400 text-xs">(Optional)</span>
              </label>
              <div className="relative">
                <KeyRound className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  id="apiKey"
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  className="input pl-10"
                  placeholder="Enter your API key"
                  disabled={loading}
                />
              </div>
              <p className="mt-1 text-xs text-gray-500">
                Leave blank to use demo mode
              </p>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                <p className="text-sm text-red-600">{error}</p>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Searching...' : 'Search'}
            </button>
          </form>

          {/* Divider */}
          <div className="relative my-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-300"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-white text-gray-500">or</span>
            </div>
          </div>

          {/* Demo Button */}
          <button
            onClick={handleDemoLogin}
            disabled={loading}
            className="w-full btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Try Demo Mode
          </button>

          {/* Signup Button */}
          <button
            onClick={handleSignup}
            disabled={loading}
            className="w-full mt-3 flex items-center justify-center gap-2 px-4 py-2 border border-primary-600 text-primary-600 rounded-lg hover:bg-primary-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <UserPlus className="w-5 h-5" />
            Create new session id
          </button>

          {/* Info */}
          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <p className="text-sm text-blue-800">
              <strong>Demo Mode:</strong> Explore the platform with sample data. No account required.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center">
          <p className="text-primary-100 text-sm">
            Autonomous memory engine with hybrid retrieval
          </p>
          <p className="text-primary-200 text-xs mt-2">
            Version 0.3.0
          </p>
        </div>
      </div>
    </div>
  );
}
