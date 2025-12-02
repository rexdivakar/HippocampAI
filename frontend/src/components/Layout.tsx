import { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Brain,
  Users,
  HeartPulse,
  BarChart3,
  Network,
  Wifi,
  WifiOff,
  LogOut,
} from 'lucide-react';
import clsx from 'clsx';

interface LayoutProps {
  children: ReactNode;
  userId: string;
  onLogout: () => void;
  wsConnected: boolean;
}

export function Layout({ children, userId, onLogout, wsConnected }: LayoutProps) {
  const location = useLocation();

  const navItems = [
    { path: '/memories', label: 'Memories', icon: Brain },
    { path: '/graph', label: 'Graph View', icon: Network },
    { path: '/collaboration', label: 'Collaboration', icon: Users },
    { path: '/health', label: 'Health', icon: HeartPulse },
    { path: '/analytics', label: 'Analytics', icon: BarChart3 },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link
              to="/memories"
              className="flex items-center space-x-3 hover:opacity-80 transition-opacity cursor-pointer"
            >
              <Brain className="w-8 h-8 text-primary-600" />
              <div>
                <h1 className="text-xl font-bold text-gray-900">HippocampAI</h1>
                <p className="text-xs text-gray-500">Memory Visualization</p>
              </div>
            </Link>

            {/* Navigation */}
            <nav className="flex space-x-1">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;

                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={clsx(
                      'flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200',
                      isActive
                        ? 'bg-primary-100 text-primary-700 font-medium'
                        : 'text-gray-600 hover:bg-gray-100'
                    )}
                  >
                    <Icon className="w-5 h-5" />
                    <span>{item.label}</span>
                  </Link>
                );
              })}
            </nav>

            {/* User info & status */}
            <div className="flex items-center space-x-4">
              {/* WebSocket status */}
              <div className="flex items-center space-x-2">
                {wsConnected ? (
                  <>
                    <Wifi className="w-5 h-5 text-green-600" />
                    <span className="text-sm text-green-600">Live</span>
                  </>
                ) : (
                  <>
                    <WifiOff className="w-5 h-5 text-gray-400" />
                    <span className="text-sm text-gray-400">Offline</span>
                  </>
                )}
              </div>

              {/* User ID */}
              <div className="text-sm text-gray-600 border-l border-gray-200 pl-4">
                <span className="font-medium">{userId.slice(0, 8)}</span>
              </div>

              {/* Logout */}
              <button
                onClick={onLogout}
                className="flex items-center space-x-2 px-3 py-2 text-sm text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors duration-200"
                title="Logout"
              >
                <LogOut className="w-4 h-4" />
                <span>Logout</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between text-sm text-gray-500">
            <p>© 2025 HippocampAI. Autonomous memory engine with hybrid retrieval.</p>
            <div className="flex items-center space-x-4">
              <a href="#" className="hover:text-primary-600">Documentation</a>
              <a href="#" className="hover:text-primary-600">API</a>
              <a href="https://github.com/anthropics/hippocampai" className="hover:text-primary-600">GitHub</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
