import { ReactNode, useState, useRef, useCallback, useEffect } from 'react';
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
  Clock,
  Play,
  Layers,
  Sparkles,
  ChevronDown,
  Eye,
  Settings,
  Menu,
  X,
  Activity,
  TrendingUp,
  Bot,
  Shield,
  Server,
  Moon,
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
  const [isVisualizationOpen, setIsVisualizationOpen] = useState(false);
  const [isAnalysisOpen, setIsAnalysisOpen] = useState(false);
  const [isManagementOpen, setIsManagementOpen] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  // Timeout refs for delayed dropdown closing
  const visualizationTimeoutRef = useRef<number | null>(null);
  const analysisTimeoutRef = useRef<number | null>(null);
  const managementTimeoutRef = useRef<number | null>(null);

  // Delayed close handlers with 200ms delay
  const handleVisualizationEnter = useCallback(() => {
    if (visualizationTimeoutRef.current) {
      clearTimeout(visualizationTimeoutRef.current);
      visualizationTimeoutRef.current = null;
    }
    setIsVisualizationOpen(true);
  }, []);

  const handleVisualizationLeave = useCallback(() => {
    visualizationTimeoutRef.current = setTimeout(() => {
      setIsVisualizationOpen(false);
    }, 200);
  }, []);

  const handleAnalysisEnter = useCallback(() => {
    if (analysisTimeoutRef.current) {
      clearTimeout(analysisTimeoutRef.current);
      analysisTimeoutRef.current = null;
    }
    setIsAnalysisOpen(true);
  }, []);

  const handleAnalysisLeave = useCallback(() => {
    analysisTimeoutRef.current = setTimeout(() => {
      setIsAnalysisOpen(false);
    }, 200);
  }, []);

  const handleManagementEnter = useCallback(() => {
    if (managementTimeoutRef.current) {
      clearTimeout(managementTimeoutRef.current);
      managementTimeoutRef.current = null;
    }
    setIsManagementOpen(true);
  }, []);

  const handleManagementLeave = useCallback(() => {
    managementTimeoutRef.current = setTimeout(() => {
      setIsManagementOpen(false);
    }, 200);
  }, []);

  const visualizationItems = [
    { path: '/timeline', label: 'Timeline', icon: Clock },
    { path: '/graph', label: 'Graph View', icon: Network },
    { path: '/cluster', label: 'Clusters', icon: Layers },
    { path: '/heatmap', label: 'Heatmap', icon: Activity },
    { path: '/concept-growth', label: 'Concept Growth', icon: TrendingUp },
  ];

  const analysisItems = [
    { path: '/replay', label: 'Usage Trace', icon: Play },
    { path: '/health', label: 'Health', icon: HeartPulse },
    { path: '/analytics', label: 'Analytics', icon: BarChart3 },
    { path: '/observability', label: 'Observability', icon: Server },
  ];

  const managementItems = [
    { path: '/hygiene', label: 'Clean-Up', icon: Sparkles },
    { path: '/collaboration', label: 'Collaboration', icon: Users },
    { path: '/agents', label: 'Agents', icon: Bot },
    { path: '/policies', label: 'Policies', icon: Shield },
  ];

  const isPathActive = (paths: string[]) => {
    return paths.some((path) => location.pathname === path);
  };

  // Click outside to close dropdowns
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (!target.closest('[data-dropdown]')) {
        setIsVisualizationOpen(false);
        setIsAnalysisOpen(false);
        setIsManagementOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      // Cleanup timeouts on unmount
      if (visualizationTimeoutRef.current) clearTimeout(visualizationTimeoutRef.current);
      if (analysisTimeoutRef.current) clearTimeout(analysisTimeoutRef.current);
      if (managementTimeoutRef.current) clearTimeout(managementTimeoutRef.current);
    };
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link
              to="/memories"
              className="flex items-center space-x-3 hover:opacity-80 transition-opacity cursor-pointer flex-shrink-0"
            >
              <Brain className="w-8 h-8 text-primary-600" />
              <div className="hidden sm:block">
                <h1 className="text-xl font-bold text-gray-900">HippocampAI</h1>
                <p className="text-xs text-gray-500">Memory Visualization</p>
              </div>
            </Link>

            {/* Desktop Navigation */}
            <nav className="hidden lg:flex items-center space-x-1">
              {/* Memories - Primary */}
              <Link
                to="/memories"
                className={clsx(
                  'flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200',
                  location.pathname === '/memories'
                    ? 'bg-primary-100 text-primary-700 font-medium'
                    : 'text-gray-600 hover:bg-gray-100'
                )}
              >
                <Brain className="w-5 h-5" />
                <span>Memories</span>
              </Link>

              {/* Sleep Phase - Primary */}
              <Link
                to="/sleep-phase"
                className={clsx(
                  'flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200',
                  location.pathname === '/sleep-phase'
                    ? 'bg-primary-100 text-primary-700 font-medium'
                    : 'text-gray-600 hover:bg-gray-100'
                )}
              >
                <Moon className="w-5 h-5" />
                <span>Sleep Phase</span>
              </Link>

              {/* Visualization Dropdown */}
              <div className="relative" data-dropdown>
                <button
                  onMouseEnter={handleVisualizationEnter}
                  onMouseLeave={handleVisualizationLeave}
                  onClick={() => setIsVisualizationOpen(!isVisualizationOpen)}
                  className={clsx(
                    'flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200',
                    isPathActive(visualizationItems.map((i) => i.path))
                      ? 'bg-primary-100 text-primary-700 font-medium'
                      : 'text-gray-600 hover:bg-gray-100'
                  )}
                >
                  <Eye className="w-5 h-5" />
                  <span>Visualize</span>
                  <ChevronDown className="w-4 h-4" />
                </button>
                {isVisualizationOpen && (
                  <div
                    onMouseEnter={handleVisualizationEnter}
                    onMouseLeave={handleVisualizationLeave}
                    className="absolute top-full left-0 pt-2 w-48"
                  >
                    <div className="bg-white rounded-lg shadow-lg border border-gray-200 py-2">
                      {visualizationItems.map((item) => {
                        const Icon = item.icon;
                        return (
                          <Link
                            key={item.path}
                            to={item.path}
                            onClick={() => setIsVisualizationOpen(false)}
                            className={clsx(
                              'flex items-center space-x-3 px-4 py-2 hover:bg-gray-50 transition-colors',
                              location.pathname === item.path
                                ? 'text-primary-700 bg-primary-50'
                                : 'text-gray-700'
                            )}
                          >
                            <Icon className="w-4 h-4" />
                            <span>{item.label}</span>
                          </Link>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>

              {/* Analysis Dropdown */}
              <div className="relative" data-dropdown>
                <button
                  onMouseEnter={handleAnalysisEnter}
                  onMouseLeave={handleAnalysisLeave}
                  onClick={() => setIsAnalysisOpen(!isAnalysisOpen)}
                  className={clsx(
                    'flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200',
                    isPathActive(analysisItems.map((i) => i.path))
                      ? 'bg-primary-100 text-primary-700 font-medium'
                      : 'text-gray-600 hover:bg-gray-100'
                  )}
                >
                  <BarChart3 className="w-5 h-5" />
                  <span>Analyze</span>
                  <ChevronDown className="w-4 h-4" />
                </button>
                {isAnalysisOpen && (
                  <div
                    onMouseEnter={handleAnalysisEnter}
                    onMouseLeave={handleAnalysisLeave}
                    className="absolute top-full left-0 pt-2 w-48"
                  >
                    <div className="bg-white rounded-lg shadow-lg border border-gray-200 py-2">
                      {analysisItems.map((item) => {
                        const Icon = item.icon;
                        return (
                          <Link
                            key={item.path}
                            to={item.path}
                            onClick={() => setIsAnalysisOpen(false)}
                            className={clsx(
                              'flex items-center space-x-3 px-4 py-2 hover:bg-gray-50 transition-colors',
                              location.pathname === item.path
                                ? 'text-primary-700 bg-primary-50'
                                : 'text-gray-700'
                            )}
                          >
                            <Icon className="w-4 h-4" />
                            <span>{item.label}</span>
                          </Link>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>

              {/* Management Dropdown */}
              <div className="relative" data-dropdown>
                <button
                  onMouseEnter={handleManagementEnter}
                  onMouseLeave={handleManagementLeave}
                  onClick={() => setIsManagementOpen(!isManagementOpen)}
                  className={clsx(
                    'flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200',
                    isPathActive(managementItems.map((i) => i.path))
                      ? 'bg-primary-100 text-primary-700 font-medium'
                      : 'text-gray-600 hover:bg-gray-100'
                  )}
                >
                  <Settings className="w-5 h-5" />
                  <span>Manage</span>
                  <ChevronDown className="w-4 h-4" />
                </button>
                {isManagementOpen && (
                  <div
                    onMouseEnter={handleManagementEnter}
                    onMouseLeave={handleManagementLeave}
                    className="absolute top-full left-0 pt-2 w-48"
                  >
                    <div className="bg-white rounded-lg shadow-lg border border-gray-200 py-2">
                      {managementItems.map((item) => {
                        const Icon = item.icon;
                        return (
                          <Link
                            key={item.path}
                            to={item.path}
                            onClick={() => setIsManagementOpen(false)}
                            className={clsx(
                              'flex items-center space-x-3 px-4 py-2 hover:bg-gray-50 transition-colors',
                              location.pathname === item.path
                                ? 'text-primary-700 bg-primary-50'
                                : 'text-gray-700'
                            )}
                          >
                            <Icon className="w-4 h-4" />
                            <span>{item.label}</span>
                          </Link>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </nav>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="lg:hidden p-2 text-gray-600 hover:bg-gray-100 rounded-lg"
            >
              {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>

            {/* User info & status - Desktop */}
            <div className="hidden lg:flex items-center space-x-4">
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

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <div className="lg:hidden border-t border-gray-200 bg-white">
            <div className="max-w-7xl mx-auto px-4 py-4 space-y-2">
              {/* Primary Links */}
              <Link
                to="/memories"
                onClick={() => setIsMobileMenuOpen(false)}
                className={clsx(
                  'flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors',
                  location.pathname === '/memories'
                    ? 'bg-primary-100 text-primary-700 font-medium'
                    : 'text-gray-700 hover:bg-gray-50'
                )}
              >
                <Brain className="w-5 h-5" />
                <span>Memories</span>
              </Link>

              <Link
                to="/sleep-phase"
                onClick={() => setIsMobileMenuOpen(false)}
                className={clsx(
                  'flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors',
                  location.pathname === '/sleep-phase'
                    ? 'bg-primary-100 text-primary-700 font-medium'
                    : 'text-gray-700 hover:bg-gray-50'
                )}
              >
                <Moon className="w-5 h-5" />
                <span>Sleep Phase</span>
              </Link>

              {/* Visualization Group */}
              <div className="space-y-1">
                <div className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                  Visualize
                </div>
                {visualizationItems.map((item) => {
                  const Icon = item.icon;
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      onClick={() => setIsMobileMenuOpen(false)}
                      className={clsx(
                        'flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors',
                        location.pathname === item.path
                          ? 'bg-primary-100 text-primary-700 font-medium'
                          : 'text-gray-700 hover:bg-gray-50'
                      )}
                    >
                      <Icon className="w-5 h-5" />
                      <span>{item.label}</span>
                    </Link>
                  );
                })}
              </div>

              {/* Analysis Group */}
              <div className="space-y-1">
                <div className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                  Analyze
                </div>
                {analysisItems.map((item) => {
                  const Icon = item.icon;
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      onClick={() => setIsMobileMenuOpen(false)}
                      className={clsx(
                        'flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors',
                        location.pathname === item.path
                          ? 'bg-primary-100 text-primary-700 font-medium'
                          : 'text-gray-700 hover:bg-gray-50'
                      )}
                    >
                      <Icon className="w-5 h-5" />
                      <span>{item.label}</span>
                    </Link>
                  );
                })}
              </div>

              {/* Management Group */}
              <div className="space-y-1">
                <div className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                  Manage
                </div>
                {managementItems.map((item) => {
                  const Icon = item.icon;
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      onClick={() => setIsMobileMenuOpen(false)}
                      className={clsx(
                        'flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors',
                        location.pathname === item.path
                          ? 'bg-primary-100 text-primary-700 font-medium'
                          : 'text-gray-700 hover:bg-gray-50'
                      )}
                    >
                      <Icon className="w-5 h-5" />
                      <span>{item.label}</span>
                    </Link>
                  );
                })}
              </div>

              {/* Mobile User Info */}
              <div className="border-t border-gray-200 pt-4 mt-4">
                <div className="flex items-center justify-between px-4 py-2">
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
                  <div className="text-sm text-gray-600">
                    <span className="font-medium">{userId.slice(0, 8)}</span>
                  </div>
                </div>
                <button
                  onClick={onLogout}
                  className="w-full flex items-center justify-center space-x-2 px-4 py-3 mt-2 text-red-600 bg-red-50 hover:bg-red-100 rounded-lg transition-colors"
                >
                  <LogOut className="w-5 h-5" />
                  <span>Logout</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </header>

      {/* Main content */}
      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
        {children}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between text-sm text-gray-500">
            <p>© 2025 HippocampAI. Autonomous memory engine with hybrid retrieval.</p>
            <div className="flex items-center space-x-4">
              <a href="https://github.com/rexdivakar/HippocampAI/tree/main/docs" target="_blank" rel="noopener noreferrer" className="hover:text-primary-600 transition-colors">Documentation</a>
              <a href="/api" target="_blank" rel="noopener noreferrer" className="hover:text-primary-600 transition-colors">API</a>
              <a href="https://github.com/rexdivakar/HippocampAI" target="_blank" rel="noopener noreferrer" className="hover:text-primary-600 transition-colors">GitHub</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
