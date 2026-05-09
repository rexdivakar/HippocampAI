import { lazy, Suspense, useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { RefreshCw } from 'lucide-react';
import { Layout } from './components/Layout';
import { ErrorBoundary } from './components/ErrorBoundary';
import { LoginPage } from './pages/LoginPage';
import { useWebSocket } from './hooks/useWebSocket';

const DashboardPage = lazy(() =>
  import('./pages/DashboardPage').then((m) => ({ default: m.DashboardPage }))
);
const MemoriesPage = lazy(() =>
  import('./pages/MemoriesPage').then((m) => ({ default: m.MemoriesPage }))
);
const MemoriesPageRedesigned = lazy(() =>
  import('./pages/MemoriesPageRedesigned').then((m) => ({ default: m.MemoriesPageRedesigned }))
);
const TimelinePage = lazy(() =>
  import('./pages/TimelinePage').then((m) => ({ default: m.TimelinePage }))
);
const ReplayPage = lazy(() =>
  import('./pages/ReplayPage').then((m) => ({ default: m.ReplayPage }))
);
const GraphViewPage = lazy(() =>
  import('./pages/GraphViewPage').then((m) => ({ default: m.GraphViewPage }))
);
const ClusterPage = lazy(() =>
  import('./pages/ClusterPage').then((m) => ({ default: m.ClusterPage }))
);
const HygienePage = lazy(() =>
  import('./pages/HygienePage').then((m) => ({ default: m.HygienePage }))
);
const CollaborationPage = lazy(() =>
  import('./pages/CollaborationPage').then((m) => ({ default: m.CollaborationPage }))
);
const HealthPage = lazy(() =>
  import('./pages/HealthPage').then((m) => ({ default: m.HealthPage }))
);
const AnalyticsPage = lazy(() =>
  import('./pages/AnalyticsPage').then((m) => ({ default: m.AnalyticsPage }))
);
const HeatmapPage = lazy(() =>
  import('./pages/HeatmapPage').then((m) => ({ default: m.HeatmapPage }))
);
const ObservabilityPage = lazy(() =>
  import('./pages/ObservabilityPage').then((m) => ({ default: m.ObservabilityPage }))
);
const AgentMemoryPage = lazy(() =>
  import('./pages/AgentMemoryPage').then((m) => ({ default: m.AgentMemoryPage }))
);
const PoliciesPage = lazy(() =>
  import('./pages/PoliciesPage').then((m) => ({ default: m.PoliciesPage }))
);
const ConceptGrowthPage = lazy(() =>
  import('./pages/ConceptGrowthPage').then((m) => ({ default: m.ConceptGrowthPage }))
);
const SleepPhasePage = lazy(() =>
  import('./pages/SleepPhasePage').then((m) => ({ default: m.SleepPhasePage }))
);
const BiTemporalPage = lazy(() =>
  import('./pages/BiTemporalPage').then((m) => ({ default: m.BiTemporalPage }))
);
const ContextAssemblyPage = lazy(() =>
  import('./pages/ContextAssemblyPage').then((m) => ({ default: m.ContextAssemblyPage }))
);
const SchemaPage = lazy(() =>
  import('./pages/SchemaPage').then((m) => ({ default: m.SchemaPage }))
);
const AgenticClassifierPage = lazy(() =>
  import('./pages/AgenticClassifierPage').then((m) => ({ default: m.AgenticClassifierPage }))
);
const FeedbackPage = lazy(() =>
  import('./pages/FeedbackPage').then((m) => ({ default: m.FeedbackPage }))
);
const TriggersPage = lazy(() =>
  import('./pages/TriggersPage').then((m) => ({ default: m.TriggersPage }))
);
const ProceduralMemoryPage = lazy(() =>
  import('./pages/ProceduralMemoryPage').then((m) => ({ default: m.ProceduralMemoryPage }))
);
const EmbeddingMigrationPage = lazy(() =>
  import('./pages/EmbeddingMigrationPage').then((m) => ({ default: m.EmbeddingMigrationPage }))
);
const ProspectiveMemoryPage = lazy(() =>
  import('./pages/ProspectiveMemoryPage').then((m) => ({ default: m.ProspectiveMemoryPage }))
);

function LoadingSpinner() {
  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center">
      <RefreshCw className="w-8 h-8 text-blue-400 animate-spin" />
    </div>
  );
}

function App() {
  const [userId, setUserId] = useState<string | null>(
    localStorage.getItem('user_id') || null
  );
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(
    !!localStorage.getItem('auth_token')
  );

  // Initialize WebSocket connection when authenticated
  const { connected } = useWebSocket({
    userId: userId || undefined,
    autoConnect: isAuthenticated,
  });

  useEffect(() => {
    // Check authentication on mount
    const token = localStorage.getItem('auth_token');
    const storedUserId = localStorage.getItem('user_id');

    if (token && storedUserId) {
      setIsAuthenticated(true);
      setUserId(storedUserId);
    }
  }, []);

  const handleLogin = (token: string, userId: string) => {
    localStorage.setItem('auth_token', token);
    localStorage.setItem('user_id', userId);
    setIsAuthenticated(true);
    setUserId(userId);
  };

  const handleLogout = () => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_id');
    setIsAuthenticated(false);
    setUserId(null);
  };

  if (!isAuthenticated) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return (
    <Router>
      <Layout
        userId={userId!}
        onLogout={handleLogout}
        wsConnected={connected}
      >
        <ErrorBoundary>
          <Suspense fallback={<LoadingSpinner />}>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<DashboardPage userId={userId!} />} />
              <Route path="/memories" element={<MemoriesPageRedesigned userId={userId!} />} />
              <Route path="/memories/classic" element={<MemoriesPage userId={userId!} />} />
              <Route path="/sleep-phase" element={<SleepPhasePage userId={userId!} />} />
              <Route path="/bitemporal" element={<BiTemporalPage userId={userId!} />} />
              <Route path="/context" element={<ContextAssemblyPage userId={userId!} />} />
              <Route path="/schema" element={<SchemaPage userId={userId!} />} />
              <Route path="/classifier" element={<AgenticClassifierPage userId={userId!} />} />
              <Route path="/timeline" element={<TimelinePage userId={userId!} />} />
              <Route path="/replay" element={<ReplayPage userId={userId!} />} />
              <Route path="/graph" element={<GraphViewPage userId={userId!} />} />
              <Route path="/cluster" element={<ClusterPage userId={userId!} />} />
              <Route path="/heatmap" element={<HeatmapPage userId={userId!} />} />
              <Route path="/concept-growth" element={<ConceptGrowthPage userId={userId!} />} />
              <Route path="/hygiene" element={<HygienePage userId={userId!} />} />
              <Route path="/collaboration" element={<CollaborationPage userId={userId!} />} />
              <Route path="/agents" element={<AgentMemoryPage userId={userId!} />} />
              <Route path="/policies" element={<PoliciesPage userId={userId!} />} />
              <Route path="/health" element={<HealthPage userId={userId!} />} />
              <Route path="/analytics" element={<AnalyticsPage userId={userId!} />} />
              <Route path="/observability" element={<ObservabilityPage userId={userId!} />} />
              <Route path="/feedback" element={<FeedbackPage userId={userId!} />} />
              <Route path="/triggers" element={<TriggersPage userId={userId!} />} />
              <Route path="/procedural" element={<ProceduralMemoryPage userId={userId!} />} />
              <Route path="/prospective" element={<ProspectiveMemoryPage userId={userId!} />} />
              <Route path="/migrations" element={<EmbeddingMigrationPage userId={userId!} />} />
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </Suspense>
        </ErrorBoundary>
      </Layout>
    </Router>
  );
}

export default App;
