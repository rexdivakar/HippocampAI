import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { Layout } from './components/Layout';
import { MemoriesPage } from './pages/MemoriesPage';
import { TimelinePage } from './pages/TimelinePage';
import { ReplayPage } from './pages/ReplayPage';
import { GraphViewPage } from './pages/GraphViewPage';
import { ClusterPage } from './pages/ClusterPage';
import { HygienePage } from './pages/HygienePage';
import { CollaborationPage } from './pages/CollaborationPage';
import { HealthPage } from './pages/HealthPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { HeatmapPage } from './pages/HeatmapPage';
import { ObservabilityPage } from './pages/ObservabilityPage';
import { AgentMemoryPage } from './pages/AgentMemoryPage';
import { PoliciesPage } from './pages/PoliciesPage';
import { ConceptGrowthPage } from './pages/ConceptGrowthPage';
import { LoginPage } from './pages/LoginPage';
import { useWebSocket } from './hooks/useWebSocket';

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
        <Routes>
          <Route path="/" element={<Navigate to="/memories" replace />} />
          <Route path="/memories" element={<MemoriesPage userId={userId!} />} />
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
          <Route path="*" element={<Navigate to="/memories" replace />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
