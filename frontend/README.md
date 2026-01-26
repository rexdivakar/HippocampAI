# HippocampAI Frontend

Interactive memory visualization and management platform for HippocampAI.

## Features

- **Memory Visualization**: Interactive views of your memory store
- **Collaboration Dashboard**: Shared memory spaces and team coordination
- **Health Monitoring**: Memory health scores and auto-healing actions
- **Predictive Analytics**: Patterns, anomalies, recommendations, and forecasts
- **Real-time Updates**: WebSocket integration for live data streaming

## Tech Stack

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: TailwindCSS
- **State Management**: Zustand + React Query
- **Charts**: Recharts + D3.js
- **Icons**: Lucide React
- **Real-time**: Socket.IO Client

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- HippocampAI backend running on `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Open http://localhost:3000
```

### Build for Production

```bash
# Build the application
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/      # Reusable UI components
│   │   └── Layout.tsx   # Main layout with navigation
│   ├── pages/           # Route pages
│   │   ├── LoginPage.tsx
│   │   ├── MemoriesPage.tsx
│   │   ├── CollaborationPage.tsx
│   │   ├── HealthPage.tsx
│   │   └── AnalyticsPage.tsx
│   ├── hooks/           # Custom React hooks
│   │   └── useWebSocket.ts
│   ├── services/        # API client and services
│   │   └── api.ts
│   ├── types/           # TypeScript type definitions
│   │   └── index.ts
│   ├── utils/           # Utility functions
│   ├── styles/          # Global styles
│   │   └── index.css
│   ├── App.tsx          # Main app component
│   └── main.tsx         # Entry point
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
└── postcss.config.js
```

## Development

### Adding New Pages

1. Create a new page component in `src/pages/`
2. Add the route in `src/App.tsx`
3. Add navigation link in `src/components/Layout.tsx`

### API Integration

All API calls go through the `apiClient` service in `src/services/api.ts`.

Example:
```typescript
import { apiClient } from '../services/api';

// Create a memory
const memory = await apiClient.createMemory({
  text: "Important fact",
  user_id: userId,
  type: "fact",
});

// Get health score
const health = await apiClient.getHealthScore(userId);
```

### WebSocket Events

Use the `useWebSocket` hook for real-time updates:

```typescript
import { useWebSocket } from '../hooks/useWebSocket';

function MyComponent() {
  const { connected, on, off } = useWebSocket({ userId });

  useEffect(() => {
    on('memory:created', (memory) => {
      console.log('New memory:', memory);
    });

    return () => {
      off('memory:created');
    };
  }, [on, off]);

  return <div>Connected: {connected ? 'Yes' : 'No'}</div>;
}
```

## Authentication

The app supports two modes:

1. **User ID + API Key**: For production use with actual authentication
2. **Demo Mode**: Generates temporary credentials for testing

Authentication state is stored in `localStorage`:
- `auth_token`: JWT or API token
- `user_id`: User identifier

## Environment Variables

Create a `.env` file:

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=http://localhost:8000
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## License

Apache 2.0
