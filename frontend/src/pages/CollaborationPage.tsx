import { Users } from 'lucide-react';

interface CollaborationPageProps {
  userId: string;
}

export function CollaborationPage({ userId }: CollaborationPageProps) {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Users className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Collaboration</h1>
            <p className="text-gray-600">Shared memory spaces and team coordination</p>
          </div>
        </div>
      </div>

      {/* Placeholder content */}
      <div className="card text-center py-12">
        <Users className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-gray-700 mb-2">Collaboration Dashboard</h2>
        <p className="text-gray-500 mb-4">
          Shared spaces, agents, and collaboration features will be displayed here
        </p>
        <p className="text-sm text-gray-400">User ID: {userId}</p>
      </div>
    </div>
  );
}
