import { useState, useMemo, useCallback, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Network,
  RefreshCw,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Filter,
  Info,
  Download,
  Search,
} from 'lucide-react';
import ForceGraph2D from 'react-force-graph-2d';
import { apiClient } from '../services/api';
import type { Memory } from '../types';
import clsx from 'clsx';

interface GraphViewPageProps {
  userId: string;
}

interface GraphNode {
  id: string;
  name: string;
  type: 'memory' | 'entity' | 'concept' | 'tag';
  val: number;
  color: string;
  memory?: Memory;
  metadata?: any;
}

interface GraphLink {
  source: string;
  target: string;
  value: number;
  type: string;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

const NODE_COLORS = {
  memory: '#3B82F6',
  entity: '#10B981',
  concept: '#8B5CF6',
  tag: '#F59E0B',
};

export function GraphViewPage({ userId }: GraphViewPageProps) {
  const [refreshKey, setRefreshKey] = useState(0);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [filterType, setFilterType] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const graphRef = useRef<any>();

  // Fetch memories
  const { data: memories = [], isLoading } = useQuery({
    queryKey: ['memories', userId, refreshKey],
    queryFn: async () => {
      const result = await apiClient.getMemories({
        user_id: userId,
        filters: {
          session_id: userId, // Pass userId as session_id to match by either field
        },
        limit: 1000,
      });
      return result;
    },
  });

  // Build graph data
  const graphData = useMemo((): GraphData => {
    const nodes: GraphNode[] = [];
    const links: GraphLink[] = [];
    const nodeMap = new Map<string, GraphNode>();

    // Extract entities and concepts from memory text
    const extractEntities = (text: string): string[] => {
      // Simple entity extraction - capitalized words
      const entities = text.match(/[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/g) || [];
      return [...new Set(entities)].slice(0, 5); // Top 5 unique entities
    };

    const extractConcepts = (text: string): string[] => {
      // Extract key concepts (words > 4 chars, excluding common words)
      const stopWords = ['this', 'that', 'with', 'from', 'have', 'more', 'will', 'your', 'their'];
      const words = text.toLowerCase().match(/\b\w{5,}\b/g) || [];
      return [...new Set(words)]
        .filter((w) => !stopWords.includes(w))
        .slice(0, 3);
    };

    // Add memory nodes
    memories.forEach((memory) => {
      const memoryNode: GraphNode = {
        id: `memory-${memory.id}`,
        name: memory.text.substring(0, 50) + '...',
        type: 'memory',
        val: memory.importance * 3,
        color: NODE_COLORS.memory,
        memory,
      };
      nodes.push(memoryNode);
      nodeMap.set(memoryNode.id, memoryNode);

      // Extract and add entities
      const entities = extractEntities(memory.text);
      entities.forEach((entity) => {
        const entityId = `entity-${entity}`;
        if (!nodeMap.has(entityId)) {
          const entityNode: GraphNode = {
            id: entityId,
            name: entity,
            type: 'entity',
            val: 5,
            color: NODE_COLORS.entity,
            metadata: { count: 1 },
          };
          nodes.push(entityNode);
          nodeMap.set(entityId, entityNode);
        } else {
          // Increment entity count
          const node = nodeMap.get(entityId)!;
          node.metadata.count++;
          node.val = node.metadata.count * 3;
        }

        // Link memory to entity
        links.push({
          source: memoryNode.id,
          target: entityId,
          value: 1,
          type: 'contains',
        });
      });

      // Extract and add concepts
      const concepts = extractConcepts(memory.text);
      concepts.forEach((concept) => {
        const conceptId = `concept-${concept}`;
        if (!nodeMap.has(conceptId)) {
          const conceptNode: GraphNode = {
            id: conceptId,
            name: concept,
            type: 'concept',
            val: 4,
            color: NODE_COLORS.concept,
            metadata: { count: 1 },
          };
          nodes.push(conceptNode);
          nodeMap.set(conceptId, conceptNode);
        } else {
          const node = nodeMap.get(conceptId)!;
          node.metadata.count++;
          node.val = node.metadata.count * 2;
        }

        links.push({
          source: memoryNode.id,
          target: conceptId,
          value: 1,
          type: 'discusses',
        });
      });

      // Add tags
      memory.tags.forEach((tag) => {
        const tagId = `tag-${tag}`;
        if (!nodeMap.has(tagId)) {
          const tagNode: GraphNode = {
            id: tagId,
            name: tag,
            type: 'tag',
            val: 6,
            color: NODE_COLORS.tag,
            metadata: { count: 1 },
          };
          nodes.push(tagNode);
          nodeMap.set(tagId, tagNode);
        } else {
          const node = nodeMap.get(tagId)!;
          node.metadata.count++;
          node.val = node.metadata.count * 4;
        }

        links.push({
          source: memoryNode.id,
          target: tagId,
          value: 2,
          type: 'tagged',
        });
      });
    });

    // Add links between similar tags (co-occurrence)
    const tagNodes = nodes.filter((n) => n.type === 'tag');
    for (let i = 0; i < tagNodes.length; i++) {
      for (let j = i + 1; j < tagNodes.length; j++) {
        const tag1 = tagNodes[i];
        const tag2 = tagNodes[j];

        // Count co-occurrence
        const coOccurrence = memories.filter((m) =>
          m.tags.includes(tag1.name) && m.tags.includes(tag2.name)
        ).length;

        if (coOccurrence > 0) {
          links.push({
            source: tag1.id,
            target: tag2.id,
            value: coOccurrence,
            type: 'related',
          });
        }
      }
    }

    return { nodes, links };
  }, [memories]);

  // Filter graph data
  const filteredGraphData = useMemo(() => {
    let filteredNodes = graphData.nodes;
    let filteredLinks = graphData.links;

    // Filter by type
    if (filterType !== 'all') {
      filteredNodes = filteredNodes.filter((n) => n.type === filterType);
      const nodeIds = new Set(filteredNodes.map((n) => n.id));
      filteredLinks = filteredLinks.filter(
        (l) => nodeIds.has(String(l.source)) && nodeIds.has(String(l.target))
      );
    }

    // Filter by search term
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filteredNodes = filteredNodes.filter((n) => n.name.toLowerCase().includes(term));
      const nodeIds = new Set(filteredNodes.map((n) => n.id));
      filteredLinks = filteredLinks.filter(
        (l) => nodeIds.has(String(l.source)) && nodeIds.has(String(l.target))
      );
    }

    return { nodes: filteredNodes, links: filteredLinks };
  }, [graphData, filterType, searchTerm]);

  const handleRefresh = () => {
    setRefreshKey((prev) => prev + 1);
  };

  const handleNodeClick = useCallback((node: any) => {
    setSelectedNode(node);
  }, []);

  const handleZoomIn = () => {
    if (graphRef.current) {
      graphRef.current.zoom(graphRef.current.zoom() * 1.2);
    }
  };

  const handleZoomOut = () => {
    if (graphRef.current) {
      graphRef.current.zoom(graphRef.current.zoom() / 1.2);
    }
  };

  const handleCenterView = () => {
    if (graphRef.current) {
      graphRef.current.zoomToFit(400);
    }
  };

  const handleExport = () => {
    const dataStr = JSON.stringify(graphData, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'knowledge-graph.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Top Bar */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Left: Title */}
          <div className="flex items-center space-x-2">
            <Network className="w-6 h-6 text-primary-600" />
            <h1 className="text-xl font-semibold text-gray-900">Knowledge Graph</h1>
            <span className="text-sm text-gray-500">
              ({filteredGraphData.nodes.length} nodes, {filteredGraphData.links.length} connections)
            </span>
          </div>

          {/* Right: Actions */}
          <div className="flex items-center space-x-2">
            <button
              onClick={handleRefresh}
              disabled={isLoading}
              className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              title="Refresh graph"
            >
              <RefreshCw className={clsx('w-4 h-4', isLoading && 'animate-spin')} />
            </button>
          </div>
        </div>
      </div>

      {/* Content Wrapper */}
      <div className="flex-1 overflow-hidden flex flex-col px-6 py-8 space-y-6">

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card bg-blue-50 border border-blue-200">
          <p className="text-sm text-blue-600 mb-1">Memories</p>
          <p className="text-2xl font-bold text-blue-900">
            {graphData.nodes.filter((n) => n.type === 'memory').length}
          </p>
        </div>
        <div className="card bg-green-50 border border-green-200">
          <p className="text-sm text-green-600 mb-1">Entities</p>
          <p className="text-2xl font-bold text-green-900">
            {graphData.nodes.filter((n) => n.type === 'entity').length}
          </p>
        </div>
        <div className="card bg-purple-50 border border-purple-200">
          <p className="text-sm text-purple-600 mb-1">Concepts</p>
          <p className="text-2xl font-bold text-purple-900">
            {graphData.nodes.filter((n) => n.type === 'concept').length}
          </p>
        </div>
        <div className="card bg-orange-50 border border-orange-200">
          <p className="text-sm text-orange-600 mb-1">Tags</p>
          <p className="text-2xl font-bold text-orange-900">
            {graphData.nodes.filter((n) => n.type === 'tag').length}
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-4">
            {/* Filter */}
            <div className="flex items-center space-x-2">
              <Filter className="w-5 h-5 text-gray-500" />
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="input py-2"
              >
                <option value="all">All Types</option>
                <option value="memory">Memories</option>
                <option value="entity">Entities</option>
                <option value="concept">Concepts</option>
                <option value="tag">Tags</option>
              </select>
            </div>

            {/* Search */}
            <div className="flex items-center space-x-2">
              <Search className="w-5 h-5 text-gray-500" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search nodes..."
                className="input py-2"
              />
            </div>
          </div>

          {/* Zoom controls */}
          <div className="flex items-center space-x-2">
            <button
              onClick={handleZoomIn}
              className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              title="Zoom In"
            >
              <ZoomIn className="w-5 h-5" />
            </button>
            <button
              onClick={handleZoomOut}
              className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              title="Zoom Out"
            >
              <ZoomOut className="w-5 h-5" />
            </button>
            <button
              onClick={handleCenterView}
              className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              title="Center View"
            >
              <Maximize2 className="w-5 h-5" />
            </button>
            <button
              onClick={handleExport}
              className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              title="Export Graph"
            >
              <Download className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center space-x-6 mb-4 pb-4 border-b border-gray-200">
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 rounded-full" style={{ backgroundColor: NODE_COLORS.memory }} />
            <span className="text-sm text-gray-600">Memory</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 rounded-full" style={{ backgroundColor: NODE_COLORS.entity }} />
            <span className="text-sm text-gray-600">Entity</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 rounded-full" style={{ backgroundColor: NODE_COLORS.concept }} />
            <span className="text-sm text-gray-600">Concept</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 rounded-full" style={{ backgroundColor: NODE_COLORS.tag }} />
            <span className="text-sm text-gray-600">Tag</span>
          </div>
        </div>

        {/* Graph */}
        {isLoading ? (
          <div className="flex items-center justify-center h-[600px]">
            <RefreshCw className="w-12 h-12 text-primary-600 animate-spin" />
          </div>
        ) : filteredGraphData.nodes.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-[600px] text-gray-400">
            <Network className="w-16 h-16 mb-4" />
            <p>No nodes to display</p>
          </div>
        ) : (
          <div className="relative bg-gray-50 rounded-lg overflow-hidden border border-gray-200">
            <ForceGraph2D
              ref={graphRef}
              graphData={filteredGraphData}
              nodeLabel="name"
              nodeColor="color"
              nodeVal="val"
              nodeRelSize={6}
              linkDirectionalParticles={2}
              linkDirectionalParticleWidth={2}
              onNodeClick={handleNodeClick}
              width={1200}
              height={600}
              backgroundColor="#f9fafb"
              nodeCanvasObject={(node: any, ctx, globalScale) => {
                const label = node.name;
                const fontSize = 12 / globalScale;
                ctx.font = `${fontSize}px Sans-Serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = node.color;
                ctx.fillText(label, node.x, node.y + 15 / globalScale);
              }}
            />
          </div>
        )}
      </div>

      {/* Selected Node Info */}
      {selectedNode && (
        <div className="card bg-gradient-to-r from-blue-50 to-purple-50">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center space-x-3">
              <Info className="w-6 h-6 text-blue-600" />
              <h2 className="text-xl font-bold text-gray-900">Node Details</h2>
            </div>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-gray-400 hover:text-gray-600"
            >
              ×
            </button>
          </div>

          <div className="space-y-3">
            <div>
              <span className="text-sm text-gray-500">Name:</span>
              <p className="font-semibold text-gray-900">{selectedNode.name}</p>
            </div>
            <div>
              <span className="text-sm text-gray-500">Type:</span>
              <p className="font-semibold text-gray-900 capitalize">{selectedNode.type}</p>
            </div>
            <div>
              <span className="text-sm text-gray-500">Connections:</span>
              <p className="font-semibold text-gray-900">
                {
                  graphData.links.filter(
                    (l) => l.source === selectedNode.id || l.target === selectedNode.id
                  ).length
                }
              </p>
            </div>
            {selectedNode.metadata && selectedNode.metadata.count && (
              <div>
                <span className="text-sm text-gray-500">Occurrences:</span>
                <p className="font-semibold text-gray-900">{selectedNode.metadata.count}</p>
              </div>
            )}
            {selectedNode.memory && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <p className="text-sm text-gray-600 mb-2">Memory Content:</p>
                <p className="text-sm text-gray-900 bg-white p-3 rounded-lg">
                  {selectedNode.memory.text}
                </p>
                <div className="flex items-center space-x-4 mt-3 text-xs text-gray-500">
                  <span>⭐ {selectedNode.memory.importance.toFixed(1)}</span>
                  <span>👁 {selectedNode.memory.access_count} views</span>
                  <span className="px-2 py-1 bg-white rounded">{selectedNode.memory.type}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      </div>
    </div>
  );
}
