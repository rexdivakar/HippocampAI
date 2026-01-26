import { useEffect, useRef, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import type { WebSocketEventType, WebSocketMessage } from '../types';

interface UseWebSocketOptions {
  userId?: string;
  spaceIds?: string[];
  agentIds?: string[];
  autoConnect?: boolean;
}

interface UseWebSocketReturn {
  socket: Socket | null;
  connected: boolean;
  subscribe: (type: 'user' | 'space' | 'agent', id: string) => void;
  unsubscribe: (type: 'user' | 'space' | 'agent', id: string) => void;
  on: <T = any>(event: WebSocketEventType, handler: (data: T) => void) => void;
  off: (event: WebSocketEventType) => void;
}

export function useWebSocket(options: UseWebSocketOptions = {}): UseWebSocketReturn {
  const { userId, spaceIds = [], agentIds = [], autoConnect = true } = options;

  const [connected, setConnected] = useState(false);
  const socketRef = useRef<Socket | null>(null);
  const handlersRef = useRef<Map<string, (data: any) => void>>(new Map());

  useEffect(() => {
    if (!autoConnect) return;

    // Connect to Socket.IO server
    const socket = io('/', {
      transports: ['websocket', 'polling'],
      auth: userId ? { user_id: userId } : undefined,
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: 5,
    });

    socketRef.current = socket;

    // Connection handlers
    socket.on('connect', () => {
      console.log('WebSocket connected:', socket.id);
      setConnected(true);

      // Subscribe to user if provided
      if (userId) {
        socket.emit('subscribe_user', { user_id: userId });
      }

      // Subscribe to spaces
      spaceIds.forEach((spaceId) => {
        socket.emit('subscribe_space', { space_id: spaceId });
      });

      // Subscribe to agents
      agentIds.forEach((agentId) => {
        socket.emit('subscribe_agent', { agent_id: agentId });
      });
    });

    socket.on('disconnect', (reason: string) => {
      console.log('WebSocket disconnected:', reason);
      setConnected(false);
    });

    socket.on('connect_error', (error: any) => {
      console.error('WebSocket connection error:', error.message);
      console.error('Make sure backend is running with: python -m uvicorn hippocampai.api.app:socket_app --port 8000');
      setConnected(false);
    });

    socket.on('error', (error: any) => {
      console.error('WebSocket error:', error);
    });

    socket.on('reconnect_attempt', (attempt: number) => {
      console.log(`WebSocket reconnection attempt ${attempt}...`);
    });

    socket.on('reconnect', (attemptNumber: number) => {
      console.log(`WebSocket reconnected after ${attemptNumber} attempts`);
      setConnected(true);
    });

    // Re-attach all registered handlers
    handlersRef.current.forEach((handler, event) => {
      socket.on(event, handler);
    });

    return () => {
      socket.disconnect();
    };
  }, [autoConnect, userId, spaceIds.join(','), agentIds.join(',')]);

  const subscribe = useCallback((type: 'user' | 'space' | 'agent', id: string) => {
    if (!socketRef.current) return;

    const eventMap = {
      user: 'subscribe_user',
      space: 'subscribe_space',
      agent: 'subscribe_agent',
    };

    const dataMap = {
      user: { user_id: id },
      space: { space_id: id },
      agent: { agent_id: id },
    };

    socketRef.current.emit(eventMap[type], dataMap[type]);
  }, []);

  const unsubscribe = useCallback((type: 'user' | 'space' | 'agent', id: string) => {
    if (!socketRef.current) return;

    socketRef.current.emit('unsubscribe', { type, id });
  }, []);

  const on = useCallback(<T = any>(event: WebSocketEventType, handler: (data: T) => void) => {
    if (!socketRef.current) return;

    // Wrap handler to extract data from message
    const wrappedHandler = (message: WebSocketMessage<T>) => {
      handler(message.data || (message as unknown as T));
    };

    handlersRef.current.set(event, wrappedHandler);
    socketRef.current.on(event, wrappedHandler);
  }, []);

  const off = useCallback((event: WebSocketEventType) => {
    if (!socketRef.current) return;

    const handler = handlersRef.current.get(event);
    if (handler) {
      socketRef.current.off(event, handler);
      handlersRef.current.delete(event);
    }
  }, []);

  return {
    socket: socketRef.current,
    connected,
    subscribe,
    unsubscribe,
    on,
    off,
  };
}
