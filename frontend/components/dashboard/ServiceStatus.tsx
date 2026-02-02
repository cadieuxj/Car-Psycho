'use client';

import { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardBody } from '@/components/ui';
import { Badge } from '@/components/ui';
import { api } from '@/lib/api';
import { ServiceHealth } from '@/lib/types';
import { Server, Database, Brain, RefreshCw, Cpu } from 'lucide-react';

interface ServiceCardProps {
  name: string;
  health: ServiceHealth;
  icon: React.ReactNode;
  details?: { label: string; value: string }[];
}

function ServiceCard({ name, health, icon, details }: ServiceCardProps) {
  return (
    <div className="flex items-start gap-4 p-4 bg-gray-50 rounded-lg">
      <div className={`p-3 rounded-lg ${health.status === 'healthy' ? 'bg-green-100' : 'bg-red-100'}`}>
        {icon}
      </div>
      <div className="flex-1">
        <div className="flex items-center justify-between mb-2">
          <h4 className="font-medium text-gray-900">{name}</h4>
          <Badge variant={health.status === 'healthy' ? 'success' : 'error'}>
            {health.status}
          </Badge>
        </div>
        {details && (
          <div className="space-y-1">
            {details.map((detail) => (
              <div key={detail.label} className="flex justify-between text-sm">
                <span className="text-gray-500">{detail.label}</span>
                <span className="text-gray-700 font-mono text-xs">
                  {detail.value === 'not configured' ? (
                    <span className="text-amber-600">Not configured</span>
                  ) : (
                    detail.value
                  )}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ServiceStatus() {
  const [health, setHealth] = useState<{
    manager: ServiceHealth;
    inference: ServiceHealth;
    data: ServiceHealth;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchHealth = async () => {
    setLoading(true);
    try {
      const result = await api.getAllServicesHealth();
      setHealth(result);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Failed to fetch health:', error);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Service Status</CardTitle>
        <button
          onClick={fetchHealth}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          disabled={loading}
        >
          <RefreshCw className={`w-4 h-4 text-gray-500 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </CardHeader>
      <CardBody className="space-y-4">
        {health ? (
          <>
            <ServiceCard
              name="Manager Service"
              health={health.manager}
              icon={<Cpu className={`w-5 h-5 ${health.manager.status === 'healthy' ? 'text-green-600' : 'text-red-600'}`} />}
              details={[
                { label: 'Database', value: health.manager.database || 'unknown' },
                { label: 'Redis', value: health.manager.redis || 'unknown' },
                { label: 'T4 VM', value: health.manager.t4_vm_host || 'not configured' },
              ]}
            />
            <ServiceCard
              name="Inference Service"
              health={health.inference}
              icon={<Brain className={`w-5 h-5 ${health.inference.status === 'healthy' ? 'text-green-600' : 'text-red-600'}`} />}
              details={[
                { label: 'ChromaDB', value: health.inference.chromadb || 'unknown' },
                { label: 'Ollama', value: health.inference.ollama || 'not configured' },
                { label: 'Redis', value: health.inference.redis || 'unknown' },
              ]}
            />
            <ServiceCard
              name="Data Service"
              health={health.data}
              icon={<Database className={`w-5 h-5 ${health.data.status === 'healthy' ? 'text-green-600' : 'text-red-600'}`} />}
              details={[
                { label: 'Teacher Model', value: health.data.teacher_model || 'not configured' },
                { label: 'ChromaDB', value: health.data.chromadb || 'unknown' },
                { label: 'Database', value: health.data.database || 'unknown' },
              ]}
            />
          </>
        ) : (
          <div className="flex items-center justify-center h-40 text-gray-500">
            {loading ? 'Checking services...' : 'Unable to connect to services'}
          </div>
        )}
        {lastUpdated && (
          <p className="text-xs text-gray-400 text-center">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </p>
        )}
      </CardBody>
    </Card>
  );
}
