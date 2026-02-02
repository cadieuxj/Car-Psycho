'use client';

import { Card, CardBody } from '@/components/ui';
import { Users, MessageSquare, Car, TrendingUp, Brain, Database } from 'lucide-react';
import { cn } from '@/lib/utils';

interface StatCardProps {
  title: string;
  value: string | number;
  change?: string;
  changeType?: 'positive' | 'negative' | 'neutral';
  icon: React.ReactNode;
  color: string;
}

function StatCard({ title, value, change, changeType = 'neutral', icon, color }: StatCardProps) {
  return (
    <Card className="relative overflow-hidden">
      <CardBody className="flex items-center gap-4">
        <div className={cn('p-3 rounded-xl', color)}>
          {icon}
        </div>
        <div className="flex-1">
          <p className="text-sm text-gray-500">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          {change && (
            <p className={cn(
              'text-xs mt-1',
              changeType === 'positive' && 'text-green-600',
              changeType === 'negative' && 'text-red-600',
              changeType === 'neutral' && 'text-gray-500'
            )}>
              {change}
            </p>
          )}
        </div>
      </CardBody>
    </Card>
  );
}

export default function StatsCards() {
  const stats: StatCardProps[] = [
    {
      title: 'Total Customers',
      value: '1,284',
      change: '+12% from last month',
      changeType: 'positive',
      icon: <Users className="w-6 h-6 text-blue-600" />,
      color: 'bg-blue-100',
    },
    {
      title: 'Active Conversations',
      value: '23',
      change: '8 new today',
      changeType: 'positive',
      icon: <MessageSquare className="w-6 h-6 text-purple-600" />,
      color: 'bg-purple-100',
    },
    {
      title: 'Cars Recommended',
      value: '456',
      change: '+28% match rate',
      changeType: 'positive',
      icon: <Car className="w-6 h-6 text-amber-600" />,
      color: 'bg-amber-100',
    },
    {
      title: 'Profiles Analyzed',
      value: '892',
      change: 'Avg confidence: 87%',
      changeType: 'neutral',
      icon: <Brain className="w-6 h-6 text-emerald-600" />,
      color: 'bg-emerald-100',
    },
    {
      title: 'Training Jobs',
      value: '3',
      change: '1 running',
      changeType: 'neutral',
      icon: <TrendingUp className="w-6 h-6 text-rose-600" />,
      color: 'bg-rose-100',
    },
    {
      title: 'Dataset Samples',
      value: '15.2K',
      change: '40/40/20 mix',
      changeType: 'neutral',
      icon: <Database className="w-6 h-6 text-cyan-600" />,
      color: 'bg-cyan-100',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {stats.map((stat) => (
        <StatCard key={stat.title} {...stat} />
      ))}
    </div>
  );
}
