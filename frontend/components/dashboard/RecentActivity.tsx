'use client';

import { Card, CardHeader, CardTitle, CardBody } from '@/components/ui';
import { Badge } from '@/components/ui';
import { MessageSquare, Brain, Car, Database, Cpu, User } from 'lucide-react';
import { formatDateTime } from '@/lib/utils';

interface Activity {
  id: string;
  type: 'conversation' | 'profile' | 'recommendation' | 'training' | 'dataset';
  title: string;
  description: string;
  timestamp: string;
  status?: 'success' | 'pending' | 'error';
}

const activities: Activity[] = [
  {
    id: '1',
    type: 'conversation',
    title: 'New Customer Conversation',
    description: 'Customer discussing SUV preferences',
    timestamp: new Date(Date.now() - 5 * 60000).toISOString(),
    status: 'success',
  },
  {
    id: '2',
    type: 'profile',
    title: 'Personality Profile Generated',
    description: 'High Conscientiousness (0.85), Moderate Openness (0.62)',
    timestamp: new Date(Date.now() - 15 * 60000).toISOString(),
    status: 'success',
  },
  {
    id: '3',
    type: 'recommendation',
    title: 'Car Recommendations Sent',
    description: '3 vehicles matched: Toyota RAV4, Honda CR-V, Mazda CX-5',
    timestamp: new Date(Date.now() - 30 * 60000).toISOString(),
    status: 'success',
  },
  {
    id: '4',
    type: 'training',
    title: 'Training Job Started',
    description: 'Fine-tuning Llama 3.2 1B with ordinal regression',
    timestamp: new Date(Date.now() - 60 * 60000).toISOString(),
    status: 'pending',
  },
  {
    id: '5',
    type: 'dataset',
    title: 'Dataset Generated',
    description: '1,000 samples via PsychSteer (GPT-4o)',
    timestamp: new Date(Date.now() - 120 * 60000).toISOString(),
    status: 'success',
  },
];

const iconMap = {
  conversation: MessageSquare,
  profile: Brain,
  recommendation: Car,
  training: Cpu,
  dataset: Database,
};

const colorMap = {
  conversation: 'bg-purple-100 text-purple-600',
  profile: 'bg-blue-100 text-blue-600',
  recommendation: 'bg-amber-100 text-amber-600',
  training: 'bg-rose-100 text-rose-600',
  dataset: 'bg-cyan-100 text-cyan-600',
};

export default function RecentActivity() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Activity</CardTitle>
      </CardHeader>
      <CardBody className="p-0">
        <div className="divide-y divide-gray-100">
          {activities.map((activity) => {
            const Icon = iconMap[activity.type];
            return (
              <div key={activity.id} className="flex items-start gap-4 p-4 hover:bg-gray-50 transition-colors">
                <div className={`p-2 rounded-lg ${colorMap[activity.type]}`}>
                  <Icon className="w-4 h-4" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium text-gray-900">{activity.title}</p>
                    {activity.status && (
                      <Badge
                        variant={
                          activity.status === 'success'
                            ? 'success'
                            : activity.status === 'pending'
                            ? 'warning'
                            : 'error'
                        }
                      >
                        {activity.status}
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm text-gray-500 truncate">{activity.description}</p>
                  <p className="text-xs text-gray-400 mt-1">
                    {formatDateTime(activity.timestamp)}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </CardBody>
    </Card>
  );
}
