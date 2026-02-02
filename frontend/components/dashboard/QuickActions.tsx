'use client';

import Link from 'next/link';
import { Card, CardHeader, CardTitle, CardBody } from '@/components/ui';
import { Button } from '@/components/ui';
import { MessageSquare, UserPlus, Database, Cpu, ArrowRight } from 'lucide-react';

interface QuickAction {
  title: string;
  description: string;
  icon: React.ReactNode;
  href: string;
  color: string;
}

const quickActions: QuickAction[] = [
  {
    title: 'Start Conversation',
    description: 'Begin a new sales conversation with AI assistance',
    icon: <MessageSquare className="w-5 h-5" />,
    href: '/assistant',
    color: 'bg-primary-600 hover:bg-primary-700',
  },
  {
    title: 'Add Customer',
    description: 'Create a new customer profile',
    icon: <UserPlus className="w-5 h-5" />,
    href: '/customers/new',
    color: 'bg-emerald-600 hover:bg-emerald-700',
  },
  {
    title: 'Generate Data',
    description: 'Create synthetic training data with PsychSteer',
    icon: <Database className="w-5 h-5" />,
    href: '/datasets',
    color: 'bg-purple-600 hover:bg-purple-700',
  },
  {
    title: 'Start Training',
    description: 'Launch a new model training job',
    icon: <Cpu className="w-5 h-5" />,
    href: '/training',
    color: 'bg-amber-600 hover:bg-amber-700',
  },
];

export default function QuickActions() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Quick Actions</CardTitle>
      </CardHeader>
      <CardBody className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {quickActions.map((action) => (
          <Link key={action.title} href={action.href}>
            <div className="group p-4 border border-gray-200 rounded-lg hover:border-primary-300 hover:shadow-sm transition-all cursor-pointer">
              <div className="flex items-start gap-3">
                <div className={`p-2 rounded-lg text-white ${action.color}`}>
                  {action.icon}
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <h4 className="font-medium text-gray-900">{action.title}</h4>
                    <ArrowRight className="w-4 h-4 text-gray-400 group-hover:text-primary-600 group-hover:translate-x-1 transition-all" />
                  </div>
                  <p className="text-sm text-gray-500 mt-1">{action.description}</p>
                </div>
              </div>
            </div>
          </Link>
        ))}
      </CardBody>
    </Card>
  );
}
