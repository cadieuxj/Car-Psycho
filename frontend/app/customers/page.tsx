'use client';

import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardBody } from '@/components/ui';
import { Button } from '@/components/ui';
import { Badge } from '@/components/ui';
import { CustomerProfile, OceanScores } from '@/lib/types';
import { formatDateTime, OCEAN_LABELS, OCEAN_COLORS } from '@/lib/utils';
import { TraitBars } from '@/components/psychometrics';
import {
  Users,
  Search,
  Filter,
  Plus,
  User,
  Mail,
  Phone,
  MessageSquare,
  Car,
  Calendar,
  MoreHorizontal,
  ChevronRight,
  Brain,
} from 'lucide-react';

// Sample customers
const sampleCustomers: CustomerProfile[] = [
  {
    id: '1',
    name: 'John Davidson',
    email: 'john.d@email.com',
    phone: '(555) 123-4567',
    personality_profile: {
      scores: { openness: 0.72, conscientiousness: 0.85, extraversion: 0.58, agreeableness: 0.76, neuroticism: 0.32 },
      confidence: { openness: 0.88, conscientiousness: 0.92, extraversion: 0.85, agreeableness: 0.90, neuroticism: 0.87 },
      timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    },
    conversation_count: 3,
    last_interaction: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    created_at: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: '2',
    name: 'Sarah Mitchell',
    email: 'sarah.m@email.com',
    phone: '(555) 234-5678',
    personality_profile: {
      scores: { openness: 0.88, conscientiousness: 0.62, extraversion: 0.78, agreeableness: 0.55, neuroticism: 0.45 },
      confidence: { openness: 0.85, conscientiousness: 0.78, extraversion: 0.90, agreeableness: 0.82, neuroticism: 0.75 },
      timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
    },
    conversation_count: 5,
    last_interaction: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
    created_at: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: '3',
    name: 'Michael Roberts',
    email: 'mike.r@email.com',
    phone: '(555) 345-6789',
    personality_profile: {
      scores: { openness: 0.52, conscientiousness: 0.92, extraversion: 0.38, agreeableness: 0.84, neuroticism: 0.28 },
      confidence: { openness: 0.82, conscientiousness: 0.95, extraversion: 0.78, agreeableness: 0.88, neuroticism: 0.85 },
      timestamp: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
    },
    conversation_count: 2,
    last_interaction: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
    created_at: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: '4',
    name: 'Emily Chen',
    email: 'emily.c@email.com',
    phone: '(555) 456-7890',
    conversation_count: 1,
    last_interaction: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    created_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: '5',
    name: 'David Thompson',
    email: 'david.t@email.com',
    phone: '(555) 567-8901',
    personality_profile: {
      scores: { openness: 0.65, conscientiousness: 0.70, extraversion: 0.82, agreeableness: 0.68, neuroticism: 0.42 },
      confidence: { openness: 0.80, conscientiousness: 0.85, extraversion: 0.92, agreeableness: 0.78, neuroticism: 0.80 },
      timestamp: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
    },
    conversation_count: 4,
    last_interaction: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
    created_at: new Date(Date.now() - 21 * 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
  },
];

function getTopTraits(scores: OceanScores): (keyof OceanScores)[] {
  return Object.entries(scores)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 2)
    .map(([key]) => key as keyof OceanScores);
}

export default function CustomersPage() {
  const [customers] = useState<CustomerProfile[]>(sampleCustomers);
  const [selectedCustomer, setSelectedCustomer] = useState<CustomerProfile | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  const filteredCustomers = customers.filter(
    (c) =>
      c.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      c.email?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Customers</h1>
          <p className="text-gray-500 mt-1">
            Manage customer profiles and psychometric data
          </p>
        </div>
        <Button icon={<Plus className="w-4 h-4" />}>Add Customer</Button>
      </div>

      {/* Search and Filter */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search by name or email..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="input pl-10 w-full"
          />
        </div>
        <Button variant="secondary" icon={<Filter className="w-4 h-4" />}>
          Filter
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Customer List */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>All Customers ({filteredCustomers.length})</CardTitle>
              </div>
            </CardHeader>
            <CardBody className="p-0">
              <div className="divide-y divide-gray-100">
                {filteredCustomers.map((customer) => (
                  <button
                    key={customer.id}
                    onClick={() => setSelectedCustomer(customer)}
                    className={`w-full p-4 text-left hover:bg-gray-50 transition-colors ${
                      selectedCustomer?.id === customer.id ? 'bg-primary-50' : ''
                    }`}
                  >
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center text-white font-semibold">
                        {customer.name?.charAt(0) || '?'}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-1">
                          <h4 className="font-medium text-gray-900">{customer.name || 'Unknown'}</h4>
                          {customer.personality_profile && (
                            <div className="flex items-center gap-1">
                              {getTopTraits(customer.personality_profile.scores).map((trait) => (
                                <div
                                  key={trait}
                                  className="w-5 h-5 rounded-full flex items-center justify-center text-white text-xs font-bold"
                                  style={{ backgroundColor: OCEAN_COLORS[trait] }}
                                  title={OCEAN_LABELS[trait]}
                                >
                                  {trait.charAt(0).toUpperCase()}
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                        <div className="flex items-center gap-4 text-sm text-gray-500">
                          <span className="flex items-center gap-1">
                            <Mail className="w-3 h-3" />
                            {customer.email}
                          </span>
                          <span className="flex items-center gap-1">
                            <MessageSquare className="w-3 h-3" />
                            {customer.conversation_count} chats
                          </span>
                        </div>
                        <p className="text-xs text-gray-400 mt-1">
                          Last active: {formatDateTime(customer.last_interaction || customer.updated_at)}
                        </p>
                      </div>
                      <ChevronRight className="w-5 h-5 text-gray-300" />
                    </div>
                  </button>
                ))}
              </div>
            </CardBody>
          </Card>
        </div>

        {/* Customer Details */}
        <div className="lg:col-span-1">
          {selectedCustomer ? (
            <div className="space-y-6">
              {/* Profile Card */}
              <Card>
                <CardBody className="text-center">
                  <div className="w-20 h-20 mx-auto rounded-full bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center text-white text-2xl font-bold mb-4">
                    {selectedCustomer.name?.charAt(0) || '?'}
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900">
                    {selectedCustomer.name || 'Unknown'}
                  </h3>
                  <div className="mt-4 space-y-2 text-sm text-gray-600">
                    {selectedCustomer.email && (
                      <div className="flex items-center justify-center gap-2">
                        <Mail className="w-4 h-4 text-gray-400" />
                        {selectedCustomer.email}
                      </div>
                    )}
                    {selectedCustomer.phone && (
                      <div className="flex items-center justify-center gap-2">
                        <Phone className="w-4 h-4 text-gray-400" />
                        {selectedCustomer.phone}
                      </div>
                    )}
                  </div>
                  <div className="flex gap-2 mt-4">
                    <Button size="sm" className="flex-1" icon={<MessageSquare className="w-4 h-4" />}>
                      Chat
                    </Button>
                    <Button size="sm" variant="secondary" className="flex-1" icon={<Car className="w-4 h-4" />}>
                      Recommend
                    </Button>
                  </div>
                </CardBody>
              </Card>

              {/* Personality Profile */}
              {selectedCustomer.personality_profile ? (
                <Card>
                  <CardHeader>
                    <div className="flex items-center gap-2">
                      <Brain className="w-4 h-4 text-purple-600" />
                      <CardTitle className="text-base">Personality Profile</CardTitle>
                    </div>
                  </CardHeader>
                  <CardBody>
                    <TraitBars
                      scores={selectedCustomer.personality_profile.scores}
                      confidence={selectedCustomer.personality_profile.confidence}
                    />
                    <p className="text-xs text-gray-400 mt-4 text-center">
                      Last updated: {formatDateTime(selectedCustomer.personality_profile.timestamp)}
                    </p>
                  </CardBody>
                </Card>
              ) : (
                <Card>
                  <CardBody className="text-center py-8">
                    <Brain className="w-12 h-12 mx-auto text-gray-300 mb-4" />
                    <p className="text-gray-500">No personality profile yet</p>
                    <p className="text-sm text-gray-400 mt-1">
                      Start a conversation to analyze personality
                    </p>
                    <Button size="sm" className="mt-4" icon={<MessageSquare className="w-4 h-4" />}>
                      Start Chat
                    </Button>
                  </CardBody>
                </Card>
              )}

              {/* Activity */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Activity</CardTitle>
                </CardHeader>
                <CardBody>
                  <div className="space-y-3 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">Conversations</span>
                      <span className="font-medium">{selectedCustomer.conversation_count}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">Customer since</span>
                      <span className="font-medium">{formatDateTime(selectedCustomer.created_at)}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">Last active</span>
                      <span className="font-medium">
                        {formatDateTime(selectedCustomer.last_interaction || selectedCustomer.updated_at)}
                      </span>
                    </div>
                  </div>
                </CardBody>
              </Card>
            </div>
          ) : (
            <Card>
              <CardBody className="flex flex-col items-center justify-center h-64 text-gray-500">
                <User className="w-12 h-12 mb-4 text-gray-300" />
                <p>Select a customer to view details</p>
              </CardBody>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
