'use client';

import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardBody } from '@/components/ui';
import { Button } from '@/components/ui';
import { Badge } from '@/components/ui';
import {
  Settings,
  Server,
  Key,
  Database,
  Brain,
  Bell,
  Shield,
  Palette,
  Save,
  RefreshCw,
  ExternalLink,
} from 'lucide-react';

export default function SettingsPage() {
  const [saving, setSaving] = useState(false);

  const handleSave = () => {
    setSaving(true);
    setTimeout(() => setSaving(false), 1500);
  };

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
          <p className="text-gray-500 mt-1">
            Configure platform settings and integrations
          </p>
        </div>
        <Button icon={<Save className="w-4 h-4" />} loading={saving} onClick={handleSave}>
          Save Changes
        </Button>
      </div>

      {/* API Configuration */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Server className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <CardTitle>API Configuration</CardTitle>
              <CardDescription>Backend service endpoints and settings</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardBody className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              API Gateway URL
            </label>
            <input
              type="text"
              defaultValue="http://localhost:8080"
              className="input w-full"
            />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Manager Service
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  defaultValue="http://localhost:8001"
                  className="input flex-1"
                />
                <Badge variant="success">Online</Badge>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Inference Service
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  defaultValue="http://localhost:8002"
                  className="input flex-1"
                />
                <Badge variant="success">Online</Badge>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Data Service
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  defaultValue="http://localhost:8003"
                  className="input flex-1"
                />
                <Badge variant="success">Online</Badge>
              </div>
            </div>
          </div>
        </CardBody>
      </Card>

      {/* API Keys */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="p-2 bg-amber-100 rounded-lg">
              <Key className="w-5 h-5 text-amber-600" />
            </div>
            <div>
              <CardTitle>API Keys</CardTitle>
              <CardDescription>Manage external service API keys</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardBody className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              OpenAI API Key
            </label>
            <div className="flex items-center gap-2">
              <input
                type="password"
                defaultValue="sk-***************************"
                className="input flex-1"
              />
              <Badge variant="success">Valid</Badge>
            </div>
            <p className="text-xs text-gray-500 mt-1">Used for GPT-4 teacher model in PsychSteer</p>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Anthropic API Key
            </label>
            <div className="flex items-center gap-2">
              <input
                type="password"
                placeholder="sk-ant-***"
                className="input flex-1"
              />
              <Badge variant="warning">Not set</Badge>
            </div>
            <p className="text-xs text-gray-500 mt-1">Used for Claude teacher model in PsychSteer</p>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Hugging Face Token
            </label>
            <div className="flex items-center gap-2">
              <input
                type="password"
                defaultValue="hf_***************************"
                className="input flex-1"
              />
              <Badge variant="success">Valid</Badge>
            </div>
            <p className="text-xs text-gray-500 mt-1">Used for downloading Llama 3.2 model</p>
          </div>
        </CardBody>
      </Card>

      {/* Model Settings */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Brain className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <CardTitle>Model Settings</CardTitle>
              <CardDescription>Configure inference and training parameters</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardBody className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Default Base Model
              </label>
              <select className="input w-full">
                <option>meta-llama/Llama-3.2-1B</option>
                <option>meta-llama/Llama-3.2-3B</option>
                <option>mistralai/Mistral-7B-v0.1</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Teacher Model (PsychSteer)
              </label>
              <select className="input w-full">
                <option>gpt-4o</option>
                <option>gpt-4-turbo</option>
                <option>claude-3.5-sonnet</option>
                <option>claude-opus-4.5</option>
              </select>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                LM Loss Weight
              </label>
              <input type="number" defaultValue="0.6" step="0.1" className="input w-full" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Personality Loss Weight
              </label>
              <input type="number" defaultValue="0.4" step="0.1" className="input w-full" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Car Domain Boost
              </label>
              <input type="number" defaultValue="1.2" step="0.1" className="input w-full" />
            </div>
          </div>
        </CardBody>
      </Card>

      {/* Database Settings */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="p-2 bg-green-100 rounded-lg">
              <Database className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <CardTitle>Database Settings</CardTitle>
              <CardDescription>PostgreSQL and ChromaDB configuration</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardBody className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              PostgreSQL Connection
            </label>
            <input
              type="text"
              defaultValue="postgresql://postgres:***@localhost:5432/carpsycho"
              className="input w-full"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              ChromaDB URL
            </label>
            <input
              type="text"
              defaultValue="http://localhost:8000"
              className="input w-full"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Redis URL
            </label>
            <input
              type="text"
              defaultValue="redis://localhost:6379"
              className="input w-full"
            />
          </div>
        </CardBody>
      </Card>

      {/* T4 VM Configuration */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="p-2 bg-rose-100 rounded-lg">
              <Server className="w-5 h-5 text-rose-600" />
            </div>
            <div>
              <CardTitle>T4 VM Configuration</CardTitle>
              <CardDescription>Remote GPU training environment settings</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardBody className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                VM Host
              </label>
              <input
                type="text"
                placeholder="your-vm-ip-or-hostname"
                className="input w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                SSH User
              </label>
              <input
                type="text"
                defaultValue="ubuntu"
                className="input w-full"
              />
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              SSH Key Path
            </label>
            <input
              type="text"
              placeholder="~/.ssh/t4_vm_key"
              className="input w-full"
            />
          </div>
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div>
              <p className="font-medium text-gray-900">VM Status</p>
              <p className="text-sm text-gray-500">Check connection to T4 VM</p>
            </div>
            <Button variant="secondary" size="sm" icon={<RefreshCw className="w-4 h-4" />}>
              Test Connection
            </Button>
          </div>
        </CardBody>
      </Card>

      {/* Notifications */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyan-100 rounded-lg">
              <Bell className="w-5 h-5 text-cyan-600" />
            </div>
            <div>
              <CardTitle>Notifications</CardTitle>
              <CardDescription>Configure alerts and notifications</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardBody className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900">Training Job Completion</p>
              <p className="text-sm text-gray-500">Get notified when training jobs finish</p>
            </div>
            <input type="checkbox" defaultChecked className="w-5 h-5 rounded text-primary-600" />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900">New Customer Conversation</p>
              <p className="text-sm text-gray-500">Alert when new conversations start</p>
            </div>
            <input type="checkbox" defaultChecked className="w-5 h-5 rounded text-primary-600" />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900">Service Health Alerts</p>
              <p className="text-sm text-gray-500">Notify on service downtime</p>
            </div>
            <input type="checkbox" defaultChecked className="w-5 h-5 rounded text-primary-600" />
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
