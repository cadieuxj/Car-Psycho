'use client';

import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardBody } from '@/components/ui';
import { Button } from '@/components/ui';
import { Badge } from '@/components/ui';
import { Progress } from '@/components/ui';
import { TrainingJob, TrainingConfig } from '@/lib/types';
import { formatDateTime } from '@/lib/utils';
import {
  Cpu,
  Play,
  Pause,
  Square,
  RefreshCw,
  Settings,
  ChevronRight,
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  Plus,
  Server,
} from 'lucide-react';

// Sample training jobs
const sampleJobs: TrainingJob[] = [
  {
    id: '1',
    name: 'Llama-3.2-1B-CarPsycho-v2',
    status: 'running',
    progress: 67,
    model_base: 'meta-llama/Llama-3.2-1B',
    dataset_id: 'dataset-001',
    created_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date().toISOString(),
    started_at: new Date(Date.now() - 1.5 * 60 * 60 * 1000).toISOString(),
    metrics: {
      loss: 0.342,
      personality_loss: 0.128,
      lm_loss: 0.214,
      epoch: 2,
      step: 1250,
      learning_rate: 0.0002,
    },
  },
  {
    id: '2',
    name: 'Llama-3.2-1B-CarPsycho-v1',
    status: 'completed',
    progress: 100,
    model_base: 'meta-llama/Llama-3.2-1B',
    dataset_id: 'dataset-001',
    created_at: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 20 * 60 * 60 * 1000).toISOString(),
    started_at: new Date(Date.now() - 23 * 60 * 60 * 1000).toISOString(),
    completed_at: new Date(Date.now() - 20 * 60 * 60 * 1000).toISOString(),
    metrics: {
      loss: 0.285,
      personality_loss: 0.095,
      lm_loss: 0.190,
      epoch: 3,
      step: 2000,
      learning_rate: 0.0001,
    },
  },
  {
    id: '3',
    name: 'Experimental-HighLR',
    status: 'failed',
    progress: 34,
    model_base: 'meta-llama/Llama-3.2-1B',
    dataset_id: 'dataset-002',
    created_at: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 46 * 60 * 60 * 1000).toISOString(),
    started_at: new Date(Date.now() - 47 * 60 * 60 * 1000).toISOString(),
    error_message: 'CUDA out of memory. Reduce batch size.',
  },
];

const statusConfig = {
  pending: { icon: Clock, color: 'text-gray-500', bg: 'bg-gray-100', badge: 'default' as const },
  running: { icon: Loader2, color: 'text-blue-500', bg: 'bg-blue-100', badge: 'info' as const },
  completed: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-100', badge: 'success' as const },
  failed: { icon: XCircle, color: 'text-red-500', bg: 'bg-red-100', badge: 'error' as const },
  cancelled: { icon: Square, color: 'text-gray-500', bg: 'bg-gray-100', badge: 'default' as const },
};

interface JobCardProps {
  job: TrainingJob;
  onSelect: () => void;
  selected: boolean;
}

function JobCard({ job, onSelect, selected }: JobCardProps) {
  const config = statusConfig[job.status];
  const Icon = config.icon;

  return (
    <button
      onClick={onSelect}
      className={`w-full p-4 text-left border rounded-lg transition-all ${
        selected
          ? 'border-primary-500 bg-primary-50 ring-2 ring-primary-200'
          : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
      }`}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${config.bg}`}>
            <Icon className={`w-4 h-4 ${config.color} ${job.status === 'running' ? 'animate-spin' : ''}`} />
          </div>
          <div>
            <h4 className="font-medium text-gray-900">{job.name}</h4>
            <p className="text-xs text-gray-500">{job.model_base}</p>
          </div>
        </div>
        <Badge variant={config.badge}>{job.status}</Badge>
      </div>

      {(job.status === 'running' || job.status === 'completed') && (
        <Progress
          value={job.progress}
          size="sm"
          color={job.status === 'completed' ? 'bg-green-500' : 'bg-blue-500'}
        />
      )}

      {job.error_message && (
        <p className="text-xs text-red-600 mt-2 truncate">{job.error_message}</p>
      )}

      <div className="flex items-center justify-between mt-3 text-xs text-gray-400">
        <span>Created: {formatDateTime(job.created_at)}</span>
        <ChevronRight className="w-4 h-4" />
      </div>
    </button>
  );
}

export default function TrainingPage() {
  const [jobs] = useState<TrainingJob[]>(sampleJobs);
  const [selectedJob, setSelectedJob] = useState<TrainingJob | null>(sampleJobs[0]);
  const [showNewJobModal, setShowNewJobModal] = useState(false);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Model Training</h1>
          <p className="text-gray-500 mt-1">
            Manage and monitor training jobs for psychometric models
          </p>
        </div>
        <Button icon={<Plus className="w-4 h-4" />} onClick={() => setShowNewJobModal(true)}>
          New Training Job
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Jobs List */}
        <div className="lg:col-span-1 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Training Jobs</CardTitle>
              <CardDescription>Select a job to view details</CardDescription>
            </CardHeader>
            <CardBody className="space-y-3">
              {jobs.map((job) => (
                <JobCard
                  key={job.id}
                  job={job}
                  onSelect={() => setSelectedJob(job)}
                  selected={selectedJob?.id === job.id}
                />
              ))}
            </CardBody>
          </Card>
        </div>

        {/* Job Details */}
        <div className="lg:col-span-2 space-y-6">
          {selectedJob ? (
            <>
              {/* Job Overview */}
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>{selectedJob.name}</CardTitle>
                      <CardDescription>{selectedJob.model_base}</CardDescription>
                    </div>
                    <div className="flex items-center gap-2">
                      {selectedJob.status === 'running' && (
                        <>
                          <Button variant="secondary" size="sm" icon={<Pause className="w-4 h-4" />}>
                            Pause
                          </Button>
                          <Button variant="danger" size="sm" icon={<Square className="w-4 h-4" />}>
                            Stop
                          </Button>
                        </>
                      )}
                      {selectedJob.status === 'pending' && (
                        <Button size="sm" icon={<Play className="w-4 h-4" />}>
                          Start
                        </Button>
                      )}
                      {(selectedJob.status === 'completed' || selectedJob.status === 'failed') && (
                        <Button variant="secondary" size="sm" icon={<RefreshCw className="w-4 h-4" />}>
                          Retry
                        </Button>
                      )}
                    </div>
                  </div>
                </CardHeader>
                <CardBody>
                  <div className="mb-6">
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-gray-500">Progress</span>
                      <span className="font-medium">{selectedJob.progress}%</span>
                    </div>
                    <Progress value={selectedJob.progress} size="lg" showLabel={false} />
                  </div>

                  {selectedJob.error_message && (
                    <div className="p-4 bg-red-50 border border-red-200 rounded-lg mb-6">
                      <p className="text-sm text-red-700">{selectedJob.error_message}</p>
                    </div>
                  )}

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <p className="text-xs text-gray-500 mb-1">Status</p>
                      <Badge variant={statusConfig[selectedJob.status].badge}>
                        {selectedJob.status}
                      </Badge>
                    </div>
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <p className="text-xs text-gray-500 mb-1">Dataset</p>
                      <p className="text-sm font-medium">{selectedJob.dataset_id}</p>
                    </div>
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <p className="text-xs text-gray-500 mb-1">Started</p>
                      <p className="text-sm font-medium">
                        {selectedJob.started_at ? formatDateTime(selectedJob.started_at) : '-'}
                      </p>
                    </div>
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <p className="text-xs text-gray-500 mb-1">Duration</p>
                      <p className="text-sm font-medium">
                        {selectedJob.started_at
                          ? `${Math.round(
                              (new Date(selectedJob.completed_at || new Date()).getTime() -
                                new Date(selectedJob.started_at).getTime()) /
                                1000 /
                                60
                            )} min`
                          : '-'}
                      </p>
                    </div>
                  </div>
                </CardBody>
              </Card>

              {/* Training Metrics */}
              {selectedJob.metrics && (
                <Card>
                  <CardHeader>
                    <CardTitle>Training Metrics</CardTitle>
                    <CardDescription>Real-time training statistics</CardDescription>
                  </CardHeader>
                  <CardBody>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      <div className="p-4 bg-blue-50 rounded-lg">
                        <p className="text-xs text-blue-600 mb-1">Total Loss</p>
                        <p className="text-2xl font-bold text-blue-900">
                          {selectedJob.metrics.loss.toFixed(4)}
                        </p>
                      </div>
                      <div className="p-4 bg-purple-50 rounded-lg">
                        <p className="text-xs text-purple-600 mb-1">Personality Loss</p>
                        <p className="text-2xl font-bold text-purple-900">
                          {selectedJob.metrics.personality_loss.toFixed(4)}
                        </p>
                      </div>
                      <div className="p-4 bg-amber-50 rounded-lg">
                        <p className="text-xs text-amber-600 mb-1">LM Loss</p>
                        <p className="text-2xl font-bold text-amber-900">
                          {selectedJob.metrics.lm_loss.toFixed(4)}
                        </p>
                      </div>
                      <div className="p-4 bg-gray-50 rounded-lg">
                        <p className="text-xs text-gray-500 mb-1">Current Epoch</p>
                        <p className="text-2xl font-bold text-gray-900">
                          {selectedJob.metrics.epoch}/3
                        </p>
                      </div>
                      <div className="p-4 bg-gray-50 rounded-lg">
                        <p className="text-xs text-gray-500 mb-1">Current Step</p>
                        <p className="text-2xl font-bold text-gray-900">
                          {selectedJob.metrics.step.toLocaleString()}
                        </p>
                      </div>
                      <div className="p-4 bg-gray-50 rounded-lg">
                        <p className="text-xs text-gray-500 mb-1">Learning Rate</p>
                        <p className="text-2xl font-bold text-gray-900">
                          {selectedJob.metrics.learning_rate.toExponential(1)}
                        </p>
                      </div>
                    </div>
                  </CardBody>
                </Card>
              )}

              {/* Training Configuration */}
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>Configuration</CardTitle>
                    <Button variant="ghost" size="sm" icon={<Settings className="w-4 h-4" />}>
                      Edit
                    </Button>
                  </div>
                </CardHeader>
                <CardBody>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <p className="text-gray-500">Base Model</p>
                      <p className="font-medium">{selectedJob.model_base}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Epochs</p>
                      <p className="font-medium">3</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Batch Size</p>
                      <p className="font-medium">4</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Learning Rate</p>
                      <p className="font-medium">2e-4</p>
                    </div>
                    <div>
                      <p className="text-gray-500">LM Loss Weight</p>
                      <p className="font-medium">0.6</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Personality Loss Weight</p>
                      <p className="font-medium">0.4</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Ordinal Loss Weight</p>
                      <p className="font-medium">0.7</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Car Domain Boost</p>
                      <p className="font-medium">1.2x</p>
                    </div>
                  </div>
                </CardBody>
              </Card>
            </>
          ) : (
            <Card>
              <CardBody className="flex flex-col items-center justify-center h-64 text-gray-500">
                <Cpu className="w-12 h-12 mb-4 text-gray-300" />
                <p>Select a training job to view details</p>
              </CardBody>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
