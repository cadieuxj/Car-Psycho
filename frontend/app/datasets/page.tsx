'use client';

import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardBody } from '@/components/ui';
import { Button } from '@/components/ui';
import { Badge } from '@/components/ui';
import { Progress } from '@/components/ui';
import { Dataset } from '@/lib/types';
import { formatDateTime } from '@/lib/utils';
import {
  Database,
  Plus,
  Download,
  Upload,
  RefreshCw,
  Trash2,
  FileJson,
  BarChart3,
  Sparkles,
} from 'lucide-react';

// Sample datasets
const sampleDatasets: Dataset[] = [
  {
    id: 'dataset-001',
    name: 'CarPsycho-Train-v1',
    description: 'Primary training dataset with 40/40/20 mix',
    total_samples: 15000,
    public_general_samples: 6000,
    car_domain_samples: 6000,
    synthetic_samples: 3000,
    teacher_model: 'gpt-4o',
    created_at: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'dataset-002',
    name: 'CarPsycho-Eval-v1',
    description: 'Evaluation dataset for model validation',
    total_samples: 2000,
    public_general_samples: 800,
    car_domain_samples: 800,
    synthetic_samples: 400,
    teacher_model: 'claude-3.5-sonnet',
    created_at: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'dataset-003',
    name: 'Synthetic-Test-Batch',
    description: 'Experimental synthetic-only dataset',
    total_samples: 5000,
    public_general_samples: 0,
    car_domain_samples: 0,
    synthetic_samples: 5000,
    teacher_model: 'gpt-4o',
    created_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
  },
];

export default function DatasetsPage() {
  const [datasets] = useState<Dataset[]>(sampleDatasets);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(sampleDatasets[0]);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerate = () => {
    setIsGenerating(true);
    setTimeout(() => setIsGenerating(false), 3000);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Datasets</h1>
          <p className="text-gray-500 mt-1">
            Manage training data with PsychSteer labeling and RegMix mixing
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="secondary" icon={<Upload className="w-4 h-4" />}>
            Import
          </Button>
          <Button icon={<Plus className="w-4 h-4" />} onClick={handleGenerate} loading={isGenerating}>
            Generate Dataset
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardBody className="flex items-center gap-4">
            <div className="p-3 bg-blue-100 rounded-xl">
              <Database className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Total Samples</p>
              <p className="text-2xl font-bold text-gray-900">
                {datasets.reduce((acc, d) => acc + d.total_samples, 0).toLocaleString()}
              </p>
            </div>
          </CardBody>
        </Card>
        <Card>
          <CardBody className="flex items-center gap-4">
            <div className="p-3 bg-green-100 rounded-xl">
              <FileJson className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Datasets</p>
              <p className="text-2xl font-bold text-gray-900">{datasets.length}</p>
            </div>
          </CardBody>
        </Card>
        <Card>
          <CardBody className="flex items-center gap-4">
            <div className="p-3 bg-purple-100 rounded-xl">
              <Sparkles className="w-6 h-6 text-purple-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Synthetic Data</p>
              <p className="text-2xl font-bold text-gray-900">
                {datasets.reduce((acc, d) => acc + d.synthetic_samples, 0).toLocaleString()}
              </p>
            </div>
          </CardBody>
        </Card>
        <Card>
          <CardBody className="flex items-center gap-4">
            <div className="p-3 bg-amber-100 rounded-xl">
              <BarChart3 className="w-6 h-6 text-amber-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Avg Mix Ratio</p>
              <p className="text-2xl font-bold text-gray-900">40/40/20</p>
            </div>
          </CardBody>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Dataset List */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>All Datasets</CardTitle>
            </CardHeader>
            <CardBody className="p-0">
              <div className="divide-y divide-gray-100">
                {datasets.map((dataset) => (
                  <button
                    key={dataset.id}
                    onClick={() => setSelectedDataset(dataset)}
                    className={`w-full p-4 text-left hover:bg-gray-50 transition-colors ${
                      selectedDataset?.id === dataset.id ? 'bg-primary-50 border-l-4 border-l-primary-600' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-gray-900">{dataset.name}</h4>
                      <Badge variant="info">{dataset.total_samples.toLocaleString()}</Badge>
                    </div>
                    <p className="text-sm text-gray-500 mb-2">{dataset.description}</p>
                    <p className="text-xs text-gray-400">
                      Created: {formatDateTime(dataset.created_at)}
                    </p>
                  </button>
                ))}
              </div>
            </CardBody>
          </Card>
        </div>

        {/* Dataset Details */}
        <div className="lg:col-span-2 space-y-6">
          {selectedDataset ? (
            <>
              {/* Overview */}
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>{selectedDataset.name}</CardTitle>
                      <CardDescription>{selectedDataset.description}</CardDescription>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button variant="secondary" size="sm" icon={<Download className="w-4 h-4" />}>
                        Export
                      </Button>
                      <Button variant="ghost" size="sm" icon={<Trash2 className="w-4 h-4" />}>
                        Delete
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardBody>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <p className="text-2xl font-bold text-gray-900">
                        {selectedDataset.total_samples.toLocaleString()}
                      </p>
                      <p className="text-sm text-gray-500">Total Samples</p>
                    </div>
                    <div className="p-4 bg-blue-50 rounded-lg">
                      <p className="text-2xl font-bold text-blue-900">
                        {selectedDataset.public_general_samples.toLocaleString()}
                      </p>
                      <p className="text-sm text-blue-600">Public General</p>
                    </div>
                    <div className="p-4 bg-green-50 rounded-lg">
                      <p className="text-2xl font-bold text-green-900">
                        {selectedDataset.car_domain_samples.toLocaleString()}
                      </p>
                      <p className="text-sm text-green-600">Car Domain</p>
                    </div>
                    <div className="p-4 bg-purple-50 rounded-lg">
                      <p className="text-2xl font-bold text-purple-900">
                        {selectedDataset.synthetic_samples.toLocaleString()}
                      </p>
                      <p className="text-sm text-purple-600">Synthetic</p>
                    </div>
                  </div>
                </CardBody>
              </Card>

              {/* RegMix Composition */}
              <Card>
                <CardHeader>
                  <CardTitle>RegMix Composition</CardTitle>
                  <CardDescription>Data source distribution following 40/40/20 strategy</CardDescription>
                </CardHeader>
                <CardBody>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-gray-600">Public General (Big5-Chat, PANDORA)</span>
                        <span className="font-medium">
                          {Math.round((selectedDataset.public_general_samples / selectedDataset.total_samples) * 100)}%
                        </span>
                      </div>
                      <Progress
                        value={(selectedDataset.public_general_samples / selectedDataset.total_samples) * 100}
                        color="bg-blue-500"
                        size="lg"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-gray-600">Car Domain Specific</span>
                        <span className="font-medium">
                          {Math.round((selectedDataset.car_domain_samples / selectedDataset.total_samples) * 100)}%
                        </span>
                      </div>
                      <Progress
                        value={(selectedDataset.car_domain_samples / selectedDataset.total_samples) * 100}
                        color="bg-green-500"
                        size="lg"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-gray-600">PsychSteer Synthetic</span>
                        <span className="font-medium">
                          {Math.round((selectedDataset.synthetic_samples / selectedDataset.total_samples) * 100)}%
                        </span>
                      </div>
                      <Progress
                        value={(selectedDataset.synthetic_samples / selectedDataset.total_samples) * 100}
                        color="bg-purple-500"
                        size="lg"
                      />
                    </div>
                  </div>
                </CardBody>
              </Card>

              {/* PsychSteer Config */}
              <Card>
                <CardHeader>
                  <CardTitle>PsychSteer Configuration</CardTitle>
                  <CardDescription>Teacher model used for synthetic data labeling</CardDescription>
                </CardHeader>
                <CardBody>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-500">Teacher Model</p>
                      <p className="font-medium">{selectedDataset.teacher_model || 'N/A'}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Labeling Method</p>
                      <p className="font-medium">PsychSteer (Car-to-Personality)</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Output Format</p>
                      <p className="font-medium">OCEAN Scores (0-1)</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Created</p>
                      <p className="font-medium">{formatDateTime(selectedDataset.created_at)}</p>
                    </div>
                  </div>
                </CardBody>
              </Card>
            </>
          ) : (
            <Card>
              <CardBody className="flex flex-col items-center justify-center h-64 text-gray-500">
                <Database className="w-12 h-12 mb-4 text-gray-300" />
                <p>Select a dataset to view details</p>
              </CardBody>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
