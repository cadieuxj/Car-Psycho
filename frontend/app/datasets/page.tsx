'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardBody } from '@/components/ui';
import { Button } from '@/components/ui';
import { Badge } from '@/components/ui';
import { Progress } from '@/components/ui';
import { Dataset, DatasetStats, DatasetPreview } from '@/lib/types';
import { api } from '@/lib/api';
import { formatDateTime, OCEAN_COLORS, OCEAN_LABELS } from '@/lib/utils';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import {
  Database,
  RefreshCw,
  Trash2,
  BarChart3,
  Eye,
  Settings,
  ChevronRight,
  AlertCircle,
  FolderSearch,
  FileJson,
  Layers,
  Info,
} from 'lucide-react';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TRAIT_KEYS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'] as const;
type TraitKey = (typeof TRAIT_KEYS)[number];

const BIN_LABELS = [
  '0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5',
  '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0',
];

const TAB_LIST = [
  { key: 'overview', label: 'Overview', icon: Info },
  { key: 'statistics', label: 'Statistics', icon: BarChart3 },
  { key: 'preview', label: 'Preview', icon: Eye },
  { key: 'config', label: 'Config', icon: Settings },
] as const;

type TabKey = (typeof TAB_LIST)[number]['key'];

const TYPE_BADGE_VARIANT: Record<string, 'default' | 'success' | 'warning' | 'error' | 'info'> = {
  training: 'info',
  evaluation: 'success',
  synthetic: 'warning',
  combined: 'default',
};

// ---------------------------------------------------------------------------
// Skeleton helpers
// ---------------------------------------------------------------------------

function SkeletonLine({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse bg-gray-200 rounded ${className}`} />;
}

function SkeletonCard() {
  return (
    <div className="p-4 space-y-3">
      <SkeletonLine className="h-4 w-3/4" />
      <SkeletonLine className="h-3 w-1/2" />
      <SkeletonLine className="h-3 w-1/3" />
    </div>
  );
}

function StatsSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="p-4 bg-gray-50 rounded-lg space-y-2">
            <SkeletonLine className="h-3 w-1/2" />
            <SkeletonLine className="h-6 w-2/3" />
          </div>
        ))}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-52 bg-gray-50 rounded-lg animate-pulse" />
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page component
// ---------------------------------------------------------------------------

export default function DatasetsPage() {
  // --- Data state ---
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [activeTab, setActiveTab] = useState<TabKey>('overview');

  // --- Loading / error state ---
  const [loadingList, setLoadingList] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [loadingStats, setLoadingStats] = useState(false);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scanMessage, setScanMessage] = useState<string | null>(null);

  const selected = datasets.find((d) => d.id === selectedId) ?? null;

  // --- Fetch dataset list ---
  const fetchDatasets = useCallback(async () => {
    setLoadingList(true);
    setError(null);
    try {
      const list = await api.listDatasets();
      setDatasets(list);
      if (list.length > 0 && !selectedId) {
        setSelectedId(list[0].id);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load datasets');
    } finally {
      setLoadingList(false);
    }
  }, [selectedId]);

  useEffect(() => {
    fetchDatasets();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- Fetch stats when selected dataset or stats tab changes ---
  useEffect(() => {
    if (!selectedId) return;
    if (activeTab !== 'statistics') return;
    let cancelled = false;
    (async () => {
      setLoadingStats(true);
      setStats(null);
      try {
        const data = await api.getDatasetStats(selectedId);
        if (!cancelled) setStats(data);
      } catch {
        if (!cancelled) setStats(null);
      } finally {
        if (!cancelled) setLoadingStats(false);
      }
    })();
    return () => { cancelled = true; };
  }, [selectedId, activeTab]);

  // --- Fetch preview when preview tab is active ---
  useEffect(() => {
    if (!selectedId) return;
    if (activeTab !== 'preview') return;
    let cancelled = false;
    (async () => {
      setLoadingPreview(true);
      setPreview(null);
      try {
        const data = await api.getDatasetPreview(selectedId);
        if (!cancelled) setPreview(data);
      } catch {
        if (!cancelled) setPreview(null);
      } finally {
        if (!cancelled) setLoadingPreview(false);
      }
    })();
    return () => { cancelled = true; };
  }, [selectedId, activeTab]);

  // --- Actions ---
  const handleScan = async () => {
    setScanning(true);
    setScanMessage(null);
    try {
      const result = await api.scanDatasets();
      setScanMessage(result.message);
      await fetchDatasets();
    } catch (err: unknown) {
      setScanMessage(err instanceof Error ? err.message : 'Scan failed');
    } finally {
      setScanning(false);
    }
  };

  const handleDelete = async () => {
    if (!selectedId) return;
    setDeleting(true);
    try {
      await api.deleteDataset(selectedId);
      setSelectedId(null);
      setStats(null);
      setPreview(null);
      await fetchDatasets();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Delete failed');
    } finally {
      setDeleting(false);
    }
  };

  const selectDataset = (id: string) => {
    setSelectedId(id);
    setStats(null);
    setPreview(null);
    setActiveTab('overview');
  };

  // =========================================================================
  // Render
  // =========================================================================

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Datasets</h1>
          <p className="text-gray-500 mt-1">
            Manage personality-labeled datasets for training and evaluation
          </p>
        </div>
        <Button
          icon={<FolderSearch className="w-4 h-4" />}
          onClick={handleScan}
          loading={scanning}
        >
          Scan for Datasets
        </Button>
      </div>

      {/* Scan feedback */}
      {scanMessage && (
        <div className="flex items-center gap-2 text-sm text-blue-700 bg-blue-50 border border-blue-200 rounded-lg px-4 py-2">
          <Info className="w-4 h-4 flex-shrink-0" />
          {scanMessage}
          <button className="ml-auto text-blue-400 hover:text-blue-600" onClick={() => setScanMessage(null)}>
            &times;
          </button>
        </div>
      )}

      {/* Global error */}
      {error && (
        <div className="flex items-center gap-2 text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-4 py-2">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          {error}
          <button className="ml-auto text-red-400 hover:text-red-600" onClick={() => setError(null)}>
            &times;
          </button>
        </div>
      )}

      {/* Two-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* ---------------------------------------------------------------- */}
        {/* Dataset list (left column)                                       */}
        {/* ---------------------------------------------------------------- */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>All Datasets</CardTitle>
                <Badge variant="info">{datasets.length}</Badge>
              </div>
            </CardHeader>
            <CardBody className="p-0">
              {loadingList ? (
                <div className="divide-y divide-gray-100">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <SkeletonCard key={i} />
                  ))}
                </div>
              ) : datasets.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-16 text-gray-400">
                  <Database className="w-10 h-10 mb-3" />
                  <p className="text-sm">No datasets found</p>
                  <p className="text-xs mt-1">Click &quot;Scan for Datasets&quot; to discover files</p>
                </div>
              ) : (
                <div className="divide-y divide-gray-100 max-h-[calc(100vh-280px)] overflow-y-auto">
                  {datasets.map((ds) => (
                    <button
                      key={ds.id}
                      onClick={() => selectDataset(ds.id)}
                      className={`w-full p-4 text-left hover:bg-gray-50 transition-colors ${
                        selectedId === ds.id
                          ? 'bg-primary-50 border-l-4 border-l-primary-600'
                          : ''
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <h4 className="font-medium text-gray-900 truncate mr-2">
                          {ds.name}
                        </h4>
                        <ChevronRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
                      </div>
                      <div className="flex items-center gap-2 mb-2 flex-wrap">
                        <Badge variant={TYPE_BADGE_VARIANT[ds.dataset_type] ?? 'default'}>
                          {ds.dataset_type}
                        </Badge>
                        <Badge variant="info">
                          {ds.num_samples.toLocaleString()} samples
                        </Badge>
                        <Badge variant="default">{ds.format}</Badge>
                      </div>
                      <p className="text-xs text-gray-400">
                        Created {formatDateTime(ds.created_at)}
                      </p>
                    </button>
                  ))}
                </div>
              )}
            </CardBody>
          </Card>
        </div>

        {/* ---------------------------------------------------------------- */}
        {/* Detail panel (right column)                                      */}
        {/* ---------------------------------------------------------------- */}
        <div className="lg:col-span-2 space-y-4">
          {!selected ? (
            <Card>
              <CardBody className="flex flex-col items-center justify-center h-64 text-gray-400">
                <Layers className="w-12 h-12 mb-4" />
                <p className="font-medium">Select a dataset to view details</p>
              </CardBody>
            </Card>
          ) : (
            <>
              {/* Tab navigation */}
              <Card>
                <CardBody className="p-0">
                  <div className="flex border-b border-gray-200">
                    {TAB_LIST.map(({ key, label, icon: Icon }) => (
                      <button
                        key={key}
                        onClick={() => setActiveTab(key)}
                        className={`flex items-center gap-2 px-5 py-3 text-sm font-medium border-b-2 transition-colors ${
                          activeTab === key
                            ? 'border-primary-600 text-primary-700'
                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                        }`}
                      >
                        <Icon className="w-4 h-4" />
                        {label}
                      </button>
                    ))}

                    {/* Delete button pushed to right */}
                    <div className="ml-auto flex items-center pr-3">
                      <Button
                        variant="danger"
                        size="sm"
                        icon={<Trash2 className="w-4 h-4" />}
                        onClick={handleDelete}
                        loading={deleting}
                      >
                        Delete
                      </Button>
                    </div>
                  </div>
                </CardBody>
              </Card>

              {/* Tab content */}
              {activeTab === 'overview' && <OverviewTab dataset={selected} />}
              {activeTab === 'statistics' && (
                <StatisticsTab stats={stats} loading={loadingStats} dataset={selected} />
              )}
              {activeTab === 'preview' && (
                <PreviewTab preview={preview} loading={loadingPreview} />
              )}
              {activeTab === 'config' && <ConfigTab dataset={selected} />}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ===========================================================================
// Overview Tab
// ===========================================================================

function OverviewTab({ dataset }: { dataset: Dataset }) {
  const fields: { label: string; value: string }[] = [
    { label: 'Name', value: dataset.name },
    { label: 'Type', value: dataset.dataset_type },
    { label: 'Source', value: dataset.source ?? 'N/A' },
    { label: 'Format', value: dataset.format },
    { label: 'Samples', value: dataset.num_samples.toLocaleString() },
    { label: 'Processed', value: dataset.is_processed ? 'Yes' : 'No' },
    { label: 'File Path', value: dataset.file_path },
    { label: 'Created', value: formatDateTime(dataset.created_at) },
    { label: 'Updated', value: formatDateTime(dataset.updated_at) },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>{dataset.name}</CardTitle>
        <CardDescription>{dataset.description || 'No description provided.'}</CardDescription>
      </CardHeader>
      <CardBody>
        {/* Quick stat boxes */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="p-4 bg-gray-50 rounded-lg text-center">
            <p className="text-2xl font-bold text-gray-900">{dataset.num_samples.toLocaleString()}</p>
            <p className="text-sm text-gray-500">Total Samples</p>
          </div>
          <div className="p-4 bg-blue-50 rounded-lg text-center">
            <p className="text-2xl font-bold text-blue-700">{dataset.format.toUpperCase()}</p>
            <p className="text-sm text-blue-600">Format</p>
          </div>
          <div className="p-4 bg-green-50 rounded-lg text-center">
            <p className="text-2xl font-bold text-green-700">{dataset.dataset_type}</p>
            <p className="text-sm text-green-600">Type</p>
          </div>
          <div className="p-4 bg-purple-50 rounded-lg text-center">
            <p className="text-2xl font-bold text-purple-700">
              {dataset.is_processed ? 'Ready' : 'Raw'}
            </p>
            <p className="text-sm text-purple-600">Status</p>
          </div>
        </div>

        {/* Detail grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-3 text-sm">
          {fields.map(({ label, value }) => (
            <div key={label} className="flex justify-between py-2 border-b border-gray-100">
              <span className="text-gray-500">{label}</span>
              <span className="font-medium text-gray-900 text-right break-all max-w-[60%]">{value}</span>
            </div>
          ))}
        </div>
      </CardBody>
    </Card>
  );
}

// ===========================================================================
// Statistics Tab
// ===========================================================================

function StatisticsTab({
  stats,
  loading,
  dataset,
}: {
  stats: DatasetStats | null;
  loading: boolean;
  dataset: Dataset;
}) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Statistics</CardTitle>
          <CardDescription>Loading dataset statistics...</CardDescription>
        </CardHeader>
        <CardBody>
          <StatsSkeleton />
        </CardBody>
      </Card>
    );
  }

  // Use embedded stats as fallback
  const effective = stats ?? dataset.stats ?? null;

  if (!effective) {
    return (
      <Card>
        <CardBody className="flex flex-col items-center justify-center h-48 text-gray-400">
          <BarChart3 className="w-10 h-10 mb-3" />
          <p className="text-sm">No statistics available for this dataset.</p>
        </CardBody>
      </Card>
    );
  }

  // --- Build chart data ---
  const traitComparison = TRAIT_KEYS.map((t) => ({
    trait: OCEAN_LABELS[t],
    mean: Number(effective.traits[t].mean.toFixed(3)),
    fill: OCEAN_COLORS[t],
  }));

  const sourceData = Object.entries(effective.source_breakdown).map(([name, count]) => ({
    name,
    count,
  }));

  return (
    <div className="space-y-6">
      {/* Summary stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <Card>
          <CardBody className="flex items-center gap-3">
            <div className="p-2 bg-gray-100 rounded-lg">
              <Database className="w-5 h-5 text-gray-600" />
            </div>
            <div>
              <p className="text-xs text-gray-500">Total Samples</p>
              <p className="text-lg font-bold text-gray-900">
                {effective.total_samples.toLocaleString()}
              </p>
            </div>
          </CardBody>
        </Card>
        {TRAIT_KEYS.map((t) => (
          <Card key={t}>
            <CardBody className="flex items-center gap-3">
              <div
                className="p-2 rounded-lg"
                style={{ backgroundColor: `${OCEAN_COLORS[t]}20` }}
              >
                <div
                  className="w-5 h-5 rounded-full"
                  style={{ backgroundColor: OCEAN_COLORS[t] }}
                />
              </div>
              <div>
                <p className="text-xs text-gray-500">{OCEAN_LABELS[t]} Mean</p>
                <p className="text-lg font-bold" style={{ color: OCEAN_COLORS[t] }}>
                  {effective.traits[t].mean.toFixed(3)}
                </p>
              </div>
            </CardBody>
          </Card>
        ))}
      </div>

      {/* Trait detail stats table */}
      <Card>
        <CardHeader>
          <CardTitle>Trait Summary</CardTitle>
          <CardDescription>Mean, standard deviation, and range for each OCEAN trait</CardDescription>
        </CardHeader>
        <CardBody>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 text-left text-gray-500">
                  <th className="pb-2 font-medium">Trait</th>
                  <th className="pb-2 font-medium">Mean +/- Std</th>
                  <th className="pb-2 font-medium">Median</th>
                  <th className="pb-2 font-medium">Min</th>
                  <th className="pb-2 font-medium">Max</th>
                  <th className="pb-2 font-medium w-40">Distribution</th>
                </tr>
              </thead>
              <tbody>
                {TRAIT_KEYS.map((t) => {
                  const ts = effective.traits[t];
                  return (
                    <tr key={t} className="border-b border-gray-100">
                      <td className="py-3 font-medium" style={{ color: OCEAN_COLORS[t] }}>
                        {OCEAN_LABELS[t]}
                      </td>
                      <td className="py-3">
                        {ts.mean.toFixed(3)} +/- {ts.std.toFixed(3)}
                      </td>
                      <td className="py-3">{ts.median.toFixed(3)}</td>
                      <td className="py-3">{ts.min.toFixed(3)}</td>
                      <td className="py-3">{ts.max.toFixed(3)}</td>
                      <td className="py-3">
                        <Progress
                          value={ts.mean * 100}
                          color={`bg-[${OCEAN_COLORS[t]}]`}
                          size="sm"
                        />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardBody>
      </Card>

      {/* Trait comparison bar chart */}
      <Card>
        <CardHeader>
          <CardTitle>Trait Comparison</CardTitle>
          <CardDescription>Mean scores across all five OCEAN traits</CardDescription>
        </CardHeader>
        <CardBody>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={traitComparison} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="trait" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                <Tooltip
                  formatter={(value: number) => [value.toFixed(3), 'Mean']}
                  contentStyle={{ borderRadius: 8, fontSize: 13 }}
                />
                <Bar dataKey="mean" radius={[4, 4, 0, 0]}>
                  {traitComparison.map((entry, i) => (
                    <rect key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardBody>
      </Card>

      {/* Histograms for each trait */}
      <Card>
        <CardHeader>
          <CardTitle>Trait Distributions</CardTitle>
          <CardDescription>
            Histogram of scores across 10 bins (0.0 to 1.0) for each personality trait
          </CardDescription>
        </CardHeader>
        <CardBody>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {TRAIT_KEYS.map((t) => {
              const histData = effective.traits[t].histogram.map((count, i) => ({
                bin: BIN_LABELS[i],
                count,
              }));
              return (
                <div key={t}>
                  <h4
                    className="text-sm font-semibold mb-2"
                    style={{ color: OCEAN_COLORS[t] }}
                  >
                    {OCEAN_LABELS[t]}
                  </h4>
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={histData} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                        <XAxis
                          dataKey="bin"
                          tick={{ fontSize: 10 }}
                          interval={0}
                          angle={-45}
                          textAnchor="end"
                          height={50}
                        />
                        <YAxis tick={{ fontSize: 10 }} />
                        <Tooltip
                          formatter={(value: number) => [value, 'Samples']}
                          labelFormatter={(label) => `Bin: ${label}`}
                          contentStyle={{ borderRadius: 8, fontSize: 12 }}
                        />
                        <Bar
                          dataKey="count"
                          fill={OCEAN_COLORS[t]}
                          radius={[3, 3, 0, 0]}
                          opacity={0.85}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              );
            })}
          </div>
        </CardBody>
      </Card>

      {/* Source breakdown */}
      {sourceData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Source Breakdown</CardTitle>
            <CardDescription>Number of samples contributed by each data source</CardDescription>
          </CardHeader>
          <CardBody>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={sourceData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip
                    formatter={(value: number) => [value.toLocaleString(), 'Samples']}
                    contentStyle={{ borderRadius: 8, fontSize: 13 }}
                  />
                  <Legend />
                  <Bar dataKey="count" name="Samples" fill="#6366f1" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardBody>
        </Card>
      )}
    </div>
  );
}

// ===========================================================================
// Preview Tab
// ===========================================================================

function PreviewTab({
  preview,
  loading,
}: {
  preview: DatasetPreview | null;
  loading: boolean;
}) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Data Preview</CardTitle>
          <CardDescription>Loading sample rows...</CardDescription>
        </CardHeader>
        <CardBody>
          <div className="space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <SkeletonLine key={i} className="h-8 w-full" />
            ))}
          </div>
        </CardBody>
      </Card>
    );
  }

  if (!preview || preview.samples.length === 0) {
    return (
      <Card>
        <CardBody className="flex flex-col items-center justify-center h-48 text-gray-400">
          <Eye className="w-10 h-10 mb-3" />
          <p className="text-sm">No preview data available.</p>
        </CardBody>
      </Card>
    );
  }

  const samples = preview.samples.slice(0, 20);
  const columns = Object.keys(samples[0]);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Data Preview</CardTitle>
            <CardDescription>
              Showing {samples.length} of {preview.total_samples.toLocaleString()} total samples
            </CardDescription>
          </div>
          <Badge variant="info">{preview.total_samples.toLocaleString()} total</Badge>
        </div>
      </CardHeader>
      <CardBody className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  #
                </th>
                {columns.map((col) => (
                  <th
                    key={col}
                    className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {samples.map((row, idx) => (
                <tr key={idx} className="hover:bg-gray-50">
                  <td className="px-4 py-2 text-gray-400">{idx + 1}</td>
                  {columns.map((col) => {
                    const val = row[col];
                    const display =
                      typeof val === 'number'
                        ? val.toFixed(3)
                        : typeof val === 'string'
                          ? val.length > 80
                            ? val.slice(0, 80) + '...'
                            : val
                          : JSON.stringify(val);
                    return (
                      <td key={col} className="px-4 py-2 text-gray-700 max-w-xs truncate">
                        {display}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardBody>
    </Card>
  );
}

// ===========================================================================
// Config Tab
// ===========================================================================

function ConfigTab({ dataset }: { dataset: Dataset }) {
  const configItems: { label: string; value: string; icon: React.ReactNode }[] = [
    {
      label: 'File Path',
      value: dataset.file_path,
      icon: <FileJson className="w-4 h-4 text-gray-400" />,
    },
    {
      label: 'Format',
      value: dataset.format.toUpperCase(),
      icon: <FileJson className="w-4 h-4 text-gray-400" />,
    },
    {
      label: 'Dataset Type',
      value: dataset.dataset_type,
      icon: <Layers className="w-4 h-4 text-gray-400" />,
    },
    {
      label: 'Source',
      value: dataset.source ?? 'N/A',
      icon: <Database className="w-4 h-4 text-gray-400" />,
    },
    {
      label: 'Processed',
      value: dataset.is_processed ? 'Yes' : 'No',
      icon: <RefreshCw className="w-4 h-4 text-gray-400" />,
    },
    {
      label: 'Number of Samples',
      value: dataset.num_samples.toLocaleString(),
      icon: <BarChart3 className="w-4 h-4 text-gray-400" />,
    },
  ];

  // Infer schema columns from stats if available
  const schemaFields: string[] = [];
  if (dataset.stats) {
    schemaFields.push('text', 'source');
    TRAIT_KEYS.forEach((t) => schemaFields.push(t));
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Dataset Configuration</CardTitle>
          <CardDescription>File location, format, and metadata</CardDescription>
        </CardHeader>
        <CardBody>
          <div className="space-y-3">
            {configItems.map(({ label, value, icon }) => (
              <div
                key={label}
                className="flex items-center gap-3 py-3 border-b border-gray-100 last:border-0"
              >
                {icon}
                <span className="text-sm text-gray-500 w-40 flex-shrink-0">{label}</span>
                <span className="text-sm font-medium text-gray-900 break-all">{value}</span>
              </div>
            ))}
          </div>
        </CardBody>
      </Card>

      {schemaFields.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Schema</CardTitle>
            <CardDescription>Inferred columns from the dataset</CardDescription>
          </CardHeader>
          <CardBody>
            <div className="flex flex-wrap gap-2">
              {schemaFields.map((field) => {
                const isTrait = TRAIT_KEYS.includes(field as TraitKey);
                return (
                  <Badge
                    key={field}
                    variant={isTrait ? 'success' : 'default'}
                    className="text-xs"
                  >
                    {field}
                  </Badge>
                );
              })}
            </div>
          </CardBody>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Timestamps</CardTitle>
        </CardHeader>
        <CardBody>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-gray-500">Created At</p>
              <p className="font-medium text-gray-900">{formatDateTime(dataset.created_at)}</p>
            </div>
            <div>
              <p className="text-gray-500">Last Updated</p>
              <p className="font-medium text-gray-900">{formatDateTime(dataset.updated_at)}</p>
            </div>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
