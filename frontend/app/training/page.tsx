'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardBody } from '@/components/ui';
import { Button } from '@/components/ui';
import { Badge } from '@/components/ui';
import { Progress } from '@/components/ui';
import { useTraining } from '@/hooks/useTraining';
import { api } from '@/lib/api';
import {
  TrainingJob,
  TrainingConfig,
  Dataset,
  LossHistoryEntry,
  EpochMetrics,
  DEFAULT_TRAINING_CONFIG,
} from '@/lib/types';
import { formatDateTime } from '@/lib/utils';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';
import {
  Cpu,
  Square,
  RefreshCw,
  ChevronRight,
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  Plus,
  Trash2,
  X,
  Wifi,
  WifiOff,
  AlertTriangle,
} from 'lucide-react';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STATUS_CONFIG = {
  pending: { icon: Clock, color: 'text-gray-500', bg: 'bg-gray-100', badge: 'default' as const },
  running: { icon: Loader2, color: 'text-blue-500', bg: 'bg-blue-100', badge: 'info' as const },
  completed: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-100', badge: 'success' as const },
  failed: { icon: XCircle, color: 'text-red-500', bg: 'bg-red-100', badge: 'error' as const },
  cancelled: { icon: Square, color: 'text-gray-500', bg: 'bg-gray-100', badge: 'default' as const },
};

const DETAIL_TABS = ['Overview', 'Metrics', 'Epoch Analysis', 'Configuration'] as const;
type DetailTab = (typeof DETAIL_TABS)[number];

const TRAIT_COLORS: Record<string, string> = {
  openness: '#8b5cf6',
  conscientiousness: '#3b82f6',
  extraversion: '#f59e0b',
  agreeableness: '#10b981',
  neuroticism: '#ef4444',
};

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

function fmtLoss(v: number | undefined): string {
  return v !== undefined ? v.toFixed(6) : '--';
}

function fmtLR(v: number | undefined): string {
  return v !== undefined ? v.toExponential(2) : '--';
}

function fmtDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

function elapsedBetween(start?: string, end?: string): string {
  if (!start) return '--';
  const s = new Date(start).getTime();
  const e = end ? new Date(end).getTime() : Date.now();
  return fmtDuration((e - s) / 1000);
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function JobCard({
  job,
  selected,
  onSelect,
}: {
  job: TrainingJob;
  selected: boolean;
  onSelect: () => void;
}) {
  const cfg = STATUS_CONFIG[job.status];
  const Icon = cfg.icon;

  return (
    <button
      onClick={onSelect}
      className={`w-full p-4 text-left border rounded-lg transition-all ${
        selected
          ? 'border-primary-500 bg-primary-50 ring-2 ring-primary-200'
          : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
      }`}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-3 min-w-0">
          <div className={`shrink-0 p-2 rounded-lg ${cfg.bg}`}>
            <Icon
              className={`w-4 h-4 ${cfg.color} ${job.status === 'running' ? 'animate-spin' : ''}`}
            />
          </div>
          <div className="min-w-0">
            <h4 className="font-medium text-gray-900 truncate">{job.name}</h4>
            <p className="text-xs text-gray-500 truncate">{job.model_base}</p>
          </div>
        </div>
        <Badge variant={cfg.badge} className="shrink-0 ml-2">
          {job.status}
        </Badge>
      </div>

      {(job.status === 'running' || job.status === 'completed') && (
        <div className="mt-2">
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>
              {job.current_epoch !== undefined && job.total_epochs
                ? `Epoch ${job.current_epoch}/${job.total_epochs}`
                : ''}
            </span>
            <span>{Math.round(job.progress)}%</span>
          </div>
          <Progress
            value={job.progress}
            size="sm"
            color={job.status === 'completed' ? 'bg-green-500' : 'bg-blue-500'}
          />
        </div>
      )}

      {job.error_message && (
        <p className="text-xs text-red-600 mt-2 truncate">{job.error_message}</p>
      )}

      <div className="flex items-center justify-between mt-3 text-xs text-gray-400">
        <span>{formatDateTime(job.created_at)}</span>
        <ChevronRight className="w-4 h-4" />
      </div>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Metric Card
// ---------------------------------------------------------------------------

function MetricCard({
  label,
  value,
  bgClass,
  textClass,
}: {
  label: string;
  value: string;
  bgClass: string;
  textClass: string;
}) {
  return (
    <div className={`p-4 rounded-lg ${bgClass}`}>
      <p className={`text-xs mb-1 ${textClass}`}>{label}</p>
      <p className={`text-xl font-bold ${textClass.replace(/\/\d+/, '/900')}`}>{value}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab: Overview
// ---------------------------------------------------------------------------

function OverviewTab({ job }: { job: TrainingJob }) {
  return (
    <div className="space-y-6">
      {/* Progress */}
      <div>
        <div className="flex justify-between text-sm mb-2">
          <span className="text-gray-500">Overall Progress</span>
          <span className="font-medium">{Math.round(job.progress)}%</span>
        </div>
        <Progress value={job.progress} size="lg" showLabel={false} />
      </div>

      {/* Error */}
      {job.error_message && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
          <div>
            <p className="font-medium text-red-800 text-sm">Training Failed</p>
            <p className="text-sm text-red-700 mt-1">{job.error_message}</p>
          </div>
        </div>
      )}

      {/* Info Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 bg-gray-50 rounded-lg">
          <p className="text-xs text-gray-500 mb-1">Status</p>
          <Badge variant={STATUS_CONFIG[job.status].badge}>{job.status}</Badge>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg">
          <p className="text-xs text-gray-500 mb-1">Dataset</p>
          <p className="text-sm font-medium truncate">{job.dataset_id}</p>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg">
          <p className="text-xs text-gray-500 mb-1">Started</p>
          <p className="text-sm font-medium">
            {job.started_at ? formatDateTime(job.started_at) : '--'}
          </p>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg">
          <p className="text-xs text-gray-500 mb-1">Duration</p>
          <p className="text-sm font-medium">
            {elapsedBetween(job.started_at, job.completed_at)}
          </p>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg">
          <p className="text-xs text-gray-500 mb-1">Created</p>
          <p className="text-sm font-medium">{formatDateTime(job.created_at)}</p>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg">
          <p className="text-xs text-gray-500 mb-1">Last Updated</p>
          <p className="text-sm font-medium">{formatDateTime(job.updated_at)}</p>
        </div>
        {job.completed_at && (
          <div className="p-4 bg-gray-50 rounded-lg">
            <p className="text-xs text-gray-500 mb-1">Completed</p>
            <p className="text-sm font-medium">{formatDateTime(job.completed_at)}</p>
          </div>
        )}
        <div className="p-4 bg-gray-50 rounded-lg">
          <p className="text-xs text-gray-500 mb-1">Base Model</p>
          <p className="text-sm font-medium truncate">{job.model_base}</p>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab: Metrics
// ---------------------------------------------------------------------------

function MetricsTab({
  job,
  liveMetrics,
  lossHistory,
}: {
  job: TrainingJob;
  liveMetrics: ReturnType<typeof useTraining>['liveMetrics'];
  lossHistory: LossHistoryEntry[];
}) {
  const m = liveMetrics ?? job.metrics;

  const chartData = useMemo(
    () =>
      lossHistory.map((e) => ({
        step: e.step,
        total_loss: e.total_loss,
        personality_loss: e.personality_loss,
        ordinal_loss: e.ordinal_loss,
        mse_loss: e.mse_loss,
        learning_rate: e.learning_rate,
      })),
    [lossHistory],
  );

  return (
    <div className="space-y-6">
      {/* Live metric cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Total Loss"
          value={fmtLoss(m?.loss)}
          bgClass="bg-blue-50"
          textClass="text-blue-600"
        />
        <MetricCard
          label="Personality Loss"
          value={fmtLoss(m?.personality_loss)}
          bgClass="bg-purple-50"
          textClass="text-purple-600"
        />
        <MetricCard
          label="Ordinal Loss"
          value={fmtLoss(m?.ordinal_loss)}
          bgClass="bg-amber-50"
          textClass="text-amber-600"
        />
        <MetricCard
          label="MSE Loss"
          value={fmtLoss(m?.mse_loss)}
          bgClass="bg-rose-50"
          textClass="text-rose-600"
        />
        <MetricCard
          label="Current Epoch"
          value={
            m?.epoch != null ? `${m.epoch}${m.total_epochs ? ` / ${m.total_epochs}` : ''}` : '--'
          }
          bgClass="bg-gray-50"
          textClass="text-gray-600"
        />
        <MetricCard
          label="Step"
          value={
            m?.step != null
              ? `${m.step.toLocaleString()}${m.total_steps ? ` / ${m.total_steps.toLocaleString()}` : ''}`
              : '--'
          }
          bgClass="bg-gray-50"
          textClass="text-gray-600"
        />
        <MetricCard
          label="Learning Rate"
          value={fmtLR(m?.learning_rate)}
          bgClass="bg-gray-50"
          textClass="text-gray-600"
        />
        {m?.samples_per_second !== undefined && (
          <MetricCard
            label="Samples / sec"
            value={m.samples_per_second.toFixed(1)}
            bgClass="bg-green-50"
            textClass="text-green-600"
          />
        )}
      </div>

      {/* Loss curve chart */}
      {chartData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Loss Curves</CardTitle>
            <CardDescription>Loss components over training steps</CardDescription>
          </CardHeader>
          <CardBody>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis
                    dataKey="step"
                    tick={{ fontSize: 12 }}
                    label={{ value: 'Step', position: 'insideBottomRight', offset: -5, fontSize: 12 }}
                  />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{ fontSize: 12, borderRadius: 8 }}
                    formatter={(value: number) => value.toFixed(6)}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line
                    type="monotone"
                    dataKey="total_loss"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                    name="Total Loss"
                  />
                  <Line
                    type="monotone"
                    dataKey="personality_loss"
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    dot={false}
                    name="Personality Loss"
                  />
                  <Line
                    type="monotone"
                    dataKey="ordinal_loss"
                    stroke="#f59e0b"
                    strokeWidth={1.5}
                    dot={false}
                    name="Ordinal Loss"
                    strokeDasharray="4 2"
                  />
                  <Line
                    type="monotone"
                    dataKey="mse_loss"
                    stroke="#ef4444"
                    strokeWidth={1.5}
                    dot={false}
                    name="MSE Loss"
                    strokeDasharray="4 2"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardBody>
        </Card>
      )}

      {/* Learning rate schedule */}
      {chartData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Learning Rate Schedule</CardTitle>
          </CardHeader>
          <CardBody>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="step" tick={{ fontSize: 12 }} />
                  <YAxis
                    tick={{ fontSize: 12 }}
                    tickFormatter={(v: number) => v.toExponential(1)}
                  />
                  <Tooltip
                    contentStyle={{ fontSize: 12, borderRadius: 8 }}
                    formatter={(value: number) => value.toExponential(2)}
                  />
                  <Line
                    type="monotone"
                    dataKey="learning_rate"
                    stroke="#10b981"
                    strokeWidth={2}
                    dot={false}
                    name="Learning Rate"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardBody>
        </Card>
      )}

      {chartData.length === 0 && (
        <div className="text-center py-12 text-gray-400">
          <Cpu className="w-10 h-10 mx-auto mb-3" />
          <p>No loss history available yet.</p>
          {job.status === 'pending' && <p className="text-sm mt-1">Metrics will appear once training starts.</p>}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab: Epoch Analysis
// ---------------------------------------------------------------------------

function EpochAnalysisTab({ epochMetrics }: { epochMetrics: EpochMetrics[] }) {
  const latestEpoch = epochMetrics.length > 0 ? epochMetrics[epochMetrics.length - 1] : null;

  const traitBarData = useMemo(() => {
    if (!latestEpoch?.trait_mae) return [];
    return Object.entries(latestEpoch.trait_mae).map(([trait, mae]) => ({
      trait: trait.charAt(0).toUpperCase() + trait.slice(1),
      mae,
      fill: TRAIT_COLORS[trait] || '#6b7280',
    }));
  }, [latestEpoch]);

  const epochLossData = useMemo(
    () =>
      epochMetrics.map((em) => ({
        epoch: em.epoch,
        train_loss: em.train_loss,
        personality_loss: em.personality_loss,
        ordinal_loss: em.ordinal_loss,
        mse_loss: em.mse_loss,
      })),
    [epochMetrics],
  );

  if (epochMetrics.length === 0) {
    return (
      <div className="text-center py-12 text-gray-400">
        <Cpu className="w-10 h-10 mx-auto mb-3" />
        <p>No epoch data available yet.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Epoch table */}
      <Card>
        <CardHeader>
          <CardTitle>Epoch Summary</CardTitle>
        </CardHeader>
        <CardBody className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-500 border-b">
                <th className="pb-2 pr-4">Epoch</th>
                <th className="pb-2 pr-4">Train Loss</th>
                <th className="pb-2 pr-4">Personality</th>
                <th className="pb-2 pr-4">Ordinal</th>
                <th className="pb-2 pr-4">MSE</th>
                <th className="pb-2 pr-4">LR</th>
                <th className="pb-2">Duration</th>
              </tr>
            </thead>
            <tbody>
              {epochMetrics.map((em) => (
                <tr key={em.epoch} className="border-b last:border-0 hover:bg-gray-50">
                  <td className="py-2 pr-4 font-medium">{em.epoch}</td>
                  <td className="py-2 pr-4 font-mono text-xs">{fmtLoss(em.train_loss)}</td>
                  <td className="py-2 pr-4 font-mono text-xs">{fmtLoss(em.personality_loss)}</td>
                  <td className="py-2 pr-4 font-mono text-xs">{fmtLoss(em.ordinal_loss)}</td>
                  <td className="py-2 pr-4 font-mono text-xs">{fmtLoss(em.mse_loss)}</td>
                  <td className="py-2 pr-4 font-mono text-xs">{fmtLR(em.learning_rate)}</td>
                  <td className="py-2 font-mono text-xs">{fmtDuration(em.duration_seconds)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardBody>
      </Card>

      {/* Epoch loss progression */}
      {epochLossData.length > 1 && (
        <Card>
          <CardHeader>
            <CardTitle>Epoch Loss Progression</CardTitle>
          </CardHeader>
          <CardBody>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={epochLossData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{ fontSize: 12, borderRadius: 8 }}
                    formatter={(value: number) => value.toFixed(6)}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line type="monotone" dataKey="train_loss" stroke="#3b82f6" strokeWidth={2} name="Train Loss" />
                  <Line type="monotone" dataKey="personality_loss" stroke="#8b5cf6" strokeWidth={2} name="Personality" />
                  <Line type="monotone" dataKey="ordinal_loss" stroke="#f59e0b" strokeWidth={1.5} name="Ordinal" strokeDasharray="4 2" />
                  <Line type="monotone" dataKey="mse_loss" stroke="#ef4444" strokeWidth={1.5} name="MSE" strokeDasharray="4 2" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardBody>
        </Card>
      )}

      {/* Per-trait MAE bar chart */}
      {traitBarData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Per-Trait MAE (Epoch {latestEpoch!.epoch})</CardTitle>
            <CardDescription>Mean Absolute Error by OCEAN trait for the latest completed epoch</CardDescription>
          </CardHeader>
          <CardBody>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={traitBarData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="trait" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{ fontSize: 12, borderRadius: 8 }}
                    formatter={(value: number) => value.toFixed(6)}
                  />
                  <Bar dataKey="mae" name="MAE" radius={[4, 4, 0, 0]}>
                    {traitBarData.map((entry, idx) => (
                      <rect key={idx} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardBody>
        </Card>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab: Configuration
// ---------------------------------------------------------------------------

function ConfigurationTab({ config }: { config: TrainingConfig }) {
  const groups: { title: string; items: { label: string; value: string }[] }[] = [
    {
      title: 'Basic',
      items: [
        { label: 'Job Name', value: config.job_name },
        { label: 'Base Model', value: config.model_base },
        { label: 'Ollama Model', value: config.ollama_model || 'llama3.2:1b' },
        { label: 'Dataset', value: config.dataset_id },
      ],
    },
    {
      title: 'Training',
      items: [
        { label: 'Epochs', value: String(config.epochs) },
        { label: 'Batch Size', value: String(config.batch_size) },
        { label: 'Learning Rate', value: config.learning_rate.toExponential(2) },
        { label: 'Gradient Accumulation', value: String(config.gradient_accumulation_steps) },
        { label: 'Warmup Steps', value: String(config.warmup_steps) },
        { label: 'Weight Decay', value: String(config.weight_decay) },
        { label: 'Save Steps', value: String(config.save_steps) },
        { label: 'Eval Steps', value: String(config.eval_steps) },
      ],
    },
    {
      title: 'Loss Weights',
      items: [
        { label: 'LM Loss Weight', value: String(config.lm_loss_weight) },
        { label: 'Personality Loss Weight', value: String(config.personality_loss_weight) },
        { label: 'Ordinal Loss Weight', value: String(config.ordinal_loss_weight) },
        { label: 'MSE Loss Weight', value: String(config.mse_loss_weight) },
      ],
    },
    {
      title: 'Advanced',
      items: [
        { label: 'Car Domain Boost', value: `${config.car_domain_boost}x` },
        { label: 'Max Seq Length', value: String(config.max_seq_length) },
      ],
    },
    {
      title: 'Model Architecture',
      items: [
        { label: 'Input Size', value: String(config.input_size) },
        { label: 'Hidden Size', value: String(config.hidden_size) },
      ],
    },
  ];

  return (
    <div className="space-y-6">
      {groups.map((g) => (
        <Card key={g.title}>
          <CardHeader>
            <CardTitle>{g.title}</CardTitle>
          </CardHeader>
          <CardBody>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              {g.items.map((item) => (
                <div key={item.label}>
                  <p className="text-gray-500">{item.label}</p>
                  <p className="font-medium mt-0.5 truncate">{item.value}</p>
                </div>
              ))}
            </div>
          </CardBody>
        </Card>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// New Training Job Modal
// ---------------------------------------------------------------------------

interface NewJobModalProps {
  open: boolean;
  onClose: () => void;
  onCreate: (config: TrainingConfig) => Promise<void>;
  datasets: Dataset[];
  datasetsLoading: boolean;
}

function NewJobModal({ open, onClose, onCreate, datasets, datasetsLoading }: NewJobModalProps) {
  const [form, setForm] = useState<TrainingConfig>({
    ...DEFAULT_TRAINING_CONFIG,
    job_name: '',
    dataset_id: '',
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const set = useCallback(
    <K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) =>
      setForm((prev) => ({ ...prev, [key]: value })),
    [],
  );

  const handleSubmit = async () => {
    if (!form.job_name.trim()) {
      setError('Job name is required.');
      return;
    }
    if (!form.dataset_id) {
      setError('Please select a dataset.');
      return;
    }
    setError(null);
    setSubmitting(true);
    try {
      await onCreate(form);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create job.');
    } finally {
      setSubmitting(false);
    }
  };

  if (!open) return null;

  const inputCls =
    'w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500';
  const labelCls = 'block text-sm font-medium text-gray-700 mb-1';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* backdrop */}
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />

      {/* modal */}
      <div className="relative bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto mx-4">
        <div className="sticky top-0 bg-white border-b px-6 py-4 flex items-center justify-between rounded-t-xl z-10">
          <h2 className="text-lg font-semibold text-gray-900">New Training Job</h2>
          <button onClick={onClose} className="p-1 rounded hover:bg-gray-100">
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        <div className="px-6 py-5 space-y-6">
          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
              {error}
            </div>
          )}

          {/* Basic */}
          <fieldset>
            <legend className="text-sm font-semibold text-gray-900 mb-3">Basic</legend>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className={labelCls}>Job Name</label>
                <input
                  className={inputCls}
                  value={form.job_name}
                  onChange={(e) => set('job_name', e.target.value)}
                  placeholder="e.g. CarPsycho-v3"
                />
              </div>
              <div>
                <label className={labelCls}>Dataset</label>
                <select
                  className={inputCls}
                  value={form.dataset_id}
                  onChange={(e) => set('dataset_id', e.target.value)}
                  disabled={datasetsLoading}
                >
                  <option value="">
                    {datasetsLoading ? 'Loading datasets...' : 'Select a dataset'}
                  </option>
                  {datasets.map((ds) => (
                    <option key={ds.id} value={ds.id}>
                      {ds.name} ({ds.num_samples.toLocaleString()} samples)
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className={labelCls}>Base Model</label>
                <input
                  className={inputCls}
                  value={form.model_base}
                  onChange={(e) => set('model_base', e.target.value)}
                />
              </div>
              <div>
                <label className={labelCls}>Ollama Model</label>
                <select
                  className={inputCls}
                  value={form.ollama_model}
                  onChange={(e) => set('ollama_model', e.target.value)}
                >
                  <option value="llama3.2:1b">Llama 3.2 1B</option>
                  <option value="llama3.2:3b">Llama 3.2 3B</option>
                  <option value="llama3.1:8b">Llama 3.1 8B</option>
                  <option value="nomic-embed-text">Nomic Embed Text</option>
                  <option value="mxbai-embed-large">MxBAI Embed Large</option>
                </select>
              </div>
            </div>
          </fieldset>

          {/* Training */}
          <fieldset>
            <legend className="text-sm font-semibold text-gray-900 mb-3">Training</legend>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <label className={labelCls}>Epochs</label>
                <input
                  type="number"
                  className={inputCls}
                  value={form.epochs}
                  onChange={(e) => set('epochs', Number(e.target.value))}
                  min={1}
                />
              </div>
              <div>
                <label className={labelCls}>Batch Size</label>
                <input
                  type="number"
                  className={inputCls}
                  value={form.batch_size}
                  onChange={(e) => set('batch_size', Number(e.target.value))}
                  min={1}
                />
              </div>
              <div>
                <label className={labelCls}>Learning Rate</label>
                <input
                  type="number"
                  className={inputCls}
                  value={form.learning_rate}
                  onChange={(e) => set('learning_rate', Number(e.target.value))}
                  step={0.0001}
                  min={0}
                />
              </div>
              <div>
                <label className={labelCls}>Grad. Accum. Steps</label>
                <input
                  type="number"
                  className={inputCls}
                  value={form.gradient_accumulation_steps}
                  onChange={(e) => set('gradient_accumulation_steps', Number(e.target.value))}
                  min={1}
                />
              </div>
            </div>
          </fieldset>

          {/* Loss Weights */}
          <fieldset>
            <legend className="text-sm font-semibold text-gray-900 mb-3">Loss Weights</legend>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <label className={labelCls}>LM Loss</label>
                <input
                  type="number"
                  className={inputCls}
                  value={form.lm_loss_weight}
                  onChange={(e) => set('lm_loss_weight', Number(e.target.value))}
                  step={0.1}
                  min={0}
                  max={1}
                />
              </div>
              <div>
                <label className={labelCls}>Personality Loss</label>
                <input
                  type="number"
                  className={inputCls}
                  value={form.personality_loss_weight}
                  onChange={(e) => set('personality_loss_weight', Number(e.target.value))}
                  step={0.1}
                  min={0}
                  max={1}
                />
              </div>
              <div>
                <label className={labelCls}>Ordinal Loss</label>
                <input
                  type="number"
                  className={inputCls}
                  value={form.ordinal_loss_weight}
                  onChange={(e) => set('ordinal_loss_weight', Number(e.target.value))}
                  step={0.1}
                  min={0}
                  max={1}
                />
              </div>
              <div>
                <label className={labelCls}>MSE Loss</label>
                <input
                  type="number"
                  className={inputCls}
                  value={form.mse_loss_weight}
                  onChange={(e) => set('mse_loss_weight', Number(e.target.value))}
                  step={0.1}
                  min={0}
                  max={1}
                />
              </div>
            </div>
          </fieldset>

          {/* Advanced */}
          <fieldset>
            <legend className="text-sm font-semibold text-gray-900 mb-3">Advanced</legend>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <label className={labelCls}>Car Domain Boost</label>
                <input
                  type="number"
                  className={inputCls}
                  value={form.car_domain_boost}
                  onChange={(e) => set('car_domain_boost', Number(e.target.value))}
                  step={0.1}
                  min={0}
                />
              </div>
              <div>
                <label className={labelCls}>Max Seq Length</label>
                <input
                  type="number"
                  className={inputCls}
                  value={form.max_seq_length}
                  onChange={(e) => set('max_seq_length', Number(e.target.value))}
                  min={1}
                />
              </div>
              <div>
                <label className={labelCls}>Warmup Steps</label>
                <input
                  type="number"
                  className={inputCls}
                  value={form.warmup_steps}
                  onChange={(e) => set('warmup_steps', Number(e.target.value))}
                  min={0}
                />
              </div>
              <div>
                <label className={labelCls}>Weight Decay</label>
                <input
                  type="number"
                  className={inputCls}
                  value={form.weight_decay}
                  onChange={(e) => set('weight_decay', Number(e.target.value))}
                  step={0.001}
                  min={0}
                />
              </div>
            </div>
          </fieldset>

          {/* Model Architecture */}
          <fieldset>
            <legend className="text-sm font-semibold text-gray-900 mb-3">Model Architecture</legend>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <label className={labelCls}>Input Size</label>
                <input
                  type="number"
                  className={inputCls}
                  value={form.input_size}
                  onChange={(e) => set('input_size', Number(e.target.value))}
                  min={1}
                />
              </div>
              <div>
                <label className={labelCls}>Hidden Size</label>
                <input
                  type="number"
                  className={inputCls}
                  value={form.hidden_size}
                  onChange={(e) => set('hidden_size', Number(e.target.value))}
                  min={1}
                />
              </div>
              <div>
                <label className={labelCls}>Save Steps</label>
                <input
                  type="number"
                  className={inputCls}
                  value={form.save_steps}
                  onChange={(e) => set('save_steps', Number(e.target.value))}
                  min={1}
                />
              </div>
              <div>
                <label className={labelCls}>Eval Steps</label>
                <input
                  type="number"
                  className={inputCls}
                  value={form.eval_steps}
                  onChange={(e) => set('eval_steps', Number(e.target.value))}
                  min={1}
                />
              </div>
            </div>
          </fieldset>
        </div>

        <div className="sticky bottom-0 bg-white border-t px-6 py-4 flex justify-end gap-3 rounded-b-xl">
          <Button variant="secondary" onClick={onClose} disabled={submitting}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            loading={submitting}
            icon={<Plus className="w-4 h-4" />}
          >
            Create Job
          </Button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Page
// ---------------------------------------------------------------------------

export default function TrainingPage() {
  const {
    jobs,
    selectedJob,
    isLoading,
    error,
    liveMetrics,
    lossHistory,
    epochMetrics,
    wsStatus,
    fetchJobs,
    selectJob,
    createJob,
    stopJob,
    deleteJob,
    refreshJob,
  } = useTraining();

  const [activeTab, setActiveTab] = useState<DetailTab>('Overview');
  const [showNewJob, setShowNewJob] = useState(false);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  // Fetch datasets for the new-job modal
  useEffect(() => {
    if (!showNewJob) return;
    let cancelled = false;
    setDatasetsLoading(true);
    api
      .listDatasets()
      .then((ds) => {
        if (!cancelled) setDatasets(ds);
      })
      .catch(() => {
        /* ignore */
      })
      .finally(() => {
        if (!cancelled) setDatasetsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [showNewJob]);

  // Reset tab when switching jobs
  useEffect(() => {
    setActiveTab('Overview');
  }, [selectedJob?.id]);

  const handleCreate = useCallback(
    async (config: TrainingConfig) => {
      await createJob(config);
    },
    [createJob],
  );

  const handleStop = useCallback(
    async (jobId: string) => {
      setActionLoading('stop');
      try {
        await stopJob(jobId);
      } finally {
        setActionLoading(null);
      }
    },
    [stopJob],
  );

  const handleDelete = useCallback(
    async (jobId: string) => {
      if (!confirm('Are you sure you want to delete this training job? This cannot be undone.')) return;
      setActionLoading('delete');
      try {
        await deleteJob(jobId);
      } finally {
        setActionLoading(null);
      }
    },
    [deleteJob],
  );

  const handleRefresh = useCallback(
    async (jobId: string) => {
      setActionLoading('refresh');
      try {
        await refreshJob(jobId);
      } finally {
        setActionLoading(null);
      }
    },
    [refreshJob],
  );

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Model Training</h1>
          <p className="text-gray-500 mt-1">
            Manage and monitor training jobs for psychometric models
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* WS indicator */}
          {selectedJob?.status === 'running' && (
            <span
              className={`flex items-center gap-1.5 text-xs px-2 py-1 rounded-full ${
                wsStatus === 'connected'
                  ? 'bg-green-50 text-green-700'
                  : 'bg-yellow-50 text-yellow-700'
              }`}
            >
              {wsStatus === 'connected' ? (
                <Wifi className="w-3 h-3" />
              ) : (
                <WifiOff className="w-3 h-3" />
              )}
              {wsStatus === 'connected' ? 'Live' : 'Polling'}
            </span>
          )}
          <Button icon={<Plus className="w-4 h-4" />} onClick={() => setShowNewJob(true)}>
            New Training Job
          </Button>
        </div>
      </div>

      {/* Global error */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* ---- Left: Job List ---- */}
        <div className="lg:col-span-1 space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Training Jobs</CardTitle>
                  <CardDescription>
                    {jobs.length} job{jobs.length !== 1 ? 's' : ''}
                  </CardDescription>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  icon={<RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />}
                  onClick={fetchJobs}
                  disabled={isLoading}
                >
                  Refresh
                </Button>
              </div>
            </CardHeader>
            <CardBody className="space-y-3">
              {isLoading && jobs.length === 0 && (
                <div className="flex items-center justify-center py-10 text-gray-400">
                  <Loader2 className="w-6 h-6 animate-spin mr-2" />
                  Loading jobs...
                </div>
              )}
              {!isLoading && jobs.length === 0 && (
                <div className="text-center py-10 text-gray-400">
                  <Cpu className="w-10 h-10 mx-auto mb-3" />
                  <p className="text-sm">No training jobs yet.</p>
                  <p className="text-xs mt-1">Create one to get started.</p>
                </div>
              )}
              {jobs.map((job) => (
                <JobCard
                  key={job.id}
                  job={job}
                  selected={selectedJob?.id === job.id}
                  onSelect={() => selectJob(job)}
                />
              ))}
            </CardBody>
          </Card>
        </div>

        {/* ---- Right: Job Detail ---- */}
        <div className="lg:col-span-2 space-y-6">
          {selectedJob ? (
            <>
              {/* Header + controls */}
              <Card>
                <CardHeader>
                  <div className="flex items-start justify-between flex-wrap gap-3">
                    <div className="min-w-0">
                      <CardTitle>{selectedJob.name}</CardTitle>
                      <CardDescription>{selectedJob.model_base}</CardDescription>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {selectedJob.status === 'running' && (
                        <Button
                          variant="danger"
                          size="sm"
                          icon={<Square className="w-4 h-4" />}
                          onClick={() => handleStop(selectedJob.id)}
                          loading={actionLoading === 'stop'}
                        >
                          Stop
                        </Button>
                      )}
                      {(selectedJob.status === 'completed' ||
                        selectedJob.status === 'failed' ||
                        selectedJob.status === 'cancelled') && (
                        <Button
                          variant="danger"
                          size="sm"
                          icon={<Trash2 className="w-4 h-4" />}
                          onClick={() => handleDelete(selectedJob.id)}
                          loading={actionLoading === 'delete'}
                        >
                          Delete
                        </Button>
                      )}
                      <Button
                        variant="secondary"
                        size="sm"
                        icon={
                          <RefreshCw
                            className={`w-4 h-4 ${actionLoading === 'refresh' ? 'animate-spin' : ''}`}
                          />
                        }
                        onClick={() => handleRefresh(selectedJob.id)}
                        disabled={actionLoading === 'refresh'}
                      >
                        Refresh
                      </Button>
                    </div>
                  </div>
                </CardHeader>

                {/* Tabs */}
                <div className="border-b px-6">
                  <nav className="flex gap-6 -mb-px">
                    {DETAIL_TABS.map((tab) => (
                      <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={`py-3 text-sm font-medium border-b-2 transition-colors ${
                          activeTab === tab
                            ? 'border-primary-600 text-primary-600'
                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                        }`}
                      >
                        {tab}
                      </button>
                    ))}
                  </nav>
                </div>

                <CardBody>
                  {activeTab === 'Overview' && <OverviewTab job={selectedJob} />}
                  {activeTab === 'Metrics' && (
                    <MetricsTab
                      job={selectedJob}
                      liveMetrics={liveMetrics}
                      lossHistory={lossHistory}
                    />
                  )}
                  {activeTab === 'Epoch Analysis' && (
                    <EpochAnalysisTab epochMetrics={epochMetrics} />
                  )}
                  {activeTab === 'Configuration' && selectedJob.config && (
                    <ConfigurationTab config={selectedJob.config} />
                  )}
                  {activeTab === 'Configuration' && !selectedJob.config && (
                    <div className="text-center py-12 text-gray-400">
                      <p>No configuration data available for this job.</p>
                    </div>
                  )}
                </CardBody>
              </Card>
            </>
          ) : (
            <Card>
              <CardBody className="flex flex-col items-center justify-center h-64 text-gray-500">
                <Cpu className="w-12 h-12 mb-4 text-gray-300" />
                <p className="font-medium">Select a training job to view details</p>
                <p className="text-sm text-gray-400 mt-1">
                  Or create a new one to begin training.
                </p>
              </CardBody>
            </Card>
          )}
        </div>
      </div>

      {/* New Job Modal */}
      <NewJobModal
        open={showNewJob}
        onClose={() => setShowNewJob(false)}
        onCreate={handleCreate}
        datasets={datasets}
        datasetsLoading={datasetsLoading}
      />
    </div>
  );
}
