'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardBody } from '@/components/ui';
import { Button } from '@/components/ui';
import { Badge } from '@/components/ui';
import { Progress } from '@/components/ui';
import { Model, PredictionResponse } from '@/lib/types';
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
  Cell,
} from 'recharts';
import {
  Box,
  RefreshCw,
  Trash2,
  Play,
  Square,
  CheckCircle,
  AlertCircle,
  Archive,
  Send,
  Loader2,
  Server,
  Info,
  Layers,
  Settings,
  FlaskConical,
} from 'lucide-react';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STATUS_CONFIG = {
  active: { badge: 'success' as const, icon: CheckCircle, label: 'Active' },
  training: { badge: 'info' as const, icon: Loader2, label: 'Training' },
  archived: { badge: 'default' as const, icon: Archive, label: 'Archived' },
};

const TRAIT_COLORS: Record<string, string> = {
  openness: OCEAN_COLORS.openness,
  conscientiousness: OCEAN_COLORS.conscientiousness,
  extraversion: OCEAN_COLORS.extraversion,
  agreeableness: OCEAN_COLORS.agreeableness,
  neuroticism: OCEAN_COLORS.neuroticism,
};

const TRAIT_LABELS: Record<string, string> = {
  openness: OCEAN_LABELS.openness,
  conscientiousness: OCEAN_LABELS.conscientiousness,
  extraversion: OCEAN_LABELS.extraversion,
  agreeableness: OCEAN_LABELS.agreeableness,
  neuroticism: OCEAN_LABELS.neuroticism,
};

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface ModelCardProps {
  model: Model;
  isActive: boolean;
  onSelect: () => void;
  selected: boolean;
  onLoad: () => void;
  onUnload: () => void;
  onDelete: () => void;
  actionLoading: string | null;
}

function ModelCard({
  model,
  isActive,
  onSelect,
  selected,
  onLoad,
  onUnload,
  onDelete,
  actionLoading,
}: ModelCardProps) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const statusCfg = STATUS_CONFIG[model.status];
  const StatusIcon = statusCfg.icon;

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={(e) => e.key === 'Enter' && onSelect()}
      className={`p-4 border rounded-lg transition-all cursor-pointer ${
        selected
          ? 'border-primary-500 bg-primary-50 ring-2 ring-primary-200'
          : isActive
          ? 'border-green-400 bg-green-50 ring-2 ring-green-200'
          : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
      }`}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2 min-w-0">
          <StatusIcon
            className={`w-4 h-4 flex-shrink-0 ${
              model.status === 'training' ? 'animate-spin text-blue-500' : ''
            } ${model.status === 'active' ? 'text-green-500' : ''} ${
              model.status === 'archived' ? 'text-gray-400' : ''
            }`}
          />
          <h4 className="font-medium text-gray-900 truncate">{model.name}</h4>
        </div>
        <Badge variant={statusCfg.badge}>{statusCfg.label}</Badge>
      </div>

      <div className="flex items-center gap-2 text-xs text-gray-500 mb-2">
        <span className="font-mono">v{model.version}</span>
        <span>|</span>
        <span className="truncate">{model.base_model}</span>
      </div>

      {isActive && (
        <div className="flex items-center gap-1 text-xs text-green-700 bg-green-100 rounded px-2 py-1 mb-2">
          <CheckCircle className="w-3 h-3" />
          <span>Currently loaded</span>
        </div>
      )}

      <div className="text-xs text-gray-400 mb-3">
        Created {formatDateTime(model.created_at)}
      </div>

      <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
        {model.is_loaded ? (
          <Button
            variant="secondary"
            size="sm"
            icon={<Square className="w-3 h-3" />}
            loading={actionLoading === `unload-${model.id}`}
            onClick={onUnload}
          >
            Unload
          </Button>
        ) : (
          <Button
            variant="primary"
            size="sm"
            icon={<Play className="w-3 h-3" />}
            loading={actionLoading === `load-${model.id}`}
            onClick={onLoad}
            disabled={model.status === 'training'}
          >
            Load
          </Button>
        )}

        {confirmDelete ? (
          <div className="flex items-center gap-1">
            <Button variant="danger" size="sm" onClick={onDelete}>
              Confirm
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setConfirmDelete(false)}
            >
              Cancel
            </Button>
          </div>
        ) : (
          <Button
            variant="ghost"
            size="sm"
            icon={<Trash2 className="w-3 h-3" />}
            onClick={() => setConfirmDelete(true)}
          >
            Delete
          </Button>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Trait MAE Bar Chart
// ---------------------------------------------------------------------------

function TraitMaeChart({ traitMae }: { traitMae: Record<string, number> }) {
  const data = Object.entries(traitMae).map(([trait, value]) => ({
    trait: TRAIT_LABELS[trait] ?? trait,
    value: Number(value.toFixed(4)),
    color: TRAIT_COLORS[trait] ?? '#6b7280',
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis dataKey="trait" tick={{ fontSize: 11 }} />
        <YAxis tick={{ fontSize: 11 }} />
        <Tooltip />
        <Bar dataKey="value" radius={[4, 4, 0, 0]}>
          {data.map((entry, idx) => (
            <Cell key={idx} fill={entry.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------------------
// OCEAN Prediction Bars
// ---------------------------------------------------------------------------

function OceanResultBars({
  scores,
  confidence,
}: {
  scores: Record<string, number>;
  confidence?: Record<string, number>;
}) {
  const traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'];

  return (
    <div className="space-y-3">
      {traits.map((trait) => {
        const score = scores[trait] ?? 0;
        const pct = Math.round(score * 100);
        const conf = confidence?.[trait];
        return (
          <div key={trait}>
            <div className="flex items-center justify-between text-sm mb-1">
              <span className="font-medium text-gray-700">
                {TRAIT_LABELS[trait] ?? trait}
              </span>
              <span className="tabular-nums text-gray-600">
                {pct}%
                {conf !== undefined && (
                  <span className="text-xs text-gray-400 ml-1">
                    ({Math.round(conf * 100)}% conf.)
                  </span>
                )}
              </span>
            </div>
            <div className="w-full h-3 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${pct}%`,
                  backgroundColor: TRAIT_COLORS[trait],
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Page
// ---------------------------------------------------------------------------

export default function ModelsPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [activeModel, setActiveModel] = useState<Model | null>(null);
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Test inference state
  const [testText, setTestText] = useState('');
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  // Detail tab state
  const [detailTab, setDetailTab] = useState<'overview' | 'metrics' | 'config' | 'test'>(
    'overview'
  );

  // ------- Data fetching -------------------------------------------------

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [modelList, active] = await Promise.all([
        api.listModels(),
        api.getActiveModel(),
      ]);
      setModels(modelList);
      setActiveModel(active);
      if (
        selectedModel &&
        !modelList.find((m) => m.id === selectedModel.id)
      ) {
        setSelectedModel(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models');
    } finally {
      setLoading(false);
    }
  }, [selectedModel]);

  useEffect(() => {
    fetchData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ------- Actions -------------------------------------------------------

  const handleLoad = async (model: Model) => {
    setActionLoading(`load-${model.id}`);
    try {
      await api.loadModel(model.id);
      await fetchData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load model');
    } finally {
      setActionLoading(null);
    }
  };

  const handleUnload = async (model: Model) => {
    setActionLoading(`unload-${model.id}`);
    try {
      await api.unloadModel(model.id);
      await fetchData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to unload model');
    } finally {
      setActionLoading(null);
    }
  };

  const handleDelete = async (model: Model) => {
    setActionLoading(`delete-${model.id}`);
    try {
      await api.deleteModel(model.id);
      if (selectedModel?.id === model.id) setSelectedModel(null);
      await fetchData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete model');
    } finally {
      setActionLoading(null);
    }
  };

  const handlePredict = async () => {
    if (!testText.trim()) return;
    setPredicting(true);
    setPredictionError(null);
    setPrediction(null);
    try {
      const result = await api.predict({ text: testText });
      setPrediction(result);
    } catch (err) {
      setPredictionError(
        err instanceof Error ? err.message : 'Prediction failed'
      );
    } finally {
      setPredicting(false);
    }
  };

  // ------- Render --------------------------------------------------------

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Model Registry</h1>
          <p className="text-gray-500 mt-1">
            Manage, inspect, and test psychometric prediction models
          </p>
        </div>
        <Button
          variant="secondary"
          icon={<RefreshCw className="w-4 h-4" />}
          loading={loading}
          onClick={fetchData}
        >
          Refresh
        </Button>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
          <button
            className="ml-auto text-red-500 hover:text-red-800"
            onClick={() => setError(null)}
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Active Model Indicator */}
      <Card>
        <CardBody>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-green-100">
                <Server className="w-5 h-5 text-green-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Currently Loaded Model</p>
                {activeModel ? (
                  <p className="font-semibold text-gray-900">
                    {activeModel.name}{' '}
                    <span className="text-xs font-mono text-gray-500">
                      v{activeModel.version}
                    </span>
                  </p>
                ) : (
                  <p className="text-sm text-gray-400 italic">
                    No model loaded
                  </p>
                )}
              </div>
            </div>
            {activeModel && (
              <Button
                variant="secondary"
                size="sm"
                icon={<Square className="w-3 h-3" />}
                loading={actionLoading === `unload-${activeModel.id}`}
                onClick={() => handleUnload(activeModel)}
              >
                Unload
              </Button>
            )}
          </div>
        </CardBody>
      </Card>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model Cards List */}
        <div className="lg:col-span-1 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Models</CardTitle>
              <CardDescription>
                {models.length} model{models.length !== 1 ? 's' : ''} registered
              </CardDescription>
            </CardHeader>
            <CardBody className="space-y-3">
              {loading && models.length === 0 ? (
                <div className="flex items-center justify-center py-12 text-gray-400">
                  <Loader2 className="w-6 h-6 animate-spin" />
                </div>
              ) : models.length === 0 ? (
                <div className="text-center py-12 text-gray-400">
                  <Box className="w-10 h-10 mx-auto mb-3" />
                  <p className="text-sm">No models found</p>
                </div>
              ) : (
                models.map((model) => (
                  <ModelCard
                    key={model.id}
                    model={model}
                    isActive={activeModel?.id === model.id}
                    selected={selectedModel?.id === model.id}
                    onSelect={() => {
                      setSelectedModel(model);
                      setDetailTab('overview');
                      setPrediction(null);
                      setPredictionError(null);
                    }}
                    onLoad={() => handleLoad(model)}
                    onUnload={() => handleUnload(model)}
                    onDelete={() => handleDelete(model)}
                    actionLoading={actionLoading}
                  />
                ))
              )}
            </CardBody>
          </Card>
        </div>

        {/* Detail Panel */}
        <div className="lg:col-span-2 space-y-6">
          {selectedModel ? (
            <>
              {/* Tab Navigation */}
              <Card>
                <CardBody className="p-0">
                  <div className="flex border-b border-gray-200">
                    {(
                      [
                        { key: 'overview', label: 'Overview', icon: Info },
                        { key: 'metrics', label: 'Metrics', icon: Layers },
                        { key: 'config', label: 'Configuration', icon: Settings },
                        { key: 'test', label: 'Test Inference', icon: FlaskConical },
                      ] as const
                    ).map(({ key, label, icon: TabIcon }) => (
                      <button
                        key={key}
                        onClick={() => setDetailTab(key)}
                        className={`flex items-center gap-2 px-5 py-3 text-sm font-medium border-b-2 transition-colors ${
                          detailTab === key
                            ? 'border-primary-600 text-primary-700'
                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                        }`}
                      >
                        <TabIcon className="w-4 h-4" />
                        {label}
                      </button>
                    ))}
                  </div>
                </CardBody>
              </Card>

              {/* ---- Overview Tab ---- */}
              {detailTab === 'overview' && (
                <Card>
                  <CardHeader>
                    <CardTitle>{selectedModel.name}</CardTitle>
                    <CardDescription>
                      {selectedModel.description || 'No description provided'}
                    </CardDescription>
                  </CardHeader>
                  <CardBody>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <p className="text-xs text-gray-500 mb-1">Version</p>
                        <p className="font-medium font-mono">{selectedModel.version}</p>
                      </div>
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <p className="text-xs text-gray-500 mb-1">Type</p>
                        <p className="font-medium">{selectedModel.model_type}</p>
                      </div>
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <p className="text-xs text-gray-500 mb-1">Base Model</p>
                        <p className="font-medium truncate">{selectedModel.base_model}</p>
                      </div>
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <p className="text-xs text-gray-500 mb-1">Status</p>
                        <Badge variant={STATUS_CONFIG[selectedModel.status].badge}>
                          {STATUS_CONFIG[selectedModel.status].label}
                        </Badge>
                      </div>
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <p className="text-xs text-gray-500 mb-1">Default</p>
                        <p className="font-medium">
                          {selectedModel.is_default ? 'Yes' : 'No'}
                        </p>
                      </div>
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <p className="text-xs text-gray-500 mb-1">Loaded</p>
                        <p className="font-medium">
                          {selectedModel.is_loaded ? 'Yes' : 'No'}
                        </p>
                      </div>
                      <div className="p-3 bg-gray-50 rounded-lg col-span-2 md:col-span-3">
                        <p className="text-xs text-gray-500 mb-1">Storage Path</p>
                        <p className="font-medium font-mono text-xs truncate">
                          {selectedModel.storage_path}
                        </p>
                      </div>
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <p className="text-xs text-gray-500 mb-1">Created</p>
                        <p className="font-medium text-xs">
                          {formatDateTime(selectedModel.created_at)}
                        </p>
                      </div>
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <p className="text-xs text-gray-500 mb-1">Updated</p>
                        <p className="font-medium text-xs">
                          {formatDateTime(selectedModel.updated_at)}
                        </p>
                      </div>
                    </div>
                  </CardBody>
                </Card>
              )}

              {/* ---- Metrics Tab ---- */}
              {detailTab === 'metrics' && (
                <Card>
                  <CardHeader>
                    <CardTitle>Training Metrics</CardTitle>
                    <CardDescription>
                      Final performance metrics for this model
                    </CardDescription>
                  </CardHeader>
                  <CardBody>
                    {selectedModel.metrics ? (
                      <div className="space-y-6">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="p-4 bg-blue-50 rounded-lg">
                            <p className="text-xs text-blue-600 mb-1">Final Loss</p>
                            <p className="text-2xl font-bold text-blue-900">
                              {(selectedModel.metrics.final_loss ?? selectedModel.metrics.total_loss ?? 0).toFixed(4)}
                            </p>
                          </div>
                          <div className="p-4 bg-purple-50 rounded-lg">
                            <p className="text-xs text-purple-600 mb-1">
                              Personality Loss
                            </p>
                            <p className="text-2xl font-bold text-purple-900">
                              {(selectedModel.metrics.personality_loss ?? 0).toFixed(4)}
                            </p>
                          </div>
                        </div>

                        {selectedModel.metrics.trait_mae &&
                          Object.keys(selectedModel.metrics.trait_mae).length > 0 && (
                            <div>
                              <h4 className="text-sm font-medium text-gray-700 mb-3">
                                Per-Trait Mean Absolute Error
                              </h4>
                              <TraitMaeChart traitMae={selectedModel.metrics.trait_mae} />
                            </div>
                          )}
                      </div>
                    ) : (
                      <div className="text-center py-12 text-gray-400">
                        <Layers className="w-10 h-10 mx-auto mb-3" />
                        <p className="text-sm">No metrics available for this model</p>
                      </div>
                    )}
                  </CardBody>
                </Card>
              )}

              {/* ---- Configuration Tab ---- */}
              {detailTab === 'config' && (
                <Card>
                  <CardHeader>
                    <CardTitle>Model Configuration</CardTitle>
                    <CardDescription>
                      Hyperparameters and settings used during training
                    </CardDescription>
                  </CardHeader>
                  <CardBody>
                    {selectedModel.config &&
                    Object.keys(selectedModel.config).length > 0 ? (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3">
                        {Object.entries(selectedModel.config).map(
                          ([key, value]) => (
                            <div
                              key={key}
                              className="flex items-center justify-between py-2 border-b border-gray-100"
                            >
                              <span className="text-sm text-gray-500">
                                {key.replace(/_/g, ' ')}
                              </span>
                              <span className="text-sm font-medium font-mono text-gray-900">
                                {typeof value === 'object'
                                  ? JSON.stringify(value)
                                  : String(value)}
                              </span>
                            </div>
                          )
                        )}
                      </div>
                    ) : (
                      <div className="text-center py-12 text-gray-400">
                        <Settings className="w-10 h-10 mx-auto mb-3" />
                        <p className="text-sm">
                          No configuration data available
                        </p>
                      </div>
                    )}
                  </CardBody>
                </Card>
              )}

              {/* ---- Test Inference Tab ---- */}
              {detailTab === 'test' && (
                <Card>
                  <CardHeader>
                    <CardTitle>Test Inference</CardTitle>
                    <CardDescription>
                      Enter text to predict OCEAN personality scores using the
                      currently loaded model
                    </CardDescription>
                  </CardHeader>
                  <CardBody>
                    <div className="space-y-4">
                      <textarea
                        value={testText}
                        onChange={(e) => setTestText(e.target.value)}
                        placeholder="Type or paste text here to analyze personality traits..."
                        rows={4}
                        className="w-full rounded-lg border border-gray-300 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 resize-none"
                      />
                      <div className="flex items-center gap-3">
                        <Button
                          icon={<Send className="w-4 h-4" />}
                          loading={predicting}
                          disabled={!testText.trim() || !activeModel}
                          onClick={handlePredict}
                        >
                          Predict
                        </Button>
                        {!activeModel && (
                          <span className="text-xs text-amber-600">
                            Load a model first to run predictions
                          </span>
                        )}
                      </div>

                      {predictionError && (
                        <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
                          <AlertCircle className="w-4 h-4 flex-shrink-0" />
                          <span>{predictionError}</span>
                        </div>
                      )}

                      {prediction && (
                        <div className="mt-4 space-y-4">
                          <h4 className="text-sm font-semibold text-gray-700">
                            OCEAN Personality Scores
                          </h4>
                          <OceanResultBars
                            scores={prediction.personality}
                            confidence={prediction.confidence}
                          />

                          {prediction.trait_descriptions &&
                            Object.keys(prediction.trait_descriptions).length >
                              0 && (
                              <div className="mt-4 p-4 bg-gray-50 rounded-lg space-y-2">
                                <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                                  Trait Descriptions
                                </h4>
                                {Object.entries(
                                  prediction.trait_descriptions
                                ).map(([trait, desc]) => (
                                  <div key={trait} className="flex gap-2 text-sm">
                                    <span
                                      className="w-2 rounded-full flex-shrink-0"
                                      style={{
                                        backgroundColor:
                                          TRAIT_COLORS[trait] ?? '#6b7280',
                                      }}
                                    />
                                    <div>
                                      <span className="font-medium text-gray-800 capitalize">
                                        {TRAIT_LABELS[trait] ?? trait}:
                                      </span>{' '}
                                      <span className="text-gray-600">
                                        {desc}
                                      </span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}
                        </div>
                      )}
                    </div>
                  </CardBody>
                </Card>
              )}
            </>
          ) : (
            <Card>
              <CardBody className="flex flex-col items-center justify-center h-64 text-gray-500">
                <Box className="w-12 h-12 mb-4 text-gray-300" />
                <p>Select a model to view details</p>
              </CardBody>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
