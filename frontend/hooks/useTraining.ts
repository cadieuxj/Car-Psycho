'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { api } from '@/lib/api';
import { useWebSocket } from './useWebSocket';
import {
  TrainingJob,
  TrainingConfig,
  TrainingMetrics,
  EpochMetrics,
  LossHistoryEntry,
  TrainingWSMessage,
} from '@/lib/types';

interface UseTrainingReturn {
  jobs: TrainingJob[];
  selectedJob: TrainingJob | null;
  isLoading: boolean;
  error: string | null;
  liveMetrics: TrainingMetrics | null;
  lossHistory: LossHistoryEntry[];
  epochMetrics: EpochMetrics[];
  wsStatus: string;
  // Actions
  fetchJobs: () => Promise<void>;
  selectJob: (job: TrainingJob | null) => void;
  createJob: (config: TrainingConfig) => Promise<TrainingJob>;
  stopJob: (jobId: string) => Promise<void>;
  deleteJob: (jobId: string) => Promise<void>;
  refreshJob: (jobId: string) => Promise<void>;
}

export function useTraining(): UseTrainingReturn {
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [selectedJob, setSelectedJob] = useState<TrainingJob | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<TrainingMetrics | null>(null);
  const [lossHistory, setLossHistory] = useState<LossHistoryEntry[]>([]);
  const [epochMetrics, setEpochMetrics] = useState<EpochMetrics[]>([]);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  // WebSocket for live training metrics
  const wsUrl = selectedJob?.status === 'running'
    ? api.getTrainingWSUrl(selectedJob.id)
    : '';

  const handleWSMessage = useCallback((data: unknown) => {
    const msg = data as TrainingWSMessage;

    if (msg.type === 'metric') {
      setLiveMetrics(msg.data);
      setLossHistory(prev => [
        ...prev,
        {
          step: msg.data.step,
          epoch: msg.data.epoch,
          total_loss: msg.data.loss,
          personality_loss: msg.data.personality_loss,
          ordinal_loss: msg.data.ordinal_loss,
          mse_loss: msg.data.mse_loss,
          learning_rate: msg.data.learning_rate,
        },
      ]);
    } else if (msg.type === 'epoch_complete') {
      setEpochMetrics(prev => [...prev, msg.data]);
    } else if (msg.type === 'status') {
      setSelectedJob(prev =>
        prev ? { ...prev, status: msg.status } : null
      );
      // Refresh job list when status changes
      fetchJobs();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const { status: wsStatus } = useWebSocket({
    url: wsUrl,
    onMessage: handleWSMessage,
    enabled: !!wsUrl,
    reconnect: true,
    reconnectInterval: 2000,
  });

  // Polling fallback for when WebSocket is not available
  useEffect(() => {
    if (selectedJob?.status === 'running' && wsStatus !== 'connected') {
      pollRef.current = setInterval(async () => {
        try {
          const job = await api.getJob(selectedJob.id);
          setSelectedJob(job);
          if (job.metrics) {
            setLiveMetrics(job.metrics);
          }
          if (job.loss_history) {
            setLossHistory(job.loss_history);
          }
          if (job.epoch_metrics) {
            setEpochMetrics(job.epoch_metrics);
          }
          if (job.status !== 'running') {
            fetchJobs();
          }
        } catch {
          // Ignore polling errors
        }
      }, 3000);
    }

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [selectedJob?.id, selectedJob?.status, wsStatus]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchJobs = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const fetchedJobs = await api.listJobs();
      setJobs(fetchedJobs);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch jobs');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const selectJob = useCallback((job: TrainingJob | null) => {
    setSelectedJob(job);
    setLiveMetrics(job?.metrics || null);
    setLossHistory(job?.loss_history || []);
    setEpochMetrics(job?.epoch_metrics || []);
  }, []);

  const createJob = useCallback(async (config: TrainingConfig): Promise<TrainingJob> => {
    setError(null);
    try {
      const job = await api.createJob(config);
      await fetchJobs();
      setSelectedJob(job);
      setLossHistory([]);
      setEpochMetrics([]);
      return job;
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to create job';
      setError(msg);
      throw err;
    }
  }, [fetchJobs]);

  const stopJob = useCallback(async (jobId: string) => {
    setError(null);
    try {
      await api.stopJob(jobId);
      await fetchJobs();
      if (selectedJob?.id === jobId) {
        const updated = await api.getJob(jobId);
        setSelectedJob(updated);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to stop job');
    }
  }, [fetchJobs, selectedJob?.id]);

  const deleteJob = useCallback(async (jobId: string) => {
    setError(null);
    try {
      await api.deleteJob(jobId);
      if (selectedJob?.id === jobId) {
        setSelectedJob(null);
        setLossHistory([]);
        setEpochMetrics([]);
        setLiveMetrics(null);
      }
      await fetchJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete job');
    }
  }, [fetchJobs, selectedJob?.id]);

  const refreshJob = useCallback(async (jobId: string) => {
    try {
      const job = await api.getJob(jobId);
      setSelectedJob(job);
      if (job.loss_history) setLossHistory(job.loss_history);
      if (job.epoch_metrics) setEpochMetrics(job.epoch_metrics);
      if (job.metrics) setLiveMetrics(job.metrics);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh job');
    }
  }, []);

  // Initial load
  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  return {
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
  };
}
