// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, Button, Form, Alert, Table, Badge, Spinner, Row, Col, Nav } from 'react-bootstrap';

const API_BASE_URL = '/api/evaluation';
const VLM_TRAINING_API_BASE_URL = '/api/vlm-training';
const DDM_TRAINING_API_BASE_URL = '/api/ddm-training';
const ANNOTATION_API_BASE_URL = '/api/annotation';

const EVAL_TYPE_PER_CHUNK = 'per-chunk';
const EVAL_TYPE_E2E = 'e2e';

const ACTIVE_STATUSES = new Set(['running', 'queued']);

const formatPercent = (v) =>
  v == null || Number.isNaN(v) ? '—' : `${(Number(v) * 100).toFixed(1)}%`;

const formatDate = (iso) => (iso ? new Date(iso).toLocaleString() : '—');

const getStatusBadgeVariant = (status) => {
  switch (status) {
    case 'running': return 'primary';
    case 'queued': return 'secondary';
    case 'completed': return 'success';
    case 'failed': return 'danger';
    case 'cancelled': return 'warning';
    default: return 'secondary';
  }
};

// Build the Levenshtein DP table for two sequences. Pure, no side effects.
const buildEditDistanceMatrix = (predicted, golden) => {
  const m = predicted.length, n = golden.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = predicted[i - 1] === golden[j - 1]
        ? dp[i - 1][j - 1]
        : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  return dp;
};

// Classify one column during the backtrace step of alignActionSequences.
// Returns the column object + how far to step (di, dj). Splitting this out
// keeps alignActionSequences below the cognitive-complexity threshold
// (SonarJS S3776).
const classifyAlignmentStep = (dp, predicted, golden, i, j) => {
  if (i > 0 && j > 0 && predicted[i - 1] === golden[j - 1]) {
    return { col: { type: 'match', golden: golden[j - 1], predicted: predicted[i - 1] }, di: 1, dj: 1 };
  }
  if (i > 0 && j > 0 && dp[i][j] === dp[i - 1][j - 1] + 1) {
    return { col: { type: 'wrong', golden: golden[j - 1], predicted: predicted[i - 1] }, di: 1, dj: 1 };
  }
  if (i > 0 && dp[i][j] === dp[i - 1][j] + 1) {
    return { col: { type: 'duplicate', predicted: predicted[i - 1] }, di: 1, dj: 0 };
  }
  return { col: { type: 'missing', golden: golden[j - 1] }, di: 0, dj: 1 };
};

// Levenshtein-with-backtrace producing per-column alignment between the
// predicted action sequence and the golden one. Mirrors the backend's
// utils.e2e_eval_utils._calculate_edit_distance so the frontend draws the
// same column structure the backend used to compute wrong/duplicate/missing.
//
// Returns an array of { type, golden?, predicted? } columns in left-to-right
// order. type is one of 'match' | 'wrong' | 'duplicate' | 'missing'.
const alignActionSequences = (predicted, golden) => {
  const dp = buildEditDistanceMatrix(predicted, golden);
  const cols = [];
  let i = predicted.length, j = golden.length;
  while (i > 0 || j > 0) {
    const { col, di, dj } = classifyAlignmentStep(dp, predicted, golden, i, j);
    cols.push(col);
    i -= di;
    j -= dj;
  }
  return cols.reverse();
};

// Color tokens for the sequence-diff visualization. Subtle so a row of
// matched chips fades into the background and the eye is drawn to the few
// columns that disagree. Uses Bootstrap's "subtle" palette equivalents.
const SEQ_CHIP_STYLES = {
  match:     { bg: '#e9ecef', color: '#495057', border: '#dee2e6' },
  wrong:     { bg: '#f8d7da', color: '#842029', border: '#f1aeb5' },
  missing:   { bg: '#cfe2ff', color: '#084298', border: '#9ec5fe' },
  duplicate: { bg: '#fff3cd', color: '#664d03', border: '#ffe69c' },
  gap:       { bg: 'transparent', color: '#adb5bd', border: '#dee2e6' },
};

// Presentational sub-components — kept at module scope so React doesn't
// remount them on every parent re-render, and to satisfy SonarJS S6478.
// We deliberately don't use prop-types (the rest of the codebase doesn't
// either); these disables are the project convention for SonarJS S6774.
// eslint-disable-next-line react/prop-types
const SeqChip = ({ value, status }) => {
  const s = SEQ_CHIP_STYLES[status] || SEQ_CHIP_STYLES.match;
  const isGap = value == null;
  return (
    <div
      title={status === 'gap' ? '(no action at this column)' : `action #${value} (${status})`}
      style={{
        width: 34,
        height: 26,
        lineHeight: '24px',
        textAlign: 'center',
        fontFamily: '"SFMono-Regular", "Menlo", monospace',
        fontSize: '0.78rem',
        fontWeight: 600,
        borderRadius: 4,
        background: s.bg,
        color: s.color,
        border: `1px ${isGap ? 'dashed' : 'solid'} ${s.border}`,
        flexShrink: 0,
      }}
    >
      {isGap ? '·' : value}
    </div>
  );
};

// Reusable metric-tile element. Big number, plain-English caption underneath.
// Hoisted to module scope so it doesn't remount on every parent re-render
// (SonarJS S6478). Same prop-validation convention as SeqChip above.
// eslint-disable-next-line react/prop-types
const MetricTile = ({ label, value, caption, accent, secondary }) => (
  <div
    className="p-3 h-100 rounded border position-relative"
    style={{
      background: '#fff',
      borderLeft: `4px solid ${accent}`,
    }}
  >
    <div
      className="text-muted text-uppercase"
      style={{ fontSize: '0.7rem', letterSpacing: '0.08em', fontWeight: 600 }}
    >
      {label}
    </div>
    <div className="d-flex align-items-baseline mt-1">
      <span className="display-6 fw-semibold" style={{ color: accent, lineHeight: 1 }}>
        {value == null ? '—' : value}
      </span>
      {secondary && (
        <span className="ms-2 text-muted" style={{ fontSize: '0.85rem' }}>
          {secondary}
        </span>
      )}
    </div>
    <div className="text-muted mt-2" style={{ fontSize: '0.78rem', lineHeight: 1.35 }}>
      {caption}
    </div>
  </div>
);

// Map a column of the alignActionSequences output to a SeqChip status
// string. `kind` is "golden" (top row of the diff) or "predicted" (bottom).
// Extracted from a nested ternary in JSX (SonarJS S3358).
const diffStatus = (col, kind) => {
  if (col.type === 'match') return 'match';
  if (col.type === 'wrong') return 'wrong';
  // missing = predicted side has no chip; duplicate = golden side has no chip.
  if (col.type === 'missing' && kind === 'golden') return 'missing';
  if (col.type === 'duplicate' && kind === 'predicted') return 'duplicate';
  return 'gap';
};

// Pick the value to display in a diff chip — null shows the "gap" dash.
const diffValue = (col, kind) => {
  if (kind === 'golden') return col.type === 'duplicate' ? null : col.golden;
  return col.type === 'missing' ? null : col.predicted;
};

// One row of the per-video diff visualisation (Golden or Predicted).
// `cols` is the alignActionSequences output; `kind` selects which side.
// eslint-disable-next-line react/prop-types
const DiffRow = ({ label, cols, kind }) => (
  <div className="d-flex align-items-center" style={kind === 'golden' ? { marginBottom: 4 } : undefined}>
    <small
      className="text-muted me-2"
      style={{ width: 80, textAlign: 'right', fontSize: '0.72rem', fontWeight: 600 }}
    >
      {label}
    </small>
    <div className="d-flex" style={{ gap: 3 }}>
      {/* Columns come from alignActionSequences and are strictly positional —
          no reorder, no insert-in-middle — so the index is the natural key.
          The rule (SonarJS S6479 / react/no-array-index-key) is intentionally
          disabled here. */}
      {cols.map((c, i) => ( // NOSONAR — see comment above re: positional cols + prop validation
        // eslint-disable-next-line react/no-array-index-key
        <SeqChip
          key={`${kind}-${i}`}
          value={diffValue(c, kind)}
          status={diffStatus(c, kind)}
        />
      ))}
    </div>
  </div>
);

const EvaluationPanel = () => {
  // Active eval mode — switches which form / table / results are shown.
  const [activeEvalType, setActiveEvalType] = useState(EVAL_TYPE_PER_CHUNK);

  // Shared dropdowns (populated once + on refresh).
  const [trainingJobs, setTrainingJobs] = useState({});
  const [ddmTrainingJobs, setDdmTrainingJobs] = useState({});
  const [annotationDatasets, setAnnotationDatasets] = useState([]);

  // Shared UI state.
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [statusMessage, setStatusMessage] = useState({ show: false, variant: 'info', message: '' });

  // Per-action-chunk form + results state.
  const [selectedTrainingJobId, setSelectedTrainingJobId] = useState('');
  const [selectedValDatasetId, setSelectedValDatasetId] = useState('');
  const [fps, setFps] = useState(8);
  const [temperature, setTemperature] = useState(0);
  const [backend, setBackend] = useState('vllm');
  const [checkpointStep, setCheckpointStep] = useState('');
  const [resolutionConfigText, setResolutionConfigText] = useState('');
  const [resolutionConfigError, setResolutionConfigError] = useState('');
  const [showResolutionConfig, setShowResolutionConfig] = useState(false);
  const [evalJobs, setEvalJobs] = useState({});
  const [selectedEvalJobId, setSelectedEvalJobId] = useState('');
  const [evalResults, setEvalResults] = useState(null);

  // GPU selector — shared between per-chunk and e2e forms. Refreshed via
  // GET /api/evaluation/api/v1/gpus. selectedGpuId/e2eSelectedGpuId hold the
  // user's chosen GPU index ('' = "Auto", which the backend treats as null
  // and lets the subprocess use whatever the container sees).
  const [gpus, setGpus] = useState([]);
  const [selectedGpuId, setSelectedGpuId] = useState('');
  const [e2eSelectedGpuId, setE2eSelectedGpuId] = useState('');

  // E2E form + results state.
  const [e2eSelectedTrainingJobId, setE2eSelectedTrainingJobId] = useState('');
  const [e2eSelectedDdmTrainingJobId, setE2eSelectedDdmTrainingJobId] = useState('');
  const [e2eSelectedValDatasetId, setE2eSelectedValDatasetId] = useState('');
  // Chunking strategy: 'ddm' uses DDM-Net learned segmentation (default,
  // requires a completed DDM training job); 'uniform' splits every video
  // into fixed-length slices of `e2eChunkLengthSec` seconds (no DDM model
  // needed). Mirrors inference-bp's chunking_options.algorithm field.
  const [e2eChunkingAlgorithm, setE2eChunkingAlgorithm] = useState('ddm');
  const [e2eChunkLengthSec, setE2eChunkLengthSec] = useState(10);
  const [e2eFps, setE2eFps] = useState(8);
  const [e2eTemperature, setE2eTemperature] = useState(0);
  const [e2eBackend, setE2eBackend] = useState('vllm');
  const [e2eCheckpointStep, setE2eCheckpointStep] = useState('');
  const [e2eDdmCheckpoint, setE2eDdmCheckpoint] = useState('');
  const [e2eScoreThreshold, setE2eScoreThreshold] = useState(0.5);
  const [e2eNmsSec, setE2eNmsSec] = useState(0);
  const [e2eDdmBatchSize, setE2eDdmBatchSize] = useState(8);
  const [e2eFramesPerSegmentHint, setE2eFramesPerSegmentHint] = useState(256);
  const [e2eResolutionConfigText, setE2eResolutionConfigText] = useState('');
  const [e2eResolutionConfigError, setE2eResolutionConfigError] = useState('');
  const [showE2eAdvanced, setShowE2eAdvanced] = useState(false);
  const [e2eEvalJobs, setE2eEvalJobs] = useState({});
  const [selectedE2eEvalJobId, setSelectedE2eEvalJobId] = useState('');
  const [e2eEvalResults, setE2eEvalResults] = useState(null);
  // Per-video diff rows the user has expanded (set of video_name strings).
  const [expandedE2eRows, setExpandedE2eRows] = useState(() => new Set());
  // Whether the secondary-detail (chunk-level / temporal-table) sections are open.
  const [showE2eSecondary, setShowE2eSecondary] = useState(false);

  const pollingIntervalRef = useRef(null);
  // Separate interval that re-fetches the upstream dropdowns (VLM training,
  // DDM training, annotation datasets) so newly-completed jobs / new datasets
  // appear in this panel without the user having to click Refresh. The
  // VLMTrainingPanel / DDMTrainingPanel only poll their *own* active jobs;
  // the Evaluation panel is the only consumer that needs cross-service
  // visibility, so it owns this poll.
  const dropdownsPollRef = useRef(null);
  const DROPDOWNS_POLL_INTERVAL_MS = 15000;

  // --- Fetchers ---------------------------------------------------------------

  const fetchAllTrainingJobs = useCallback(async () => {
    try {
      const response = await fetch(`${VLM_TRAINING_API_BASE_URL}/api/v1/fine-tuning/all_jobs`);
      if (!response.ok) {
        console.error('Failed to fetch training jobs: HTTP', response.status);
        return;
      }
      setTrainingJobs(await response.json());
    } catch (error) {
      console.error('Failed to fetch training jobs:', error);
    }
  }, []);

  const fetchAllDdmTrainingJobs = useCallback(async () => {
    try {
      const response = await fetch(`${DDM_TRAINING_API_BASE_URL}/api/v1/fine-tuning/all_jobs`);
      if (!response.ok) {
        console.error('Failed to fetch DDM training jobs: HTTP', response.status);
        return;
      }
      setDdmTrainingJobs(await response.json());
    } catch (error) {
      console.error('Failed to fetch DDM training jobs:', error);
    }
  }, []);

  const fetchAnnotationDatasets = useCallback(async () => {
    try {
      const response = await fetch(`${ANNOTATION_API_BASE_URL}/api/v1/datasets`);
      if (!response.ok) {
        console.error('Failed to fetch annotation datasets: HTTP', response.status);
        return;
      }
      const data = await response.json();
      const datasets = Array.isArray(data) ? data : Object.keys(data);
      const filtered = datasets.filter(ds => {
        const id = typeof ds === 'string' ? ds : (ds.dataset_id || ds.id || '');
        return !id.includes('_augmented');
      });
      setAnnotationDatasets(filtered);
    } catch (error) {
      console.error('Failed to fetch annotation datasets:', error);
    }
  }, []);

  const fetchGpus = useCallback(async () => {
    // Best-effort — if eval_ms isn't reachable yet, just leave gpus empty
    // and the dropdown falls back to "Auto" only. Mirrors the same lenient
    // pattern used by fetchAnnotationDatasets etc.
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/gpus`);
      if (!response.ok) return;
      const data = await response.json();
      setGpus(Array.isArray(data?.gpus) ? data.gpus : []);
    } catch (error) {
      console.error('Failed to fetch GPU list:', error);
    }
  }, []);

  const fetchAllEvalJobs = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/all_jobs`);
      if (!response.ok) return;
      setEvalJobs(await response.json());
    } catch (error) {
      console.error('Failed to fetch evaluation jobs:', error);
    }
  }, []);

  const fetchAllE2eEvalJobs = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/e2e-evaluation/all_jobs`);
      if (!response.ok) return;
      setE2eEvalJobs(await response.json());
    } catch (error) {
      console.error('Failed to fetch e2e evaluation jobs:', error);
    }
  }, []);

  const fetchEvalResults = useCallback(async (evalJobId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/results/${evalJobId}`);
      if (!response.ok) return;
      setEvalResults(await response.json());
      setSelectedEvalJobId(evalJobId);
    } catch (error) {
      console.error('Failed to fetch eval results:', error);
    }
  }, []);

  const fetchE2eEvalResults = useCallback(async (evalJobId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/e2e-evaluation/results/${evalJobId}`);
      if (!response.ok) return;
      setE2eEvalResults(await response.json());
      setSelectedE2eEvalJobId(evalJobId);
    } catch (error) {
      console.error('Failed to fetch e2e eval results:', error);
    }
  }, []);

  // --- Polling ----------------------------------------------------------------

  const stopPolling = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  };

  // Reconcile one /all_jobs response: update state, fetch results for jobs
  // that just finished. Extracted from pollActive so the parent stays under
  // the cognitive-complexity threshold (SonarJS S3776).
  const reconcileJobs = useCallback(async (response, prev, setJobs, fetchResults) => {
    if (!response.ok) return prev;
    const data = await response.json();
    setJobs(data);
    for (const [id, job] of Object.entries(data)) {
      if (job.status === 'completed' && prev[id]?.status !== 'completed') {
        await fetchResults(id);
      }
    }
    return data;
  }, []);

  const pollActive = useCallback(async (prevPerChunk, prevE2e) => {
    try {
      const [r1, r2] = await Promise.all([
        fetch(`${API_BASE_URL}/api/v1/evaluation/all_jobs`),
        fetch(`${API_BASE_URL}/api/v1/e2e-evaluation/all_jobs`),
      ]);
      const nextPerChunk = await reconcileJobs(r1, prevPerChunk, setEvalJobs, fetchEvalResults);
      const nextE2e = await reconcileJobs(r2, prevE2e, setE2eEvalJobs, fetchE2eEvalResults);

      const stillActive =
        Object.values(nextPerChunk).some(j => ACTIVE_STATUSES.has(j.status)) ||
        Object.values(nextE2e).some(j => ACTIVE_STATUSES.has(j.status));
      if (!stillActive) stopPolling();
    } catch (error) {
      console.error('Polling error:', error);
    }
  }, [reconcileJobs, fetchEvalResults, fetchE2eEvalResults]);

  const startPolling = useCallback(() => {
    stopPolling();
    // Snapshot current state for "newly completed" comparison on each tick.
    pollingIntervalRef.current = setInterval(() => {
      pollActive(evalJobs, e2eEvalJobs);
    }, 15000);
  }, [pollActive, evalJobs, e2eEvalJobs]);

  // --- Mount / unmount --------------------------------------------------------

  useEffect(() => {
    const init = async () => {
      await Promise.all([
        fetchAllTrainingJobs(),
        fetchAllDdmTrainingJobs(),
        fetchAnnotationDatasets(),
        fetchAllEvalJobs(),
        fetchAllE2eEvalJobs(),
        fetchGpus(),
      ]);
    };
    init();

    // Background refresh for the upstream dropdown lists. Quiet — no UI
    // loading state; the user shouldn't see anything unless the dropdown
    // contents change. Errors are swallowed by each individual fetch.
    // GPU list is also refreshed here so free-memory readings stay current.
    const refreshUpstream = () => {
      fetchAllTrainingJobs();
      fetchAllDdmTrainingJobs();
      fetchAnnotationDatasets();
      fetchGpus();
    };
    dropdownsPollRef.current = setInterval(refreshUpstream, DROPDOWNS_POLL_INTERVAL_MS);

    return () => {
      stopPolling();
      if (dropdownsPollRef.current) {
        clearInterval(dropdownsPollRef.current);
        dropdownsPollRef.current = null;
      }
    };
  }, [
    fetchAllTrainingJobs,
    fetchAllDdmTrainingJobs,
    fetchAnnotationDatasets,
    fetchAllEvalJobs,
    fetchAllE2eEvalJobs,
    fetchGpus,
  ]);

  // Kick off polling whenever there are active jobs on either track.
  useEffect(() => {
    const anyActive =
      Object.values(evalJobs).some(j => ACTIVE_STATUSES.has(j.status)) ||
      Object.values(e2eEvalJobs).some(j => ACTIVE_STATUSES.has(j.status));
    if (anyActive && !pollingIntervalRef.current) {
      startPolling();
    } else if (!anyActive && pollingIntervalRef.current) {
      stopPolling();
    }
  }, [evalJobs, e2eEvalJobs, startPolling]);

  const refreshAll = async () => {
    setIsRefreshing(true);
    try {
      await Promise.all([
        fetchAllTrainingJobs(),
        fetchAllDdmTrainingJobs(),
        fetchAnnotationDatasets(),
        fetchAllEvalJobs(),
        fetchAllE2eEvalJobs(),
        fetchGpus(),
      ]);
    } finally {
      setIsRefreshing(false);
    }
  };

  // --- Resolution-config helper (shared by both forms) ------------------------

  const parseResolutionConfig = (text, setError) => {
    const trimmed = text.trim();
    if (trimmed === '') {
      setError('');
      return { valid: true, value: undefined };
    }
    try {
      const parsed = JSON.parse(trimmed);
      if (typeof parsed !== 'object' || Array.isArray(parsed)) {
        setError('Must be a JSON object, e.g. {"max_frames": 40, "total_pixels": 16572416}');
        return { valid: false };
      }
      setError('');
      return { valid: true, value: parsed };
    } catch {
      setError('Invalid JSON');
      return { valid: false };
    }
  };

  const handleResolutionConfigChange = (val) => {
    setResolutionConfigText(val);
    parseResolutionConfig(val, setResolutionConfigError);
  };

  const handleE2eResolutionConfigChange = (val) => {
    setE2eResolutionConfigText(val);
    parseResolutionConfig(val, setE2eResolutionConfigError);
  };

  // --- Per-chunk actions ------------------------------------------------------

  const handleRunEvaluation = async () => {
    if (!selectedTrainingJobId) {
      setStatusMessage({ show: true, variant: 'warning', message: 'Please select a training experiment.' });
      return;
    }
    if (!selectedValDatasetId) {
      setStatusMessage({ show: true, variant: 'warning', message: 'Please select a validation dataset.' });
      return;
    }
    const { valid, value: resolutionConfig } =
      parseResolutionConfig(resolutionConfigText, setResolutionConfigError);
    if (!valid) {
      setStatusMessage({ show: true, variant: 'warning', message: 'Please fix the resolution config JSON before submitting.' });
      return;
    }

    setIsSubmitting(true);
    setStatusMessage({ show: true, variant: 'info', message: 'Submitting evaluation job...' });

    try {
      const body = {
        training_job_id: selectedTrainingJobId,
        val_dataset_id: selectedValDatasetId,
        fps: Number(fps),
        temperature: Number(temperature),
        backend,
      };
      if (checkpointStep !== '' && checkpointStep !== null) {
        body.checkpoint_step = Number(checkpointStep);
      }
      if (resolutionConfig !== undefined) body.resolution_config = resolutionConfig;
      // gpu_id: empty string = "Auto" (omit field so backend gets null).
      if (selectedGpuId !== '') body.gpu_id = Number(selectedGpuId);

      const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', accept: 'application/json' },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to start evaluation: ${response.status}`);
      }
      const result = await response.json();
      setEvalJobs(prev => ({
        ...prev,
        [result.eval_job_id]: {
          training_job_id: selectedTrainingJobId,
          val_dataset_id: selectedValDatasetId,
          status: result.status,
          overall_accuracy: null,
          checkpoint_step: checkpointStep === '' ? null : Number(checkpointStep),
          created_at: result.created_at,
          updated_at: result.created_at,
        },
      }));
      setStatusMessage({
        show: true,
        variant: 'success',
        message: `Evaluation job started. Job ID: ${result.eval_job_id}`,
      });
    } catch (error) {
      setStatusMessage({ show: true, variant: 'danger', message: `Failed to start evaluation: ${error.message}` });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancelEval = async (evalJobId) => {
    if (!window.confirm('Cancel this evaluation job?')) return;
    setIsCancelling(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/cancel/${evalJobId}`, {
        method: 'POST',
        headers: { accept: 'application/json', 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }
      setEvalJobs(prev => ({
        ...prev,
        [evalJobId]: { ...prev[evalJobId], status: 'cancelled', updated_at: new Date().toISOString() },
      }));
      setStatusMessage({ show: true, variant: 'warning', message: `Evaluation job ${evalJobId} cancelled.` });
    } catch (error) {
      setStatusMessage({ show: true, variant: 'danger', message: `Failed to cancel: ${error.message}` });
    } finally {
      setIsCancelling(false);
    }
  };

  const handleViewEvalResults = async (evalJobId) => {
    if (evalJobs[evalJobId]?.status !== 'completed') return;
    await fetchEvalResults(evalJobId);
  };

  // --- E2E actions ------------------------------------------------------------

  // Validate the e2e form. Returns { ok: true, resolutionConfig } on success
  // or { ok: false, message } on failure. Extracted so handleRunE2eEvaluation
  // stays under the cognitive-complexity threshold (SonarJS S3776).
  const validateE2eForm = () => {
    if (!e2eSelectedTrainingJobId) {
      return { ok: false, message: 'Please select a VLM training experiment.' };
    }
    if (e2eChunkingAlgorithm === 'ddm' && !e2eSelectedDdmTrainingJobId) {
      return { ok: false, message: 'Please select a DDM training experiment (or switch to uniform chunking).' };
    }
    if (e2eChunkingAlgorithm === 'uniform' && (!e2eChunkLengthSec || Number(e2eChunkLengthSec) <= 0)) {
      return { ok: false, message: 'Please set a positive chunk length (sec) for uniform chunking.' };
    }
    if (!e2eSelectedValDatasetId) {
      return { ok: false, message: 'Please select a validation dataset.' };
    }
    const { valid, value: resolutionConfig } =
      parseResolutionConfig(e2eResolutionConfigText, setE2eResolutionConfigError);
    if (!valid) {
      return { ok: false, message: 'Please fix the resolution config JSON before submitting.' };
    }
    return { ok: true, resolutionConfig };
  };

  // Build the POST body for /api/v1/e2e-evaluation/start. Pure — no setState.
  const buildE2eRequestBody = (resolutionConfig) => {
    const body = {
      training_job_id: e2eSelectedTrainingJobId,
      val_dataset_id: e2eSelectedValDatasetId,
      fps: Number(e2eFps),
      temperature: Number(e2eTemperature),
      backend: e2eBackend,
      chunking_algorithm: e2eChunkingAlgorithm,
      frames_per_segment_hint: Number(e2eFramesPerSegmentHint),
    };
    if (e2eChunkingAlgorithm === 'ddm') {
      body.ddm_training_job_id = e2eSelectedDdmTrainingJobId;
      body.score_threshold = Number(e2eScoreThreshold);
      body.nms_sec = Number(e2eNmsSec);
      body.ddm_batch_size = Number(e2eDdmBatchSize);
      if (e2eDdmCheckpoint.trim() !== '') body.ddm_checkpoint = e2eDdmCheckpoint.trim();
    } else {
      body.chunk_length_sec = Number(e2eChunkLengthSec);
    }
    if (e2eCheckpointStep !== '' && e2eCheckpointStep !== null) {
      body.checkpoint_step = Number(e2eCheckpointStep);
    }
    if (resolutionConfig !== undefined) body.resolution_config = resolutionConfig;
    // gpu_id: empty string = "Auto" (omit field so backend gets null).
    if (e2eSelectedGpuId !== '') body.gpu_id = Number(e2eSelectedGpuId);
    return body;
  };

  const handleRunE2eEvaluation = async () => {
    const v = validateE2eForm();
    if (!v.ok) {
      setStatusMessage({ show: true, variant: 'warning', message: v.message });
      return;
    }

    setIsSubmitting(true);
    setStatusMessage({ show: true, variant: 'info', message: 'Submitting e2e evaluation job...' });

    try {
      const body = buildE2eRequestBody(v.resolutionConfig);
      const response = await fetch(`${API_BASE_URL}/api/v1/e2e-evaluation/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', accept: 'application/json' },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to start e2e evaluation: ${response.status}`);
      }
      const result = await response.json();
      setE2eEvalJobs(prev => ({
        ...prev,
        [result.eval_job_id]: {
          training_job_id: e2eSelectedTrainingJobId,
          ddm_training_job_id: e2eChunkingAlgorithm === 'ddm' ? e2eSelectedDdmTrainingJobId : null,
          val_dataset_id: e2eSelectedValDatasetId,
          status: result.status,
          overall_accuracy: null,
          avg_f1: null,
          checkpoint_step: e2eCheckpointStep === '' ? null : Number(e2eCheckpointStep),
          chunking_algorithm: e2eChunkingAlgorithm,
          chunk_length_sec: e2eChunkingAlgorithm === 'uniform' ? Number(e2eChunkLengthSec) : null,
          created_at: result.created_at,
          updated_at: result.created_at,
        },
      }));
      setStatusMessage({
        show: true,
        variant: 'success',
        message: `E2E evaluation job started. Job ID: ${result.eval_job_id}`,
      });
    } catch (error) {
      setStatusMessage({ show: true, variant: 'danger', message: `Failed to start e2e evaluation: ${error.message}` });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancelE2eEval = async (evalJobId) => {
    if (!window.confirm('Cancel this e2e evaluation job?')) return;
    setIsCancelling(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/e2e-evaluation/cancel/${evalJobId}`, {
        method: 'POST',
        headers: { accept: 'application/json', 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }
      setE2eEvalJobs(prev => ({
        ...prev,
        [evalJobId]: { ...prev[evalJobId], status: 'cancelled', updated_at: new Date().toISOString() },
      }));
      setStatusMessage({ show: true, variant: 'warning', message: `E2E evaluation job ${evalJobId} cancelled.` });
    } catch (error) {
      setStatusMessage({ show: true, variant: 'danger', message: `Failed to cancel: ${error.message}` });
    } finally {
      setIsCancelling(false);
    }
  };

  const handleViewE2eEvalResults = async (evalJobId) => {
    if (e2eEvalJobs[evalJobId]?.status !== 'completed') return;
    await fetchE2eEvalResults(evalJobId);
  };

  // --- Derived ---------------------------------------------------------------

  const completedTrainingJobs = Object.entries(trainingJobs).filter(
    ([, job]) => job.status === 'completed'
  );
  const completedDdmTrainingJobs = Object.entries(ddmTrainingJobs).filter(
    ([, job]) => job.status === 'completed'
  );
  const sortedEvalJobs = Object.entries(evalJobs).sort(
    ([, a], [, b]) => new Date(b.created_at || 0) - new Date(a.created_at || 0)
  );
  const sortedE2eEvalJobs = Object.entries(e2eEvalJobs).sort(
    ([, a], [, b]) => new Date(b.created_at || 0) - new Date(a.created_at || 0)
  );

  // --- Render helpers --------------------------------------------------------

  const renderDatasetOptions = () =>
    annotationDatasets.map((ds) => {
      const id = typeof ds === 'string' ? ds : (ds.dataset_id || ds.id || '');
      const label = typeof ds === 'string' ? ds : (ds.name || ds.dataset_id || ds.id || id);
      return <option key={id} value={id}>{label}</option>;
    });

  const renderResolutionConfigField = (text, err, onChange, disabled) => (
    <div className="mt-2">
      <Form.Control
        as="textarea"
        rows={3}
        placeholder={
          'Leave blank for default  {"max_frames": 40, "total_pixels": 16572416}\n' +
          '(matches assets/config/train_config.toml — same resolution as training)\n' +
          'Example override: {"resized_height": 567, "resized_width": 1008, "max_frames": 40}'
        }
        value={text}
        onChange={(e) => onChange(e.target.value)}
        isInvalid={!!err}
        disabled={disabled}
        style={{ fontFamily: 'monospace', fontSize: '0.85rem' }}
      />
      {err
        ? <Form.Control.Feedback type="invalid">{err}</Form.Control.Feedback>
        : (
          <Form.Text className="text-muted">
            Common keys: <code>max_frames</code>, <code>total_pixels</code>,
            {' '}<code>resized_height</code>, <code>resized_width</code>,
            {' '}<code>max_pixels</code>, <code>min_pixels</code>
          </Form.Text>
        )
      }
    </div>
  );

  // Render a GPU selector dropdown bound to the given (value, setValue) pair.
  // Reads `gpus` state populated by /api/v1/gpus. Empty value = "Auto" (the
  // backend treats null gpu_id as "use whatever the container sees").
  const renderGpuSelector = ({ value, setValue, disabled }) => {
    const fmtMb = (mb) => (mb == null ? '?' : `${(mb / 1024).toFixed(1)} GiB`);
    // Extracted from a nested ternary in JSX (SonarJS S3358).
    const fmtGpuMemory = (g) => {
      if (g.free_memory_mb != null && g.total_memory_mb != null) {
        return ` (${fmtMb(g.free_memory_mb)} free / ${fmtMb(g.total_memory_mb)})`;
      }
      if (g.total_memory_mb != null) {
        return ` (${fmtMb(g.total_memory_mb)})`;
      }
      return '';
    };
    const gpusPlural = gpus.length === 1 ? '' : 's';
    const helpText = gpus.length === 0
      ? 'No GPUs visible to the eval container — pinning disabled.'
      : `${gpus.length} GPU${gpusPlural} visible. "Auto" lets the subprocess use whatever it sees.`;
    return (
      <Form.Group className="mb-3">
        <Form.Label>GPU</Form.Label>
        <Form.Select
          value={value}
          onChange={(e) => setValue(e.target.value)}
          disabled={disabled || gpus.length === 0}
        >
          <option value="">Auto (any visible GPU)</option>
          {gpus.map((g) => (
            <option key={g.index} value={String(g.index)}>
              GPU {g.index} — {g.name}{fmtGpuMemory(g)}
            </option>
          ))}
        </Form.Select>
        <Form.Text className="text-muted">{helpText}</Form.Text>
      </Form.Group>
    );
  };

  // --- Per-chunk view ---------------------------------------------------------

  const renderPerChunkView = () => (
    <>
      <div className="mb-3 pb-2 border-bottom">
        <div className="d-flex align-items-center">
          <Badge bg="info" className="me-2">Per-action-chunk</Badge>
          <small className="text-muted">
            VLM inference on pre-chunked validation videos. Reports per-action accuracy.
          </small>
        </div>
      </div>

      <Row className="mb-3">
        <Col md={6}>
          <Form.Group className="mb-3">
            <Form.Label>Training Experiment</Form.Label>
            <Form.Select
              value={selectedTrainingJobId}
              onChange={(e) => setSelectedTrainingJobId(e.target.value)}
              disabled={isSubmitting}
            >
              <option value="">Choose a completed training job…</option>
              {completedTrainingJobs.map(([jobId, job]) => (
                <option key={jobId} value={jobId}>
                  {jobId}{job.aug_dataset_id ? ` (dataset: ${job.aug_dataset_id})` : ''}
                </option>
              ))}
            </Form.Select>
            {completedTrainingJobs.length === 0 && (
              <Form.Text className="text-muted">
                No completed VLM training jobs yet. Use the refresh button above if you just finished one.
              </Form.Text>
            )}
          </Form.Group>
        </Col>
        <Col md={6}>
          <Form.Group className="mb-3">
            <Form.Label>Validation Dataset</Form.Label>
            <Form.Select
              value={selectedValDatasetId}
              onChange={(e) => setSelectedValDatasetId(e.target.value)}
              disabled={isSubmitting}
            >
              <option value="">Choose a validation dataset…</option>
              {renderDatasetOptions()}
            </Form.Select>
          </Form.Group>
        </Col>
      </Row>

      <Row className="mb-3">
        <Col md={4}>
          <Form.Group className="mb-3">
            <Form.Label>Backend</Form.Label>
            <Form.Select
              value={backend}
              onChange={(e) => setBackend(e.target.value)}
              disabled={isSubmitting}
            >
              <option value="vllm">vllm</option>
              <option value="transformers">transformers</option>
            </Form.Select>
          </Form.Group>
        </Col>
        <Col md={2}>
          <Form.Group className="mb-3">
            <Form.Label>FPS</Form.Label>
            <Form.Control
              type="number"
              value={fps}
              min={1}
              max={30}
              onChange={(e) => setFps(e.target.value)}
              disabled={isSubmitting}
            />
          </Form.Group>
        </Col>
        <Col md={2}>
          <Form.Group className="mb-3">
            <Form.Label>Temperature</Form.Label>
            <Form.Control
              type="number"
              value={temperature}
              min={0}
              max={2}
              step={0.1}
              onChange={(e) => setTemperature(e.target.value)}
              disabled={isSubmitting}
            />
          </Form.Group>
        </Col>
        <Col md={4}>
          <Form.Group className="mb-3">
            <Form.Label>Checkpoint Step <small className="text-muted">(optional)</small></Form.Label>
            <Form.Control
              type="number"
              value={checkpointStep}
              min={1}
              placeholder="Latest checkpoint"
              onChange={(e) => setCheckpointStep(e.target.value)}
              disabled={isSubmitting}
            />
          </Form.Group>
        </Col>
      </Row>

      <Row className="mb-3">
        <Col md={6}>
          {renderGpuSelector({
            value: selectedGpuId,
            setValue: setSelectedGpuId,
            disabled: isSubmitting,
          })}
        </Col>
      </Row>

      <div className="mb-3">
        <Button
          variant="link"
          className="p-0 text-decoration-none text-secondary"
          size="sm"
          onClick={() => setShowResolutionConfig(v => !v)}
        >
          <i className={`bi bi-chevron-${showResolutionConfig ? 'up' : 'down'} me-1`} /> Advanced: Resolution Config
        </Button>
        {showResolutionConfig && renderResolutionConfigField(
          resolutionConfigText,
          resolutionConfigError,
          handleResolutionConfigChange,
          isSubmitting,
        )}
      </div>

      <div className="mb-4">
        <Button
          variant="success"
          onClick={handleRunEvaluation}
          disabled={!selectedTrainingJobId || !selectedValDatasetId || isSubmitting}
        >
          {isSubmitting ? (
            <>
              <Spinner animation="border" size="sm" className="me-2" />
              Starting Evaluation…
            </>
          ) : 'Run Evaluation'}
        </Button>
      </div>

      <h5 className="mb-3">Per-action-chunk Jobs</h5>
      {sortedEvalJobs.length === 0 ? (
        <Alert variant="info">No evaluation jobs yet. Configure and run an evaluation above.</Alert>
      ) : (
        <Table bordered hover responsive size="sm" className="mb-4">
          <thead>
            <tr>
              <th>Eval Job ID</th>
              <th>Training Experiment</th>
              <th>Val Dataset</th>
              <th>Status</th>
              <th>Accuracy</th>
              <th>Created</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {sortedEvalJobs.map(([evalJobId, job]) => (
              <tr
                key={evalJobId}
                onClick={() => handleViewEvalResults(evalJobId)}
                style={{ cursor: job.status === 'completed' ? 'pointer' : 'default' }}
                className={selectedEvalJobId === evalJobId ? 'table-active' : ''}
              >
                <td><small className="font-monospace">{evalJobId}</small></td>
                <td><small>{job.training_job_id}</small></td>
                <td><small>{job.val_dataset_id}</small></td>
                <td>
                  <Badge bg={getStatusBadgeVariant(job.status)}>
                    {job.status ? job.status.toUpperCase() : 'UNKNOWN'}
                  </Badge>
                  {ACTIVE_STATUSES.has(job.status) && (
                    <Spinner animation="border" size="sm" className="ms-2" role="status">
                      <span className="visually-hidden">Running…</span>
                    </Spinner>
                  )}
                </td>
                <td>{formatPercent(job.overall_accuracy)}</td>
                <td><small>{formatDate(job.created_at)}</small></td>
                <td onClick={(e) => e.stopPropagation()}>
                  {ACTIVE_STATUSES.has(job.status) && (
                    <Button
                      size="sm"
                      variant="outline-danger"
                      onClick={() => handleCancelEval(evalJobId)}
                      disabled={isCancelling}
                    >
                      {isCancelling ? <Spinner animation="border" size="sm" /> : 'Cancel'}
                    </Button>
                  )}
                  {job.status === 'completed' && (
                    <Button
                      size="sm"
                      variant="outline-primary"
                      onClick={() => handleViewEvalResults(evalJobId)}
                    >
                      View Results
                    </Button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}

      {selectedEvalJobId && evalResults && (
        <Card className="mt-3 border-primary">
          <Card.Header>
            <strong>Results: {selectedEvalJobId}</strong>
          </Card.Header>
          <Card.Body>
            <div className="mb-3">
              <h5>
                Overall Accuracy:{' '}
                <span className="text-success">{formatPercent(evalResults.overall_accuracy)}</span>
              </h5>
            </div>
            {evalResults.per_action && Object.keys(evalResults.per_action).length > 0 && (
              <Table bordered responsive size="sm">
                <thead>
                  <tr>
                    <th>Action #</th>
                    <th>Label</th>
                    <th>Accuracy</th>
                    <th>n (correct / total)</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(evalResults.per_action)
                    .sort(([a], [b]) => Number(a) - Number(b))
                    .map(([actionNum, d]) => (
                      <tr key={actionNum}>
                        <td>{actionNum}</td>
                        <td>{d.label || '—'}</td>
                        <td>{formatPercent(d.accuracy)}</td>
                        <td>
                          {d.correct == null || d.total == null
                            ? '—' : `${d.correct} / ${d.total}`}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </Table>
            )}
          </Card.Body>
        </Card>
      )}
    </>
  );

  // --- E2E view ---------------------------------------------------------------

  const renderE2eResults = () => {
    if (!selectedE2eEvalJobId || !e2eEvalResults) return null;
    const r = e2eEvalResults;

    // ─── Schema normalization ─────────────────────────────────────────────
    // New schema (post sequence-level eval upgrade):
    //   { temporal_segmentation: {avg_f1, per_video: {...}}, action_recognition: {...rich...} }
    // Legacy schema (older jobs): metrics at top level.
    const ts = r.temporal_segmentation || {};
    const ar = r.action_recognition || r;  // legacy: ar might be the root dict
    const isNewSchema = ar.sequence_accuracy !== undefined;

    // Headline values (new schema preferred; falls through to legacy keys).
    const seqAcc = ar.sequence_accuracy;
    const actAcc = ar.action_accuracy ?? ar.overall_accuracy;
    const tempF1 = ts.avg_f1 ?? r.avg_f1;
    const totalVideos = ar.total_videos ?? null;
    const totalVideosOk = ar.total_videos_dist_0 ?? null;
    const totalActions = ar.total_actions ?? null;

    // Per-video table data: merge action-recognition diff with temporal F1 by video name.
    const arPerVideo = Array.isArray(ar.per_video) ? ar.per_video : [];
    const tsPerVideo = ts.per_video || {};
    const perVideoRows = arPerVideo.map(v => ({
      ...v,
      f1: tsPerVideo[v.video]?.f1,
      precision: tsPerVideo[v.video]?.precision,
      recall: tsPerVideo[v.video]?.recall,
    }));

    const toggleRow = (video) => {
      setExpandedE2eRows(prev => {
        const next = new Set(prev);
        if (next.has(video)) next.delete(video); else next.add(video);
        return next;
      });
    };

    return (
      <Card className="mt-3 border-primary">
        <Card.Header className="d-flex justify-content-between align-items-center">
          <div>
            <strong>Results</strong>
            <span className="ms-2 font-monospace text-muted" style={{ fontSize: '0.85rem' }}>
              {selectedE2eEvalJobId}
            </span>
          </div>
          {!isNewSchema && (
            <Badge bg="secondary" className="ms-2" title="This job was evaluated before the sequence-level upgrade.">
              Legacy result
            </Badge>
          )}
        </Card.Header>
        <Card.Body>

          {/* ─── HERO METRICS ────────────────────────────────────────── */}
          <Row className="g-3 mb-4">
            {isNewSchema ? (
              <>
                <Col md={4}>
                  <MetricTile
                    label="Sequence Accuracy"
                    value={formatPercent(seqAcc)}
                    secondary={
                      totalVideos != null
                      && `${totalVideosOk}/${totalVideos} videos`
                    }
                    caption="Fraction of videos where the predicted action sequence matches the ground truth end-to-end (edit distance = 0)."
                    accent="#198754"
                  />
                </Col>
                <Col md={4}>
                  <MetricTile
                    label="Action Accuracy"
                    value={formatPercent(actAcc)}
                    secondary={
                      totalActions != null
                      && `${totalActions - (ar.wrong || 0) - (ar.duplicate || 0) - (ar.missing || 0)}/${totalActions} actions`
                    }
                    caption="Percentage of individual actions correctly identified, accounting for wrong, extra, and missing detections."
                    accent="#0d6efd"
                  />
                </Col>
                <Col md={4}>
                  <MetricTile
                    label="Temporal F1"
                    value={formatPercent(tempF1)}
                    caption="How well DDM-Net detected action boundaries on the timeline (F1 between predicted and golden boundaries)."
                    accent="#6610f2"
                  />
                </Col>
              </>
            ) : (
              // Legacy fallback: only the two metrics the old backend emitted.
              <>
                <Col md={6}>
                  <MetricTile
                    label="Overall Action Accuracy (chunk-level)"
                    value={formatPercent(actAcc)}
                    caption="Fraction of segmented chunks the VLM classified correctly. (Legacy chunk-level metric.)"
                    accent="#198754"
                  />
                </Col>
                <Col md={6}>
                  <MetricTile
                    label="Temporal F1"
                    value={formatPercent(tempF1)}
                    caption="How well DDM-Net detected action boundaries on the timeline."
                    accent="#6610f2"
                  />
                </Col>
              </>
            )}
          </Row>

          {/* ─── ERROR BREAKDOWN ─────────────────────────────────────── */}
          {isNewSchema && (
            <div
              className="mb-4 p-3 rounded border"
              style={{ background: '#fafbfc' }}
            >
              <div
                className="text-muted text-uppercase mb-2"
                style={{ fontSize: '0.7rem', letterSpacing: '0.08em', fontWeight: 600 }}
              >
                Error Breakdown
              </div>
              <Row className="g-3">
                <Col md={4}>
                  <div className="d-flex align-items-baseline">
                    <span
                      className="me-2 px-2 py-1 rounded"
                      style={{
                        background: SEQ_CHIP_STYLES.wrong.bg,
                        color: SEQ_CHIP_STYLES.wrong.color,
                        border: `1px solid ${SEQ_CHIP_STYLES.wrong.border}`,
                        fontWeight: 700,
                        minWidth: 36,
                        textAlign: 'center',
                      }}
                    >
                      {ar.wrong ?? 0}
                    </span>
                    <strong>Wrong</strong>
                  </div>
                  <div className="text-muted small mt-1" style={{ fontSize: '0.78rem' }}>
                    Action position predicted, but with the wrong action number.
                  </div>
                </Col>
                <Col md={4}>
                  <div className="d-flex align-items-baseline">
                    <span
                      className="me-2 px-2 py-1 rounded"
                      style={{
                        background: SEQ_CHIP_STYLES.duplicate.bg,
                        color: SEQ_CHIP_STYLES.duplicate.color,
                        border: `1px solid ${SEQ_CHIP_STYLES.duplicate.border}`,
                        fontWeight: 700,
                        minWidth: 36,
                        textAlign: 'center',
                      }}
                    >
                      {ar.duplicate ?? 0}
                    </span>
                    <strong>Duplicate</strong>
                  </div>
                  <div className="text-muted small mt-1" style={{ fontSize: '0.78rem' }}>
                    Extra prediction with no corresponding action in the ground truth.
                  </div>
                </Col>
                <Col md={4}>
                  <div className="d-flex align-items-baseline">
                    <span
                      className="me-2 px-2 py-1 rounded"
                      style={{
                        background: SEQ_CHIP_STYLES.missing.bg,
                        color: SEQ_CHIP_STYLES.missing.color,
                        border: `1px solid ${SEQ_CHIP_STYLES.missing.border}`,
                        fontWeight: 700,
                        minWidth: 36,
                        textAlign: 'center',
                      }}
                    >
                      {ar.missing ?? 0}
                    </span>
                    <strong>Missing</strong>
                  </div>
                  <div className="text-muted small mt-1" style={{ fontSize: '0.78rem' }}>
                    Ground-truth action that the model failed to predict at all.
                  </div>
                </Col>
              </Row>
            </div>
          )}

          {/* ─── PER-VIDEO DIFF (PRIMARY) ────────────────────────────── */}
          {isNewSchema && perVideoRows.length > 0 && (
            <>
              <div className="d-flex justify-content-between align-items-baseline mb-2">
                <h6
                  className="m-0 text-muted text-uppercase"
                  style={{ fontSize: '0.7rem', letterSpacing: '0.08em', fontWeight: 600 }}
                >
                  Per-video diff
                </h6>
                <small className="text-muted" style={{ fontSize: '0.75rem' }}>
                  Click any row to expand
                </small>
              </div>
              <Table bordered responsive size="sm" className="mb-4">
                <thead>
                  <tr>
                    <th style={{ width: '32%' }}>Video</th>
                    <th>Edit dist.</th>
                    <th>Wrong</th>
                    <th>Dup.</th>
                    <th>Missing</th>
                    <th>Temporal F1</th>
                    <th style={{ width: 30 }}></th>
                  </tr>
                </thead>
                <tbody>
                  {perVideoRows.map((v) => {
                    const isOpen = expandedE2eRows.has(v.video);
                    const perfect = v.edit_distance === 0;
                    return (
                      <React.Fragment key={v.video}>
                        <tr
                          onClick={() => toggleRow(v.video)}
                          style={{ cursor: 'pointer' }}
                          className={isOpen ? 'table-active' : ''}
                        >
                          <td><small className="font-monospace">{v.video}</small></td>
                          <td>
                            <Badge bg={perfect ? 'success' : 'warning'}>
                              {v.edit_distance}
                            </Badge>
                          </td>
                          <td>
                            {v.wrong > 0
                              ? <span style={{ color: SEQ_CHIP_STYLES.wrong.color, fontWeight: 600 }}>{v.wrong}</span>
                              : <span className="text-muted">0</span>}
                          </td>
                          <td>
                            {v.duplicate > 0
                              ? <span style={{ color: SEQ_CHIP_STYLES.duplicate.color, fontWeight: 600 }}>{v.duplicate}</span>
                              : <span className="text-muted">0</span>}
                          </td>
                          <td>
                            {v.missing > 0
                              ? <span style={{ color: SEQ_CHIP_STYLES.missing.color, fontWeight: 600 }}>{v.missing}</span>
                              : <span className="text-muted">0</span>}
                          </td>
                          <td>{formatPercent(v.f1)}</td>
                          <td className="text-muted text-center">{isOpen ? '▾' : '▸'}</td>
                        </tr>
                        {isOpen && (
                          <tr>
                            <td colSpan={7} style={{ background: '#fafbfc' }}>
                              <div className="p-2">
                                {/* Sequence diff visualization */}
                                <div
                                  className="text-muted mb-2"
                                  style={{ fontSize: '0.7rem', letterSpacing: '0.06em', fontWeight: 600, textTransform: 'uppercase' }}
                                >
                                  Action sequence alignment
                                </div>
                                {(() => {
                                  const cols = alignActionSequences(v.predicted || [], v.golden || []);
                                  if (cols.length === 0) {
                                    return <div className="text-muted small">No actions to compare.</div>;
                                  }
                                  return (
                                    <div style={{ overflowX: 'auto', paddingBottom: 6 }}>
                                      <DiffRow label="Golden" cols={cols} kind="golden" />
                                      <DiffRow label="Predicted" cols={cols} kind="predicted" />
                                    </div>
                                  );
                                })()}

                                {/* Error trace */}
                                {v.steps && v.steps.length > 0 && (
                                  <>
                                    <div
                                      className="text-muted mt-3 mb-1"
                                      style={{ fontSize: '0.7rem', letterSpacing: '0.06em', fontWeight: 600, textTransform: 'uppercase' }}
                                    >
                                      Errors ({v.steps.length})
                                    </div>
                                    <ul className="mb-0 small font-monospace" style={{ fontSize: '0.78rem' }}>
                                      {/* Error trace lines are positional, freshly built each
                                          render; the string content itself is stable per row. */}
                                      {v.steps.map((s) => (
                                        <li key={s}>{s}</li>
                                      ))}
                                    </ul>
                                  </>
                                )}
                                {(!v.steps || v.steps.length === 0) && (
                                  <div className="text-success small mt-3">
                                    Predicted sequence matches golden exactly.
                                  </div>
                                )}
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    );
                  })}
                </tbody>
              </Table>
            </>
          )}

          {/* ─── SECONDARY DETAILS (collapsed by default) ────────────── */}
          {((ar.per_action && Object.keys(ar.per_action).length > 0) || Object.keys(tsPerVideo).length > 0) && (
            <>
              <Button
                variant="link"
                className="p-0 text-decoration-none text-secondary mb-2"
                size="sm"
                onClick={() => setShowE2eSecondary(v => !v)}
              >
                <i className={`bi bi-chevron-${showE2eSecondary ? 'up' : 'down'} me-1`} /> {showE2eSecondary ? 'Hide' : 'Show'} secondary details (chunk-level accuracy, per-video temporal F1)
              </Button>

              {showE2eSecondary && (
                <div className="ps-3 ms-1" style={{ borderLeft: '2px solid #e9ecef' }}>
                  {/* Per-action chunk accuracy table — chunk-level (kept for reference) */}
                  {ar.per_action && Object.keys(ar.per_action).length > 0 && (
                    <>
                      <h6
                        className="text-muted text-uppercase mt-2"
                        style={{ fontSize: '0.7rem', letterSpacing: '0.08em', fontWeight: 600 }}
                      >
                        Per-action accuracy <small className="text-muted text-lowercase ms-1" style={{ letterSpacing: 0 }}>
                          (chunk-level — each predicted chunk matched to its expected action)
                        </small>
                      </h6>
                      <Table bordered responsive size="sm">
                        <thead>
                          <tr>
                            <th style={{ width: 70 }}>Action #</th>
                            <th>Label</th>
                            <th style={{ width: 110 }}>Accuracy</th>
                            <th style={{ width: 130 }}>n (correct / total)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(ar.per_action)
                            .sort(([a], [b]) => Number(a) - Number(b))
                            .map(([actionNum, d]) => (
                              <tr key={actionNum}>
                                <td>{actionNum}</td>
                                <td><small>{d.label || '—'}</small></td>
                                <td>{formatPercent(d.accuracy)}</td>
                                <td>
                                  {d.correct == null || d.total == null
                                    ? '—' : `${d.correct} / ${d.total}`}
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </Table>
                    </>
                  )}

                  {/* Per-video temporal F1 table */}
                  {Object.keys(tsPerVideo).length > 0 && (
                    <>
                      <h6
                        className="text-muted text-uppercase mt-3"
                        style={{ fontSize: '0.7rem', letterSpacing: '0.08em', fontWeight: 600 }}
                      >
                        Per-video temporal segmentation
                      </h6>
                      <Table bordered responsive size="sm" className="mb-2">
                        <thead>
                          <tr>
                            <th>Video</th>
                            <th>F1</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>Predicted boundaries</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(tsPerVideo).map(([videoName, v]) => (
                            <tr key={videoName}>
                              <td><small className="font-monospace">{videoName}</small></td>
                              <td>{formatPercent(v.f1)}</td>
                              <td>{formatPercent(v.precision)}</td>
                              <td>{formatPercent(v.recall)}</td>
                              <td>{Array.isArray(v.boundaries) ? v.boundaries.length : '—'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </>
                  )}
                </div>
              )}
            </>
          )}

          {/* ─── ON-DISK REFERENCES ──────────────────────────────────── */}
          <Alert variant="light" className="mt-3 mb-0 border">
            <strong className="text-muted" style={{ fontSize: '0.85rem' }}>Output files</strong>
            <ul className="small text-muted mt-1 mb-0" style={{ fontSize: '0.78rem' }}>
              <li>
                Boundary visualization PNGs:{' '}
                <code>&lt;RESULTS_ROOT&gt;/{selectedE2eEvalJobId}/outputs_temporal_segmentation/*.png</code>
              </li>
              <li>
                Sequence accuracy report:{' '}
                <code>&lt;RESULTS_ROOT&gt;/{selectedE2eEvalJobId}/outputs_action_recognition/accuracy.json</code>
              </li>
              <li>
                Combined results:{' '}
                <code>&lt;RESULTS_ROOT&gt;/{selectedE2eEvalJobId}/e2e_results.json</code>
              </li>
            </ul>
          </Alert>
        </Card.Body>
      </Card>
    );
  };

  const renderE2eView = () => (
    <>
      <div className="mb-3 pb-2 border-bottom">
        <div className="d-flex align-items-center">
          <Badge bg="dark" className="me-2">End-to-end</Badge>
          <small className="text-muted">
            DDM temporal segmentation → VLM action recognition on raw full-length videos.
            Reports temporal F1 and per-action accuracy.
          </small>
        </div>
      </div>

      <Row className="mb-3">
        <Col md={6}>
          <Form.Group className="mb-3">
            <Form.Label>Chunking Strategy</Form.Label>
            <Form.Select
              value={e2eChunkingAlgorithm}
              onChange={(e) => setE2eChunkingAlgorithm(e.target.value)}
              disabled={isSubmitting}
            >
              <option value="ddm">DDM-Net (learned temporal segmentation)</option>
              <option value="uniform">Uniform (fixed-length chunks)</option>
            </Form.Select>
            <Form.Text className="text-muted">
              {e2eChunkingAlgorithm === 'ddm'
                ? 'Uses a trained DDM-Net to detect action boundaries in each video.'
                : 'Splits each video into fixed-length slices — no DDM training job required.'}
            </Form.Text>
          </Form.Group>
        </Col>
        {e2eChunkingAlgorithm === 'uniform' && (
          <Col md={6}>
            <Form.Group className="mb-3">
              <Form.Label>Chunk Length (sec)</Form.Label>
              <Form.Control
                type="number"
                value={e2eChunkLengthSec}
                min={0.1}
                step={0.5}
                onChange={(e) => setE2eChunkLengthSec(e.target.value)}
                disabled={isSubmitting}
              />
              <Form.Text className="text-muted">
                Each chunk is this many seconds long; the last chunk in a video may be shorter.
              </Form.Text>
            </Form.Group>
          </Col>
        )}
      </Row>

      <Row className="mb-3">
        <Col md={6}>
          <Form.Group className="mb-3">
            <Form.Label>VLM Training Experiment</Form.Label>
            <Form.Select
              value={e2eSelectedTrainingJobId}
              onChange={(e) => setE2eSelectedTrainingJobId(e.target.value)}
              disabled={isSubmitting}
            >
              <option value="">Choose a completed VLM training job…</option>
              {completedTrainingJobs.map(([jobId, job]) => (
                <option key={jobId} value={jobId}>
                  {jobId}{job.aug_dataset_id ? ` (dataset: ${job.aug_dataset_id})` : ''}
                </option>
              ))}
            </Form.Select>
          </Form.Group>
        </Col>
        {e2eChunkingAlgorithm === 'ddm' && (
          <Col md={6}>
            <Form.Group className="mb-3">
              <Form.Label>DDM Training Experiment</Form.Label>
              <Form.Select
                value={e2eSelectedDdmTrainingJobId}
                onChange={(e) => setE2eSelectedDdmTrainingJobId(e.target.value)}
                disabled={isSubmitting}
              >
                <option value="">Choose a completed DDM training job…</option>
                {completedDdmTrainingJobs.map(([jobId, job]) => (
                  <option key={jobId} value={jobId}>
                    {jobId}{job.dataset_id ? ` (dataset: ${job.dataset_id})` : ''}
                  </option>
                ))}
              </Form.Select>
              {completedDdmTrainingJobs.length === 0 && (
                <Form.Text className="text-muted">
                  No completed DDM training jobs yet. Train one in the DDM panel first.
                </Form.Text>
              )}
            </Form.Group>
          </Col>
        )}
      </Row>

      <Row className="mb-3">
        <Col md={6}>
          <Form.Group className="mb-3">
            <Form.Label>Validation Dataset (raw full videos)</Form.Label>
            <Form.Select
              value={e2eSelectedValDatasetId}
              onChange={(e) => setE2eSelectedValDatasetId(e.target.value)}
              disabled={isSubmitting}
            >
              <option value="">Choose a validation dataset…</option>
              {renderDatasetOptions()}
            </Form.Select>
          </Form.Group>
        </Col>
        <Col md={3}>
          <Form.Group className="mb-3">
            <Form.Label>Backend</Form.Label>
            <Form.Select
              value={e2eBackend}
              onChange={(e) => setE2eBackend(e.target.value)}
              disabled={isSubmitting}
            >
              <option value="vllm">vllm</option>
              <option value="transformers">transformers</option>
            </Form.Select>
          </Form.Group>
        </Col>
        <Col md={3}>
          <Form.Group className="mb-3">
            <Form.Label>FPS</Form.Label>
            <Form.Control
              type="number"
              value={e2eFps}
              min={1}
              max={30}
              onChange={(e) => setE2eFps(e.target.value)}
              disabled={isSubmitting}
            />
          </Form.Group>
        </Col>
      </Row>

      <Row className="mb-3">
        <Col md={6}>
          {renderGpuSelector({
            value: e2eSelectedGpuId,
            setValue: setE2eSelectedGpuId,
            disabled: isSubmitting,
          })}
        </Col>
      </Row>

      <div className="mb-3">
        <Button
          variant="link"
          className="p-0 text-decoration-none text-secondary"
          size="sm"
          onClick={() => setShowE2eAdvanced(v => !v)}
        >
          <i className={`bi bi-chevron-${showE2eAdvanced ? 'up' : 'down'} me-1`} /> Advanced: segmentation & VLM overrides
        </Button>
        {showE2eAdvanced && (
          <div className="mt-3 p-3 border rounded bg-light">
            <Row>
              <Col md={4}>
                <Form.Group className="mb-3">
                  <Form.Label>Temperature</Form.Label>
                  <Form.Control
                    type="number"
                    value={e2eTemperature}
                    min={0}
                    max={2}
                    step={0.1}
                    onChange={(e) => setE2eTemperature(e.target.value)}
                    disabled={isSubmitting}
                  />
                </Form.Group>
              </Col>
              <Col md={4}>
                <Form.Group className="mb-3">
                  <Form.Label>VLM Checkpoint Step <small className="text-muted">(optional)</small></Form.Label>
                  <Form.Control
                    type="number"
                    value={e2eCheckpointStep}
                    min={1}
                    placeholder="Latest"
                    onChange={(e) => setE2eCheckpointStep(e.target.value)}
                    disabled={isSubmitting}
                  />
                </Form.Group>
              </Col>
              {e2eChunkingAlgorithm === 'ddm' && (
                <Col md={4}>
                  <Form.Group className="mb-3">
                    <Form.Label>DDM Checkpoint <small className="text-muted">(optional)</small></Form.Label>
                    <Form.Control
                      type="text"
                      value={e2eDdmCheckpoint}
                      placeholder="Override filename (e.g. best.ckpt)"
                      onChange={(e) => setE2eDdmCheckpoint(e.target.value)}
                      disabled={isSubmitting}
                    />
                  </Form.Group>
                </Col>
              )}
            </Row>

            {e2eChunkingAlgorithm === 'ddm' && (
              <Row>
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>
                      Boundary score threshold:{' '}
                      <span className="font-monospace text-primary">
                        {Number(e2eScoreThreshold).toFixed(2)}
                      </span>
                    </Form.Label>
                    <Form.Range
                      min={0}
                      max={1}
                      step={0.01}
                      value={e2eScoreThreshold}
                      onChange={(e) => setE2eScoreThreshold(e.target.value)}
                      disabled={isSubmitting}
                    />
                    <Form.Text className="text-muted">
                      Min DDM score to accept as a boundary. Higher = fewer, more confident cuts.
                    </Form.Text>
                  </Form.Group>
                </Col>
                <Col md={3}>
                  <Form.Group className="mb-3">
                    <Form.Label>NMS window (sec)</Form.Label>
                    <Form.Control
                      type="number"
                      value={e2eNmsSec}
                      min={0}
                      step={0.1}
                      onChange={(e) => setE2eNmsSec(e.target.value)}
                      disabled={isSubmitting}
                    />
                  </Form.Group>
                </Col>
                <Col md={3}>
                  <Form.Group className="mb-3">
                    <Form.Label>DDM batch size</Form.Label>
                    <Form.Control
                      type="number"
                      value={e2eDdmBatchSize}
                      min={1}
                      onChange={(e) => setE2eDdmBatchSize(e.target.value)}
                      disabled={isSubmitting}
                    />
                  </Form.Group>
                </Col>
              </Row>
            )}

            <Row>
              {e2eChunkingAlgorithm === 'ddm' && (
                <Col md={4}>
                  <Form.Group className="mb-3">
                    <Form.Label>Frames per segment hint</Form.Label>
                    <Form.Control
                      type="number"
                      value={e2eFramesPerSegmentHint}
                      min={32}
                      step={32}
                      onChange={(e) => setE2eFramesPerSegmentHint(e.target.value)}
                      disabled={isSubmitting}
                    />
                    <Form.Text className="text-muted">
                      Affects DDM window stride.
                    </Form.Text>
                  </Form.Group>
                </Col>
              )}
              <Col md={e2eChunkingAlgorithm === 'ddm' ? 8 : 12}>
                <Form.Label>Resolution config <small className="text-muted">(optional JSON)</small></Form.Label>
                {renderResolutionConfigField(
                  e2eResolutionConfigText,
                  e2eResolutionConfigError,
                  handleE2eResolutionConfigChange,
                  isSubmitting,
                )}
              </Col>
            </Row>
          </div>
        )}
      </div>

      <div className="mb-4">
        <Button
          variant="success"
          onClick={handleRunE2eEvaluation}
          disabled={
            !e2eSelectedTrainingJobId ||
            (e2eChunkingAlgorithm === 'ddm' && !e2eSelectedDdmTrainingJobId) ||
            (e2eChunkingAlgorithm === 'uniform' && (!e2eChunkLengthSec || Number(e2eChunkLengthSec) <= 0)) ||
            !e2eSelectedValDatasetId ||
            isSubmitting
          }
        >
          {isSubmitting ? (
            <>
              <Spinner animation="border" size="sm" className="me-2" />
              Starting E2E Evaluation…
            </>
          ) : 'Run E2E Evaluation'}
        </Button>
      </div>

      <h5 className="mb-3">E2E Evaluation Jobs</h5>
      {sortedE2eEvalJobs.length === 0 ? (
        <Alert variant="info">
          No e2e evaluation jobs yet. Configure and run an e2e evaluation above.
        </Alert>
      ) : (
        <Table bordered hover responsive size="sm" className="mb-4">
          <thead>
            <tr>
              <th>Eval Job ID</th>
              <th>VLM Experiment</th>
              <th>Chunker</th>
              <th>Val Dataset</th>
              <th>Status</th>
              <th title="Fraction of videos where the predicted action sequence matches the ground truth end-to-end (edit distance = 0).">
                Sequence Acc
              </th>
              <th title="How well DDM-Net detected action boundaries on the timeline (avg F1 between predicted and golden boundaries).">
                Temporal F1
              </th>
              <th>Created</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {sortedE2eEvalJobs.map(([evalJobId, job]) => {
              const ar = job.results_json?.action_recognition;
              const ts = job.results_json?.temporal_segmentation;
              const seqAcc = ar?.sequence_accuracy;
              const tempF1 = ts?.avg_f1 ?? job.avg_f1;
              const isLegacy = job.status === 'completed'
                && job.results_json
                && (seqAcc === undefined || seqAcc === null);
              return (
              <tr
                key={evalJobId}
                onClick={() => handleViewE2eEvalResults(evalJobId)}
                style={{ cursor: job.status === 'completed' ? 'pointer' : 'default' }}
                className={selectedE2eEvalJobId === evalJobId ? 'table-active' : ''}
              >
                <td><small className="font-monospace">{evalJobId}</small></td>
                <td><small>{job.training_job_id}</small></td>
                <td>
                  {job.chunking_algorithm === 'uniform' ? (
                    <small title={`Uniform chunks of ${job.chunk_length_sec}s`}>
                      uniform · {job.chunk_length_sec}s
                    </small>
                  ) : (
                    <small title={job.ddm_training_job_id || ''}>
                      DDM · <span className="font-monospace">{job.ddm_training_job_id || '—'}</span>
                    </small>
                  )}
                </td>
                <td><small>{job.val_dataset_id}</small></td>
                <td>
                  <Badge bg={getStatusBadgeVariant(job.status)}>
                    {job.status ? job.status.toUpperCase() : 'UNKNOWN'}
                  </Badge>
                  {ACTIVE_STATUSES.has(job.status) && (
                    <Spinner animation="border" size="sm" className="ms-2" role="status">
                      <span className="visually-hidden">Running…</span>
                    </Spinner>
                  )}
                </td>
                <td>
                  {seqAcc !== undefined && seqAcc !== null
                    ? formatPercent(seqAcc)
                    : <span className="text-muted">—</span>}
                  {isLegacy && (
                    <Badge bg="secondary" className="ms-2" style={{ fontSize: '0.65rem' }}
                      title="Legacy job — pre-sequence-accuracy schema. Open the result for the chunk-level number.">
                      legacy
                    </Badge>
                  )}
                </td>
                <td>
                  {tempF1 !== undefined && tempF1 !== null
                    ? formatPercent(tempF1)
                    : <span className="text-muted">—</span>}
                </td>
                <td><small>{formatDate(job.created_at)}</small></td>
                <td onClick={(e) => e.stopPropagation()}>
                  {ACTIVE_STATUSES.has(job.status) && (
                    <Button
                      size="sm"
                      variant="outline-danger"
                      onClick={() => handleCancelE2eEval(evalJobId)}
                      disabled={isCancelling}
                    >
                      {isCancelling ? <Spinner animation="border" size="sm" /> : 'Cancel'}
                    </Button>
                  )}
                  {job.status === 'completed' && (
                    <Button
                      size="sm"
                      variant="outline-primary"
                      onClick={() => handleViewE2eEvalResults(evalJobId)}
                    >
                      View Results
                    </Button>
                  )}
                </td>
              </tr>
              );
            })}
          </tbody>
        </Table>
      )}

      {renderE2eResults()}
    </>
  );

  // --- Main render -----------------------------------------------------------

  return (
    <Card className="mb-4">
      <Card.Header>
        <div className="d-flex justify-content-between align-items-center flex-wrap gap-2">
          <div>
            <h4 className="mb-0">Evaluation</h4>
            <small className="text-muted">
              Measure finetuned model accuracy on held-out data
            </small>
          </div>

          <Nav
            variant="pills"
            activeKey={activeEvalType}
            onSelect={(key) => { if (key) setActiveEvalType(key); }}
          >
            <Nav.Item>
              <Nav.Link eventKey={EVAL_TYPE_PER_CHUNK} className="py-1 px-3">
                Per-action-chunk
              </Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey={EVAL_TYPE_E2E} className="py-1 px-3">
                End-to-end
              </Nav.Link>
            </Nav.Item>
          </Nav>

          <Button
            variant="outline-secondary"
            size="sm"
            onClick={refreshAll}
            disabled={isRefreshing}
            className="d-flex align-items-center"
            title="Re-fetch training jobs, DDM jobs, datasets, and evaluation jobs"
          >
            {isRefreshing ? (
              <>
                <Spinner animation="border" size="sm" className="me-2" role="status">
                  <span className="visually-hidden">Refreshing…</span>
                </Spinner>
                Refreshing…
              </>
            ) : (
              <>
                <i className="bi bi-arrow-clockwise me-1" /> Refresh
              </>
            )}
          </Button>
        </div>
      </Card.Header>

      <Card.Body>
        {statusMessage.show && (
          <Alert
            variant={statusMessage.variant}
            dismissible
            onClose={() => setStatusMessage({ ...statusMessage, show: false })}
            className="mb-3"
          >
            {statusMessage.message}
          </Alert>
        )}

        {activeEvalType === EVAL_TYPE_PER_CHUNK ? renderPerChunkView() : renderE2eView()}
      </Card.Body>
    </Card>
  );
};

export default EvaluationPanel;
