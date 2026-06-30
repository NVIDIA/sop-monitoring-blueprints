// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import React, { useState, useEffect, useRef } from 'react';
import { Card, Button, Form, Alert, ProgressBar, Spinner, Row, Col, Badge } from 'react-bootstrap';

const API_BASE_URL = '/api/vlm-training';

const VLMTrainingPanel = ({ augmentedDatasets, onTrainingComplete }) => {
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [trainingJobs, setTrainingJobs] = useState({});
  const [activeTrainingJobId, setActiveTrainingJobId] = useState('');
  const [trainingStatus, setTrainingStatus] = useState({
    show: false,
    variant: 'info',
    message: ''
  });
  const [isSubmittingJob, setIsSubmittingJob] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Ref for polling interval
  const statusPollingRef = useRef(null);

  // Available augmented datasets
  const availableDatasets = Object.keys(augmentedDatasets).filter(
    datasetId => augmentedDatasets[datasetId].status === 'completed'
  );

  // Load training jobs from training microservice API on component mount
  useEffect(() => {
    fetchAllTrainingJobs();
  }, []);

  // Function to fetch training jobs from training microservice
  const fetchAllTrainingJobs = async () => {
    try {
      console.log('Fetching training jobs from training microservice...');

      const response = await fetch(`${API_BASE_URL}/api/v1/fine-tuning/all_jobs`);

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Failed to fetch training jobs:', errorData.detail || `HTTP ${response.status}`);
        return false;
      }

      const trainingJobsData = await response.json();
      console.log('Fetched training jobs from microservice:', trainingJobsData);

      // Transform the backend data format to match our frontend format
      const transformedTrainingJobs = {};

      for (const [jobId, jobInfo] of Object.entries(trainingJobsData)) {
        transformedTrainingJobs[jobId] = {
          jobId: jobId,
          datasetId: jobInfo.aug_dataset_id,
          status: jobInfo.status,
          totalEpochs: jobInfo.total_epochs,
          currentEpoch: jobInfo.current_epoch,
          totalSteps: jobInfo.total_steps,
          currentStep: jobInfo.current_step,
          progress: jobInfo.progress,
          loss: jobInfo.loss,
          eta: jobInfo.eta,
          startedAt: jobInfo.created_at,
          lastUpdated: jobInfo.updated_at
        };
      }

      console.log('Transformed training jobs for frontend:', transformedTrainingJobs);
      setTrainingJobs(transformedTrainingJobs);

      // Find active job and start polling if needed
      const activeJob = Object.entries(transformedTrainingJobs).find(([_, job]) =>
        job.status === 'running' || job.status === 'queued'
      );
      if (activeJob) {
        setActiveTrainingJobId(activeJob[0]);
        startStatusPolling(activeJob[0]);
      }

      return true;
    } catch (error) {
      console.error('Failed to fetch training jobs from microservice:', error);
      return false;
    }
  };

  // Function to refresh training jobs from microservice
  const refreshAllTrainingJobs = async () => {
    return await fetchAllTrainingJobs();
  };

  // Start polling for training status
  const startStatusPolling = (jobId) => {
    // Clear any existing interval first
    if (statusPollingRef.current) {
      clearInterval(statusPollingRef.current);
    }

    statusPollingRef.current = setInterval(async () => {
      await checkTrainingStatus(jobId);
    }, 30000); // Poll every 30 seconds
  };

  // Stop polling
  const stopStatusPolling = () => {
    if (statusPollingRef.current) {
      clearInterval(statusPollingRef.current);
      statusPollingRef.current = null;
    }
  };

  // Check training status
  const checkTrainingStatus = async (jobId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/fine-tuning/status/${jobId}`);

      if (!response.ok) {
        throw new Error(`Failed to check training status: ${response.status}`);
      }

      const statusData = await response.json();
      console.log('Training status response:', statusData);

      // Update training job with real status data
      setTrainingJobs(prev => ({
        ...prev,
        [jobId]: {
          ...prev[jobId],
          status: statusData.status,
          progress: statusData.progress,
          currentEpoch: statusData.current_epoch,
          totalEpochs: statusData.total_epochs,
          currentStep: statusData.current_step,
          totalSteps: statusData.total_steps,
          loss: statusData.loss,
          eta: statusData.eta,
          lastUpdated: statusData.updated_at || new Date().toISOString()
        }
      }));

      // Check if training is complete
      if (statusData.status === 'completed' || statusData.status === 'failed' || statusData.status === 'cancelled') {
        stopStatusPolling();
        setActiveTrainingJobId('');

        if (statusData.status === 'completed') {
          setTrainingStatus({
            show: true,
            variant: 'success',
            message: `Training completed successfully! Job ID: ${jobId}`
          });

          if (onTrainingComplete) {
            onTrainingComplete(jobId, statusData);
          }
        } else if (statusData.status === 'failed') {
          setTrainingStatus({
            show: true,
            variant: 'danger',
            message: `Training failed for job: ${jobId}`
          });
        } else if (statusData.status === 'cancelled') {
          setTrainingStatus({
            show: true,
            variant: 'warning',
            message: `Training cancelled for job: ${jobId}`
          });
        }
      }

    } catch (error) {
      console.error('Failed to check training status:', error);
      setTrainingStatus({
        show: true,
        variant: 'warning',
        message: `Failed to check training status: ${error.message}`
      });
    }
  };

  // Start training
  const handleStartTraining = async () => {
    if (!selectedDatasetId) {
      setTrainingStatus({
        show: true,
        variant: 'warning',
        message: 'Please select an augmented dataset first.'
      });
      return;
    }

    setIsSubmittingJob(true);
    setTrainingStatus({
      show: true,
      variant: 'info',
      message: `Submitting training request for dataset: ${selectedDatasetId}`
    });

    try {
      // Prepare training configuration

      const response = await fetch(`${API_BASE_URL}/api/v1/fine-tuning/start?dataset_id=${selectedDatasetId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'accept': 'application/json'
        },
        body: JSON.stringify({config: {}})
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData || `Failed to start training: ${response.status}`);
      }

      const result = await response.json();
      console.log('Training start response:', result);

      // Create new training job with real data
      const newTrainingJob = {
        jobId: result.job_id,
        datasetId: selectedDatasetId,
        status: result.status,
        progress: 0,
        startedAt: result.created_at,
        lastUpdated: result.created_at,
      };

      setTrainingJobs(prev => ({
        ...prev,
        [result.job_id]: newTrainingJob
      }));

      setActiveTrainingJobId(result.job_id);
      setIsSubmittingJob(false);
      setTrainingStatus({
        show: true,
        variant: 'success',
        message: `Training started successfully! Job ID: ${result.job_id}`
      });

      // Start polling for status
      startStatusPolling(result.job_id);

    } catch (error) {
      console.error('Failed to start training:', error);
      setIsSubmittingJob(false);
      setTrainingStatus({
        show: true,
        variant: 'danger',
        message: `Failed to start training: ${error.message}`
      });
    }
  };

  // Cancel training
  const handleCancelTraining = async (jobId) => {
    if (!window.confirm('Are you sure you want to cancel this training job?')) {
      return;
    }

    setIsCancelling(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/fine-tuning/cancel/${jobId}`, {
        method: 'POST',
        headers: {
          'accept': 'application/json',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData || `Failed to cancel training: ${response.status}`);
      }

      const result = await response.json();
      console.log('Training cancel response:', result);

      // Update job status to cancelled
      setTrainingJobs(prev => ({
        ...prev,
        [jobId]: {
          ...prev[jobId],
          status: 'cancelled',
          lastUpdated: new Date().toISOString()
        }
      }));

      if (activeTrainingJobId === jobId) {
        setActiveTrainingJobId('');
        stopStatusPolling();
      }

      setTrainingStatus({
        show: true,
        variant: 'warning',
        message: result.message || `Training job ${jobId} has been cancelled.`
      });

    } catch (error) {
      console.error('Failed to cancel training:', error);
      setTrainingStatus({
        show: true,
        variant: 'danger',
        message: `Failed to cancel training: ${error.message}`
      });
    } finally {
      setIsCancelling(false);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStatusPolling();
    };
  }, []);

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

  const activeJob = activeTrainingJobId ? trainingJobs[activeTrainingJobId] : null;

  return (
    <Card className="mb-4">
      <Card.Header className="d-flex justify-content-between align-items-center">
        <div>
          <h4 className="mb-0">🧠 VLM Training</h4>
          <small className="text-muted">
            Fine-tune Vision Language Model with augmented datasets
          </small>
        </div>
        <Button
          variant="secondary"
          size="sm"
          onClick={async () => {
            setIsRefreshing(true);
            try {
              await refreshAllTrainingJobs();
            } finally {
              setIsRefreshing(false);
            }
          }}
          disabled={isRefreshing}
          className="d-flex align-items-center"
        >
          {isRefreshing ? (
            <>
              <div className="spinner-border spinner-border-sm me-2" role="status">
                <span className="visually-hidden">Loading...</span>
              </div>
              Refreshing...
            </>
          ) : (
            <>
              <i className="bi bi-arrow-clockwise me-2"></i>
              Refresh Jobs
            </>
          )}
        </Button>
      </Card.Header>
      <Card.Body>
        {availableDatasets.length === 0 ? (
          <Alert variant="info">
            No augmented datasets available. Please complete data augmentation first.
          </Alert>
        ) : (
          <>
            <Row>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Select Augmented Dataset</Form.Label>
                  <Form.Select
                    value={selectedDatasetId}
                    onChange={(e) => setSelectedDatasetId(e.target.value)}
                    disabled={isSubmittingJob || !!activeTrainingJobId}
                  >
                    <option value="">Choose a dataset...</option>
                    {availableDatasets.map(datasetId => {
                      const dataset = augmentedDatasets[datasetId];
                      return (
                        <option key={datasetId} value={datasetId}>
                          {datasetId} ({dataset.videoCount} videos, {dataset.totalClips} clips)
                        </option>
                      );
                    })}
                  </Form.Select>
                </Form.Group>

                {selectedDatasetId && augmentedDatasets[selectedDatasetId] && (
                  <div className="mb-3">
                    <h6>Dataset Information:</h6>
                    <ul className="list-unstyled small text-muted">
                      <li>📹 Videos: {augmentedDatasets[selectedDatasetId].videoCount}</li>
                      <li>✂️ Total Clips: {augmentedDatasets[selectedDatasetId].totalClips}</li>
                    </ul>
                  </div>
                )}
              </Col>

              <Col md={6}>
                <h6>Training Jobs</h6>
                {Object.keys(trainingJobs).length === 0 ? (
                  <p className="text-muted small">No training jobs yet</p>
                ) : (
                  <div className="small" style={{ maxHeight: '200px', overflowY: 'auto' }}>
                    {Object.entries(trainingJobs)
                      .sort(([,a], [,b]) => new Date(b.startedAt) - new Date(a.startedAt))
                      .map(([jobId, job]) => (
                        <div key={jobId} className="mb-2 p-2 bg-light rounded">
                          <div className="d-flex justify-content-between align-items-center">
                            <div>
                              <strong>{jobId}</strong>
                              <br />
                              <Badge variant={getStatusBadgeVariant(job.status)}>
                                {job.status.toUpperCase()}
                              </Badge>
                              {job.status === 'running' && (
                                <span className="text-muted ms-2">
                                  {Math.round(job.progress)}%
                                </span>
                              )}
                            </div>
                            {(job.status === 'running' || job.status === 'queued') && (
                              <Button
                                size="sm"
                                variant="outline-danger"
                                onClick={() => handleCancelTraining(jobId)}
                                disabled={isCancelling}
                              >
                                Cancel
                              </Button>
                            )}
                          </div>
                          <small className="text-muted">
                            Started: {new Date(job.startedAt).toLocaleString()}
                            {job.loss && ` | Loss: ${job.loss.toFixed(3)}`}
                            {job.eta && ` | ETA: ${job.eta}`}
                          </small>
                        </div>
                      ))}
                  </div>
                )}
              </Col>
            </Row>

            {trainingStatus.show && (
              <Alert
                variant={trainingStatus.variant}
                dismissible
                onClose={() => setTrainingStatus({...trainingStatus, show: false})}
                className="mb-3"
              >
                {trainingStatus.message}
              </Alert>
            )}

            {activeJob && (
              <div className="mb-3">
                <div className="d-flex justify-content-between align-items-center mb-2">
                  <span>Training Progress</span>
                  <span>Job ID: {activeJob.jobId}</span>
                  <span>Dataset ID: {activeJob.datasetId}</span>
                  <span>{Math.round(activeJob.progress)}%</span>
                </div>
                <ProgressBar
                  now={activeJob.progress}
                  variant="primary"
                  animated
                  striped
                />
                <small className="text-muted">
                  Job ID: {activeJob.jobId} | Status: {activeJob.status} |
                  {activeJob.currentEpoch && ` Epoch: ${activeJob.currentEpoch.toFixed(2)}/${activeJob.totalEpochs} |`}
                  {activeJob.currentStep && ` Step: ${activeJob.currentStep}/${activeJob.totalSteps} |`}
                  {activeJob.loss && ` Loss: ${activeJob.loss.toFixed(3)} |`}
                  {activeJob.eta && ` ETA: ${activeJob.eta} |`}
                  Last Updated: {new Date(activeJob.lastUpdated).toLocaleString()}
                </small>
              </div>
            )}

            <div className="d-flex gap-2">
              <Button
                variant="success"
                onClick={handleStartTraining}
                disabled={!selectedDatasetId || isSubmittingJob || !!activeTrainingJobId}
              >
                {isSubmittingJob ? (
                  <>
                    <Spinner size="sm" className="me-2" />
                    Starting Training...
                  </>
                ) : (
                  'Start Training'
                )}
              </Button>

              {activeTrainingJobId && (
                <Button
                  variant="outline-danger"
                  onClick={() => handleCancelTraining(activeTrainingJobId)}
                  disabled={isCancelling}
                >
                  {isCancelling ? (
                    <>
                      <Spinner size="sm" className="me-2" />
                      Cancelling...
                    </>
                  ) : (
                    'Cancel Training'
                  )}
                </Button>
              )}
            </div>

            <div className="mt-3 p-3 bg-light rounded">
              <small className="text-muted">
                <strong>Training Configuration:</strong> Training configuration can be found in train_config.toml<br />
                <strong>Note:</strong> Training progress is monitored automatically.
                You can safely navigate away and return to check status.
              </small>
            </div>
          </>
        )}
      </Card.Body>
    </Card>
  );
};

export default VLMTrainingPanel;