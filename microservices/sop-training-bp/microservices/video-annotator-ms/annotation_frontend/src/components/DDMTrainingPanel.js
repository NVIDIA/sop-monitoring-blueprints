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

const API_BASE_URL = '/api/ddm-training';

const DDMTrainingPanel = ({ allVideoResults, onTrainingComplete }) => {
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [selectedValidationDatasetId, setSelectedValidationDatasetId] = useState('');
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

  // Available datasets from annotation results (not augmented datasets!)
  // Filter out augmented datasets - DDM trains on original annotated data only
  
  // Debug: Print all dataset IDs before filtering
  console.log('=== DDM Training Panel: All Dataset IDs ===');
  console.log('All dataset IDs in allVideoResults:', Object.keys(allVideoResults));
  console.log('allVideoResults object:', allVideoResults);
  
  const availableDatasets = Object.keys(allVideoResults).filter(
    datasetId => {
      // Ensure the dataset has videos
      const hasVideos = Object.keys(allVideoResults[datasetId] || {}).length > 0;
      const videoCount = Object.keys(allVideoResults[datasetId] || {}).length;
      
      // Exclude augmented datasets (format: xxx_augmented, xxx_augmented_0, xxx_augmented_1, etc.)
      // More strict filtering: exclude any dataset containing '_augmented'
      const isNotAugmented = !datasetId.includes('_augmented');
      
      // Debug: Print filtering details for all datasets
      console.log(`[DDM Filter] Dataset: "${datasetId}" | Videos: ${videoCount} | hasVideos: ${hasVideos} | isNotAugmented: ${isNotAugmented} | Include: ${hasVideos && isNotAugmented}`);
      
      return hasVideos && isNotAugmented;
    }
  );
  
  console.log('=== DDM Training Panel: Available Datasets After Filtering ===');
  console.log('Available datasets:', availableDatasets);
  console.log('======================================================');

  // Load training jobs from training microservice API on component mount
  useEffect(() => {
    fetchAllTrainingJobs();
  }, []);

  // Function to fetch training jobs from training microservice
  const fetchAllTrainingJobs = async () => {
    try {
      console.log('Fetching DDM training jobs from training microservice...');

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
          message: `DDM Training completed successfully! Job ID: ${jobId}`
        });

          if (onTrainingComplete) {
            onTrainingComplete(jobId, statusData);
          }
        } else if (statusData.status === 'failed') {
          setTrainingStatus({
            show: true,
            variant: 'danger',
            message: `DDM Training failed for job: ${jobId}`
          });
        } else if (statusData.status === 'cancelled') {
          setTrainingStatus({
            show: true,
            variant: 'warning',
            message: `DDM Training cancelled for job: ${jobId}`
          });
        }
      }

    } catch (error) {
      console.error('Failed to check DDM training status:', error);
      setTrainingStatus({
        show: true,
        variant: 'warning',
        message: `Failed to check DDM training status: ${error.message}`
      });
    }
  };

  // Start training
  const handleStartTraining = async () => {
    if (!selectedDatasetId) {
      setTrainingStatus({
        show: true,
        variant: 'warning',
        message: 'Please select an annotated dataset first.'
      });
      return;
    }

    if (!selectedValidationDatasetId) {
      setTrainingStatus({
        show: true,
        variant: 'warning',
        message: 'Please select a validation dataset first.'
      });
      return;
    }

    setIsSubmittingJob(true);
    setTrainingStatus({
      show: true,
      variant: 'info',
      message: `Submitting DDM training request for dataset: ${selectedDatasetId}, validation: ${selectedValidationDatasetId}`
    });

    try {
      // Build URL with both training and validation dataset IDs
      const url = `${API_BASE_URL}/api/v1/fine-tuning/start?dataset_id=${selectedDatasetId}&validation_dataset_id=${selectedValidationDatasetId}`;
      
      const response = await fetch(url, {
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
        message: `DDM Training started successfully! Job ID: ${result.job_id}`
      });

      // Start polling for status
      startStatusPolling(result.job_id);

    } catch (error) {
      console.error('Failed to start DDM training:', error);
      setIsSubmittingJob(false);
      setTrainingStatus({
        show: true,
        variant: 'danger',
        message: `Failed to start DDM training: ${error.message}`
      });
    }
  };

  // Cancel training
  const handleCancelTraining = async (jobId) => {
    if (!window.confirm('Are you sure you want to cancel this DDM training job?')) {
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
        message: result.message || `DDM Training job ${jobId} has been cancelled.`
      });

    } catch (error) {
      console.error('Failed to cancel DDM training:', error);
      setTrainingStatus({
        show: true,
        variant: 'danger',
        message: `Failed to cancel DDM training: ${error.message}`
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
          <h4 className="mb-0">🎯 Action Segmentation Model Training</h4>
          <small className="text-muted">
            Fine-tune DDM-net with full videos and annotated timestamps.
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
            No annotated datasets available for DDM training. 
            <br />
            Please complete video annotation (Steps 1-3) first.
            <br />
            <small className="text-muted">
              Note: DDM uses original annotated data, not augmented datasets.
            </small>
          </Alert>
        ) : (
          <>
            <Row>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Select Training Dataset</Form.Label>
                  <Form.Select
                    value={selectedDatasetId}
                    onChange={(e) => {
                      const newDatasetId = e.target.value;
                      setSelectedDatasetId(newDatasetId);
                      // Auto-sync validation dataset to match training dataset by default
                      setSelectedValidationDatasetId(newDatasetId);
                    }}
                    disabled={isSubmittingJob || !!activeTrainingJobId}
                  >
                    <option value="">Choose a dataset...</option>
                    {availableDatasets.map(datasetId => {
                      const dataset = allVideoResults[datasetId];
                      // allVideoResults has videos nested, need to count them
                      const videoCount = dataset?.videos ? Object.keys(dataset.videos).length : 0;
                      const totalClips = dataset?.videos 
                        ? Object.values(dataset.videos).reduce((sum, video) => sum + (video.clips?.length || 0), 0)
                        : 0;
                      return (
                        <option key={datasetId} value={datasetId}>
                          {datasetId} ({videoCount} videos, {totalClips} clips)
                        </option>
                      );
                    })}
                  </Form.Select>
                </Form.Group>

                {selectedDatasetId && allVideoResults[selectedDatasetId] && (
                  <div className="mb-3">
                    <h6>Training Dataset Information:</h6>
                    <ul className="list-unstyled small text-muted">
                      <li>📹 Videos: {allVideoResults[selectedDatasetId].videos ? Object.keys(allVideoResults[selectedDatasetId].videos).length : 0}</li>
                      <li>✂️ Total Clips: {allVideoResults[selectedDatasetId].videos 
                        ? Object.values(allVideoResults[selectedDatasetId].videos).reduce((sum, video) => sum + (video.clips?.length || 0), 0)
                        : 0}</li>
                    </ul>
                  </div>
                )}

                <Form.Group className="mb-3">
                  <Form.Label>Select Validation Dataset</Form.Label>
                  <Form.Select
                    value={selectedValidationDatasetId}
                    onChange={(e) => setSelectedValidationDatasetId(e.target.value)}
                    disabled={isSubmittingJob || !!activeTrainingJobId || !selectedDatasetId}
                  >
                    <option value="">Choose a dataset...</option>
                    {availableDatasets.map(datasetId => {
                      const dataset = allVideoResults[datasetId];
                      // allVideoResults has videos nested, need to count them
                      const videoCount = dataset?.videos ? Object.keys(dataset.videos).length : 0;
                      const totalClips = dataset?.videos 
                        ? Object.values(dataset.videos).reduce((sum, video) => sum + (video.clips?.length || 0), 0)
                        : 0;
                      return (
                        <option key={datasetId} value={datasetId}>
                          {datasetId} ({videoCount} videos, {totalClips} clips)
                        </option>
                      );
                    })}
                  </Form.Select>
                </Form.Group>

                {selectedValidationDatasetId && allVideoResults[selectedValidationDatasetId] && (
                  <div className="mb-3">
                    <h6>Validation Dataset Information:</h6>
                    <ul className="list-unstyled small text-muted">
                      <li>📹 Videos: {allVideoResults[selectedValidationDatasetId].videos ? Object.keys(allVideoResults[selectedValidationDatasetId].videos).length : 0}</li>
                      <li>✂️ Total Clips: {allVideoResults[selectedValidationDatasetId].videos 
                        ? Object.values(allVideoResults[selectedValidationDatasetId].videos).reduce((sum, video) => sum + (video.clips?.length || 0), 0)
                        : 0}</li>
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
                              <Badge bg={getStatusBadgeVariant(job.status)}>
                                {job.status ? job.status.toUpperCase() : 'UNKNOWN'}
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
                <strong>Training Configuration:</strong> Training configuration can be found in ddm_train_config.yaml<br />
                <strong>Note</strong>
                <ul className="mb-0">
                  <li>
                    Training progress is automatically monitored. You can safely navigate away and return to check the status.
                  </li>
                  <li>
                    To adjust hyperparameters, we recommend modifying the resolution, learning rate, and batch size first.
                  </li>
                  <li>
                    Since our action segmentation model is a lightweight CV solution, we recommend selecting a validation dataset from a similar domain as the training dataset.
                  </li>
                </ul>
              </small>
            </div>
          </>
        )}
      </Card.Body>
    </Card>
  );
};

export default DDMTrainingPanel;

