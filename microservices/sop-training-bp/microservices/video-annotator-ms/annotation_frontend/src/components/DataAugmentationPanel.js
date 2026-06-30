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
import PropTypes from 'prop-types';
import { Card, Button, Form, Alert, ProgressBar, Spinner, Row, Col } from 'react-bootstrap';

const API_BASE_URL = '/api/augmentation';

const DataAugmentationPanel = ({ allVideoResults, augmentedDatasets, onAugmentationComplete }) => {
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [isAugmenting, setIsAugmenting] = useState(false);
  const [augmentationProgress, setAugmentationProgress] = useState(0);
  const [augmentationStatus, setAugmentationStatus] = useState({
    show: false,
    variant: 'info',
    message: ''
  });
  const statusPollRef = useRef(null);

  // Get available dataset IDs
  const availableDatasetIds = Object.keys(allVideoResults).filter(
    datasetId => {
        const dataset = allVideoResults[datasetId];
        // Support new structure {videos: {...}} or old structure {...}
        const videos = (dataset && dataset.videos) ? dataset.videos : dataset;
        return videos && Object.keys(videos).length > 0;
    }
  );

  // Clear any existing polling interval (used on unmount and when job completes/fails)
  const clearStatusPolling = () => {
    if (statusPollRef.current) {
      clearInterval(statusPollRef.current);
      statusPollRef.current = null;
    }
  };

  // Start polling backend for real-time augmentation status/progress
  const startStatusPolling = (datasetId) => {
    clearStatusPolling();
    // Poll every 2 seconds
    statusPollRef.current = setInterval(async () => {
      try {
        const resp = await fetch(`${API_BASE_URL}/api/v1/augmentation_status/${datasetId}`, {
          method: 'GET',
          headers: {
            'accept': 'application/json'
          }
        });
        if (!resp.ok) {
          const errorText = await resp.text();
          throw new Error(`Status check failed: ${resp.status} ${resp.statusText} - ${errorText}`);
        }
        const statusPayload = await resp.json();
        // Expecting: { dataset_id, status, progress }
        if (typeof statusPayload.progress === 'number') {
          setAugmentationProgress(statusPayload.progress);
        }

        // Normalize status for robust comparisons
        const normalizedStatus = (statusPayload.status || '').toString().toUpperCase();

        // Handle terminal states
        if (normalizedStatus === 'COMPLETED') {
          clearStatusPolling();
          setAugmentationProgress(100);
          setIsAugmenting(false);
          setAugmentationStatus({
            show: true,
            variant: 'success',
            message: `Data augmentation completed for dataset: ${datasetId}`
          });

          // Save augmented dataset info and notify parent
          const datasetInfo = allVideoResults[selectedDatasetId] || {};
          const videos = (datasetInfo.videos && typeof datasetInfo.videos === 'object') ? datasetInfo.videos : datasetInfo;

          const newAugmentedDataset = {
            status: 'completed',
            videoCount: Object.keys(videos).length,
            totalClips: Object.values(videos)
              .reduce((sum, video) => sum + (video.clips ? video.clips.length : 0), 0)
          };
          if (onAugmentationComplete) {
            onAugmentationComplete(datasetId, newAugmentedDataset);
          }
        } else if (normalizedStatus === 'FAILED') {
          clearStatusPolling();
          setIsAugmenting(false);
          setAugmentationStatus({
            show: true,
            variant: 'danger',
            message: `Augmentation failed for dataset: ${datasetId}`
          });
        } else {
          // Intermediate states (e.g., PENDING/RUNNING)
          setAugmentationStatus({
            show: true,
            variant: 'info',
            message: `Augmentation ${normalizedStatus ? normalizedStatus.toLowerCase() : 'in progress'}...`
          });
        }
      } catch (err) {
        // On polling error, stop and surface the error
        clearStatusPolling();
        setIsAugmenting(false);
        setAugmentationStatus({
          show: true,
          variant: 'danger',
          message: `Failed to poll augmentation status: ${err.message}`
        });
      }
    }, 10000); // Poll every 10 seconds
  };

  useEffect(() => {
    return () => {
      clearStatusPolling();
    };
  }, []);

  const handleStartAugmentation = async () => {
    if (!selectedDatasetId) {
      setAugmentationStatus({
        show: true,
        variant: 'warning',
        message: 'Please select a dataset ID first.'
      });
      return;
    }

    setIsAugmenting(true);
    setAugmentationProgress(0);
    setAugmentationStatus({
      show: true,
      variant: 'info',
      message: `Starting data augmentation for dataset: ${selectedDatasetId}`
    });

    try {
      // Determine if the selected dataset has two-operator mode enabled
      const datasetInfo = allVideoResults[selectedDatasetId] || {};
      const isTwoOperatorMode = datasetInfo.twoOperatorMode || false;

      const queryParams = new URLSearchParams({
        label_data_id: selectedDatasetId,
        two_operator_mode: isTwoOperatorMode,
      });

      // Call the data augmentation backend API
      const response = await fetch(`${API_BASE_URL}/api/v1/augment?${queryParams.toString()}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'accept': 'application/json'
        },
        body: '' // Empty body as per your API specification
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to start augmentation: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const result = await response.json();
      console.log('Augmentation API response:', result);

      // Check if the response contains the expected fields
      if (!result.dataset_id || !result.message) {
        throw new Error('Invalid response format from augmentation API');
      }

      // Update status to reflect submission and begin polling real progress
      setAugmentationStatus({
        show: true,
        variant: 'info',
        message: `${result.message}. Tracking progress...`
      });
      startStatusPolling(result.dataset_id);

    } catch (error) {
      console.error('Augmentation failed:', error);
      setIsAugmenting(false);
      setAugmentationStatus({
        show: true,
        variant: 'danger',
        message: `Augmentation failed: ${error.message}`
      });
    }
  };

  const getDatasetStats = (datasetId) => {
    const datasetInfo = allVideoResults[datasetId];
    if (!datasetInfo) return { videoCount: 0, totalClips: 0, totalDuration: 0 };

    // Handle new structure {videos: {...}} vs old structure (direct videos object)
    const datasetResults = (datasetInfo.videos && typeof datasetInfo.videos === 'object')
      ? datasetInfo.videos
      : datasetInfo;

    const videoCount = Object.keys(datasetResults).length;

    const totalClips = Object.values(datasetResults)
      .reduce((sum, video) => {
        // Ensure video is an object and has clips array
        if (video && Array.isArray(video.clips)) {
            return sum + video.clips.length;
        }
        return sum;
      }, 0);

    const totalDuration = Object.values(datasetResults)
      .reduce((sum, video) => {
        // Ensure video is an object and has totalDuration
        if (video && typeof video.totalDuration === 'number') {
            return sum + video.totalDuration;
        }
        return sum;
      }, 0);

    return { videoCount, totalClips, totalDuration };
  };

  return (
    <Card className="mb-4">
      <Card.Header>
        <h4>🔄 Data Augmentation</h4>
        <small className="text-muted">
          Generate multiple format of QA pairs from annotated video clips
        </small>
      </Card.Header>
      <Card.Body>
        {availableDatasetIds.length === 0 ? (
          <Alert variant="info">
            No annotated datasets available. Please complete video annotation first.
          </Alert>
        ) : (
          <>
            <Row>
              <Col>
                <Form.Group className="mb-3">
                  <Form.Label>Select Dataset ID</Form.Label>
                  <Form.Select
                    value={selectedDatasetId}
                    onChange={(e) => setSelectedDatasetId(e.target.value)}
                    disabled={isAugmenting}
                  >
                    <option value="">Choose a dataset...</option>
                    {availableDatasetIds.map(datasetId => {
                      const stats = getDatasetStats(datasetId);
                      return (
                        <option key={datasetId} value={datasetId}>
                          {datasetId} ({stats.videoCount} videos, {stats.totalClips} clips)
                        </option>
                      );
                    })}
                  </Form.Select>
                </Form.Group>

                {selectedDatasetId && (
                  <div className="mb-3">
                    <h6>Dataset Statistics:</h6>
                    <ul className="list-unstyled small text-muted">
                      <li>Videos: {getDatasetStats(selectedDatasetId).videoCount}</li>
                      <li>Total Clips: {getDatasetStats(selectedDatasetId).totalClips}</li>
                      <li>Total Duration: {getDatasetStats(selectedDatasetId).totalDuration.toFixed(2)}s</li>
                      <li>Two-Operator Mode: {(allVideoResults[selectedDatasetId]?.twoOperatorMode) ? 'Enabled' : 'Disabled'}</li>
                    </ul>
                  </div>
                )}

                {selectedDatasetId && (allVideoResults[selectedDatasetId]?.twoOperatorMode) && (
                  <Alert variant="light" className="mb-3 py-2 border">
                    <small>
                      <strong>Multi-operator pipeline active.</strong> Small chunk merging, spatial localization, and frame drop are configured via augment_config.yaml.
                    </small>
                  </Alert>
                )}
              </Col>
            </Row>

            {augmentationStatus.show && (
              <Alert
                variant={augmentationStatus.variant}
                dismissible
                onClose={() => setAugmentationStatus({...augmentationStatus, show: false})}
                className="mb-3"
              >
                {augmentationStatus.message}
              </Alert>
            )}

            {isAugmenting && (
              <div className="mb-3">
                <div className="d-flex justify-content-between align-items-center mb-2">
                  <span>Augmentation Progress</span>
                  <span>{Math.round(augmentationProgress)}%</span>
                </div>
                <ProgressBar
                  now={augmentationProgress}
                  variant="info"
                  animated
                  striped
                />
              </div>
            )}

            <div className="d-flex gap-2">
              <Button
                variant="primary"
                onClick={handleStartAugmentation}
                disabled={!selectedDatasetId || isAugmenting}
              >
                {isAugmenting ? (
                  <>
                    <Spinner size="sm" className="me-2" />
                    Augmenting...
                  </>
                ) : (
                  'Start Augmentation'
                )}
              </Button>

              {selectedDatasetId && augmentedDatasets[selectedDatasetId] && (
                <Button variant="outline-success" disabled>
                  ✓ Already Augmented
                </Button>
              )}
            </div>

            <div className="mt-3 p-3 bg-light rounded">
              <small className="text-muted">
                <strong>Note:</strong> Data augmentation will process the annotated video clips
                and generate multiple different QA pairs in LLava format dataset suitable for VLM training. This process
                may take several minutes depending on the dataset size.
              </small>
            </div>
          </>
        )}
      </Card.Body>
    </Card>
  );
};

DataAugmentationPanel.propTypes = {
  allVideoResults: PropTypes.object,
  augmentedDatasets: PropTypes.object,
  onAugmentationComplete: PropTypes.func,
};

export default DataAugmentationPanel;