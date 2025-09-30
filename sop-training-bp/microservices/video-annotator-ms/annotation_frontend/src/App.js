// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Alert, Card, Form, Button } from 'react-bootstrap';
import VideoUploader from './components/VideoUploader';
import ActionTimestampEditor from './components/ActionTimestampEditor';
import AllClipsViewer from './components/AllClipsViewer';
import DataAugmentationPanel from './components/DataAugmentationPanel';
import VLMTrainingPanel from './components/VLMTrainingPanel';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

// Use nginx proxy path to avoid CORS issues
const API_BASE_URL = '/api/annotation';
const AUGMENTATION_API_BASE_URL = '/api/augmentation';

function App() {
  // Workflow state management
  const [workflowState, setWorkflowState] = useState('INITIAL'); // INITIAL, ACTIONS_LOADED, VIDEO_LOADED, TIMESTAMPS_SET

  // Action file handling
  const [actionsFile, setActionsFile] = useState(null);
  const [actions, setActions] = useState([]);
  const [error, setError] = useState('');
  const [currentDataId, setCurrentDataId] = useState(null);

  // Video state
  const [uploadedVideo, setUploadedVideo] = useState(null); // Represents the currently active video for step 3
  const [uploaderKey, setUploaderKey] = useState(0); // Key to reset VideoUploader

  // Batch processing states
  const [videoQueue, setVideoQueue] = useState([]);
  const [currentVideoIndex, setCurrentVideoIndex] = useState(0);
  const [isBatchModeActive, setIsBatchModeActive] = useState(false);
  const [fileForUploader, setFileForUploader] = useState(null); // File to be auto-uploaded by VideoUploader
  const [batchOverallStatusMessage, setBatchOverallStatusMessage] = useState('');

  // Global video results management
  const [allVideoResults, setAllVideoResults] = useState({});

  // Data augmentation and VLM training states
  const [augmentedDatasets, setAugmentedDatasets] = useState({});

  // Load results from backend API on component mount
  useEffect(() => {
    fetchAllVideoResults();
  }, []);

  // Function to fetch historical results from backend
  // setLoadingState: Optional function to manage loading state (e.g., setIsRefreshing)
  const fetchAllVideoResults = async (setLoadingState = null) => {
    try {
      if (setLoadingState) setLoadingState(true);
      console.log('Fetching historical results from backend...');
      const response = await fetch(`${API_BASE_URL}/api/v1/datasets`);

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Failed to fetch historical results:', errorData.detail || `HTTP ${response.status}`);
        return false;
      }

      const datasetsData = await response.json();
      console.log('Fetched historical results from backend:', datasetsData);

      // Transform the backend data format to match our frontend format
      const transformedResults = {};

      for (const [dataId, videosData] of Object.entries(datasetsData)) {
        transformedResults[dataId] = {};

        for (const [videoId, videoData] of Object.entries(videosData)) {
          transformedResults[dataId][videoId] = {
            originalFilename: videoData.original_file_name,
            clips: videoData.clips.map(clip => ({
              id: clip.id,
              filename: clip.filename,
              start_time: clip.start_time,
              end_time: clip.end_time,
              duration: clip.duration,
              action_description: clip.action_description,
              created_at: videoData.processed_at // Use processed_at as created_at for clips
            })),
            totalDuration: videoData.total_duration,
            processedAt: videoData.processed_at
          };
        }
      }

      console.log('Transformed results for frontend:', transformedResults);
      setAllVideoResults(transformedResults);
      return true;
    } catch (error) {
      console.error('Failed to fetch historical results from backend:', error);
      return false;
    } finally {
      if (setLoadingState) setLoadingState(false);
    }
  };

  // Function to refresh historical results from backend (wrapper for manual refresh)
  const refreshAllVideoResults = async (setLoadingState = null) => {
    return await fetchAllVideoResults(setLoadingState);
  };

  // Load augmented datasets from augmentation microservice API on component mount
  useEffect(() => {
    fetchAllAugmentedDatasets();
  }, []);

  // Function to fetch augmented datasets from augmentation microservice
  // setLoadingState: Optional function to manage loading state
  const fetchAllAugmentedDatasets = async (setLoadingState = null) => {
    try {
      if (setLoadingState) setLoadingState(true);
      console.log('Fetching augmented datasets from augmentation microservice...');

      const response = await fetch(`${AUGMENTATION_API_BASE_URL}/api/v1/augmented_datasets`);

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Failed to fetch augmented datasets:', errorData.detail || `HTTP ${response.status}`);
        return false;
      }

      const augmentedDatasetsData = await response.json();
      console.log('Fetched augmented datasets from microservice:', augmentedDatasetsData);

      // Transform the backend data format to match our frontend format
      const transformedAugmentedDatasets = {};

      for (const [augmentedDataId, datasetInfo] of Object.entries(augmentedDatasetsData)) {
        transformedAugmentedDatasets[augmentedDataId] = {
          status: datasetInfo.status,
          videoCount: datasetInfo.video_count,
          totalClips: datasetInfo.total_clips
        };
      }

      console.log('Transformed augmented datasets for frontend:', transformedAugmentedDatasets);
      setAugmentedDatasets(transformedAugmentedDatasets);
      return true;
    } catch (error) {
      console.error('Failed to fetch augmented datasets from microservice:', error);
      return false;
    } finally {
      if (setLoadingState) setLoadingState(false);
    }
  };

  // Function to refresh augmented datasets from microservice (wrapper for manual refresh)
  const refreshAllAugmentedDatasets = async (setLoadingState = null) => {
    return await fetchAllAugmentedDatasets(setLoadingState);
  };

  // Add or update video results
  const addVideoResult = (videoId, videoData, clips) => {
    const totalDuration = clips.reduce((sum, clip) => sum + parseFloat(clip.duration), 0);

    const result = {
      originalFilename: videoData.filename,
      clips: clips.map(clip => ({
        id: clip.id,
        filename: clip.filename,
        start_time: clip.start_time,
        end_time: clip.end_time,
        duration: clip.duration,
        action_description: clip.action_description,
        created_at: clip.created_at || new Date().toISOString()
      })),
      totalDuration: totalDuration,
      processedAt: clips[clips.length - 1]?.created_at || new Date().toISOString()
    };

    // Group results by data_id
    const dataIdToUse = currentDataId || 'unknown';

    setAllVideoResults(prev => ({
      ...prev,
      [dataIdToUse]: {
        ...(prev[dataIdToUse] || {}),
        [videoId]: result
      }
    }));
  };

  // Handle augmentation completion
  const handleAugmentationComplete = async (datasetId, augmentationInfo) => {
    console.log('Augmentation completed for dataset:', datasetId, augmentationInfo);
    setAugmentedDatasets(prev => ({
      ...prev,
      [datasetId]: augmentationInfo
    }));
  };

  // Handle training completion
  const handleTrainingComplete = (jobId, trainingInfo) => {
    console.log('Training completed for job:', jobId, trainingInfo);
    // You can add additional logic here if needed
  };

  // Clear all results or specific data_id results
  const clearAllResults = (specificDataId = null) => {
    if (specificDataId) {
      // Clear specific data_id version
      if (window.confirm(`Are you sure you want to clear all results for dataset ${specificDataId}? This cannot be undone.`)) {
        setAllVideoResults(prev => {
          const newResults = { ...prev };
          delete newResults[specificDataId];
          return newResults;
        });

        // Also clear corresponding augmented dataset if exists
        if (augmentedDatasets[specificDataId]) {
          setAugmentedDatasets(prev => {
            const newAugmented = { ...prev };
            delete newAugmented[specificDataId];
            return newAugmented;
          });
        }
      }
    } else {
      // Clear all results
      if (window.confirm('Are you sure you want to clear all split results? This cannot be undone.')) {
        setAllVideoResults({});
        // Also clear all augmented datasets
        setAugmentedDatasets({});
      }
    }
  };

  // Reset uploader and related video states
  const resetStep2Uploader = () => {
    console.log('Resetting Step 2 Uploader');
    setUploaderKey(prev => prev + 1);
    setUploadedVideo(null); // Clear any previously processed single video
    setFileForUploader(null); // Clear any pending auto-upload file
    // Don't change workflowState here, let the calling function decide
  };

  // Process the next video in the queue
  const processNextVideoInQueue = (index, queue) => {
    if (index >= queue.length) {
      setBatchOverallStatusMessage('Batch processing completed! All videos processed.');
      setIsBatchModeActive(false);
      setVideoQueue([]);
      setCurrentVideoIndex(0);
      setWorkflowState('ACTIONS_LOADED'); // Ready for new single/batch
      resetStep2Uploader(); // Reset uploader for a fresh start

      // Clear the batch status message after a delay
      setTimeout(() => {
        setBatchOverallStatusMessage('');
      }, 5000); // Clear message after 5 seconds

      return;
    }

    const fileToProcess = queue[index];
    setBatchOverallStatusMessage(`Batch: Processing video ${index + 1} of ${queue.length}: "${fileToProcess.name}"`);
    setUploadedVideo(null); // Clear previous video data
    setError(''); // Clear previous errors
    setFileForUploader(fileToProcess); // Pass this file to VideoUploader for auto-upload
    // setUploaderKey(prev => prev + 1); // VideoUploader's useEffect for fileToAutoUpload should handle it.
                                     // Or, if issues, uncomment to force re-mount.
    setWorkflowState('ACTIONS_LOADED'); // Ensure uploader is enabled
  };

  // Handle actions.json file upload
  const handleActionsFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setActionsFile(file);
    const reader = new FileReader();

    reader.onload = async (event) => {
      try {
        const jsonData = JSON.parse(event.target.result);
        if (jsonData && jsonData.actions && Array.isArray(jsonData.actions)) {
          setActions(jsonData.actions);
          setError('');

          if (isBatchModeActive) {
             // If actions change during a batch, it's complex. Simplest is to cancel batch.
            if(window.confirm("Changing actions will stop the current batch processing. Continue?")) {
                setIsBatchModeActive(false);
                setVideoQueue([]);
                setCurrentVideoIndex(0);
                setFileForUploader(null);
                setBatchOverallStatusMessage('Batch mode cancelled due to actions change.');
            } else {
                // User cancelled changing actions, revert actionsFile if possible or do nothing.
                // For simplicity, we allow actions to change and batch cancels.
                 e.target.value = null; // Try to clear the file input
                 setActionsFile(null); // Clear state
                return;
            }
          }

          // Upload actions.json file to backend
          try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${API_BASE_URL}/api/v1/actions/upload`, {
              method: 'POST',
              body: formData,
            });

            if (!response.ok) {
              const errorData = await response.json();
              throw new Error(errorData.detail || 'Failed to upload actions file to server');
            }

            const uploadResult = await response.json();
            console.log('Actions file uploaded to backend successfully:', uploadResult);

            // Save the data_id from backend response
            if (uploadResult.data_id) {
              setCurrentDataId(uploadResult.data_id);
              console.log('Set current data_id:', uploadResult.data_id);
            }

            // Continue with the normal flow
            setWorkflowState('ACTIONS_LOADED'); // Proceed to video upload step
            setUploadedVideo(null); // Reset any previously uploaded video
            resetStep2Uploader(); // Reset uploader for new actions

          } catch (uploadError) {
            console.error('Failed to upload actions file to backend:', uploadError);
            setError(`Actions loaded locally but failed to upload to server: ${uploadError.message}. You can still proceed with video annotation.`);

            // Even if upload fails, allow user to proceed with local actions
            setWorkflowState('ACTIONS_LOADED');
            setUploadedVideo(null);
            resetStep2Uploader();
          }

        } else {
          setError('File format error: actions array not found in JSON');
        }
      } catch (err) {
        console.error('JSON parsing failed:', err);
        setError('JSON file parsing failed. Please ensure the file format is correct');
      }
    };
    reader.onerror = () => setError('Failed to read actions file');
    reader.readAsText(file);
  };

  // Called by VideoUploader when a folder is selected
  const handleFolderSelected = (files, skippedCount = 0) => {
    if (workflowState === 'INITIAL') {
        setError("Please load an actions.json file (Step 1) before selecting a video folder.");
        return;
    }
    console.log('App.js: Folder selected with', files.length, 'videos.');
    if (skippedCount > 0) {
        console.log('App.js: Skipped', skippedCount, 'oversized files.');
    }

    // Clear any previous batch status message
    setBatchOverallStatusMessage('');

    setVideoQueue(files);
    setCurrentVideoIndex(0);
    setIsBatchModeActive(true);
    setUploadedVideo(null);
    setError('');

    // Create appropriate message based on whether files were skipped
    let message = `Batch mode started with ${files.length} videos.`;
    if (skippedCount > 0) {
        message = `Batch mode started with ${files.length} videos. (Skipped ${skippedCount} files that exceed size limit)`;
    }
    setBatchOverallStatusMessage(message);
    processNextVideoInQueue(0, files);
  };

  // Called by VideoUploader when its internal auto-upload process begins for a file
  const handleUploadProcessStartedInUploader = (filename) => {
    if (isBatchModeActive) {
        setBatchOverallStatusMessage(`Batch: Uploading video ${currentVideoIndex + 1} of ${videoQueue.length}: "${filename}"...`);
    }
  };

  // Handle successful video upload event (from VideoUploader)
  const handleVideoUploaded = (videoData, uploadErrorMessage) => {
    setFileForUploader(null); // Clear the trigger for VideoUploader

    if (videoData) {
      console.log('App.js: Video uploaded successfully:', videoData);
      setUploadedVideo(videoData);
      setWorkflowState('VIDEO_LOADED');
      setError(''); // Clear previous errors
      if (isBatchModeActive) {
        setBatchOverallStatusMessage(`Batch: Video "${videoData.filename}" uploaded. Ready for timestamping.`);
      } else {
        // Clear any lingering batch status message for single file upload
        setBatchOverallStatusMessage('');
      }
    } else {
      // Upload failed
      const failedFileName = isBatchModeActive && videoQueue[currentVideoIndex] ? videoQueue[currentVideoIndex].name : "selected video";
      const errMsg = uploadErrorMessage || `Failed to upload "${failedFileName}".`;
      setError(errMsg);
      console.error('App.js: Video upload failed.', errMsg);

      if (isBatchModeActive) {
        if (window.confirm(`Error uploading "${failedFileName}". Stop batch processing?`)) {
            setBatchOverallStatusMessage(`Batch processing stopped due to upload error for "${failedFileName}".`);
            setIsBatchModeActive(false);
            setVideoQueue([]);
            setCurrentVideoIndex(0);
            setFileForUploader(null);
            setWorkflowState('ACTIONS_LOADED');
        } else {
            // Skip this video and continue to next
            setBatchOverallStatusMessage(`Skipped "${failedFileName}" due to upload error. Continuing to next video...`);
            const nextIndex = currentVideoIndex + 1;
            setCurrentVideoIndex(nextIndex);
            processNextVideoInQueue(nextIndex, videoQueue);
            return; // Don't set workflowState to ACTIONS_LOADED in this case
        }
      } else {
        setWorkflowState('ACTIONS_LOADED'); // Revert to a state where user can try again or select new actions/video.
      }
    }
  };

  // Handle timestamp submission (from ActionTimestampEditor)
  const handleTimestampsSubmitted = (success, clipsCount, submissionErrorMsg, clips = []) => {
    if (!isBatchModeActive) {
        if (success && clips.length > 0) {
            // Save to global results for single video mode
            addVideoResult(uploadedVideo.id, uploadedVideo, clips);
            console.log(`Timestamps for ${uploadedVideo?.filename} submitted, ${clipsCount} clips created and saved.`);

            // For single video mode: automatically go back to step 2 after successful submission
            setUploadedVideo(null);
            setWorkflowState('ACTIONS_LOADED');
            resetStep2Uploader();
            setError(''); // Clear any previous errors
        } else if (!success) {
            setError(`Timestamp submission failed for ${uploadedVideo?.filename}: ${submissionErrorMsg}`);
        }
      // For single video, user manually proceeds or resets.
      return;
    }

    // --- In Batch Mode ---
    const processedVideoName = uploadedVideo?.filename || videoQueue[currentVideoIndex]?.name || "current video";

    if (success && clips.length > 0) {
      // Save to global results for batch mode
      addVideoResult(uploadedVideo.id, uploadedVideo, clips);
      setBatchOverallStatusMessage(`Batch: Timestamps for "${processedVideoName}" submitted (${clipsCount} clips saved).`);
    } else if (!success) {
      setError(`Timestamp submission failed for "${processedVideoName}" in batch: ${submissionErrorMsg}`);
      // Decide if batch should stop on timestamp error. For now, let's allow it to continue to the next video.
      setBatchOverallStatusMessage(`Batch: Error with timestamps for "${processedVideoName}". Moving to next video if available.`);
    }

    const nextIndex = currentVideoIndex + 1;
    setCurrentVideoIndex(nextIndex);
    setUploadedVideo(null);       // Clear current video details
    resetStep2Uploader();         // Reset uploader visuals and key
    setWorkflowState('ACTIONS_LOADED'); // Prepare for next upload cycle

    // Crucially, call processNextVideoInQueue with the updated index and existing queue
    processNextVideoInQueue(nextIndex, videoQueue);
  };

  // Reset workflow
  const handleReset = async () => {
    if (window.confirm('Are you sure you want to reset the workflow? All current progress (including batch) will be lost and a new dataset annotation will be started.')) {
      try {
        // Call backend to reset data_id
        const response = await fetch(`${API_BASE_URL}/api/v1/actions/reset`, {
          method: 'POST',
        });

        if (response.ok) {
          const resetResult = await response.json();
          console.log('Backend data_id reset successfully:', resetResult);
        } else {
          console.warn('Failed to reset backend data_id, but continuing with frontend reset');
        }
      } catch (error) {
        console.warn('Error calling backend reset API:', error);
        // Continue with frontend reset even if backend call fails
      }

      // Reset frontend state
      setActionsFile(null);
      setActions([]);
      setUploadedVideo(null);
      setWorkflowState('INITIAL');
      setError('');
      setCurrentDataId(null);

      // Reset batch states
      setIsBatchModeActive(false);
      setVideoQueue([]);
      setCurrentVideoIndex(0);
      setFileForUploader(null);
      setBatchOverallStatusMessage('');

      resetStep2Uploader();
    }
  };

  // When user clicks "Back to Step 2" button (from timestamp editor)
  const handleBackToStep2 = () => {
    if (isBatchModeActive) {
      if(window.confirm("Going back will stop the current batch processing. Are you sure?")) {
        setIsBatchModeActive(false);
        setVideoQueue([]);
        setCurrentVideoIndex(0);
        setFileForUploader(null);
        setBatchOverallStatusMessage('Batch mode cancelled by user.');
        // Fall through to normal "back to step 2" logic
      } else {
        return; // User cancelled, do nothing
      }
    }
    setUploadedVideo(null);
    setWorkflowState('ACTIONS_LOADED');
    setError('');
    resetStep2Uploader();
  };

  return (
    <div className="App">
      <Container fluid="md" className="my-4">
        <Row>
          <Col>
            <header className="mb-4 text-center">
              <h1>SOP Monitoring Training</h1>
              <p className="text-muted">
                {isBatchModeActive
                  ? batchOverallStatusMessage || "Processing video batch..."
                  : "Upload actions.json, then upload video(s), and finally set timestamps."}
              </p>
              {currentDataId && (
                <p className="text-info small">
                  <strong>Current Dataset ID:</strong> {currentDataId}
                </p>
              )}
            </header>
          </Col>
        </Row>

        {/* Workflow status indicator - May need adjustment for batch mode clarity */}
        <Row className="mb-4">
          <Col>
            <Card>
              <Card.Body>
                <div className="d-flex justify-content-between">
                  {/* Step 1 */}
                  <div className={`workflow-step ${workflowState !== 'INITIAL' || isBatchModeActive ? 'completed' : 'active'}`}>
                    <div className="step-number">1</div>
                    <div className="step-text">Upload Actions</div>
                  </div>
                  <div className="workflow-connector"></div>
                  {/* Step 2 */}
                  <div className={`workflow-step ${
                    workflowState === 'ACTIONS_LOADED' && !isBatchModeActive ? 'active' :
                    (workflowState === 'VIDEO_LOADED' || workflowState === 'TIMESTAMPS_SET' || (isBatchModeActive && workflowState !== 'INITIAL')) ? 'completed' : ''
                  }`}>
                    <div className="step-number">2</div>
                    <div className="step-text">Upload Video(s)</div>
                  </div>
                  <div className="workflow-connector"></div>
                  {/* Step 3 */}
                  <div className={`workflow-step ${
                    workflowState === 'VIDEO_LOADED' ? 'active' :
                    workflowState === 'TIMESTAMPS_SET' ? 'completed' : ''
                  }`}>
                    <div className="step-number">3</div>
                    <div className="step-text">Set Timestamps</div>
                  </div>
                </div>
              </Card.Body>
            </Card>
          </Col>
        </Row>

        {error && (
          <Row className="mb-3">
            <Col><Alert variant="danger" dismissible onClose={() => setError('')}>{error}</Alert></Col>
          </Row>
        )}

        {isBatchModeActive && batchOverallStatusMessage && !error && (
             <Row className="mb-3">
                <Col><Alert variant="info">{batchOverallStatusMessage}</Alert></Col>
          </Row>
        )}

        {/* Step 1: Upload actions.json */}
        <Row className="mb-4">
          <Col>
            <Card className={(workflowState === 'INITIAL' && !isBatchModeActive) ? 'border-primary' : ''}>
              <Card.Header><h4>Step 1: Upload Actions File</h4></Card.Header>
              <Card.Body>
                <Form.Group controlId="actionsJsonUpload" className="mb-3">
                  <Form.Label>Upload actions.json file</Form.Label>
                  <Form.Control
                    type="file"
                    accept=".json"
                    onChange={handleActionsFileUpload}
                    disabled={workflowState !== 'INITIAL' && !isBatchModeActive } // Allow changing actions even if batch started, with warning
                    key={actionsFile ? 'file-selected' : 'no-file'} // To reset input if file state changes
                  />
                  <Form.Text className="text-muted">
                    Please upload a JSON file containing an "actions" array.
                  </Form.Text>
                </Form.Group>
                {actions.length > 0 && (
                  <>
                    <h5>Loaded Actions:</h5>
                    <ul className="list-group mb-3">
                      {actions.map((action, idx) => (
                        <li key={idx} className="list-group-item">{`${idx + 1}. ${action}`}</li>
                      ))}
                    </ul>
                  </>
                )}
                {(workflowState !== 'INITIAL' || isBatchModeActive) && (
                  <div className="text-end">
                    <Button variant="outline-secondary" size="sm" onClick={handleReset}>
                      Reset Workflow & Start another dataset annotation
                    </Button>
                  </div>
                )}
              </Card.Body>
            </Card>
          </Col>
        </Row>

        {/* Step 2: Upload Video (only if actions are loaded) */}
        {workflowState !== 'INITIAL' && (
          <Row className="mb-4">
            <Col>
              <Card className={(workflowState === 'ACTIONS_LOADED' && !fileForUploader && !isBatchModeActive) ? 'border-primary' : ''}>
                <Card.Header><h4>Step 2: Upload Video File(s)</h4></Card.Header>
                <Card.Body>
                  <VideoUploader
                    key={`uploader-${uploaderKey}`} // Key to force re-mount/reset
                    onVideoUploaded={handleVideoUploaded}
                    isEnabled={workflowState !== 'INITIAL' && actions.length > 0} // Enabled if actions are loaded
                    onFolderSelected={handleFolderSelected}
                    fileToAutoUpload={fileForUploader}
                    isBatchActive={isBatchModeActive}
                    onUploadProcessStarted={handleUploadProcessStartedInUploader}
                  />
                </Card.Body>
              </Card>
            </Col>
          </Row>
        )}

        {/* Step 3: Set Timestamps (only if a video is successfully loaded) */}
        {uploadedVideo && workflowState === 'VIDEO_LOADED' && (
          <Row className="mb-4">
            <Col>
              <Card className={workflowState === 'VIDEO_LOADED' ? 'border-primary' : ''}>
                <Card.Header className="d-flex justify-content-between align-items-center">
                  <h4>
                    Step 3: Set Timestamps for "{uploadedVideo.filename}"
                    {isBatchModeActive && videoQueue.length > 0 && ` (Video ${currentVideoIndex + 1} of ${videoQueue.length})`}
                  </h4>
                  <Button variant="outline-secondary" size="sm" onClick={handleBackToStep2} disabled={isBatchModeActive && currentVideoIndex > 0 /* Allow go back for first video in batch, or if not batch*/}>
                    Back to Video Upload
                  </Button>
                </Card.Header>
                <Card.Body>
                  <ActionTimestampEditor
                    actions={actions}
                    uploadedVideoId={uploadedVideo.id}
                    videoUrl={uploadedVideo.url}
                    key={`editor-${uploadedVideo.id}`} // Force recreate component when video changes
                    onTimestampsSubmitted={handleTimestampsSubmitted}
                  />
                </Card.Body>
              </Card>
            </Col>
          </Row>
        )}

        {/* All Split Results Viewer - Always visible when there are results */}
        <Row>
          <Col>
            <AllClipsViewer
              allVideoResults={allVideoResults}
              onClearAllResults={clearAllResults}
              onRefreshResults={refreshAllVideoResults}
            />
          </Col>
        </Row>

        {/* Data Augmentation Panel - Visible when there are annotated results */}
        {Object.keys(allVideoResults).length > 0 && (
          <Row>
            <Col>
              <DataAugmentationPanel
                allVideoResults={allVideoResults}
                augmentedDatasets={augmentedDatasets}
                onAugmentationComplete={handleAugmentationComplete}
              />
            </Col>
          </Row>
        )}

        {/* VLM Training Panel - Visible when there are augmented datasets */}
        {Object.keys(augmentedDatasets).length > 0 && (
          <Row>
            <Col>
              <VLMTrainingPanel
                augmentedDatasets={augmentedDatasets}
                onTrainingComplete={handleTrainingComplete}
              />
            </Col>
          </Row>
        )}
      </Container>

      {/* Inline styles for workflow steps remain the same, not re-pasting */}
      <style jsx="true">{`
        .workflow-step {
          display: flex;
          flex-direction: column;
          align-items: center;
          position: relative;
          width: 120px; /* Adjusted for three steps */
        }
        .step-number {
          width: 40px; height: 40px; border-radius: 50%;
          background-color: #f8f9fa; border: 2px solid #dee2e6;
          display: flex; align-items: center; justify-content: center;
          font-weight: bold; margin-bottom: 8px;
        }
        .workflow-step.active .step-number { background-color: #007bff; color: white; border-color: #007bff; }
        .workflow-step.completed .step-number { background-color: #28a745; color: white; border-color: #28a745; }
        .workflow-connector { flex-grow: 1; height: 2px; background-color: #dee2e6; margin-top: 20px; }
        .border-primary { box-shadow: 0 0 0 0.25rem rgba(13, 110, 253, 0.25); }
      `}</style>
    </div>
  );
}

export default App;