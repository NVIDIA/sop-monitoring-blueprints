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

import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Alert, Card, Form, Button } from 'react-bootstrap';
import VideoUploader from './components/VideoUploader';
import ActionTimestampEditor from './components/ActionTimestampEditor';
import AllClipsViewer from './components/AllClipsViewer';
import DataAugmentationPanel from './components/DataAugmentationPanel';
import VLMTrainingPanel from './components/VLMTrainingPanel';
import DDMTrainingPanel from './components/DDMTrainingPanel';
import EvaluationPanel from './components/EvaluationPanel';
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

  // Two-operator mode (dataset-level, persists across all video uploads)
  const [twoOperatorMode, setTwoOperatorMode] = useState(false);

  // Data augmentation and VLM training states
  const [augmentedDatasets, setAugmentedDatasets] = useState({});

  // Temporary context for single video re-annotation
  const [tempReAnnotationContext, setTempReAnnotationContext] = useState(null);

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

      for (const [dataId, datasetInfo] of Object.entries(datasetsData)) {
        // Handle new structure where datasetInfo contains actions and videos
        // Add safety check for datasetInfo being null/undefined
        if (!datasetInfo) continue;

        let videosData = {};
        let datasetActions = [];

        // Check if it's the new structure (has 'videos' property) or old structure (is the videos object itself)
        if (datasetInfo.videos && typeof datasetInfo.videos === 'object') {
            videosData = datasetInfo.videos;
            datasetActions = datasetInfo.actions || [];
        } else {
            // Assume it's the old structure where the value itself is the videos object
            // But we need to be careful not to treat simple properties as video objects
            videosData = datasetInfo;
        }

        transformedResults[dataId] = {
          actions: datasetActions,
          twoOperatorMode: datasetInfo.two_operator_mode || false,
          videos: {}
        };

        for (const [videoId, videoData] of Object.entries(videosData)) {
          // Defensive check: ensure videoData is an object and has expected fields
          if (!videoData || typeof videoData !== 'object') continue;

          // Skip if it looks like metadata (e.g. 'actions' array in old structure mix)
          if (videoId === 'actions' || Array.isArray(videoData)) continue;

          transformedResults[dataId].videos[videoId] = {
            id: videoId, // Ensure ID is preserved
            originalFilename: videoData.original_file_name || "Unknown",
            clips: Array.isArray(videoData.clips) ? videoData.clips.map(clip => ({
              id: clip.id,
              filename: clip.filename || "",
              start_time: clip.start_time || 0,
              end_time: clip.end_time || 0,
              duration: clip.duration || 0,
              action_description: clip.action_description || "",
              action_index: clip.action_index, // Capture action_index
              created_at: videoData.processed_at
            })) : [],
            totalDuration: videoData.total_duration || 0,
            processedAt: videoData.processed_at || new Date().toISOString()
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

  // Sync two-operator mode to backend and local allVideoResults when toggled
  const handleTwoOperatorModeChange = async (newMode) => {
    setTwoOperatorMode(newMode);

    // Update allVideoResults so DataAugmentationPanel sees the change immediately
    if (currentDataId) {
      setAllVideoResults(prev => {
        const existing = prev[currentDataId];
        if (existing) {
          return { ...prev, [currentDataId]: { ...existing, twoOperatorMode: newMode } };
        }
        return prev;
      });

      try {
        await fetch(`${API_BASE_URL}/api/v1/dataset/${currentDataId}/set_two_operator_mode`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ two_operator_mode: newMode }),
        });
        console.log(`Synced two_operator_mode=${newMode} for dataset ${currentDataId}`);
      } catch (err) {
        console.error('Failed to sync two_operator_mode to backend:', err);
      }
    }
  };

  // Add or update video results
  const addVideoResult = (videoId, videoData, clips) => {
    const totalDuration = clips.reduce((sum, clip) => sum + parseFloat(clip.duration), 0);

    const result = {
      id: videoId,
      originalFilename: videoData.filename,
      clips: clips.map(clip => ({
        id: clip.id,
        filename: clip.filename,
        start_time: clip.start_time,
        end_time: clip.end_time,
        duration: clip.duration,
        action_description: clip.action_description,
        action_index: clip.action_index,
        created_at: clip.created_at || new Date().toISOString()
      })),
      totalDuration: totalDuration,
      processedAt: clips[clips.length - 1]?.created_at || new Date().toISOString()
    };

    // Group results by data_id
    const dataIdToUse = currentDataId || 'unknown';

    setAllVideoResults(prev => {
        const prevDataset = prev[dataIdToUse] || { actions: actions, twoOperatorMode: twoOperatorMode, videos: {} };
        return {
            ...prev,
            [dataIdToUse]: {
                ...prevDataset,
                twoOperatorMode: prevDataset.twoOperatorMode ?? twoOperatorMode,
                videos: {
                    ...prevDataset.videos,
                    [videoId]: result
                }
            }
        };
    });
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
  const handleTimestampsSubmitted = async (success, clipsCount, submissionErrorMsg, clips = []) => {
    if (!isBatchModeActive) {
        if (success && clips.length > 0) {
            // Save to global results for single video mode
            addVideoResult(uploadedVideo.id, uploadedVideo, clips);
            console.log(`Timestamps for ${uploadedVideo?.filename} submitted, ${clipsCount} clips created and saved.`);

            // Check if we need to restore previous context (from single video re-annotation)
            if (tempReAnnotationContext) {
                console.log("Restoring previous context:", tempReAnnotationContext);

                // Try to sync backend FIRST before switching frontend state
                if (tempReAnnotationContext.dataId) {
                     try {
                        const response = await fetch(`${API_BASE_URL}/api/v1/dataset/${tempReAnnotationContext.dataId}/set_current`, {
                            method: 'POST'
                        });

                        if (!response.ok) {
                            const errorData = await response.json();
                            throw new Error(errorData.detail || "Server returned error");
                        }
                     } catch (err) {
                        console.error("Failed to restore backend context:", err);
                        alert(`Warning: Timestamp saved successfully, BUT failed to return to previous dataset context.\n\nTo ensure consistency, the workflow will be reset to Step 1. Please re-select your dataset or upload actions again.\n\nError: ${err.message}`);

                        // Fallback: Reset to INITIAL state for safety
                        setTempReAnnotationContext(null);
                        setUploadedVideo(null);
                        setActionsFile(null);
                        setActions([]);
                        setCurrentDataId(null);
                        setWorkflowState('INITIAL');
                        resetStep2Uploader();
                        setError('');
                        return;
                     }
                }

                // If sync successful, restore frontend state
                setCurrentDataId(tempReAnnotationContext.dataId);
                setActions(tempReAnnotationContext.actions);
                setWorkflowState(tempReAnnotationContext.workflowState);

                setTempReAnnotationContext(null);
                setUploadedVideo(null);
                resetStep2Uploader();
                setError('');
                return;
            }

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

  // Re-annotate a specific video
  const handleReAnnotateVideo = (datasetId, videoId) => {
    const dataset = allVideoResults[datasetId];
    if (!dataset || !dataset.videos[videoId]) return;

    const videoData = dataset.videos[videoId];

    // Save current context ONLY if we are not already in a re-annotation session
    if (!tempReAnnotationContext) {
        setTempReAnnotationContext({
            dataId: currentDataId,
            actions: actions,
            workflowState: workflowState
        });
    }

    // Set context
    setCurrentDataId(datasetId);
    if (dataset.actions && dataset.actions.length > 0) {
        setActions(dataset.actions);
    }
    setTwoOperatorMode(dataset.twoOperatorMode || false);

    // Prepare initial timestamps from existing clips
    // Sort by start time to be safe
    const sortedClips = [...videoData.clips].sort((a, b) => parseFloat(a.start_time) - parseFloat(b.start_time));
    const initialTimestamps = sortedClips.map(clip => ({
        timestamp: parseFloat(clip.end_time),
        actionIndex: clip.action_index !== undefined ? clip.action_index : 0
    }));

    // Set video state
    setUploadedVideo({
        id: videoId,
        filename: videoData.originalFilename,
        url: `${API_BASE_URL}/api/v1/videos/${videoId}/download`,
        initialTimestamps: initialTimestamps
    });

    // Move to Step 3
    setWorkflowState('VIDEO_LOADED');
    setError('');

    // Scroll to top
    window.scrollTo(0, 0);
  };

  // Load a dataset context (Re-annotate dataset)
  const handleLoadDataset = async (datasetId) => {
    const dataset = allVideoResults[datasetId];
    if (!dataset) return;

    if (window.confirm(`Load dataset "${datasetId}" for annotation? This will set the current actions and allow you to add more videos.`)) {
        // Call backend to update current_data_id
        try {
            const response = await fetch(`${API_BASE_URL}/api/v1/dataset/${datasetId}/set_current`, {
                method: 'POST',
            });

            if (!response.ok) {
                const errorData = await response.json();
                console.error('Failed to set current dataset on backend:', errorData);
                // Alert user and STOP flow to prevent inconsistency
                alert(`Error: Failed to sync dataset context with server. Cannot switch dataset.\n\nServer Error: ${errorData.detail || 'Unknown error'}`);
                return;
            } else {
                console.log(`Successfully set backend context to dataset ${datasetId}`);
            }
        } catch (e) {
            console.error('Error calling set_current_dataset API:', e);
            alert(`Network Error: Failed to communicate with server. Cannot switch dataset.\n\nPlease check your connection or server status.`);
            return;
        }

        // Only proceed if backend sync was successful
        setCurrentDataId(datasetId);
        if (dataset.actions && dataset.actions.length > 0) {
            setActions(dataset.actions);
        }
        setTwoOperatorMode(dataset.twoOperatorMode || false);

        // Clear any temporary re-annotation context as we are explicitly switching dataset
        setTempReAnnotationContext(null);

        setWorkflowState('ACTIONS_LOADED');
        setUploadedVideo(null);
        resetStep2Uploader();
        setError('');
        window.scrollTo(0, 0);
    }
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
      setTwoOperatorMode(false);

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
                    <Form.Group className="mb-3 mt-3 p-3 bg-light rounded">
                      <Form.Check
                        type="switch"
                        id="two-operator-mode-global"
                        label="Two-Operator Mode (Allow Concurrent Actions)"
                        checked={twoOperatorMode}
                        onChange={(e) => handleTwoOperatorModeChange(e.target.checked)}
                      />
                      <Form.Text className="text-muted">
                        {twoOperatorMode
                          ? "Enabled: All videos in this dataset will allow overlapping timestamps for concurrent actions. Spatial localization and frame drop augmentations will be included."
                          : "Disabled: All videos use sequential timestamps (single operator)."}
                      </Form.Text>
                    </Form.Group>
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
                    twoOperatorMode={twoOperatorMode}
                    videoUrl={uploadedVideo.url}
                    initialTimestamps={uploadedVideo.initialTimestamps} // Pass initial timestamps
                    key={`editor-${uploadedVideo.id}-${uploadedVideo.initialTimestamps ? 'reanno' : 'new'}`} // Force recreate component
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
              onReAnnotateVideo={handleReAnnotateVideo}
              onLoadDataset={handleLoadDataset}
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

        {/* DDM Training Panel - Visible when there are annotated results */}
        {Object.keys(allVideoResults).length > 0 && (
          <Row>
            <Col>
              <DDMTrainingPanel
                allVideoResults={allVideoResults}
                onTrainingComplete={handleTrainingComplete}
              />
            </Col>
          </Row>
        )}

        {/* Evaluation Panel */}
        <Row>
          <Col>
            <EvaluationPanel />
          </Col>
        </Row>
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