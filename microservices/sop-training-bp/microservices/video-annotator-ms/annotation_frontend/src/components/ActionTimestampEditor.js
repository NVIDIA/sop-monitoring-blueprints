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

import React, { useState, useRef, useEffect, useCallback } from 'react';
import PropTypes from 'prop-types';
import { Form, Button, Alert, Card, Row, Col, ProgressBar } from 'react-bootstrap';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';

// Use nginx proxy path to avoid CORS issues
const API_BASE_URL = '/api/annotation';

const ActionTimestampEditor = ({ actions = [], uploadedVideoId, videoUrl, initialTimestamps, onTimestampsSubmitted, twoOperatorMode = false }) => {
  // State for dynamic timestamp blocks
  const [timestampBlocks, setTimestampBlocks] = useState([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [videoDuration, setVideoDuration] = useState(0);
  const [fps, setFps] = useState(30);

  // Continuous coverage: auto-snap to prevent gaps in timeline (only active in two-operator mode)
  const [continuousCoverage, setContinuousCoverage] = useState(true);

  // Simplified preview system with direct video elements
  const videoRef = useRef(null);
  // Dual video elements for two-operator mode: one for start frame, one for end frame
  const [startVideoElements, setStartVideoElements] = useState([]);
  const [endVideoElements, setEndVideoElements] = useState([]);
  const timeUpdateThrottleRef = useRef(null);
  // Track seeks and metadata readiness per mini preview
  const seekInFlightRef = useRef({});
  const metadataReadyRef = useRef({});

  // Calculate frame-based time step
  const frameTimeStep = 1 / fps;

  // Helper function to seek a video element to a specific time
  const seekVideoToTime = useCallback(async (videoEl, targetTime, refKey) => {
    if (!videoEl?.current) return;

    const video = videoEl.current;

    try {
      const hasMeta = (video.readyState >= 1 && Number.isFinite(video.duration));
      if (!hasMeta) return;

      const knownDuration = Number.isFinite(video.duration) ? video.duration : videoDuration;
      const clampedTime = knownDuration !== undefined
        ? Math.max(0, Math.min(Number(targetTime) || 0, knownDuration))
        : Math.max(0, Number(targetTime) || 0);

      // Avoid overlapping seeks
      if (seekInFlightRef.current[refKey]) return;
      seekInFlightRef.current[refKey] = true;

      await new Promise((resolve) => {
        const done = () => {
          video.removeEventListener('seeked', done);
          video.removeEventListener('canplay', done);
          resolve();
        };
        video.addEventListener('seeked', done, { once: true });
        video.addEventListener('canplay', done, { once: true });
        video.currentTime = clampedTime;
      });

    } catch (error) {
      console.warn('Failed to seek video:', error);
    } finally {
      seekInFlightRef.current[refKey] = false;
    }
  }, [videoDuration]);

  // Update START preview for a specific block
  const updateStartPreview = useCallback(async (index, targetTime) => {
    const videoEl = startVideoElements[index];
    await seekVideoToTime(videoEl, targetTime, `start_${index}`);
  }, [startVideoElements, seekVideoToTime]);

  // Update END preview for a specific block
  const updateEndPreview = useCallback(async (index, targetTime) => {
    const videoEl = endVideoElements[index];
    await seekVideoToTime(videoEl, targetTime, `end_${index}`);
  }, [endVideoElements, seekVideoToTime]);

  // Legacy updatePreview for single-operator mode (uses end video element)
  const updatePreview = useCallback(async (index, targetTime) => {
    await updateEndPreview(index, targetTime);
  }, [updateEndPreview]);

  // Handler for range slider (two-operator mode) - sets both start and end times
  // Implements smart snapping to prevent gaps in the timeline
  const handleRangeChange = useCallback((index, values) => {
    // values is [startTime, endTime] from the range slider
    let [newStart, newEnd] = values;

    const oldBlock = timestampBlocks[index];
    const oldStart = oldBlock?.startTime || 0;
    const oldEnd = oldBlock?.endTime || 0;

    // Smart snapping logic for two-operator mode with continuous coverage
    if (twoOperatorMode && continuousCoverage) {
      // Check if newStart falls WITHIN any other action's duration (concurrent case)
      const isConcurrent = timestampBlocks.some((block, i) =>
        i !== index &&
        newStart >= block.startTime &&
        newStart < block.endTime
      );

      if (!isConcurrent) {
        // newStart is NOT overlapping with any action - potential gap!
        // Find the maximum end time of all other actions
        const otherBlocks = timestampBlocks.filter((_, i) => i !== index);

        if (otherBlocks.length > 0) {
          const maxPreviousEnd = Math.max(...otherBlocks.map(block => block.endTime));

          // If there's a gap (newStart > maxPreviousEnd), snap to close it
          if (newStart > maxPreviousEnd) {
            console.log(`Auto-snapping start from ${newStart.toFixed(3)}s to ${maxPreviousEnd.toFixed(3)}s (closing gap)`);
            newStart = maxPreviousEnd;
          }
        }
      }
      // If isConcurrent is true, allow free positioning (this is the overlap we want!)
    }

    const newTimestampBlocks = [...timestampBlocks];
    newTimestampBlocks[index] = {
      ...newTimestampBlocks[index],
      startTime: newStart,
      endTime: newEnd
    };
    setTimestampBlocks(newTimestampBlocks);

    // Update the appropriate preview(s) based on what changed
    // With dual previews, each preview updates independently
    if (Math.abs(newStart - oldStart) > 0.001) {
      updateStartPreview(index, newStart);
    }
    if (Math.abs(newEnd - oldEnd) > 0.001) {
      updateEndPreview(index, newEnd);
    }
  }, [timestampBlocks, updateStartPreview, updateEndPreview, twoOperatorMode, continuousCoverage]);

  // Simplified timestamp change handler with reduced debounce
  // In single-operator mode: handles end time only (start time is auto-calculated)
  // In two-operator mode: this is used for compatibility, but handleRangeChange is preferred
  const handleTimestampChange = useCallback((index, value) => {
    // Cancel previous timer
    if (timeUpdateThrottleRef.current) {
      clearTimeout(timeUpdateThrottleRef.current);
    }

    const newTimestampBlocks = [...timestampBlocks];
    const newTime = Number.parseFloat(value);

    // Validate time is within video bounds
    const validTime = Math.max(0, Math.min(newTime, videoDuration || 100));

    let finalTime = validTime;

    // In single-operator mode: Ensure timestamps are in correct order (sequential)
    // In two-operator mode: Allow overlapping timestamps for concurrent actions
    if (!twoOperatorMode) {
      if (index > 0 && validTime <= timestampBlocks[index - 1].endTime) {
        finalTime = timestampBlocks[index - 1].endTime + 0.1;
    }

      // Handle subsequent timestamps (only in single-operator mode)
      if (index < timestampBlocks.length - 1 && finalTime >= timestampBlocks[index + 1].endTime) {
        const delta = finalTime - timestampBlocks[index].endTime;
      for (let i = index + 1; i < timestampBlocks.length; i++) {
        newTimestampBlocks[i] = {
          ...newTimestampBlocks[i],
            startTime: newTimestampBlocks[i-1].endTime,
            endTime: Math.max(
              timestampBlocks[i].endTime + delta,
              newTimestampBlocks[i-1].endTime + 0.1
          )
        };
      }
    }
    }

    // For single-operator mode, startTime = previous endTime
    const startTime = index === 0 ? 0 : (timestampBlocks[index - 1]?.endTime || 0);

    newTimestampBlocks[index] = {
      ...newTimestampBlocks[index],
      startTime: twoOperatorMode ? newTimestampBlocks[index].startTime : startTime,
      endTime: finalTime
    };
    setTimestampBlocks(newTimestampBlocks);

    // Reduced debounce for smoother experience
    timeUpdateThrottleRef.current = setTimeout(async () => {
      try {
        // Update current preview
        await updatePreview(index, finalTime);

        // Update affected subsequent previews (only in single-operator mode)
        if (!twoOperatorMode && index < timestampBlocks.length - 1 && finalTime >= timestampBlocks[index + 1].endTime) {
          for (let i = index + 1; i < timestampBlocks.length; i++) {
            await updatePreview(i, newTimestampBlocks[i].endTime);
          }
        }
      } catch (error) {
        console.error('Failed to update preview:', error);
      }
    }, 100); // Reduced from 300ms to 100ms for smoother response
  }, [timestampBlocks, updatePreview, videoDuration, twoOperatorMode]);

  // Clear timeouts when component unmounts
  const clearTimeouts = useCallback(() => {
    if (timeUpdateThrottleRef.current) {
      clearTimeout(timeUpdateThrottleRef.current);
    }
  }, []);

  // Reset state when video changes
  useEffect(() => {
    if (uploadedVideoId) {
      // New structure: { startTime, endTime, actionIndex }
      setTimestampBlocks([{ startTime: 0, endTime: 0, actionIndex: 0 }]);
      clearTimeouts();
    }
  }, [uploadedVideoId, clearTimeouts]);

  // Log the current frame rate and step
  useEffect(() => {
    console.log(`Using frame rate: ${fps}fps, Frame duration: ${frameTimeStep.toFixed(5)}s`);
  }, [fps, frameTimeStep]);

  // Initialize timestamp blocks when actions are loaded and video duration is available
  useEffect(() => {
    if (actions.length > 0 && videoDuration > 0) {
        // Check if we have initial timestamps passed (from re-annotation)
        if (initialTimestamps && initialTimestamps.length > 0) {
             // Only initialize if we haven't modified it yet (check against default state)
             if (timestampBlocks.length === 1 && timestampBlocks[0].endTime === 0) {
                 console.log("Loading initial timestamps for re-annotation:", initialTimestamps);
                 // Convert old format to new format if needed
                 const convertedBlocks = initialTimestamps.map((ts, idx) => {
                   const fallbackStartTime = idx === 0 ? 0 : initialTimestamps[idx-1].timestamp || initialTimestamps[idx-1].endTime;
                   return {
                     startTime: ts.startTime === undefined ? fallbackStartTime : ts.startTime,
                     endTime: ts.endTime === undefined ? ts.timestamp : ts.endTime,
                     actionIndex: ts.actionIndex
                   };
                 });
                 setTimestampBlocks(convertedBlocks);
                 setMessage("Loaded existing annotations. You can modify them and submit to overwrite.");
             }
        } else if (timestampBlocks.length === 1 && timestampBlocks[0].endTime === 0) {
          // Initialize the first block with a reasonable start/end time
          const initialEnd = Math.min(videoDuration / 2, videoDuration - 1);
          setTimestampBlocks([{ startTime: 0, endTime: initialEnd, actionIndex: 0 }]);
    }
    }
  }, [actions, videoDuration, initialTimestamps]);

  // Initialize video element arrays when timestampBlocks change
  useEffect(() => {
    if (timestampBlocks.length > 0) {
      // Create video element references for start and end previews
      const startVideoEls = timestampBlocks.map(() => React.createRef());
      const endVideoEls = timestampBlocks.map(() => React.createRef());
      setStartVideoElements(startVideoEls);
      setEndVideoElements(endVideoEls);
    }
  }, [timestampBlocks.length]);

  // Load video when URL is provided
  useEffect(() => {
    // videoRef would be initialzied by JSX render to actual HTML video element
    if (videoUrl && videoRef.current) {
      videoRef.current.src = videoUrl;

      // Handle video metadata loaded
      const handleMetadataLoaded = () => {
        setVideoDuration(videoRef.current.duration);
        console.log('Video duration:', videoRef.current.duration);
      };

      videoRef.current.addEventListener('loadedmetadata', handleMetadataLoaded);

      // Check if already loaded
      if (videoRef.current.readyState >= 2) {
        setVideoDuration(videoRef.current.duration);
      }

      return () => {
        if (videoRef.current) {
          videoRef.current.removeEventListener('loadedmetadata', handleMetadataLoaded);
        }
      };
    }
  }, [videoUrl]);

  // Listen for main video time updates
  useEffect(() => {
    if (!videoRef || !videoRef.current) return;

    const updateTime = () => {
      setCurrentTime(videoRef.current.currentTime);
    };

    videoRef.current.addEventListener('timeupdate', updateTime);
    return () => {
      if (videoRef && videoRef.current) {
        videoRef.current.removeEventListener('timeupdate', updateTime);
      }
    };
  }, [videoRef]);

  // Handle action selection change
  const handleActionChange = useCallback((index, actionIndex) => {
    const newTimestampBlocks = [...timestampBlocks];
    newTimestampBlocks[index] = {
      ...newTimestampBlocks[index],
      actionIndex: parseInt(actionIndex)
    };
    setTimestampBlocks(newTimestampBlocks);
  }, [timestampBlocks]);

  // Compute position when appending a new block at the end
  const computeAppendPosition = () => {
    const lastBlock = timestampBlocks.length > 0 ? timestampBlocks[timestampBlocks.length - 1] : null;
    const newStartTime = lastBlock ? lastBlock.endTime : 0;
    const newEndTime = Math.min(newStartTime + (videoDuration / 10), videoDuration - 0.1);
    return { insertIndex: timestampBlocks.length, newStartTime, newEndTime };
  };

  // Compute position when inserting a new block after a specific index
  const computeInsertPosition = (afterIndex) => {
    const insertIndex = afterIndex + 1;
    const currentBlock = timestampBlocks[afterIndex];
    const nextBlock = insertIndex < timestampBlocks.length ? timestampBlocks[insertIndex] : null;
    let newStartTime, newEndTime;
    if (twoOperatorMode) {
      // In two-operator mode, allow flexible positioning
      newStartTime = currentBlock.endTime;
      newEndTime = nextBlock
        ? Math.min(currentBlock.endTime + 2, nextBlock.startTime - 0.1)
        : Math.min(currentBlock.endTime + 2, videoDuration - 0.1);
    } else {
      // In single-operator mode, sequential positioning
      newStartTime = currentBlock.endTime;
      const nextEndTime = nextBlock ? nextBlock.endTime : videoDuration;
      newEndTime = Math.min(
        (currentBlock.endTime + nextEndTime) / 2,
        videoDuration - 0.1
      );
    }
    return { insertIndex, newStartTime, newEndTime };
  };

  // Add timestamp block at specific position
  const addTimestampBlock = (afterIndex = -1) => {
    const { insertIndex, newStartTime, newEndTime } = afterIndex === -1
      ? computeAppendPosition()
      : computeInsertPosition(afterIndex);

    const newBlock = {
      startTime: newStartTime,
      endTime: Math.max(newEndTime, newStartTime + 0.5), // Ensure minimum duration
      actionIndex: 0 // Default to first action
    };

    // Insert the new block at the specified position
    const newTimestampBlocks = [...timestampBlocks];
    newTimestampBlocks.splice(insertIndex, 0, newBlock);
    setTimestampBlocks(newTimestampBlocks);
  };

  // Remove timestamp block
  const removeTimestampBlock = (index) => {
    if (timestampBlocks.length > 1) {
      const newTimestampBlocks = timestampBlocks.filter((_, i) => i !== index);
      setTimestampBlocks(newTimestampBlocks);
    }
  };

  // Automatically track the nearest timestamp when main video time changes
  useEffect(() => {
    if (!videoRef || !videoRef.current || timestampBlocks.length === 0) return;

    // When user watches the main video, automatically track time updates for the nearest timestamp
    const mainTime = videoRef.current.currentTime;

    // Find the timestamp block closest to current time (using endTime for comparison)
    let closestIndex = 0;
    let minDiff = Math.abs(mainTime - timestampBlocks[0].endTime);

    for (let i = 1; i < timestampBlocks.length; i++) {
      const diff = Math.abs(mainTime - timestampBlocks[i].endTime);
      if (diff < minDiff) {
        minDiff = diff;
        closestIndex = i;
      }
    }

    // If main video time is very close to a timestamp (within 0.5s) and user is playing video
    if (minDiff < 0.5 && !videoRef.current.paused) {
      // Highlight the approaching timestamp
      // Style changes or other visual cues can be added here
    }
  }, [currentTime, timestampBlocks, videoRef]);

  // Submit timestamps to the backend
  const submitTimestamps = async (videoId, timestampData, isTwoOperatorMode = false) => {
    setMessage('Submitting timestamp data...');
    console.log(`Submitting timestamp data to backend, video ID: ${videoId}`);
    console.log(`Two-operator mode: ${isTwoOperatorMode}`);
    console.log('Detailed timestamp data being sent:', JSON.stringify(timestampData, null, 2));

    // Log each timestamp entry for debugging
    timestampData.forEach((entry, index) => {
      console.log(`Entry ${index}:`, {
        start: entry.start,
        end: entry.end,
        actionIndex: entry.actionIndex,
        actionDescription: entry.actionDescription
      });
    });

    try {
      // Include two-operator mode flag in request body; merge threshold is read
      // server-side from augment_config.yaml.
      const requestBody = {
        timestamps: timestampData,
        twoOperatorMode: isTwoOperatorMode,
      };
      console.log('Request body:', JSON.stringify(requestBody, null, 2));

      const response = await fetch(`${API_BASE_URL}/api/v1/videos/${videoId}/split`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });

      let result;
      const contentType = response.headers.get('content-type');

      if (contentType && contentType.includes('application/json')) {
        result = await response.json();
      } else {
        // Handle non-JSON response
        const text = await response.text();
        console.error('Non-JSON response received:', text);
        throw new Error(`Server returned non-JSON response: ${text.substring(0, 200)}`);
      }

      if (!response.ok) {
        const errorMessage = result.detail || result.message || `HTTP error! status: ${response.status}`;
        console.error('Backend error response:', result);
        throw new Error(errorMessage);
      }

      console.log('Timestamp submission successful, response:', result);
      setMessage(`Timestamps submitted and ${result.clips ? result.clips.length : 0} clips created successfully!`);
      if (onTimestampsSubmitted) {
        onTimestampsSubmitted(true, result.clips ? result.clips.length : 0, null, result.clips || []); // Pass clips data
      }
    } catch (error) {
      console.error('Video split failed:', error);
      const errorMessage = error.message || 'Unknown error occurred';
      setError(`Video split failed: ${errorMessage}`);
      setMessage('');
      if (onTimestampsSubmitted) {
        onTimestampsSubmitted(false, 0, errorMessage); // Notify parent: failure, 0 clips, error message
      }
    }
  };

  const handleSubmit = () => {
    // Check if we have an uploaded video ID
    if (!uploadedVideoId) {
      setError('Please upload a video and get its ID first');
      return;
    }

    console.log(`Using uploaded video ID: ${uploadedVideoId}`);
    console.log(`Two-operator mode: ${twoOperatorMode}`);

    // Validate video and timestamps
    if (!videoRef || !videoRef.current) {
      setError('No video loaded');
      return;
    }

    if (timestampBlocks.length === 0) {
      setError('Please add timestamp blocks first');
      return;
    }

    // Build timestamp data - now using explicit start/end times
    const timestampData = [];

    for (let i = 0; i < timestampBlocks.length; i++) {
      const block = timestampBlocks[i];
      const start = Number.parseFloat(block.startTime);
      const end = Number.parseFloat(block.endTime);

      // Validate duration
      if (end - start < 0.1) {
        setError(`Event ${i + 1} has invalid duration (less than 0.1 seconds). Start: ${start.toFixed(2)}s, End: ${end.toFixed(2)}s`);
        return;
      }

      // Validate bounds
      if (start < 0 || end > videoDuration) {
        setError(`Event ${i + 1} has timestamps outside video bounds (0 - ${videoDuration.toFixed(2)}s)`);
        return;
      }

        timestampData.push({
          start: start,
          end: end,
        actionIndex: block.actionIndex,
        actionDescription: actions[block.actionIndex]
        });
    }

    if (timestampData.length === 0) {
      setError('No valid segments to create. Please check your timestamps.');
      return;
    }

    console.log('Timestamp data to be sent:', timestampData);
    console.log('Two-operator mode:', twoOperatorMode);

    // Submit timestamps to backend with two-operator mode flag
    submitTimestamps(uploadedVideoId, timestampData, twoOperatorMode);
  };

  return (
    <div className="action-timestamp-editor">
      {/* FPS control */}
      <Form.Group className="mb-3">
        <Form.Label>Video Frame Rate (fps)</Form.Label>
        <Form.Control
          type="number"
          min="1"
          max="120"
          value={fps}
          onChange={(e) => setFps(parseInt(e.target.value))}
          style={{ width: '100px' }}
        />
        <Form.Text className="text-muted">
          Specify the video frame rate for precise timestamp control
        </Form.Text>
      </Form.Group>

      {/* Two-Operator Mode indicator (controlled from Step 1) */}
      {twoOperatorMode && (
        <Alert variant="info" className="mb-3 py-2">
          <small><strong>Two-Operator Mode</strong> is active for this dataset. Overlapping timestamps for concurrent actions are allowed.</small>
        </Alert>
      )}

      {/* Continuous Coverage Toggle - only visible in two-operator mode */}
      {twoOperatorMode && (
        <Form.Group className="mb-3 ms-4" style={{ borderLeft: '3px solid #28a745', paddingLeft: '12px' }}>
          <Form.Check
            type="switch"
            id="continuous-coverage"
            label="Continuous Timeline Coverage (Auto-fix Gaps)"
            checked={continuousCoverage}
            onChange={(e) => setContinuousCoverage(e.target.checked)}
          />
          <Form.Text className="text-muted">
            {continuousCoverage
              ? "Enabled: Gaps are automatically closed by snapping to previous action's end time. Overlapping (concurrent) actions are still allowed."
              : "Disabled: Free positioning allowed - gaps may occur between non-overlapping actions."}
          </Form.Text>
        </Form.Group>
      )}

      {/* Error notifications */}
      {error && (
        <Alert variant="danger" dismissible onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      {/* Message notifications */}
      {message && (
        <Alert variant="success" dismissible onClose={() => setMessage('')}>
          {message}
        </Alert>
      )}

      {/* Video Preview */}
      <div className="video-preview mb-4">
        <h4>Video Preview</h4>
        <video
          ref={videoRef}
          controls
          className="preview-video"
          style={{ maxWidth: '100%', maxHeight: '400px' }}
        />
        {videoDuration > 0 && (
          <div className="mt-2">
            <small className="text-muted">
              Duration: {Math.floor(videoDuration / 60)}:{Math.floor(videoDuration % 60).toString().padStart(2, '0')}
            </small>
          </div>
        )}
      </div>

      {/* Current playback time display */}
      {videoRef && videoRef.current && (
        <div className="mb-3">
          <h5>Current Video Time: {currentTime.toFixed(2)}s</h5>
          <ProgressBar
            now={(currentTime / (videoDuration || 1)) * 100}
            variant="info"
            className="mb-3"
          />
        </div>
      )}

      {/* Timestamp editing section */}
      {actions.length > 0 && (
        <div className="timestamp-editor">
          <div className="mb-3">
            <h4>Set Action Timestamps (Frame Rate: {fps}fps)</h4>
          </div>

          {timestampBlocks.map((block, index) => (
            <Card key={index} className="mb-4">
              <Card.Header className="d-flex justify-content-between align-items-center">
                <h5 className="mb-0">{`Event ${index + 1}`}</h5>
                {timestampBlocks.length > 1 && (
                  <Button
                    variant="outline-danger"
                    size="sm"
                    onClick={() => removeTimestampBlock(index)}
                  >
                    Remove
                  </Button>
                )}
              </Card.Header>
              <Card.Body>
                <Row>
                  <Col lg={6}>
                    {/* Video preview section */}
                    {twoOperatorMode ? (
                      /* TWO-OPERATOR MODE: Dual video previews stacked vertically */
                      <div className="mb-3">
                        {/* START TIME Preview */}
                        <div className="action-video-preview mb-3" style={{ borderLeft: '4px solid #28a745', paddingLeft: '12px', background: 'linear-gradient(135deg, rgba(40,167,69,0.05) 0%, rgba(40,167,69,0.02) 100%)', borderRadius: '8px', padding: '12px' }}>
                          <div className="d-flex align-items-center mb-2">
                            <span className="badge bg-success me-2">START</span>
                            <small className="text-success"><strong>{(block.startTime || 0).toFixed(2)}s</strong></small>
                          </div>
                          <div className="preview-container">
                            <video
                              ref={startVideoElements[index]}
                              className="preview-video"
                              width="280"
                              height="158"
                              src={videoUrl}
                              preload="metadata"
                              muted
                              playsInline
                              style={{
                                maxWidth: '100%',
                                height: 'auto',
                                borderRadius: '8px',
                                boxShadow: '0 3px 10px rgba(40,167,69,0.25)',
                                border: '2px solid #28a745'
                              }}
                              onLoadedMetadata={() => {
                                metadataReadyRef.current[`start_${index}`] = true;
                                if (timestampBlocks[index]) {
                                  updateStartPreview(index, timestampBlocks[index].startTime);
                                }
                              }}
                            />
                          </div>
                        </div>

                        {/* END TIME Preview */}
                        <div className="action-video-preview" style={{ borderLeft: '4px solid #dc3545', paddingLeft: '12px', background: 'linear-gradient(135deg, rgba(220,53,69,0.05) 0%, rgba(220,53,69,0.02) 100%)', borderRadius: '8px', padding: '12px' }}>
                          <div className="d-flex align-items-center mb-2">
                            <span className="badge bg-danger me-2">END</span>
                            <small className="text-danger"><strong>{(block.endTime || 0).toFixed(2)}s</strong></small>
                          </div>
                          <div className="preview-container">
                            <video
                              ref={endVideoElements[index]}
                              className="preview-video"
                              width="280"
                              height="158"
                              src={videoUrl}
                              preload="metadata"
                              muted
                              playsInline
                              style={{
                                maxWidth: '100%',
                                height: 'auto',
                                borderRadius: '8px',
                                boxShadow: '0 3px 10px rgba(220,53,69,0.25)',
                                border: '2px solid #dc3545'
                              }}
                              onLoadedMetadata={() => {
                                metadataReadyRef.current[`end_${index}`] = true;
                                if (timestampBlocks[index]) {
                                  updateEndPreview(index, timestampBlocks[index].endTime);
                                }
                              }}
                            />
                          </div>
                        </div>
                      </div>
                    ) : (
                      /* SINGLE-OPERATOR MODE: Single video preview (end time only) */
                    <div className="action-video-preview mb-3">
                      <div className="preview-container">
                        <video
                            ref={endVideoElements[index]}
                          className="preview-video"
                          width="320"
                          height="180"
                          src={videoUrl}
                          preload="metadata"
                          muted
                          playsInline
                          style={{
                            maxWidth: '100%',
                            height: 'auto',
                            borderRadius: '8px',
                            boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                          }}
                          onLoadedMetadata={() => {
                              metadataReadyRef.current[`end_${index}`] = true;
                            if (timestampBlocks[index]) {
                                updateEndPreview(index, timestampBlocks[index].endTime);
                            }
                          }}
                        />
                        <div className="preview-time-indicator">
                            {(block.endTime || 0).toFixed(2)}s
                        </div>
                      </div>
                    </div>
                    )}
                  </Col>
                  <Col lg={6}>
                    <h5 className="mb-3">Action Selection</h5>
                    <Form.Group className="mb-3">
                      <Form.Label>Choose Action</Form.Label>
                      <Form.Select
                        value={block.actionIndex}
                        onChange={(e) => handleActionChange(index, e.target.value)}
                      >
                        {actions.map((action, actionIndex) => (
                          <option key={actionIndex} value={actionIndex}>
                            Action {actionIndex + 1}: {action.length > 50 ? action.substring(0, 50) + '...' : action}
                          </option>
                        ))}
                      </Form.Select>
                    </Form.Group>

                    <h5 className="mb-3">Description</h5>
                    <p className="action-description">{actions[block.actionIndex]}</p>

                    <h5 className="mt-4 mb-3">Set Time Range</h5>

                    {twoOperatorMode ? (
                      /* TWO-OPERATOR MODE: Range Slider with two handles */
                    <Form.Group>
                        <Form.Label>Start & End Time (drag both handles)</Form.Label>
                        <div style={{ padding: '10px 20px', marginTop: '15px' }}>
                          <Slider
                            range
                            min={0}
                            max={videoDuration || 100}
                            step={frameTimeStep}
                            value={[block.startTime || 0, block.endTime || 0]}
                            onChange={(values) => handleRangeChange(index, values)}
                            onChangeComplete={(values) => updatePreview(index, values[1])}
                            styles={{
                              track: { backgroundColor: '#007bff', height: 8 },
                              rail: { backgroundColor: '#e9ecef', height: 8 },
                              handle: {
                                borderColor: '#007bff',
                                height: 20,
                                width: 20,
                                marginTop: -6,
                                boxShadow: '0 2px 6px rgba(0,0,0,0.3)'
                              }
                            }}
                          />
                        </div>
                        <div className="d-flex justify-content-between mt-2">
                          <small className="text-success">
                            <strong>Start:</strong> {(block.startTime || 0).toFixed(2)}s
                          </small>
                          <small className="text-danger">
                            <strong>End:</strong> {(block.endTime || 0).toFixed(2)}s
                          </small>
                        </div>
                        <div className="d-flex justify-content-between mt-1">
                          <small className="text-muted">0s</small>
                          <small className="text-muted">Duration: {((block.endTime || 0) - (block.startTime || 0)).toFixed(2)}s</small>
                          <small className="text-muted">{videoDuration ? `${videoDuration.toFixed(1)}s` : '100s'}</small>
                        </div>

                        {/* Manual input fields for precise control */}
                        <Row className="mt-3">
                          <Col xs={6}>
                            <Form.Group>
                              <Form.Label className="small text-success">Start Time (s)</Form.Label>
                              <Form.Control
                                type="number"
                                min="0"
                                max={(block.endTime || 0) - 0.1}
                                step={frameTimeStep}
                                value={block.startTime || 0}
                                onChange={(e) => handleRangeChange(index, [Number.parseFloat(e.target.value), block.endTime])}
                                size="sm"
                              />
                            </Form.Group>
                          </Col>
                          <Col xs={6}>
                            <Form.Group>
                              <Form.Label className="small text-danger">End Time (s)</Form.Label>
                              <Form.Control
                                type="number"
                                min={(block.startTime || 0) + 0.1}
                                max={videoDuration || 100}
                                step={frameTimeStep}
                                value={block.endTime || 0}
                                onChange={(e) => handleRangeChange(index, [block.startTime, Number.parseFloat(e.target.value)])}
                                size="sm"
                              />
                            </Form.Group>
                          </Col>
                        </Row>
                      </Form.Group>
                    ) : (
                      /* SINGLE-OPERATOR MODE: Original single slider (end time only) */
                      <Form.Group>
                        <Form.Label>Completion Time (seconds)</Form.Label>
                      <Form.Control
                        type="range"
                        min="0"
                        max={videoDuration || 100}
                          step={frameTimeStep}
                          value={block.endTime || 0}
                        onChange={(e) => handleTimestampChange(index, e.target.value)}
                          onMouseUp={() => updatePreview(index, block.endTime || 0)}
                          onTouchEnd={() => updatePreview(index, block.endTime || 0)}
                        className="timestamp-slider"
                      />
                      <div className="d-flex justify-content-between">
                        <small>0s</small>
                        <small>{videoDuration ? `${videoDuration.toFixed(1)}s` : '100s'}</small>
                      </div>
                    <Form.Group className="mt-3">
                      <Form.Control
                        type="number"
                        min="0"
                        max={videoDuration || 100}
                        step={frameTimeStep}
                            value={block.endTime || 0}
                        onChange={(e) => handleTimestampChange(index, e.target.value)}
                        className="timestamp-input"
                      />
                      <Form.Text className="text-muted">seconds</Form.Text>
                    </Form.Group>
                        <Form.Text className="text-muted d-block mt-2">
                          Start: {index === 0 ? '0.00' : (timestampBlocks[index-1]?.endTime || 0).toFixed(2)}s → End: {(block.endTime || 0).toFixed(2)}s
                        </Form.Text>
                      </Form.Group>
                    )}
                  </Col>
                </Row>

                {/* Add Event button for this timestamp block */}
                <div className="mt-3 text-center">
                  <Button
                    variant="success"
                    onClick={() => addTimestampBlock(index)}
                    className="me-2"
                  >
                    Add Event After This
                  </Button>
                  <Form.Text className="text-muted d-block mt-2">
                    Click to add a new event after Event {index + 1}
                  </Form.Text>
                </div>
              </Card.Body>
            </Card>
          ))}

          <div className="d-grid gap-2 mt-4">
            <Button variant="primary" onClick={handleSubmit}>
              Submit Timestamp Data
            </Button>
          </div>
        </div>
      )}

      {/* Updated CSS for smooth native video preview system */}
      <style jsx="true">{`
        .preview-container {
          position: relative;
          display: inline-block;
          border-radius: 12px;
          overflow: hidden;
          transition: all 0.3s ease;
        }

        .preview-container:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        }

        .preview-video {
          display: block;
          max-width: 100%;
          height: auto;
          background-color: #000;
          border-radius: 8px;
          transition: all 0.3s ease;
        }

        .preview-video:hover {
          transform: scale(1.02);
        }

        .preview-time-indicator {
          position: absolute;
          bottom: 8px;
          right: 8px;
          background: rgba(0, 0, 0, 0.8);
          color: white;
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 12px;
          font-weight: bold;
          font-family: monospace;
          pointer-events: none;
        }

        .action-video-preview {
          display: flex;
          justify-content: center;
          background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 15px;
          border: 1px solid #dee2e6;
        }

        .action-description {
          padding: 15px;
          background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
          border-left: 4px solid #007bff;
          border-radius: 8px;
          font-size: 16px;
          line-height: 1.5;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .timestamp-slider {
          height: 12px;
          padding: 0;
          margin-top: 15px;
          background: linear-gradient(90deg, #007bff, #28a745);
          border-radius: 6px;
          transition: all 0.3s ease;
        }

        .timestamp-slider:hover {
          transform: scaleY(1.2);
        }

        .timestamp-slider::-webkit-slider-thumb {
          height: 24px;
          width: 24px;
          background: #007bff;
          border-radius: 50%;
          cursor: pointer;
          box-shadow: 0 2px 8px rgba(0,0,0,0.3);
          transition: all 0.2s ease;
        }

        .timestamp-slider::-webkit-slider-thumb:hover {
          transform: scale(1.2);
          box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }

        .timestamp-input {
          width: 120px;
          margin: 0 auto;
          text-align: center;
          font-weight: bold;
          font-size: 16px;
          border: 2px solid #007bff;
          border-radius: 8px;
          padding: 8px;
          transition: all 0.3s ease;
        }

        .timestamp-input:focus {
          border-color: #0056b3;
          box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
          transform: scale(1.05);
        }

        /* Enhanced card styling */
        .card {
          box-shadow: 0 4px 12px rgba(0,0,0,0.1);
          transition: all 0.3s ease;
          border: none;
          border-radius: 12px;
          overflow: hidden;
        }

        .card:hover {
          transform: translateY(-4px);
          box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        }

        .card-header {
          background: linear-gradient(135deg, #f0f5ff 0%, #e6f3ff 100%);
          border-bottom: 1px solid #d1e3ff;
          padding: 16px 20px;
        }

        .card-body {
          padding: 24px;
        }

        /* Button enhancements */
        .btn {
          transition: all 0.3s ease;
          border-radius: 8px;
          font-weight: 500;
        }

        .btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }

        /* Form enhancements */
        .form-control, .form-select {
          border-radius: 8px;
          border: 2px solid #e9ecef;
          transition: all 0.3s ease;
        }

        .form-control:focus, .form-select:focus {
          border-color: #007bff;
          box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
        }

        .form-label {
          font-weight: 600;
          color: #495057;
          margin-bottom: 8px;
        }

        /* Alert styling */
        .alert-info {
          border-left: 4px solid #007bff;
        }

        .alert-success {
          border-left: 4px solid #28a745;
        }

        /* Range slider styles for two-operator mode */
        .rc-slider {
          margin: 10px 0;
        }

        .rc-slider-track {
          background-color: #007bff !important;
          height: 8px !important;
        }

        .rc-slider-rail {
          background-color: #e9ecef !important;
          height: 8px !important;
        }

        .rc-slider-handle {
          border: 2px solid #007bff !important;
          height: 20px !important;
          width: 20px !important;
          margin-top: -6px !important;
          box-shadow: 0 2px 6px rgba(0,0,0,0.2);
          opacity: 1 !important;
        }

        .rc-slider-handle:nth-child(3) {
          border-color: #28a745 !important;
          background-color: #28a745 !important;
        }

        .rc-slider-handle:nth-child(4) {
          border-color: #dc3545 !important;
          background-color: #dc3545 !important;
        }

        .rc-slider-handle:hover {
          border-color: #0056b3 !important;
        }

        .rc-slider-handle-dragging {
          box-shadow: 0 0 0 5px rgba(0, 123, 255, 0.25) !important;
        }
      `}</style>
    </div>
  );
};

ActionTimestampEditor.propTypes = {
  actions: PropTypes.arrayOf(PropTypes.string),
  uploadedVideoId: PropTypes.string,
  videoUrl: PropTypes.string,
  initialTimestamps: PropTypes.array,
  onTimestampsSubmitted: PropTypes.func,
  twoOperatorMode: PropTypes.bool,
};

export default ActionTimestampEditor;
