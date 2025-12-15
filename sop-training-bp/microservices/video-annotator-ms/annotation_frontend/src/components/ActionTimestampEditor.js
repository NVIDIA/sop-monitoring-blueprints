// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Form, Button, Alert, Card, Row, Col, ProgressBar } from 'react-bootstrap';

// Use nginx proxy path to avoid CORS issues
const API_BASE_URL = '/api/annotation';

const ActionTimestampEditor = ({ actions = [], uploadedVideoId, videoUrl, initialTimestamps, onTimestampsSubmitted }) => {
  // State for dynamic timestamp blocks
  const [timestampBlocks, setTimestampBlocks] = useState([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [videoDuration, setVideoDuration] = useState(0);
  const [fps, setFps] = useState(30);

  // Simplified preview system with direct video elements
  const videoRef = useRef(null);
  const [videoElements, setVideoElements] = useState([]);
  const timeUpdateThrottleRef = useRef(null);
  // Track seeks and metadata readiness per mini preview
  const seekInFlightRef = useRef({});
  const metadataReadyRef = useRef({});

  // Calculate frame-based time step
  const frameTimeStep = 1 / fps;

  // Simplified preview update - just seek the video directly
  const updatePreview = useCallback(async (index, targetTime) => {
    const videoEl = videoElements[index];
    if (!videoEl?.current) return;

    const video = videoEl.current;

    try {
      // If metadata isn't ready yet, defer seeking to the onLoadedMetadata path
      const hasMeta = (video.readyState >= 1 && Number.isFinite(video.duration)) || metadataReadyRef.current[index];
      if (!hasMeta) return;

      // Clamp only if we know a finite duration
      const knownDuration = Number.isFinite(video.duration)
        ? video.duration
        : (Number.isFinite(videoDuration) ? videoDuration : undefined);
      const clampedTime = knownDuration !== undefined
        ? Math.max(0, Math.min(Number(targetTime) || 0, knownDuration))
        : Math.max(0, Number(targetTime) || 0);

      // Avoid overlapping seeks; wait for seek completion
      if (seekInFlightRef.current[index]) return;
      seekInFlightRef.current[index] = true;

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
      console.warn('Failed to update preview:', error);
    } finally {
      seekInFlightRef.current[index] = false;
    }
  }, [videoElements, videoDuration]);

  // Simplified timestamp change handler with reduced debounce
  const handleTimestampChange = useCallback((index, value) => {
    // Cancel previous timer
    if (timeUpdateThrottleRef.current) {
      clearTimeout(timeUpdateThrottleRef.current);
    }

    const newTimestampBlocks = [...timestampBlocks];
    const newTime = parseFloat(value);

    // Validate time is within video bounds
    const validTime = Math.max(0, Math.min(newTime, videoDuration || 100));

    // Ensure timestamps are in correct order
    let finalTime = validTime;
    if (index > 0 && validTime <= timestampBlocks[index - 1].timestamp) {
      finalTime = timestampBlocks[index - 1].timestamp + 0.1;
    }

    // Handle subsequent timestamps
    if (index < timestampBlocks.length - 1 && finalTime >= timestampBlocks[index + 1].timestamp) {
      const delta = finalTime - timestampBlocks[index].timestamp;
      for (let i = index + 1; i < timestampBlocks.length; i++) {
        newTimestampBlocks[i] = {
          ...newTimestampBlocks[i],
          timestamp: Math.max(
            timestampBlocks[i].timestamp + delta,
            newTimestampBlocks[i-1].timestamp + 0.1
          )
        };
      }
    }

    newTimestampBlocks[index] = {
      ...newTimestampBlocks[index],
      timestamp: finalTime
    };
    setTimestampBlocks(newTimestampBlocks);

    // Reduced debounce for smoother experience
    timeUpdateThrottleRef.current = setTimeout(async () => {
      try {
        // Update current preview
        await updatePreview(index, finalTime);

        // Update affected subsequent previews
        if (index < timestampBlocks.length - 1 && finalTime >= timestampBlocks[index + 1].timestamp) {
          for (let i = index + 1; i < timestampBlocks.length; i++) {
            await updatePreview(i, newTimestampBlocks[i].timestamp);
          }
        }
      } catch (error) {
        console.error('Failed to update preview:', error);
      }
    }, 100); // Reduced from 300ms to 100ms for smoother response
  }, [timestampBlocks, updatePreview, videoDuration]);

  // Clear timeouts when component unmounts
  const clearTimeouts = useCallback(() => {
    if (timeUpdateThrottleRef.current) {
      clearTimeout(timeUpdateThrottleRef.current);
    }
  }, []);

  // Reset state when video changes
  useEffect(() => {
    if (uploadedVideoId) {
      setTimestampBlocks([{ timestamp: 0, actionIndex: 0 }]);
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
             if (timestampBlocks.length === 1 && timestampBlocks[0].timestamp === 0) {
                 console.log("Loading initial timestamps for re-annotation:", initialTimestamps);
                 setTimestampBlocks(initialTimestamps);
                 setMessage("Loaded existing annotations. You can modify them and submit to overwrite.");
             }
        } else if (timestampBlocks.length === 1 && timestampBlocks[0].timestamp === 0) {
      // Initialize the first block with a reasonable timestamp
      const initialTimestamp = Math.min(videoDuration / 2, videoDuration - 1);
      setTimestampBlocks([{ timestamp: initialTimestamp, actionIndex: 0 }]);
    }
    }
  }, [actions, videoDuration, initialTimestamps]);

  // Initialize video element array when timestampBlocks change
  useEffect(() => {
    if (timestampBlocks.length > 0) {
      // Create video element references for each timestamp block
      const videoEls = timestampBlocks.map(() => React.createRef());
      setVideoElements(videoEls);
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

  // Add timestamp block at specific position
  const addTimestampBlock = (afterIndex = -1) => {
    let insertIndex;
    let newTimestamp;

    if (afterIndex === -1) {
      // Add at the end (fallback for existing behavior)
      insertIndex = timestampBlocks.length;
      const lastTimestamp = timestampBlocks.length > 0 ? timestampBlocks[timestampBlocks.length - 1].timestamp : 0;
      newTimestamp = Math.min(lastTimestamp + (videoDuration / 10), videoDuration - 1);
    } else {
      // Insert after the specified index
      insertIndex = afterIndex + 1;
      const currentTimestamp = timestampBlocks[afterIndex].timestamp;
      const nextTimestamp = insertIndex < timestampBlocks.length ? timestampBlocks[insertIndex].timestamp : videoDuration;

      // Set new timestamp to just after the previous event's end time (+ a few frames worth)
      // This allows user to drag forward naturally following the video timeline
      const smallOffset = Math.min(0.5, (nextTimestamp - currentTimestamp) * 0.1); // 0.5 sec or 10% of gap, whichever is smaller
      newTimestamp = Math.min(
        currentTimestamp + smallOffset,
        nextTimestamp - 0.1, // Ensure it's at least 0.1s before the next event
        videoDuration - 0.1
      );
    }

    const newBlock = {
      timestamp: newTimestamp,
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

    // Find the timestamp closest to current time
    let closestIndex = 0;
    let minDiff = Math.abs(mainTime - timestampBlocks[0].timestamp);

    for (let i = 1; i < timestampBlocks.length; i++) {
      const diff = Math.abs(mainTime - timestampBlocks[i].timestamp);
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
  const submitTimestamps = async (videoId, timestampData) => {
    setMessage('Submitting timestamp data...');
    console.log(`Submitting timestamp data to backend, video ID: ${videoId}`);
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
      const requestBody = { timestamps: timestampData };
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

    // Validate video and timestamps
    if (!videoRef || !videoRef.current) {
      setError('No video loaded');
      return;
    }

    if (timestampBlocks.length === 0) {
      setError('Please add timestamp blocks first');
      return;
    }

    // Sort timestamp blocks by timestamp value
    const sortedBlocks = [...timestampBlocks].sort((a, b) => a.timestamp - b.timestamp);

    // Build timestamp data structure needed by the backend
    // Only create segments for user-defined events, not gaps
    const timestampData = [];

    // For each timestamp block, create a segment that represents the action
    // We'll create segments of reasonable duration around each timestamp
    for (let i = 0; i < sortedBlocks.length; i++) {
      const block = sortedBlocks[i];
      const timestamp = parseFloat(block.timestamp);

      // Create a segment around the timestamp
      // If it's the first timestamp, start from beginning
      // If it's the last timestamp, end at video end
      // Otherwise, create segments between consecutive timestamps

      let start, end;

      if (i === 0) {
        // First event: from start to this timestamp
        start = 0;
        end = timestamp;
      } else {
        // Subsequent events: from previous timestamp to this timestamp
        start = parseFloat(sortedBlocks[i - 1].timestamp);
        end = timestamp;
      }

      // Only add segments that have meaningful duration (at least 0.1 seconds)
      if (end - start >= 0.1) {
        timestampData.push({
          start: start,
          end: end,
          actionIndex: block.actionIndex, // Include action index information
          actionDescription: actions[block.actionIndex] // Include action description for reference
        });
      }
    }

    if (timestampData.length === 0) {
      setError('No valid segments to create. Please check your timestamps.');
      return;
    }

    console.log('Timestamp data to be sent:', timestampData);

    // Submit timestamps to backend
    submitTimestamps(uploadedVideoId, timestampData);
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
                    {/* Smooth native video preview */}
                    <div className="action-video-preview mb-3">
                      <div className="preview-container">
                        <video
                          ref={videoElements[index]}
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
                            metadataReadyRef.current[index] = true;
                            if (timestampBlocks[index]) {
                              updatePreview(index, timestampBlocks[index].timestamp);
                            }
                          }}
                        />

                        {/* Time indicator overlay */}
                        <div className="preview-time-indicator">
                          {block.timestamp.toFixed(2)}s
                        </div>
                      </div>
                    </div>
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

                    <h5 className="mt-4 mb-3">Set Timestamp</h5>
                    <Form.Group>
                      <Form.Label>{`Completion Time (seconds)`}</Form.Label>
                      <Form.Control
                        type="range"
                        min="0"
                        max={videoDuration || 100}
                        step={frameTimeStep} // Use frame-rate based step
                        value={block.timestamp || 0}
                        onChange={(e) => handleTimestampChange(index, e.target.value)}
                        onMouseUp={() => updatePreview(index, timestampBlocks[index]?.timestamp || 0)}
                        onTouchEnd={() => updatePreview(index, timestampBlocks[index]?.timestamp || 0)}
                        className="timestamp-slider"
                      />
                      <div className="d-flex justify-content-between">
                        <small>0s</small>
                        <small>{videoDuration ? `${videoDuration.toFixed(1)}s` : '100s'}</small>
                      </div>
                    </Form.Group>

                    <Form.Group className="mt-3">
                      <Form.Control
                        type="number"
                        min="0"
                        max={videoDuration || 100}
                        step={frameTimeStep}
                        value={block.timestamp || 0}
                        onChange={(e) => handleTimestampChange(index, e.target.value)}
                        className="timestamp-input"
                      />
                      <Form.Text className="text-muted">seconds</Form.Text>
                    </Form.Group>
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
      `}</style>
    </div>
  );
};

export default ActionTimestampEditor;