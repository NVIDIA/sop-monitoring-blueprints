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
import { Card, Button, Row, Col, ListGroup, Accordion, Badge, Form, InputGroup, Tabs, Tab } from 'react-bootstrap';

// Use nginx proxy path to avoid CORS issues
const API_BASE_URL = '/api/annotation';

const AllClipsViewer = ({ allVideoResults, onClearAllResults, onRefreshResults, onReAnnotateVideo, onLoadDataset }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('newest'); // newest, oldest, filename, duration
  const [isClearing, setIsClearing] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [activeDataId, setActiveDataId] = useState('');

  // Get all data_ids from results
  const dataIds = Object.keys(allVideoResults || {});

  // Set default active tab to the most recent data_id whenever dataIds changes
  // Also ensure activeDataId is valid
  useEffect(() => {
    if (dataIds.length > 0) {
      // If no active ID, or current active ID is not in the list anymore, select the first one
      if (!activeDataId || !dataIds.includes(activeDataId)) {
      setActiveDataId(dataIds[0]);
      }
    } else {
        setActiveDataId('');
    }
  }, [dataIds, activeDataId]);

  if (!allVideoResults || Object.keys(allVideoResults).length === 0) {
    return (
      <Card className="mt-4">
        <Card.Body className="text-center py-5">
          <div className="text-muted">
            <i className="bi bi-film" style={{ fontSize: '3rem' }}></i>
            <h5 className="mt-3">No Split Results Yet</h5>
            <p>Process some videos with timestamps to see split results here.</p>
          </div>
        </Card.Body>
      </Card>
    );
  }

  // Get results for active data_id
  const currentDataInfo = allVideoResults[activeDataId] || {};
  // Handle both old format (direct object) and new format ({videos: {}}) for backward compatibility if needed,
  // though we fully switched in App.js
  const currentDataResults = (currentDataInfo.videos && typeof currentDataInfo.videos === 'object')
    ? currentDataInfo.videos
    : {};

  // Filter and sort results for current dataset
  const filteredResults = Object.entries(currentDataResults).filter(([videoId, result]) => {
    if (!result) return false; // Safety check
    if (!searchTerm) return true;
    const searchLower = searchTerm.toLowerCase();
    return (
      (result.originalFilename && result.originalFilename.toLowerCase().includes(searchLower)) ||
      (result.clips && result.clips.some(clip => clip.filename && clip.filename.toLowerCase().includes(searchLower)))
    );
  });

  const sortedResults = filteredResults.sort(([, a], [, b]) => {
    switch (sortBy) {
      case 'oldest':
        return new Date(a.processedAt) - new Date(b.processedAt);
      case 'filename':
        return a.originalFilename.localeCompare(b.originalFilename);
      case 'duration':
        return b.totalDuration - a.totalDuration;
      case 'newest':
      default:
        return new Date(b.processedAt) - new Date(a.processedAt);
    }
  });

  // Calculate total statistics for current dataset
  const totalVideos = Object.keys(currentDataResults).length;
  const totalClips = Object.values(currentDataResults).reduce((sum, result) => sum + result.clips.length, 0);
  const totalDuration = Object.values(currentDataResults).reduce((sum, result) => sum + result.totalDuration, 0);

  // Calculate overall statistics across all datasets
  const overallStats = dataIds.reduce((stats, dataId) => {
    const dataInfo = allVideoResults[dataId] || {};
    const dataResults = dataInfo.videos || {};
    const videos = Object.keys(dataResults).length;
    const clips = Object.values(dataResults).reduce((sum, result) => sum + result.clips.length, 0);
    return {
      totalVideos: stats.totalVideos + videos,
      totalClips: stats.totalClips + clips
    };
  }, { totalVideos: 0, totalClips: 0 });

  // Download all clips for a video as ZIP
  const downloadVideoClips = (videoId, filename) => {
    const downloadUrl = `${API_BASE_URL}/api/v1/videos/${videoId}/download-all`;
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `${filename}_clips.zip`;
    link.target = '_blank';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

    // Download individual clip
  const downloadClip = async (clipId, filename) => {
    try {
      // First check if the clip exists using chunk endpoint
      const checkUrl = `${API_BASE_URL}/api/v1/chunks/${clipId}`;
      const response = await fetch(checkUrl);

      if (!response.ok) {
        console.error(`Clip ${clipId} not found on server`);
        alert(`Error: The clip "${filename}" no longer exists on the server. It may have been deleted.`);
        return;
      }

      // If exists, proceed with download using chunk download endpoint
      const downloadUrl = `${API_BASE_URL}/api/v1/chunks/${clipId}/download`;
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename;
      link.target = '_blank';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Download error:', error);
      alert('Failed to download clip. Please try again.');
    }
  };

  // Clear specific dataset version
  const clearDatasetFiles = async (dataId) => {
    const dataInfo = allVideoResults[dataId] || {};
    const dataResults = dataInfo.videos || {};
    const videosCount = Object.keys(dataResults).length;
    const clipsCount = Object.values(dataResults).reduce((sum, result) => sum + result.clips.length, 0);

    const confirmMessage = `⚠️ DATASET CLEANUP WARNING ⚠️

This will permanently delete dataset "${dataId}":
• ${videosCount} video files from server storage
• ${clipsCount} generated clip files
• All video subdirectories for this dataset
• All database records for this dataset
• Local browser data for this dataset

This action CANNOT be undone!

Are you sure you want to proceed?`;

    if (!window.confirm(confirmMessage)) {
      return;
    }

    setIsClearing(true);

    try {
      // Call backend API to clear specific dataset
      const response = await fetch(`${API_BASE_URL}/api/v1/videos/clear-dataset/${dataId}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || `Server error: ${response.status}`);
      }

      console.log('Dataset cleanup result:', result);

      // Clear frontend data for this specific dataset
      if (onClearAllResults) {
        onClearAllResults(dataId);
      }

      // Switch to another tab if current one was deleted
      const remainingDataIds = dataIds.filter(id => id !== dataId);
      if (remainingDataIds.length > 0) {
        setActiveDataId(remainingDataIds[0]);
      }

      alert(`✅ Dataset Cleanup Completed!

Dataset "${dataId}" has been completely removed:
• ${result.deleted_count || 0} video records removed
• ${result.files_deleted || 0} files deleted

Local data cleared from browser.`);

    } catch (error) {
      console.error('Failed to clear dataset:', error);

      // Even if server cleanup fails, still clear local data
      if (onClearAllResults) {
        onClearAllResults(dataId);
      }

      alert(`⚠️ Dataset Cleanup Warning

Local browser data has been cleared for dataset "${dataId}", but there was an issue with server cleanup:

Error: ${error.message}

You may need to manually check the server storage or contact an administrator.`);
    } finally {
      setIsClearing(false);
    }
  };

  // Enhanced clear all function that clears both frontend and backend
  const clearAllResultsAndFiles = async () => {
    const confirmMessage = `⚠️ COMPLETE CLEANUP WARNING ⚠️

This will permanently delete ALL datasets:
• ${overallStats.totalVideos} video files from server storage
• ${overallStats.totalClips} generated clip files
• All video subdirectories and their contents
• All database records
• All local browser data

This action CANNOT be undone!

Are you absolutely sure you want to proceed?`;

    if (!window.confirm(confirmMessage)) {
      return;
    }

    // Second confirmation for safety
    if (!window.confirm('Last chance! This will delete ALL video files and data. Continue?')) {
      return;
    }

    setIsClearing(true);

    try {
      // Call backend API to clear all videos and files
      const response = await fetch(`${API_BASE_URL}/api/v1/videos/clear-all-datasets`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || `Server error: ${response.status}`);
      }

      console.log('Server cleanup result:', result);

      // Clear frontend data
      if (onClearAllResults) {
        onClearAllResults();
      }

      // Show success message with details
      alert(`✅ Complete Cleanup Completed Successfully!

Server cleanup:
• ${result.deleted_count || 0} video records removed
• ${result.files_deleted || 0} files deleted

All local data cleared from browser.

The system is now completely clean.`);

    } catch (error) {
      console.error('Failed to clear all videos and files:', error);

      // Even if server cleanup fails, still clear local data
      if (onClearAllResults) {
        onClearAllResults();
      }

      alert(`⚠️ Cleanup Warning

Local browser data has been cleared, but there was an issue with server cleanup:

Error: ${error.message}

You may need to manually check the server storage or contact an administrator.`);
    } finally {
      setIsClearing(false);
    }
  };

  return (
    <Card className="mt-4">
      <Card.Header>
        <Row className="align-items-center">
          <Col>
            <h4 className="mb-0">
              <i className="bi bi-collection-play me-2"></i>
              All Split Results
            </h4>
            <small className="text-muted">
              {dataIds.length} dataset{dataIds.length !== 1 ? 's' : ''} • {overallStats.totalVideos} videos • {overallStats.totalClips} clips total
            </small>
          </Col>
          <Col xs="auto" className="d-flex align-items-center gap-2">
            {activeDataId && onLoadDataset && (
                 <Button
                    variant="outline-primary"
                    size="sm"
                    onClick={() => onLoadDataset(activeDataId)}
                    title="Load this dataset to add more videos or modify actions"
                    className="d-flex align-items-center"
                  >
                    <i className="bi bi-pencil-square me-2"></i>
                    Re-annotate Dataset
                  </Button>
            )}
            {activeDataId && (
              <Button
                variant="warning"
                size="sm"
                onClick={() => clearDatasetFiles(activeDataId)}
                disabled={isClearing}
                className="d-flex align-items-center"
              >
                {isClearing ? (
                  <>
                    <div className="spinner-border spinner-border-sm me-2" role="status">
                      <span className="visually-hidden">Loading...</span>
                    </div>
                    Clearing...
                  </>
                ) : (
                  <>
                    <i className="bi bi-trash3 me-2"></i>
                    Clear This Dataset
                  </>
                )}
              </Button>
            )}
            <Button
              variant="danger"
              size="sm"
              onClick={clearAllResultsAndFiles}
              disabled={isClearing}
              className="d-flex align-items-center"
            >
              {isClearing ? (
                <>
                  <div className="spinner-border spinner-border-sm me-2" role="status">
                    <span className="visually-hidden">Loading...</span>
                  </div>
                  Clearing...
                </>
              ) : (
                <>
                  <i className="bi bi-trash3-fill me-2"></i>
                  Clear All Datasets
                </>
              )}
            </Button>
            {onRefreshResults && (
              <Button
                variant="secondary"
                size="sm"
                onClick={async () => {
                  setIsRefreshing(true);
                  try {
                    await onRefreshResults(setIsRefreshing);
                  } finally {
                    setIsRefreshing(false);
                  }
                }}
                disabled={isRefreshing}
                className="d-flex align-items-center"
                title="Sync with server and remove stale references"
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
                    Refresh Results
                  </>
                )}
              </Button>
            )}
          </Col>
        </Row>
      </Card.Header>

      {/* Dataset Tabs */}
      {dataIds.length > 1 && (
        <Card.Body className="border-bottom py-2">
          <Tabs
            activeKey={activeDataId}
            onSelect={(k) => setActiveDataId(k)}
            className="mb-0"
            variant="pills"
          >
            {dataIds.map(dataId => {
              const dataInfo = allVideoResults[dataId] || {};
              const dataResults = dataInfo.videos || {};
              const videosCount = Object.keys(dataResults).length;
              const clipsCount = Object.values(dataResults).reduce((sum, result) => sum + result.clips.length, 0);

              return (
                <Tab
                  key={dataId}
                  eventKey={dataId}
                  title={
                    <span>
                      Dataset {dataId.substring(0, 8)}...
                      <Badge bg="info" className="ms-2">
                        {videosCount}v • {clipsCount}c
                      </Badge>
                    </span>
                  }
                />
              );
            })}
          </Tabs>
        </Card.Body>
      )}

      {/* Current Dataset Info */}
      {activeDataId && (
        <Card.Body className="border-bottom py-2">
          <Row className="align-items-center">
            <Col>
              <small className="text-muted">
                <strong>Current Dataset:</strong> {activeDataId}
              </small>
            </Col>
            <Col xs="auto">
              <Badge bg="primary" className="fs-6">
                {totalVideos} Videos • {totalClips} Clips • {Math.round(totalDuration)}s Total
              </Badge>
            </Col>
          </Row>
        </Card.Body>
      )}

      {/* Search and Filter Controls */}
      <Card.Body className="border-bottom">
        <Row className="g-3">
          <Col md={8}>
            <InputGroup>
              <InputGroup.Text>
                <i className="bi bi-search"></i>
              </InputGroup.Text>
              <Form.Control
                type="text"
                placeholder="Search videos or clips in current dataset..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </InputGroup>
          </Col>
          <Col md={4}>
            <Form.Select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
              <option value="newest">Newest First</option>
              <option value="oldest">Oldest First</option>
              <option value="filename">Filename A-Z</option>
              <option value="duration">Longest Duration</option>
            </Form.Select>
          </Col>
        </Row>
      </Card.Body>

      {/* Results Display for Current Dataset */}
      <div style={{ maxHeight: '600px', overflowY: 'auto' }}>
        {totalVideos === 0 ? (
          <Card.Body className="text-center py-5">
            <div className="text-muted">
              <i className="bi bi-film" style={{ fontSize: '2rem' }}></i>
              <h6 className="mt-3">No Results in Current Dataset</h6>
              <p>Process some videos to see results for this dataset.</p>
            </div>
          </Card.Body>
        ) : (
          <Accordion>
            {sortedResults.map(([videoId, result], index) => (
              <Accordion.Item eventKey={videoId} key={videoId}>
                <Accordion.Header>
                  <div className="w-100 d-flex justify-content-between align-items-center me-3">
                    <div>
                      <strong>{result.originalFilename}</strong>
                      <div className="text-muted small">
                        {result.clips.length} clips •
                        {Math.round(result.totalDuration)}s total •
                        Processed: {new Date(result.processedAt).toLocaleString()}
                      </div>
                    </div>
                    <div className="d-flex align-items-center">
                      {onReAnnotateVideo && (
                        <Button
                          variant="outline-primary"
                          size="sm"
                          className="me-3"
                          onClick={(e) => {
                            e.stopPropagation(); // Prevent accordion toggle
                            onReAnnotateVideo(activeDataId, videoId);
                          }}
                        >
                          <i className="bi bi-pencil me-1"></i>
                          Re-annotate
                        </Button>
                      )}
                      <Badge bg="info" className="me-2">
                        {result.clips.length} clips
                      </Badge>
                      <i className="bi bi-chevron-down"></i>
                    </div>
                  </div>
                </Accordion.Header>
                <Accordion.Body>
                  <Row className="mb-3">
                    <Col>
                      <div className="d-flex gap-2 flex-wrap">
                        <Button
                          variant="success"
                          size="sm"
                          onClick={() => downloadVideoClips(videoId, result.originalFilename)}
                        >
                          <i className="bi bi-download me-1"></i>
                          Download All Clips (ZIP)
                        </Button>
                        <Button
                          variant="outline-info"
                          size="sm"
                          href={`${API_BASE_URL}/api/v1/videos/${videoId}`}
                          target="_blank"
                        >
                          <i className="bi bi-info-circle me-1"></i>
                          Video Details
                        </Button>
                      </div>
                    </Col>
                  </Row>

                  {/* Individual Clips */}
                  <ListGroup variant="flush">
                    {result.clips.map((clip, clipIndex) => (
                      <ListGroup.Item key={clip.id} className="px-0">
                        <Row className="align-items-center">
                          <Col md={1} className="text-center">
                            <Badge bg="secondary" className="rounded-pill">
                              {clipIndex + 1}
                            </Badge>
                          </Col>
                          <Col md={6}>
                            <div>
                              <strong>{clip.filename}</strong>
                              <div className="text-muted small">
                                {parseFloat(clip.start_time).toFixed(2)}s - {parseFloat(clip.end_time).toFixed(2)}s
                                <span className="ms-2">
                                  ({parseFloat(clip.duration).toFixed(2)}s duration)
                                </span>
                              </div>
                              {clip.action_description && (
                                <div className="text-info small mt-1">
                                  <i className="bi bi-tag me-1"></i>
                                  {clip.action_description}
                                </div>
                              )}
                            </div>
                          </Col>
                          <Col md={3} className="text-center">
                            <div className="progress" style={{ height: '6px' }}>
                              <div
                                className="progress-bar bg-info"
                                style={{
                                  width: `${(parseFloat(clip.duration) / result.totalDuration) * 100}%`
                                }}
                              ></div>
                            </div>
                            <small className="text-muted">
                              {((parseFloat(clip.duration) / result.totalDuration) * 100).toFixed(1)}% of total
                            </small>
                          </Col>
                          <Col md={2} className="text-end">
                            <div className="d-flex gap-1 justify-content-end">
                              <Button
                                variant="primary"
                                size="sm"
                                onClick={() => downloadClip(clip.id, clip.filename)}
                                title="Download this clip"
                              >
                                <i className="bi bi-download"></i>
                              </Button>
                              <Button
                                variant="outline-secondary"
                                size="sm"
                                href={`${API_BASE_URL}/api/v1/chunks/${clip.id}`}
                                target="_blank"
                                title="View clip details"
                              >
                                <i className="bi bi-eye"></i>
                              </Button>
                            </div>
                          </Col>
                        </Row>
                      </ListGroup.Item>
                    ))}
                  </ListGroup>
                </Accordion.Body>
              </Accordion.Item>
            ))}
          </Accordion>
        )}
      </div>

      {/* Footer with bulk actions */}
      <Card.Footer>
        <Row className="align-items-center">
          <Col>
            <small className="text-muted">
              Current dataset: {totalVideos} videos processed, {totalClips} clips created, {Math.round(totalDuration)} seconds of content
            </small>
          </Col>
          <Col xs="auto">
            <Button
              variant="outline-primary"
              href={`${API_BASE_URL}/api/v1/videos`}
              target="_blank"
            >
              <i className="bi bi-list me-1"></i>
              View All in API
            </Button>
          </Col>
        </Row>
      </Card.Footer>

      {/* Enhanced Styling */}
      <style jsx="true">{`
        .accordion-button:not(.collapsed) {
          background-color: #f8f9fa;
          border-color: #dee2e6;
        }

        .accordion-button:focus {
          border-color: #86b7fe;
          box-shadow: 0 0 0 0.25rem rgba(13, 110, 253, 0.25);
        }

        .progress {
          border-radius: 3px;
          background-color: #e9ecef;
        }

        .list-group-item {
          border-left: none;
          border-right: none;
          border-top: 1px solid #dee2e6;
        }

        .list-group-item:first-child {
          border-top: none;
        }

        .bi {
          font-size: 1em;
        }

        .accordion-body {
          padding: 1rem 1.25rem;
        }

        .badge {
          font-size: 0.75em;
        }

        .btn-sm {
          font-size: 0.8rem;
          padding: 0.25rem 0.5rem;
        }

        .nav-pills .nav-link {
          border-radius: 0.375rem;
        }

        .nav-pills .nav-link.active {
          background-color: #0d6efd;
        }
      `}</style>
    </Card>
  );
};

export default AllClipsViewer;