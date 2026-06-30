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

import React, { useState, useRef, useEffect } from 'react';
import { Form, Button, Alert, Spinner, ProgressBar, ButtonGroup } from 'react-bootstrap';

// Use nginx proxy path to avoid CORS issues
const API_BASE_URL = '/api/annotation';

// Maximum file size in bytes (2GB to match nginx config)
const MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024; // 2GB
const MAX_FILE_SIZE_MB = MAX_FILE_SIZE / (1024 * 1024); // For display

const VideoUploader = ({
  onVideoUploaded,
  isEnabled = true,
  // showTimestampEditor = false, // This prop seems unused, consider removing if confirmed
  onFolderSelected, // New prop: (files: File[], skippedCount?: number) => void
  fileToAutoUpload, // New prop: File | null
  isBatchActive,    // New prop: boolean
  onUploadProcessStarted, // New prop: (filename: string) => void
}) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState({
    show: false,
    variant: 'info',
    message: ''
  });
  const videoRef = useRef(null);
  const [duration, setDuration] = useState(0);
  const [isVideoLoaded, setIsVideoLoaded] = useState(false);
  const fileInputRef = useRef(null); // Ref for single file input
  const folderInputRef = useRef(null); // Ref for folder input
  const [selectionMode, setSelectionMode] = useState('file'); // 'file' or 'folder'

  // Effect to handle auto-uploading of a file passed by prop
  useEffect(() => {
    if (fileToAutoUpload && isBatchActive) {
      console.log('VideoUploader: fileToAutoUpload received', fileToAutoUpload.name);

      // Check file size before processing
      if (fileToAutoUpload.size > MAX_FILE_SIZE) {
        const fileSizeMB = (fileToAutoUpload.size / (1024 * 1024)).toFixed(1);
        setUploadStatus({
          show: true,
          variant: 'danger',
          message: `File "${fileToAutoUpload.name}" (${fileSizeMB} MB) exceeds the maximum limit of ${MAX_FILE_SIZE_MB} MB.`
        });
        if (onVideoUploaded) {
          onVideoUploaded(null, `File size exceeds ${MAX_FILE_SIZE_MB} MB limit`);
        }
        return;
      }

      // Reset internal state for the new file
      setSelectedFile(fileToAutoUpload);
      if (preview) {
        URL.revokeObjectURL(preview);
      }
      const fileUrl = URL.createObjectURL(fileToAutoUpload);
      setPreview(fileUrl);
      setUploadStatus({ show: false, variant: 'info', message: '' });
      setProgress(0);
      setIsVideoLoaded(false); // Ensure video metadata is reloaded

      if (onUploadProcessStarted) {
        onUploadProcessStarted(fileToAutoUpload.name);
      }
      handleUpload(fileToAutoUpload); // Automatically trigger upload for this file
    }
  }, [fileToAutoUpload, isBatchActive]); // Rely on isBatchActive as well

  // Reset state when component is remounted or key changes (e.g. via resetStep2 in App.js)
  useEffect(() => {
    console.log('VideoUploader component mounted/re-keyed, resetting state');
    // Don't reset selectedFile if it's being set by fileToAutoUpload
    if (!fileToAutoUpload) {
      setSelectedFile(null);
      if (preview) URL.revokeObjectURL(preview);
      setPreview('');
    }
    setUploadStatus({ show: false, variant: 'info', message: '' });
    setProgress(0);
    setDuration(0);
    setIsVideoLoaded(false);
    setSelectionMode('file'); // Reset to file mode
    if (fileInputRef.current) {
      fileInputRef.current.value = ""; // Clear file input
    }
    if (folderInputRef.current) {
      folderInputRef.current.value = ""; // Clear folder input
    }

    // Cleanup function
    return () => {
      console.log('VideoUploader component unmounting');
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, []); // Empty dependency array means this runs on mount and unmount

  // Handle video metadata loaded to get duration
  const handleVideoLoaded = () => {
    if (videoRef.current) {
      console.log('Video metadata loaded, duration:', videoRef.current.duration);
      setDuration(videoRef.current.duration);
      setIsVideoLoaded(true);
    }
  };

  const handleFileChange = (e) => {
    if (isBatchActive) {
      // Prevent file changes if batch mode is active and driven by App.js
      console.log("Batch mode active, manual file selection blocked.");
      return;
    }

    const files = Array.from(e.target.files);
    if (!files || files.length === 0) return;

    console.log('Files selected:', files.map(f => f.name));

    // Filter for video files only
    const videoFiles = files.filter(file => file.type.match('video.*'));

    if (videoFiles.length === 0) {
      setUploadStatus({
        show: true,
        variant: 'warning',
        message: 'No video files found in the selection.'
      });
      setSelectedFile(null);
      if (preview) URL.revokeObjectURL(preview);
      setPreview('');
      return;
    }

    // Check if this is from folder input (multiple files) or single file input
    if (selectionMode === 'folder' && videoFiles.length > 0 && onFolderSelected) {
      console.log('Folder selected with', videoFiles.length, 'videos.');

      // Check file sizes and filter out oversized files
      const oversizedFiles = videoFiles.filter(file => file.size > MAX_FILE_SIZE);
      const validFiles = videoFiles.filter(file => file.size <= MAX_FILE_SIZE);

      if (oversizedFiles.length > 0) {
        const fileNames = oversizedFiles.map(f => `${f.name} (${(f.size / (1024 * 1024)).toFixed(1)} MB)`).join(', ');
        console.log(`Skipping ${oversizedFiles.length} oversized files: ${fileNames}`);

        if (validFiles.length === 0) {
          // All files are oversized
          setUploadStatus({
            show: true,
            variant: 'danger',
            message: `All selected files exceed the maximum size limit of ${MAX_FILE_SIZE_MB} MB. No files to process.`
          });
          return;
        } else {
          // Some files are valid - show warning and delay batch start
          const warningMessage = `Skipping ${oversizedFiles.length} file(s) that exceed ${MAX_FILE_SIZE_MB} MB limit. Processing ${validFiles.length} valid video(s).`;
          setUploadStatus({
            show: true,
            variant: 'warning',
            message: warningMessage
          });

          // Pass skipped file count to parent
          onFolderSelected(validFiles, oversizedFiles.length);

          return; // Exit early to prevent immediate state clearing below
        }
      }

      onFolderSelected(validFiles);
      // Reset VideoUploader's state as App.js will drive uploads
      setSelectedFile(null);
      if (preview) URL.revokeObjectURL(preview);
      setPreview('');

      // Only update status if we haven't already shown a warning about skipped files
      if (oversizedFiles.length === 0) {
        setUploadStatus({
          show: true,
          variant: 'info',
          message: `${validFiles.length} videos selected. The application will process them sequentially.`
        });
      }
    } else if (selectionMode === 'file' && videoFiles.length > 0) {
      // Single file selection
      console.log('Single file selected:', videoFiles[0].name);
      const file = videoFiles[0];

      // Check file size
      if (file.size > MAX_FILE_SIZE) {
        const fileSizeMB = (file.size / (1024 * 1024)).toFixed(1);
        setUploadStatus({
          show: true,
          variant: 'danger',
          message: `File size (${fileSizeMB} MB) exceeds the maximum limit of ${MAX_FILE_SIZE_MB} MB. Please select a smaller file or consider compressing the video.`
        });
        setSelectedFile(null);
        if (preview) URL.revokeObjectURL(preview);
        setPreview('');
        return;
      }

      setSelectedFile(file);
      if (preview) URL.revokeObjectURL(preview);
      const fileUrl = URL.createObjectURL(file);
      setPreview(fileUrl);
      setUploadStatus({ show: false, variant: 'info', message: '' }); // Clear any previous status
    }
  };

  const handleUpload = async (fileToUploadImmediately = null) => {
    const currentFileToUpload = fileToUploadImmediately || selectedFile;

    if (!currentFileToUpload) {
      setUploadStatus({
        show: true,
        variant: 'warning',
        message: 'Please select a video file first'
      });
      return;
    }

    console.log('Starting upload for file:', currentFileToUpload.name);
    setIsUploading(true);
    setProgress(0);
    // Keep existing success/error messages from folder selection if relevant, or clear for new upload
    // setUploadStatus({ show: false, variant: 'info', message: '' }); // Let's clear for new upload action

    const formData = new FormData();
    // For files from folders, we need to handle the filename properly
    // The file.name might contain path separators, so we extract just the filename
    const actualFileName = currentFileToUpload.name.split('/').pop() || currentFileToUpload.name;

    // Create a new File object with just the filename (without path)
    const fileToUpload = new File([currentFileToUpload], actualFileName, {
      type: currentFileToUpload.type,
      lastModified: currentFileToUpload.lastModified
    });

    formData.append('file', fileToUpload);

    try {
      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const percentComplete = Math.round((event.loaded / event.total) * 100);
          console.log(`Upload progress: ${percentComplete}%`);
          setProgress(percentComplete);
        }
      });

      xhr.onload = () => {
        console.log(`Upload completed with status: ${xhr.status}`);
        if (xhr.status === 200) {
          const response = JSON.parse(xhr.responseText);
          console.log('Upload successful, response:', response);

          setUploadStatus({
            show: true,
            variant: 'success',
            message: `Video "${actualFileName}" has been successfully uploaded. ID: ${response.file_id}`
          });

          if (onVideoUploaded) {
            onVideoUploaded({
              id: response.file_id,
              filename: actualFileName,
              url: `${API_BASE_URL}/api/v1/videos/${response.file_id}/download`
            });
          }
        } else {
          let errorMessage = `Upload failed for "${actualFileName}": ${xhr.statusText}`;
          try {
            const errorResponse = JSON.parse(xhr.responseText);
            if (errorResponse && errorResponse.detail) {
              errorMessage = `Upload failed for "${actualFileName}": ${errorResponse.detail}`;
            }
          } catch (e) { /* Ignore if response is not valid JSON */ }
          console.error('Upload failed:', errorMessage);
          setUploadStatus({ show: true, variant: 'danger', message: errorMessage });
          if (onVideoUploaded) { // Notify parent of failure too
             onVideoUploaded(null, errorMessage); // Pass error message
          }
        }
        setIsUploading(false);
      };

      xhr.onerror = (event) => {
        const errorMsg = `Upload failed for "${actualFileName}" due to network error`;
        console.error('XHR error event:', event);
        console.error(errorMsg);
        setUploadStatus({ show: true, variant: 'danger', message: errorMsg });
        setIsUploading(false);
        if (onVideoUploaded) { onVideoUploaded(null, errorMsg); }
      };

      xhr.ontimeout = () => {
        const errorMsg = `Upload timed out for "${actualFileName}", please check your network connection and try again`;
        console.error(errorMsg);
        setUploadStatus({ show: true, variant: 'danger', message: errorMsg });
        setIsUploading(false);
        if (onVideoUploaded) { onVideoUploaded(null, errorMsg); }
      };

      xhr.onreadystatechange = () => {
        console.log(`XHR state changed: ${xhr.readyState}, status: ${xhr.status}`);
      };

      console.log(`Opening connection to: ${API_BASE_URL}/api/v1/upload`);
      xhr.open('POST', `${API_BASE_URL}/api/v1/upload`, true); // Upload endpoint for video files
      xhr.timeout = 1200000; // 20 minutes timeout (increased from 10 minutes)

      console.log(`Starting upload for file: ${actualFileName}, size: ${currentFileToUpload.size} bytes`);
      xhr.send(formData);
    } catch (err) {
      const errorMsg = `Upload error for "${actualFileName}": ${err.message}`;
      console.error('Upload error:', err);
      setUploadStatus({ show: true, variant: 'danger', message: errorMsg });
      setIsUploading(false);
      if (onVideoUploaded) { onVideoUploaded(null, errorMsg); }
    }
  };

  // Optimize video preview
  const enhanceVideoPreview = () => {
    if (videoRef.current) {
      videoRef.current.style.objectFit = 'contain';
      videoRef.current.style.backgroundColor = '#000';
      videoRef.current.preload = "metadata"; // Changed to metadata for faster preview load
    }
  };

  useEffect(() => {
    if (preview && videoRef.current) {
      enhanceVideoPreview();
      // Ensure metadata is loaded for the preview
      videoRef.current.load(); // This might be needed if src changes often
    }
  }, [preview]);

  // Handle selection mode change
  const handleSelectionModeChange = (mode) => {
    if (isBatchActive) return; // Don't allow changing mode during batch

    setSelectionMode(mode);
    setSelectedFile(null);
    if (preview) {
      URL.revokeObjectURL(preview);
      setPreview('');
    }
    setUploadStatus({ show: false, variant: 'info', message: '' });

    // Clear both input values
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (folderInputRef.current) folderInputRef.current.value = "";
  };

  return (
    <div>
      {/* Selection Mode Toggle */}
      {!isBatchActive && (
        <div className="mb-3">
          <Form.Label>Selection Mode:</Form.Label>
          <ButtonGroup className="d-block">
            <Button
              variant={selectionMode === 'file' ? 'primary' : 'outline-primary'}
              onClick={() => handleSelectionModeChange('file')}
              disabled={!isEnabled || isUploading}
            >
              Single Video File
            </Button>
            <Button
              variant={selectionMode === 'folder' ? 'primary' : 'outline-primary'}
              onClick={() => handleSelectionModeChange('folder')}
              disabled={!isEnabled || isUploading}
            >
              Video Folder (Batch)
            </Button>
          </ButtonGroup>
        </div>
      )}

      <Form.Group controlId="videoUpload" className="mb-3">
        <Form.Label>
          {isBatchActive
            ? "Video processing is in batch mode"
            : selectionMode === 'file'
              ? "Select Video File"
              : "Select Video Folder"}
        </Form.Label>

        {/* Single File Input */}
        {selectionMode === 'file' && (
          <Form.Control
            type="file"
            ref={fileInputRef}
            accept="video/*"
            onChange={handleFileChange}
            disabled={!isEnabled || isUploading || isBatchActive}
          />
        )}

        {/* Folder Input */}
        {selectionMode === 'folder' && (
          <Form.Control
            type="file"
            ref={folderInputRef}
            webkitdirectory="true"
            multiple
            accept="video/*"
            onChange={handleFileChange}
            disabled={!isEnabled || isUploading || isBatchActive}
          />
        )}

        <Form.Text className="text-muted">
          {isBatchActive
            ? "Batch processing is managed by the application."
            : selectionMode === 'file'
              ? `Select a single video file to upload. Maximum file size: ${MAX_FILE_SIZE_MB} MB.`
              : `Select a folder containing multiple video files for batch processing. Maximum file size per video: ${MAX_FILE_SIZE_MB} MB.`}
        </Form.Text>
      </Form.Group>

      {preview && selectedFile && ( // Only show preview if a file is manually selected or auto-selected
        <div className="mb-3 text-center">
          <h5>Preview: {selectedFile.name.split('/').pop()}</h5>
          <video
            ref={videoRef}
            src={preview}
            controls
            style={{ maxWidth: '100%', maxHeight: '300px', backgroundColor: '#000' }}
            onLoadedMetadata={handleVideoLoaded}
            onCanPlay={() => enhanceVideoPreview()}
          />
          {isVideoLoaded && duration > 0 && (
            <div className="mt-1">
              <small className="text-muted d-block">
                Duration: {Math.floor(duration / 60)}:{(Math.floor(duration % 60)).toString().padStart(2, '0')}
              </small>
              <small className="text-muted d-block">
                File size: {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
              </small>
            </div>
          )}
        </div>
      )}

      {!isBatchActive && selectedFile && !isUploading && isEnabled && (
        <div className="d-grid mb-3">
          <Button
            variant="success"
            onClick={() => handleUpload()} // Trigger manual upload
            disabled={isUploading || !selectedFile}
          >
            Upload "{selectedFile.name.split('/').pop()}"
          </Button>
        </div>
      )}

      {(isUploading || (isBatchActive && progress > 0)) && ( // Show progress if uploading or batch is active with progress
        <div className="mb-3">
          <ProgressBar
            now={progress}
            label={`${progress}%`}
            animated
            variant={progress === 100 ? "success" : "info"}
          />
          {isBatchActive && fileToAutoUpload && progress < 100 && (
            <p className="text-center mt-1">Uploading: {fileToAutoUpload.name.split('/').pop()}...</p>
          )}
        </div>
      )}

      {uploadStatus.show && (
        <Alert variant={uploadStatus.variant} dismissible onClose={() => setUploadStatus({ ...uploadStatus, show: false })}>
          {uploadStatus.message}
        </Alert>
      )}
    </div>
  );
};

export default VideoUploader;