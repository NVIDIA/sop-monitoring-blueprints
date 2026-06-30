# Video Annotation Frontend

A React-based frontend application for the SOP (Standard Operating Procedure) video annotation system. This application provides an intuitive interface for uploading videos, managing actions, and setting timestamp annotations.

## Features

- **Video Upload & Management**: Support for individual video upload and batch folder processing
- **Action Configuration**: Upload and manage actions.json files for annotation templates
- **Timestamp Annotation**: Interactive timestamp editor for precise video annotation
- **Batch Processing**: Process multiple videos with the same action configuration
- **Results Management**: View and manage annotation results with local storage persistence
- **Responsive Design**: Modern UI built with React Bootstrap

## Technology Stack

- **Frontend Framework**: React 18.2.0
- **UI Components**: Bootstrap 5.3.2 + React Bootstrap 2.9.1
- **HTTP Client**: Axios 1.6.2
- **Build Tool**: React Scripts 5.0.1
- **Containerization**: Docker + Nginx (production)

## Project Structure

```
annotation_frontend/
├── public/                     # Static files
├── src/                        # Source code
│   ├── components              # React components
│   ├── App.js                  # Main application component
│   ├── App.css                 # Application styles
│   ├── index.js                # Application entry point
│   └── index.css               # Global styles
├── Dockerfile                  # Docker build configuration
├── docker-compose.yml          # Docker Compose configuration
├── nginx.conf                  # Nginx configuration for production
├── package.json                # Dependencies and scripts
└── README.md                   # This file
```


## Prerequisites

- Node.js 18+ (for development)
- Docker and Docker Compose (for deployment)

## Installation and Setup

This service can not be setup individually, it needs also data augmentation MS and VLM fine-tuning MS.


## Application Usage

### Workflow Overview

1. **Step 1: Upload Actions File**
   - Upload a JSON file containing action definitions
   - Format: `{"actions": ["(1) Pick up ...", "(2) Put down ..."]}`

2. **Step 2: Upload Video(s)**
   - Upload individual videos or select a folder for batch processing
   - Supported formats: MP4 video format

3. **Step 3: Set Timestamps**
   - Use the interactive timestamp editor to annotate video segments
   - Set start and end times for each action
   - Preview video segments in real-time

4. **Step 4: View Results**
   - Review generated clips and annotations
   - Download processed results

### Batch Processing

The application supports batch processing multiple videos with the same action configuration:

1. Load actions file in Step 1
2. Select a folder containing multiple videos in Step 2
3. The application will process each video sequentially
4. Results are stored and can be viewed in the All Clips Viewer

## API Integration

The frontend communicates with the backend API for:
- Video upload and storage
- Video metadata management
- Video splitting based on timestamps
- Health checks and status monitoring

## License
This project is dual-licensed under the `CC-BY-4.0 AND Apache-2.0` terms in the top-level [`LICENSE`](../../../../../LICENSE) file: source code under Apache-2.0, documentation under CC-BY-4.0. Bundled third-party software is listed in [`THIRD_PARTY_NOTICES.md`](../../../../../THIRD_PARTY_NOTICES.md).