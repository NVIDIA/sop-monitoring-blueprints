# Video Annotation System

A comprehensive video annotation system consisting of a React frontend application and a FastAPI backend service for processing and annotating video content with timestamp-based action segmentation.

## System Architecture

- **Backend** (`annotation_backend/`): FastAPI service handling video upload, management, and processing
- **Frontend** (`annotation_frontend/`): React web application providing the user interface for uploading action, video annotation, triggering data augmentation and VLM fine-tuning

## Prerequisites

- Docker 28.2.2 or later
- Docker Compose v2.36.2 or later

## Installation and Setup

This service can not be setup individually, it needs also data augmentation MS and VLM fine-tuning MS.

Please refer to training BP deployment for running the whole service.


## Microservices APIs
1. **Annotation backend**: [api spec](annotation_backend/api_spec/openapi_spec.json)


## Usage Workflow

### 1. Action Configuration
- Upload `actions.json` file containing action definitions
- Format: `{"actions": ["(1) picking up ...", "(2) putting down ..."], "actions_can_be_skipped": ["(10) doing action not belong to SOP"]}`
    * Each action should look like "(N) ...."
    * Each action should start with present continuous tense
- Actions serve as instructions & choices for video annotation

### 2. Video Upload
- **Single Video**: Upload individual video files
- **Batch Processing**: Select folders containing multiple videos
- Supported formats: MP4

### 3. Timestamp Annotation
- Use the interactive timestamp editor
- Set end times for each event
- Preview video segments in real-time

### 4. Video Processing
- System splits videos based on timestamp annotations
- Generates individual clips for each annotated segment
- Maintains original video quality and format

### 5. Results Management
- View all processed videos and clips
- Provides download links for processed clips
- Clear results and delete all uploaded videos

## License
This project is dual-licensed under the `CC-BY-4.0 AND Apache-2.0` terms in the top-level [`LICENSE`](../../../../LICENSE) file: source code under Apache-2.0, documentation under CC-BY-4.0. Bundled third-party software is listed in [`THIRD_PARTY_NOTICES.md`](../../../../THIRD_PARTY_NOTICES.md).
