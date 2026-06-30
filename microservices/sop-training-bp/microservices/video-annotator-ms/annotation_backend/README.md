# Video Annotation Backend

A FastAPI-based backend service for the video annotation system. This service handles video upload, storage, management, and processing tasks including timestamp-based video splitting.


## Prerequisites

- Python 3.10+ (for development)
- Docker and Docker Compose (for deployment)

## Installation and Setup

```bash
# Navigate to backend directory
cd annotation_backend

# Run service
docker compose up
```

The API will be available at http://<server_ip>:8100

You can adjust the API port with `SERVICDE_PORT` environment variable


## API Documentation

1. **Annotation**: [api spec](api_spec/openapi_spec.json)


## Configuration

* Environment Variables
  - `SERVICE_PORT`: Service port (default: 8100)
  - `VIDEO_ROOT`: Video data root (default: ./assets/data)
  - `LOG_ROOT`: Log root (default: ./assets/logs)



## Troubleshooting
- Ensure you have Docker and Docker Compose installed
- Verify GPU drivers and nvidia-docker are properly configured
- Check that all required NGC credentials are set in the `.env` file
- Ensure the required directories exist and have proper permissions

## License
This project is dual-licensed under the `CC-BY-4.0 AND Apache-2.0` terms in the top-level [`LICENSE`](../../../../../LICENSE) file: source code under Apache-2.0, documentation under CC-BY-4.0. Bundled third-party software is listed in [`THIRD_PARTY_NOTICES.md`](../../../../../THIRD_PARTY_NOTICES.md).
