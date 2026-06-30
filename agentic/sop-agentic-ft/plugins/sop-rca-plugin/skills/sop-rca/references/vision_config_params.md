# Vision Configuration Parameters

This document describes the vision processing configuration for Cosmos Reason
VLM inference and training.

Source: `cosmos_reason2_utils/cosmos_reason2_utils/vision.py`

## VisionConfig

The VisionConfig dataclass controls how video frames are processed before being
fed to the VLM.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `total_pixels` | int | 16572416 | Total pixel budget across all frames. Controls the resolution-frame tradeoff. |
| `max_pixels` | int | 1003520 | Maximum pixels per individual frame |
| `min_pixels` | int | 100352 | Minimum pixels per individual frame |
| `fps` | float | 8.0 | Target frame rate for video processing |
| `max_frames` | int | 40 | Maximum number of frames to extract from a video |
| `use_fps_or_nframes` | str | "fps" | "fps" = sample at target fps; "nframes" = sample fixed number of frames |

## Frame Extraction Logic

1. Load video and get total frames and native FPS
2. If `use_fps_or_nframes == "fps"`:
   - Calculate desired frames = `video_duration * fps`
   - Cap at `max_frames`
   - Uniformly sample that many frames from the video
3. If `use_fps_or_nframes == "nframes"`:
   - Use `max_frames` as the fixed number of frames to extract
   - Uniformly sample from the video

## Resolution Computation

For each frame, the resolution is determined by the pixel budget:
- `pixels_per_frame = total_pixels / num_frames`
- Capped between `min_pixels` and `max_pixels`
- The actual resolution is computed to maintain aspect ratio while fitting the pixel budget

## Key Constants and Formulas

- `IMAGE_PATCH_SIZE` = 16
- `SPATIAL_MERGE_SIZE` = 2
- `PATCH_FACTOR` = IMAGE_PATCH_SIZE * SPATIAL_MERGE_SIZE = 32
- `PIXELS_PER_TOKEN` = PATCH_FACTOR^2 = **1024** (1024 pixels = 1 visual token)

### total_pixels Derivation (in SOP custom dataset)
If `total_pixels` is not explicitly set in config:
```python
total_pixels = int(model_max_length * PIXELS_PER_TOKEN * 0.9)
# e.g., model_max_length=81920 -> total_pixels = 81920 * 1024 * 0.9 = 75,497,472
# e.g., model_max_length=4096  -> total_pixels = 4096 * 1024 * 0.9 = 3,774,873
```

### SOP CustomConfig Defaults (before TOML override)
The SOP custom dataset script defines these defaults:
- `fps`: 8
- `max_pixels`: 81920 (~286x286 per frame)
- `max_frames`: 8

These defaults are typically overridden by the training TOML config (e.g., `max_frames=40`, `total_pixels=16572416`).

## Impact on SOP Monitoring

### Chunk Duration vs Frame Coverage
| Chunk Duration | Frames at 8fps | Effective Frames (max=40) | Temporal Resolution |
|---------------|----------------|---------------------------|---------------------|
| 2s | 16 | 16 | Full |
| 5s | 40 | 40 | Full |
| 10s | 80 | 40 | 1 frame/0.25s |
| 25s | 200 | 40 | 1 frame/0.625s |
| 60s | 480 | 40 | 1 frame/1.5s |

Key insight: Chunks longer than `max_frames / fps` (= 5 seconds at defaults) get subsampled.
For very long chunks (>20s), temporal resolution drops significantly, making it hard for the
VLM to distinguish multiple short actions.

### Recommendations
- For SOP monitoring, `fps=8` and `max_frames=40` covers up to 5 seconds at full resolution
- Most SOP action chunks are 2-8 seconds, which is adequate
- DDM under-segmentation creating 20+ second chunks causes severe subsampling
- Consider increasing `max_frames` if DDM issues cannot be resolved (tradeoff: more compute)
