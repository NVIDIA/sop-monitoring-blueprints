"""
Module for uboco action segmentation.

This module is responsible for segmenting actions from a video using uboco.
"""

import os
import logging
import json
import ast
import sys
import time
import traceback
from pathlib import Path
from fractions import Fraction
import shutil

import cv2
import pandas as pd

_LOGGER = logging.getLogger(__name__)


def float_to_fraction_string(value: float) -> str:
    """Convert a float to a fraction string (e.g., 0.333 -> "1/3")."""
    try:
        frac = Fraction(value).limit_denominator(1000)  # Limit denominator to avoid very large numbers
        return f"{frac.numerator}/{frac.denominator}"
    except Exception:
        # Fallback to decimal string if conversion fails
        return str(value)


def create_single_video_csv(video_path: str, output_dir: str) -> tuple[str, str, str]:
    """Create CSV files for a single video for uboco processing."""
    os.makedirs(output_dir, exist_ok=True)

    # Create a temporary CSV with the single video
    video_data = pd.DataFrame({
        'video_path': [video_path]
    })

    # Generate feature paths
    video_name = Path(video_path).stem

    # SlowFast features
    slowfast_csv = os.path.join(output_dir, "slowfast_features_input.csv")
    video_data['feature_path'] = f'{output_dir}/slowfast_features/{video_name}.npz'
    video_data.to_csv(slowfast_csv, index=False)

    # ViCLIP features
    viclip_csv = os.path.join(output_dir, "viclip_features_input.csv")
    video_data['feature_path'] = f'{output_dir}/viclip_features/{video_name}.npz'
    video_data.to_csv(viclip_csv, index=False)

    # Video IDs file
    video_ids_path = os.path.join(output_dir, "input_video_ids.txt")
    with open(video_ids_path, 'w') as f:
        f.write(f"{video_name}\n")

    # Benchmark file
    try:
        import ffmpeg
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream:
            duration = float(video_stream['duration'])
            avg_frame_rate = video_stream['avg_frame_rate']
            num, den = map(int, avg_frame_rate.split('/'))
            fps = num / den

            benchmark_data = {
                "vid": video_name,
                "duration": duration,
                "fps": fps
            }

            benchmark_path = os.path.join(output_dir, "benchmark_test_single.jsonl")
            with open(benchmark_path, 'w') as f:
                f.write(json.dumps(benchmark_data) + '\n')

            return slowfast_csv, viclip_csv, benchmark_path
    except Exception as e:
        _LOGGER.warning("Could not get video info: %s", e)
        return slowfast_csv, viclip_csv, None


def extract_viclip_features(csv_path: str, output_dir: str, uboco_base_path: str, viclip_path: str, is_deterministic: bool, cached_model=None, fps: float = 3.0):
    """Extract ViCLIP features."""
    clip_len = 1.0 / fps  # Convert fps to clip_len
    _LOGGER.info("Extracting ViCLIP features with fps=%.1f (clip_len=%.3f seconds)...", fps, clip_len)
    
    # Debug: Check CSV file
    _LOGGER.info("CSV path: %s", csv_path)
    _LOGGER.info("CSV exists: %s", os.path.exists(csv_path))
    
    # Check CSV content
    try:
        df = pd.read_csv(csv_path)
        _LOGGER.info("CSV content: %s", df.to_string())
        _LOGGER.info("CSV columns: %s", list(df.columns))
        _LOGGER.info("CSV shape: %s", df.shape)
    except Exception as e:
        _LOGGER.error("Error reading CSV: %s", e)
        raise

    sys.path = [uboco_base_path] + sys.path
    
    # Import the extraction function
    from clip.extract_intern_at_startframe_corner import extract_viclip_features_direct
    
    returned_model = extract_viclip_features_direct(
        csv_path=csv_path,
        viclip_path=viclip_path,
        is_deterministic=is_deterministic,
        cached_model=cached_model,
        clip_len=clip_len  # Pass the calculated clip_len
    )
    
    return returned_model


def extract_slowfast_features(csv_path: str, output_dir: str, uboco_base_path: str, slowfast_path: str, cached_model=None, fps: float = 3.0):
    """Extract SlowFast features."""
    clip_len = 1.0 / fps  # Convert fps to clip_len
    clip_len_str = float_to_fraction_string(clip_len)  # Convert to fraction string
    
    _LOGGER.info("Extracting SlowFast features with fps=%.1f (clip_len=%s seconds)...", 
                 fps, clip_len_str)
    
    # Add the uboco slowfast directory to path
    uboco_slowfast_path = os.path.join(uboco_base_path, "slowfast")
    if uboco_slowfast_path not in sys.path:
        sys.path.insert(0, uboco_slowfast_path)
    
    # Import the extraction function
    from extract_feature.extract import extract_slowfast_features_direct
    
    # Call the function directly with clip_len as fraction string
    returned_model = extract_slowfast_features_direct(
        csv_path=csv_path,
        slowfast_path=slowfast_path,
        cached_model=cached_model,
        uboco_base_path=uboco_base_path,
        clip_len=clip_len_str  # Pass as fraction string instead of float
    )
    
    return returned_model


def get_event_boundaries(output_dir: str, benchmark_path: str, uboco_base_path: str, extracted_fps: float, min_segment_seconds: float = 2.0, threshold: float = 0.2) -> str:
    """Get event boundaries by merging features."""
    # Convert seconds to frames: T1_frames = min_segment_seconds * extracted_fps
    rtp_T1 = int(min_segment_seconds * extracted_fps)
    _LOGGER.info("Getting event boundaries with RTP parameters: min_segment_seconds=%.1fs (T1=%d frames), threshold=%.2f...", 
                 min_segment_seconds, rtp_T1, threshold)
    
    viclip_dir = os.path.join(output_dir, "viclip_features")
    slowfast_dir = os.path.join(output_dir, "slowfast_features")
    result_path = os.path.join(output_dir, "uboco_intervals.txt")

    # Add the uboco merge_features directory to path
    uboco_merge_path = os.path.join(uboco_base_path, "merge_features")
    if uboco_merge_path not in sys.path:
        sys.path.insert(0, uboco_merge_path)
    
    # Import the direct function
    from pseudo_event_boundaries_inference import get_event_boundaries_direct
    
    # Call the function directly with RTP parameters
    result_path = get_event_boundaries_direct(
        v_feat_dirs=[viclip_dir, slowfast_dir],
        eval_path=benchmark_path,
        result_path=result_path,
        data_path="benchmark",
        rtp_T1=rtp_T1,
        rtp_T2=threshold
    )
    
    return result_path


def convert_boundaries_to_json(intervals_path: str, output_dir: str, video_path: str, uboco_base_path: str) -> str:
    """Convert boundaries to JSON format."""
    _LOGGER.info("Converting boundaries to JSON...")
    video_name = Path(video_path).stem
    output_json = os.path.join(output_dir, "uboco_intervals.json")
    
    # Read video IDs
    video_ids_path = os.path.join(output_dir, "input_video_ids.txt")
    with open(video_ids_path, "r") as f:
        video_ids = [x.strip() for x in f.readlines()]
    
    # Read intervals data
    with open(intervals_path, "r") as f:
        data = [ast.literal_eval(x.strip()) for x in f.readlines()]
    
    # Convert to the expected format
    video_data = {}
    for idx, item in enumerate(data):
        times = []
        for timestamp in item:
            times.extend(timestamp)
        vid = video_ids[idx]
        video_data[os.path.join(str(Path(video_path).parent), vid + ".mp4")] = {
            "url": "",
            "time": times,
        }
    
    # Write output JSON
    with open(output_json, "w") as f:
        json.dump(video_data, f)
    
    return output_json


def chunk_video_uboco(video_path: str, intervals_json: str, output_dir: str) -> list[str]:
    """Chunk video based on predicted boundaries using uboco."""
    _LOGGER.info("Chunking video using uboco...")

    # Read the intervals JSON
    with open(intervals_json, 'r') as f:
        data = json.load(f)

    video_name = Path(video_path).name
    if video_path not in data:
        _LOGGER.warning("Video path %s not found in intervals data", video_path)
        return []

    timestamps = data[video_path]["time"]

    # Convert flat list to pairs of start/end times
    intervals = []
    for i in range(0, len(timestamps), 2):
        if i + 1 < len(timestamps):
            intervals.append([timestamps[i], timestamps[i + 1]])

    if not intervals:
        _LOGGER.warning("No intervals found for chunking")
        return []

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _LOGGER.error("Failed to open video: %s", video_path)
        return []

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    # Add final interval if video is longer than last timestamp
    if video_duration > intervals[-1][1]:
        intervals.append([intervals[-1][1], video_duration])

    _LOGGER.info("Chunking video into %d segments...", len(intervals))

    # Create output directory for chunks
    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    chunk_filenames = []

    # Process each chunk
    for i, (start, end) in enumerate(intervals, 1):
        output_path = os.path.join(chunks_dir, f"{i:02d}_{Path(video_path).stem}.mp4")

        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Set start frame
        start_frame = int(start * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Read and write frames until end time
        while cap.get(cv2.CAP_PROP_POS_FRAMES) < end * fps:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        chunk_filenames.append(output_path)
        _LOGGER.debug("Created chunk %d: %s (%.2fs - %.2fs)", i, output_path, start, end)

    cap.release()
    _LOGGER.info("Video chunking completed. Created %d chunks", len(chunk_filenames))
    return chunk_filenames


def print_timing_summary(timings: dict):
    """Print a formatted timing summary."""
    total_time = sum(timings.values())
    
    _LOGGER.info("=" * 50)
    _LOGGER.info("UBOCO PIPELINE TIMING SUMMARY")
    _LOGGER.info("=" * 50)
    
    for step, duration in timings.items():
        percentage = (duration / total_time) * 100 if total_time > 0 else 0
        _LOGGER.info(f"{step:<30} {duration:.2f}s ({percentage:5.1f}%)")
    
    _LOGGER.info("-" * 50)
    _LOGGER.info(f"{'TOTAL':<30} {total_time:.2f}s")
    _LOGGER.info("=" * 50)


def process_video_with_uboco(video_path: str,
                             output_dir: str,
                             uboco_base_path: str,
                             viclip_path: str,
                             slowfast_path: str,
                             is_deterministic: bool,
                             cached_viclip_model=None,
                             cached_slowfast_model=None,
                             return_cached_model=False,
                             extracted_fps: float = 3.0,
                             min_segment_seconds: float = 2.0,
                             threshold: float = 0.2):
    """Process video using uboco to get intelligent chunks."""
    _LOGGER.info("Processing video with uboco: %s (extracted_fps=%.1f, min_segment_seconds=%.1fs, threshold=%.2f)", 
                 video_path, extracted_fps, min_segment_seconds, threshold)
    
    # Initialize timing dictionary
    timings = {}

    try:
        # Step 1: Create input files
        _LOGGER.info("Creating input files...")
        start_time = time.time()
        slowfast_csv, viclip_csv, benchmark_path = create_single_video_csv(video_path, output_dir)
        timings["Create input files"] = time.time() - start_time

        # Step 2: Extract ViCLIP features
        start_time = time.time()
        returned_viclip_model = extract_viclip_features(viclip_csv,
                                output_dir,
                                uboco_base_path,
                                viclip_path,
                                is_deterministic,
                                cached_model=cached_viclip_model,
                                fps=extracted_fps)
        timings["Extract ViCLIP features"] = time.time() - start_time

        # Step 3: Extract SlowFast features
        start_time = time.time()
        returned_slowfast_model = extract_slowfast_features(slowfast_csv,
                                  output_dir,
                                  uboco_base_path,
                                  slowfast_path,
                                  cached_model=cached_slowfast_model,
                                  fps=extracted_fps)
        timings["Extract SlowFast features"] = time.time() - start_time

        # Step 4: Get event boundaries
        _LOGGER.info("Getting event boundaries...")
        start_time = time.time()
        intervals_path = get_event_boundaries(output_dir,
                                              benchmark_path,
                                              uboco_base_path,
                                              extracted_fps=extracted_fps,
                                              min_segment_seconds=min_segment_seconds,
                                              threshold=threshold)
        timings["Get event boundaries"] = time.time() - start_time

        # Step 5: Convert boundaries to JSON
        _LOGGER.info("Converting boundaries to JSON...")
        start_time = time.time()
        intervals_json = convert_boundaries_to_json(intervals_path,
                                                    output_dir,
                                                    video_path,
                                                    uboco_base_path)
        timings["Convert boundaries to JSON"] = time.time() - start_time

        # Step 6: Extract chunk boundaries (without creating video chunks)
        _LOGGER.info("Extracting chunk boundaries...")
        start_time = time.time()
        chunk_start_seconds, chunk_end_seconds = get_chunk_boundaries_from_json(intervals_json, video_path)
        timings["Extract chunk boundaries"] = time.time() - start_time

        # Print timing summary
        print_timing_summary(timings)

        # Return boundaries and optionally the cached models
        if return_cached_model:
            return chunk_start_seconds, chunk_end_seconds, returned_viclip_model, returned_slowfast_model
        else:
            return chunk_start_seconds, chunk_end_seconds

    except Exception as e:
        _LOGGER.error("Unexpected error during uboco processing: %s", e)
        _LOGGER.error("Full traceback: %s", traceback.format_exc())
        raise


def get_chunk_boundaries_from_json(intervals_json: str, video_path: str) -> tuple[list[float], list[float]]:
    """Extract chunk boundaries from intervals JSON without creating video chunks.
    
    Returns:
        tuple: (chunk_start_seconds, chunk_end_seconds)
    """
    _LOGGER.info("Extracting chunk boundaries from intervals...")

    # Read the intervals JSON
    with open(intervals_json, 'r') as f:
        data = json.load(f)

    if video_path not in data:
        _LOGGER.warning("Video path %s not found in intervals data", video_path)
        return [], []

    timestamps = data[video_path]["time"]

    # Convert flat list to pairs of start/end times
    intervals = []
    for i in range(0, len(timestamps), 2):
        if i + 1 < len(timestamps):
            intervals.append([timestamps[i], timestamps[i + 1]])

    if not intervals:
        _LOGGER.warning("No intervals found for chunking")
        return [], []

    # Get video duration to add final interval if needed
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _LOGGER.error("Failed to open video: %s", video_path)
        return [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    cap.release()

    # Add final interval if video is longer than last timestamp
    if video_duration > intervals[-1][1]:
        intervals.append([intervals[-1][1], video_duration])

    _LOGGER.info("Found %d chunk boundaries", len(intervals))

    # Extract start and end times
    chunk_start_seconds = [interval[0] for interval in intervals]
    chunk_end_seconds = [interval[1] for interval in intervals]

    _LOGGER.info("Chunk boundaries extracted successfully")
    return chunk_start_seconds, chunk_end_seconds
