from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils.span_utils import span_cxw_to_xx
from new_start_end_dataset import NewStartEndDataset, start_end_collate
from rtp_algorithm_simple import SimpleRTPAlgorithm

import logging

# Computational complexity check
import argparse
import pdb

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def get_event_boundaries_direct(v_feat_dirs, eval_path, result_path, data_path="benchmark", start_time=0, max_v_l=150, 
                               ctx_mode="video_tef", clip_length=1.0, data_ratio=1.0, normalize_v=True, 
                               eval_bsz=64, num_workers=1, pin_memory=True, device="cuda", rtp_T1=2, rtp_T2=0.2):
    """
    Direct function to get event boundaries using simplified RTP algorithm.
    
    Args:
        v_feat_dirs: List of feature directories (ViCLIP and SlowFast)
        eval_path: Path to evaluation file
        result_path: Path to save results
        data_path: Data path (default: "benchmark")
        start_time: Start time offset
        max_v_l: Maximum video length
        ctx_mode: Context mode
        clip_length: Clip length
        data_ratio: Data ratio
        normalize_v: Whether to normalize video features
        eval_bsz: Evaluation batch size
        num_workers: Number of workers
        pin_memory: Whether to pin memory
        device: Device to use
        rtp_T1: Minimum segment size threshold
        rtp_T2: Score threshold for boundary detection
    
    Returns:
        Path to the result file
    """
    logger.info("Setup config, data and model for simplified RTP algorithm...")
    cudnn.benchmark = True
    cudnn.deterministic = False
    
    # Create args-like object
    class Args:
        def __init__(self):
            self.data_path = data_path
            self.start_time = start_time
            self.eval_path = eval_path
            self.result_path = result_path
            self.v_feat_dirs = v_feat_dirs
            self.max_v_l = max_v_l
            self.ctx_mode = ctx_mode
            self.clip_length = clip_length
            self.data_ratio = data_ratio
            self.normalize_v = normalize_v
            self.eval_bsz = eval_bsz
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.device = device
            self.rtp_T1 = rtp_T1
            self.rtp_T2 = rtp_T2
    
    args = Args()
    print(args.__dict__)
    
    eval_dataset = NewStartEndDataset(
        data_path=args.data_path,
        eval_path=args.eval_path,
        v_feat_dirs=args.v_feat_dirs,
        max_v_l=args.max_v_l,
        ctx_mode=args.ctx_mode,
        data_ratio=args.data_ratio,
        normalize_v=args.normalize_v,
        clip_len=args.clip_length,
        txt_drop_ratio=0
    )

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=args.eval_bsz,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.pin_memory
    )

    res = []

    for batch in tqdm(eval_loader, desc="compute boundaries with simplified RTP"):
        metas = batch[0]

        src_vid = batch[1]["video_feat"][0].to(args.device,non_blocking=args.pin_memory)
        src_vid_mask=batch[1]["video_feat"][1].to(args.device,non_blocking=args.pin_memory)
        
        # Use simplified RTP algorithm instead of simple convolution
        pseudo_event_spans = generate_pseudo_event_simple_rtp(
            src_vid, src_vid_mask, 
            T1=args.rtp_T1, T2=args.rtp_T2
        )

        for idx, (pseudo_spans,meta) in enumerate(zip(pseudo_event_spans,metas)):
            if len(pseudo_spans) > 0:
                # Convert spans to the expected format
                spans_tensor = torch.stack(pseudo_spans)
                spans_tensor = span_cxw_to_xx(spans_tensor.cpu()) * (meta["duration"]-args.start_time)+args.start_time
                pseudo_span = []
                for i in range(len(spans_tensor)):
                    if spans_tensor[i][0] > spans_tensor[i][1]:
                        logging.info(f"Found invalid pseudotimestamp for {meta['vid']}")
                    else:
                        pseudo_span.append([round(num,4) for num in spans_tensor[i].tolist()])
            else:
                pseudo_span = []
                
            res.append(dict(vid=meta["vid"],pseudo_event_spans=pseudo_span))
    
    with open(args.result_path, "w") as f:
        for data in res:
            f.write(str(data['pseudo_event_spans']) + "\n")
    
    return result_path


def generate_pseudo_event_simple_rtp(src_vid, src_vid_mask, T1=2, T2=0.2):
    """
    Generate pseudo events using simplified RTP algorithm.
    
    Args:
        src_vid: Video features [batch_size, seq_len, feature_dim]
        src_vid_mask: Video mask [batch_size, seq_len]
        T1: Minimum segment size threshold
        T2: Score threshold for boundary detection
        
    Returns:
        List of pseudo event spans for each video in the batch
    """
    bsz, L_src, _ = src_vid.size()
    
    # Normalize video features
    norm_vid = src_vid / (src_vid.norm(dim=2, keepdim=True) + 1e-8)
    
    # Compute Temporal Self-similarity Matrix (TSM)
    tsm = torch.bmm(norm_vid, norm_vid.transpose(1, 2))
    
    # Initialize simplified RTP algorithm
    rtp = SimpleRTPAlgorithm(T1=T1, T2=T2)
    
    pseudo_event_spans = []
    
    # Process each video in the batch
    for i in range(bsz):
        # Get TSM for current video
        video_tsm = tsm[i]
        video_mask = src_vid_mask[i]
        
        # Get valid length
        L_vid = torch.count_nonzero(video_mask)
        
        # Extract valid TSM segment
        valid_tsm = video_tsm[:L_vid, :L_vid]
        
        # Detect boundaries using simplified RTP
        boundaries = rtp.detect_boundaries(valid_tsm)
        
        # Convert boundaries to event spans with complete coverage
        spans = []
        
        if len(boundaries) >= 2:
            # Add start span (from 0 to first boundary)
            if boundaries[0] > 0:
                start_center = boundaries[0] / 2
                start_width = boundaries[0]
                start_span = torch.tensor([start_center, start_width], device=video_mask.device, dtype=video_mask.dtype)
                spans.append(start_span)
            
            # Create spans from consecutive boundaries
            for j in range(len(boundaries) - 1):
                start = boundaries[j]
                end = boundaries[j + 1]
                center = (start + end) / 2
                width = end - start
                # Create tensor on the same device as video_mask
                span_tensor = torch.tensor([center, width], device=video_mask.device, dtype=video_mask.dtype)
                spans.append(span_tensor)
            
            # Add end span (from last boundary to video end)
            if boundaries[-1] < L_vid:
                end_center = (boundaries[-1] + L_vid) / 2
                end_width = L_vid - boundaries[-1]
                end_span = torch.tensor([end_center, end_width], device=video_mask.device, dtype=video_mask.dtype)
                spans.append(end_span)
        
        elif len(boundaries) == 1:
            # Only one boundary detected
            boundary = boundaries[0]
            
            # Add start span (from 0 to boundary)
            if boundary > 0:
                start_center = boundary / 2
                start_width = boundary
                start_span = torch.tensor([start_center, start_width], device=video_mask.device, dtype=video_mask.dtype)
                spans.append(start_span)
            
            # Add end span (from boundary to video end)
            if boundary < L_vid:
                end_center = (boundary + L_vid) / 2
                end_width = L_vid - boundary
                end_span = torch.tensor([end_center, end_width], device=video_mask.device, dtype=video_mask.dtype)
                spans.append(end_span)
        
        else:
            # No boundaries detected - create single span for entire video
            center = L_vid / 2
            width = L_vid
            full_span = torch.tensor([center, width], device=video_mask.device, dtype=video_mask.dtype)
            spans.append(full_span)
        
        # Normalize by video length (ensure same device)
        L_vid_tensor = L_vid.to(device=video_mask.device, dtype=video_mask.dtype)
        spans = [span / L_vid_tensor for span in spans]
        pseudo_event_spans.append(spans)
    
    return pseudo_event_spans

def start_inference_simple_rtp(args):
    logger.info("Setup config, data and model for simplified RTP algorithm...")
    cudnn.benchmark = True
    cudnn.deterministic = False
   
    print(args)
    eval_dataset = NewStartEndDataset(
        data_path=args.data_path,
        eval_path=args.eval_path,
        v_feat_dirs=args.v_feat_dirs,
        max_v_l=args.max_v_l,
        ctx_mode=args.ctx_mode,
        data_ratio=args.data_ratio,
        normalize_v=args.normalize_v,
        clip_len=args.clip_length,
        txt_drop_ratio=0
    )

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=args.eval_bsz,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.pin_memory
    )

    res = []

    for batch in tqdm(eval_loader, desc="compute boundaries with simplified RTP"):
        metas = batch[0]

        src_vid = batch[1]["video_feat"][0].to(args.device,non_blocking=args.pin_memory)
        src_vid_mask=batch[1]["video_feat"][1].to(args.device,non_blocking=args.pin_memory)
        
        # Use simplified RTP algorithm instead of simple convolution
        pseudo_event_spans = generate_pseudo_event_simple_rtp(
            src_vid, src_vid_mask, 
            T1=args.rtp_T1, T2=args.rtp_T2
        )

        for idx, (pseudo_spans,meta) in enumerate(zip(pseudo_event_spans,metas)):
            if len(pseudo_spans) > 0:
                # Convert spans to the expected format
                spans_tensor = torch.stack(pseudo_spans)
                spans_tensor = span_cxw_to_xx(spans_tensor.cpu()) * (meta["duration"]-args.start_time)+args.start_time
                pseudo_span = []
                for i in range(len(spans_tensor)):
                    if spans_tensor[i][0] > spans_tensor[i][1]:
                        logging.info(f"Found invalid pseudotimestamp for {meta['vid']}")
                    else:
                        pseudo_span.append([round(num,4) for num in spans_tensor[i].tolist()])
            else:
                pseudo_span = []
                
            res.append(dict(vid=meta["vid"],pseudo_event_spans=pseudo_span))
    
    with open(args.result_path, "w") as f:
        for data in res:
            f.write(str(data['pseudo_event_spans']) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video clip captioning with simplified RTP algorithm")
    parser.add_argument("--data_path", type=str, default="/home/alhu/nv-label-studio-video/data/benchmark")
    parser.add_argument("--start_time", type=float, default=0)
    parser.add_argument("--eval_path", type=str)
    parser.add_argument("--result_path", type=str, default="results/benchmark_test_rtp.txt")
    parser.add_argument("--v_feat_dirs", type=str, nargs="+", default=["/home/alhu/nv-label-studio-video/HERO_Video_Feature_Extractor/clip/output/clip-vit_cliplen1_features","/home/alhu/nv-label-studio-video/HERO_Video_Feature_Extractor/slowfast/output/slowfast_features_cliplen1"])
    parser.add_argument("--max_v_l", type=int, default=150)
    parser.add_argument("--ctx_mode", type=str, default="video_tef")
    parser.add_argument("--clip_length", type=float, default=1.0)
    parser.add_argument("--data_ratio", type=float, default=1.0)
    parser.add_argument("--normalize_v", type=bool, default=True)
    parser.add_argument("--eval_bsz", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Simplified RTP algorithm parameters
    parser.add_argument("--rtp_T1", type=int, default=2, help="Minimum segment size threshold")
    parser.add_argument("--rtp_T2", type=float, default=0.6, help="Score threshold for boundary detection")

    args = parser.parse_args()
    start_inference_simple_rtp(args) 