import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    level=logging.INFO)


class SimpleRTPAlgorithm:
    """
    Simplified RTP (Recursive TSM Parsing) algorithm.
    
    The key insight: Instead of one-time convolution, recursively divide TSM
    into smaller segments and apply convolution to each segment.
    """
    
    def __init__(self, T1: int = 10, T2: float = 0.2):
        """
        Initialize simple RTP algorithm.
        
        Args:
            T1: Minimum segment size threshold
            T2: Score threshold for boundary detection
        """
        self.T1 = T1
        self.T2 = T2
        
        # Simple contrastive kernel (same as original)
        # self.mask = torch.tensor([[1., 0., -1.],
        #                          [0., 0., 0.],
        #                          [-1., 0., 1.]])
        self.mask = torch.tensor([[1., 1., 0., -1., -1.],
                            [1., 1., 0., -1., -1.],
                            [0., 0., 0., 0., 0.],
                            [-1., -1., 0., 1., 1.],
                            [-1., -1., 0., 1., 1.]])
    def _apply_convolution(self, tsm_segment: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution to TSM segment with proper normalization.
        
        Normalization is crucial to avoid bias in convolution:
        - Without normalization, the diagonal kernel creates artificial high scores
        - Positive parts of kernel applied to TSM values, negative parts to zero-padding
        - Normalizing to zero mean ensures fair comparison across all positions
        
        Args:
            tsm_segment: TSM segment
            
        Returns:
            Boundary scores
        """
        # Normalize TSM segment to have zero mean to avoid bias
        # tsm_mean = tsm_segment.mean()
        # tsm_normalized = tsm_segment - tsm_mean
        tsm_normalized = tsm_segment

        # Move mask to same device as tsm_segment
        mask = self.mask.to(device=tsm_segment.device, dtype=tsm_segment.dtype)
        mask_size = mask.size(0)
        pad_tsm = torch.nn.ZeroPad2d(mask_size//2)(tsm_normalized)
        
        # Apply convolution
        score = torch.diagonal(F.conv2d(pad_tsm.unsqueeze(0), 
                                       mask.unsqueeze(0).unsqueeze(0)).squeeze(0), 
                             dim1=0, dim2=1)
        
        return score
    
    def _find_local_maxima(self, score: torch.Tensor) -> List[int]:
        """
        Find local maxima in scores (same as original approach).
        
        Args:
            score: Boundary scores
            
        Returns:
            List of local maxima indices
        """
        # Move to CPU for numpy operations
        scores = score.cpu().numpy()
        maxima = []
        for i in range(1, len(scores)-1):
            if scores[i] > scores[i-1] and scores[i] > scores[i+1]:
                maxima.append(i)
        return maxima
    
    def _recursive_parse(self, tsm_segment: torch.Tensor, start_idx: int, 
                        boundaries: List[int], depth: int = 0) -> None:
        """
        Recursive TSM parsing - the key difference from original.
        
        Instead of one-time convolution on entire TSM, recursively:
        1. Apply convolution to current segment
        2. Find local maxima
        3. If maxima found, split at best boundary and recurse
        4. If no clear boundaries, stop
        
        Args:
            tsm_segment: Current TSM segment
            start_idx: Starting index in original TSM
            boundaries: List to store detected boundaries
            depth: Recursion depth
        """
        indent = "  " * depth
        
        # Check if segment is too small
        if len(tsm_segment) < self.T1:
            logger.info(f"{indent}🛑 STOP: Segment too small ({len(tsm_segment)} < {self.T1})")
            return
        
        # print(f"{indent}🔄 Processing segment [{start_idx}:{start_idx+len(tsm_segment)}], size={len(tsm_segment)}, tsm_segment= {tsm_segment}")
        logger.info(f"{indent}🔄 Processing segment [{start_idx}:{start_idx+len(tsm_segment)}], size={len(tsm_segment)}")

        # Apply convolution (same as original approach)
        score = self._apply_convolution(tsm_segment)
        
        # Find local maxima (same as original approach)
        local_maxima = self._find_local_maxima(score)
        
        if not local_maxima:
            print(f"{indent}🛑 STOP: No local maxima found")
            return
        
        # Find the strongest boundary
        scores = score.cpu().numpy()
        best_idx = max(local_maxima, key=lambda i: scores[i])
        best_score = scores[best_idx]
        
        print(f"{indent}🎯 Best boundary: frame {best_idx} (score={best_score:.4f}) and this tsm scores= {scores}")
        # print(f"{indent}🎯 Best boundary: frame {best_idx} (score={best_score:.4f})")

        # Check if score is above threshold
        # if best_score - scores.mean() < self.T2:
        if best_score < self.T2:
            print(f"{indent}🛑 STOP: Best score ({best_score:.4f}) - scores.mean() ({scores.mean():.4f}) < threshold ({self.T2})")
            return
        
        # Add boundary
        global_boundary_idx = start_idx + best_idx
        if global_boundary_idx not in boundaries:
            boundaries.append(global_boundary_idx)
            print(f"{indent}✅ ADDED BOUNDARY: Global frame {global_boundary_idx}")
            
            # Split TSM and recurse
            left_segment = tsm_segment[:best_idx, :best_idx]
            right_segment = tsm_segment[best_idx:, best_idx:]
            
            print(f"{indent}📂 SPLITTING: Left[{start_idx}:{start_idx+best_idx}], Right[{start_idx+best_idx}:{start_idx+len(tsm_segment)}]")
            
            # Recursive calls
            self._recursive_parse(left_segment, start_idx, boundaries, depth + 1)
            self._recursive_parse(right_segment, start_idx + best_idx, boundaries, depth + 1)
        else:
            print(f"{indent}⚠️  SKIPPED: Boundary {global_boundary_idx} already detected")
    
    def detect_boundaries(self, tsm: torch.Tensor) -> List[int]:
        """
        Detect boundaries using simplified RTP.
        
        Args:
            tsm: Temporal Self-similarity Matrix
            
        Returns:
            List of boundary frame indices
        """
        print(f"\n🚀 SIMPLE RTP ALGORITHM")
        print(f"📊 Parameters: T1={self.T1}, T2={self.T2}")
        print(f"📏 TSM Size: {len(tsm)}x{len(tsm)}")
        print("=" * 50)
        
        boundaries = []
        self._recursive_parse(tsm, 0, boundaries)
        
        print("=" * 50)
        print(f"🎯 RTP COMPLETED: Found {len(boundaries)} boundaries")
        print(f"📍 Final boundaries: {sorted(boundaries)}")
        
        return sorted(boundaries)
