import torch as th
import math
import numpy as np
import logging
from .video_loader import BetterVideoLoader
from torch.utils.data import DataLoader
import argparse
from .preprocessing import Preprocessing, PreprocessingIntern
from .random_sequence_shuffler import RandomSequenceSampler
import torch.nn.functional as F
from tqdm import tqdm
import os
# import clip
from .viclip import ViCLIP
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

import pdb

_LOGGER = logging.getLogger(__name__)

# Set deterministic behavior
def set_deterministic_environment():
    """Set environment for deterministic behavior."""
    import random
    seed = 42  # Fixed seed for reproducibility
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    
    # Set deterministic algorithms
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    
    return seed


def extract_viclip_features_direct(csv_path, viclip_path, is_deterministic=False, cached_model=None, clip_len=1/3):
    """Direct function to extract ViCLIP features."""
    # Set deterministic environment if requested
    if is_deterministic:
        seed = set_deterministic_environment()
        _LOGGER.info("set deterministic mode with seed: %d", seed)

    # Parameters
    output_feat_size = 768
    # Use the passed clip_len parameter instead of hardcoded value
    size = 224
    corner = 0  # top left corner
    batch_size = 64
    
    _LOGGER.info("Using clip_len: %.3f seconds (fps: %.1f)", clip_len, 1/clip_len)
    
    # Create dataset and loader
    dataset = BetterVideoLoader(
        csv_path,
        framerate=1/clip_len,  # Use the configurable clip_len
        size=size,
        centercrop=True,
        overwrite=False,
        start_time=0
    )
    n_dataset = len(dataset)
    _LOGGER.info("Dataset size: %d", n_dataset)
    # Update the sampler creation
    sampler = RandomSequenceSampler(n_dataset, 10, deterministic=is_deterministic)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        sampler=sampler if n_dataset > 10 else None,
    )
    preprocess = PreprocessingIntern()

    # Create or use cached model
    model_created = False
    if cached_model is not None:
        _LOGGER.info("Using cached ViCLIP model...")
        model = cached_model
    else:
        _LOGGER.info("Creating ViCLIP model...")
        tokenizer = _Tokenizer()
        model = ViCLIP(tokenizer, pretrain=viclip_path).cuda()
        model_created = True

    if is_deterministic:
        seed = set_deterministic_environment()
        _LOGGER.info("set deterministic mode again to make sure the cached ViCLIP and the initial ViCLIP are both deterministic with seed: %d", seed)

    # Set model to evaluation mode and disable dropout if deterministic
    if is_deterministic:
        model.eval()
        # Disable dropout in the model
        for module in model.modules():
            if isinstance(module, th.nn.Dropout):
                module.p = 0.0
            elif hasattr(module, 'drop_path_rate'):
                module.drop_path_rate = 0.0

    totatl_num_frames = 0
    with th.no_grad():
        for k, data in enumerate(tqdm(loader)):
            input_file = data['input'][0]
            output_file = data['output'][0]
            if os.path.isfile(output_file):
                _LOGGER.debug("Video %s already processed.", input_file)
                continue
            elif not os.path.isfile(input_file):
                _LOGGER.warning("%s does not exist.", input_file)
            elif len(data['video'].shape) > 4:
                video = data['video'].squeeze(0)
                slice_y = slice(None, 224) if corner < 2 else slice(-224,None)
                slice_x = slice(None,224) if corner in [0,3] else slice(-224, None)
                video_chunk = preprocess(video[...,slice_y,slice_x])
                n_chunk = len(video_chunk)
                features = th.cuda.FloatTensor(
                    n_chunk, output_feat_size).fill_(0)
                n_iter = int(math.ceil(n_chunk / float(batch_size)))
                for i in range(n_iter):
                    min_ind = i * batch_size
                    max_ind = (i + 1) * batch_size
                    # pdb.set_trace()
                    video_batch = video_chunk[min_ind:max_ind].cuda()
                    batch_features = model.encode_vision(video_batch)
                    features[min_ind:max_ind] = batch_features
                    # _LOGGER.debug("Features shape: %s", features.shape)
                features = features.cpu().numpy()
                features = features.astype('float16')
                totatl_num_frames += features.shape[0]
                # safeguard output path before saving
                dirname = os.path.dirname(output_file)
                if not os.path.exists(dirname):
                    _LOGGER.info("ViCLIP Output directory %s does not exist, creating...", dirname)
                    os.makedirs(dirname)
                np.savez(output_file, features=features)
            else:
                _LOGGER.error("%s failed at ffprobe.", input_file)

    _LOGGER.info("Total number of frames: %d", totatl_num_frames)
    
    # Return the model if it was created (not cached)
    if model_created:
        return model
    else:
        return None
