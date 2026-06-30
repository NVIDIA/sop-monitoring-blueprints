######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is based on DDM-Net (https://github.com/MCG-NJU/DDM),
# Copyright (c) 2021 Mike Zheng Shou, licensed under the MIT License.
# Modifications Copyright (c) NVIDIA CORPORATION & AFFILIATES.
######################################################################################################

"""
Configuration management for DDM Lightning Training using OmegaConf.

Supports both YAML/TOML config files and command-line arguments.
Priority: CLI Arguments > YAML/TOML Config > Default Values (System Defaults)
"""

import argparse
from typing import Dict, Any, Tuple
import yaml
import toml
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

def get_system_defaults() -> DictConfig:
    """
    Define the base layer of configuration (System Defaults).
    Using OmegaConf for structured, typed configuration.
    """
    base_conf = OmegaConf.create({
        'dataset_config': {
            'dataset': 'DDMDataset',
            'resolution': 224,
            'frames_per_side': 5,
            'downsample': 1,
            'min_change_dur': 0.3,
            'num_classes': 2,
            'seed': 42,
            'video_backend': 'pyav',
            'processor_name_or_path': None,
            'use_cache': False,
            'batch_size': 2,
            'workers': 4,
            'train_config': {
                'mode': 'train',
                'anno_path': None,
                'data_root': None,
            },
            'val_config': {
                'mode': 'val',
                'anno_path': None,
                'data_root': None,
            },
        },
        'model_config': {
            'model_name': 'multiframes_resnet',
            'backbone': 'resnet50',
            'pretrained': None,
            'freeze_backbone': False,
            'num_classes': 2,
            'img_size': 224,
        },
        'training_config': {
            'optimizer': 'adamw',
            'learning_rate': 0.01,
            'weight_decay': 0.0001,
            'momentum': 0.9,
            'opt_eps': None,
            'opt_betas': None,
            'scheduler': 'step',
            'warmup_epochs': 2,
            'warmup_lr': 0.0001,
            'decay_epochs': 30,
            'decay_rate': 0.1,
            'min_lr': 1e-5,
            'patience_epochs': 10,
            'epochs': 30,
            'eval_freq': 1,
            'seed': 42,
            'clip_grad': None,
            'clip_mode': 'norm',
            'amp': False,
            'model_ema': False,
            'model_ema_decay': 0.9998,
            'model_ema_start_epoch': 0,
            'eval_metric': 'f1_score',
            'save_visualizations': False,
            'log_interval': 50,
            'output': './output_lightning',
            'exp_name': 'exp',
            'resume': None,
            'checkpoint_top_k': 3,
            'num_gpus': 1,
            'num_nodes': 1,
            'strategy': 'auto',
        },
        'logging': {}
    })
    return base_conf


def get_parser():
    """
    Get argument parser.
    Defaults are SUPPRESSED to enable proper merge logic,
    but we inject the default values into the help strings dynamically.
    """
    # 1. Load System Defaults to inject into help strings
    # This ensures --help always shows the actual values defined in get_system_defaults()
    cfg = get_system_defaults()
    d_cfg = cfg.dataset_config
    m_cfg = cfg.model_config
    t_cfg = cfg.training_config

    parser = argparse.ArgumentParser(
        description="PyTorch Lightning 2.x Training for DDM",
        formatter_class=argparse.RawTextHelpFormatter # Changed to RawText to allow better formatting if needed
    )

    # ============================================================================
    # Config file
    # ============================================================================
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML or TOML config file.'
    )

    # ============================================================================
    # Model parameters
    # ============================================================================
    parser.add_argument('--model', default=argparse.SUPPRESS, type=str, 
                        help=f'Model architecture name (default: {m_cfg.model_name})')
    parser.add_argument('--backbone', default=argparse.SUPPRESS, type=str, 
                        help=f'Backbone network name (default: {m_cfg.backbone})')
    parser.add_argument('--pretrained', type=str, default=argparse.SUPPRESS, 
                        help=f'Path to pretrained backbone weights (default: {m_cfg.pretrained})')
    parser.add_argument('--freeze-backbone', action='store_true', default=argparse.SUPPRESS, 
                        help=f'Freeze backbone weights (default: {m_cfg.freeze_backbone})')
    parser.add_argument('--num-classes', type=int, default=argparse.SUPPRESS, 
                        help=f'Number of classes (default: {m_cfg.num_classes})')
    parser.add_argument('--img-size', type=int, default=argparse.SUPPRESS, 
                        help=f'Input image size (default: {d_cfg.resolution})')

    # ============================================================================
    # Dataset parameters
    # ============================================================================
    parser.add_argument('--train-anno-path', type=str, default=argparse.SUPPRESS, 
                        help='Path to training annotation JSON file')
    parser.add_argument('--val-anno-path', type=str, default=argparse.SUPPRESS, 
                        help='Path to validation annotation JSON file')
    parser.add_argument('--train-dataroot', type=str, default=argparse.SUPPRESS, 
                        help='Root directory containing training video files')
    parser.add_argument('--test-dataroot', type=str, default=argparse.SUPPRESS, 
                        help='Root directory containing test video files')
    parser.add_argument('--frames-per-side', type=int, default=argparse.SUPPRESS, 
                        help=f'Number of frames per side (default: {d_cfg.frames_per_side})')
    parser.add_argument('--downsample', type=int, default=argparse.SUPPRESS, 
                        help=f'Temporal downsampling rate (default: {d_cfg.downsample})')
    parser.add_argument('--min-change-dur', type=float, default=argparse.SUPPRESS, 
                        help=f'Minimum change duration (default: {d_cfg.min_change_dur})')
    parser.add_argument('--video-backend', type=str, default=argparse.SUPPRESS, choices=['pyav', 'torchcodec'],
                        help=f'Video decoding backend (default: {d_cfg.video_backend})')
    parser.add_argument('--use-cache', action='store_true', default=argparse.SUPPRESS, 
                        help=f'Cache validation videos (default: {d_cfg.use_cache})')
    parser.add_argument('--processor-name-or-path', type=str, default=argparse.SUPPRESS, 
                        help=f'Processor path (default: {d_cfg.processor_name_or_path})')

    # ============================================================================
    # DataLoader parameters
    # ============================================================================
    parser.add_argument('--batch-size', '-b', type=int, default=argparse.SUPPRESS, 
                        help=f'Batch size (default: {d_cfg.batch_size})')
    parser.add_argument('--num-workers', '-j', type=int, default=argparse.SUPPRESS, 
                        help=f'Number of workers (default: {d_cfg.workers})')
    parser.add_argument('--pin-memory', action='store_true', default=argparse.SUPPRESS,
                        help='Pin memory in DataLoader')

    # ============================================================================
    # Optimizer parameters
    # ============================================================================
    parser.add_argument('--optimizer', '--opt', default=argparse.SUPPRESS, type=str, 
                        help=f'Optimizer name (default: {t_cfg.optimizer})')
    parser.add_argument('--learning-rate', '--lr', type=float, default=argparse.SUPPRESS, 
                        help=f'Learning rate (default: {t_cfg.learning_rate})')
    parser.add_argument('--weight-decay', type=float, default=argparse.SUPPRESS, 
                        help=f'Weight decay (default: {t_cfg.weight_decay})')
    parser.add_argument('--momentum', type=float, default=argparse.SUPPRESS, 
                        help=f'Momentum (default: {t_cfg.momentum})')
    parser.add_argument('--opt-eps', type=float, default=argparse.SUPPRESS, 
                        help=f'Optimizer epsilon (default: {t_cfg.opt_eps})')
    parser.add_argument('--opt-betas', type=float, nargs='+', default=argparse.SUPPRESS, 
                        help=f'Optimizer betas (default: {t_cfg.opt_betas})')

    # ============================================================================
    # Scheduler parameters
    # ============================================================================
    parser.add_argument('--scheduler', '--sched', default=argparse.SUPPRESS, type=str, 
                        help=f'Scheduler (default: {t_cfg.scheduler})')
    parser.add_argument('--warmup-epochs', type=int, default=argparse.SUPPRESS, 
                        help=f'Warmup epochs (default: {t_cfg.warmup_epochs})')
    parser.add_argument('--warmup-lr', type=float, default=argparse.SUPPRESS, 
                        help=f'Warmup LR (default: {t_cfg.warmup_lr})')
    parser.add_argument('--decay-epochs', type=int, default=argparse.SUPPRESS, 
                        help=f'Decay epochs (default: {t_cfg.decay_epochs})')
    parser.add_argument('--decay-rate', type=float, default=argparse.SUPPRESS, 
                        help=f'Decay rate (default: {t_cfg.decay_rate})')
    parser.add_argument('--min-lr', type=float, default=argparse.SUPPRESS, 
                        help=f'Min LR (default: {t_cfg.min_lr})')
    parser.add_argument('--patience-epochs', type=int, default=argparse.SUPPRESS, 
                        help=f'Patience epochs (default: {t_cfg.patience_epochs})')

    # ============================================================================
    # Training parameters
    # ============================================================================
    parser.add_argument('--epochs', type=int, default=argparse.SUPPRESS, 
                        help=f'Number of epochs (default: {t_cfg.epochs})')
    parser.add_argument('--eval-freq', type=int, default=argparse.SUPPRESS, 
                        help=f'Eval frequency (default: {t_cfg.eval_freq})')
    parser.add_argument('--seed', type=int, default=argparse.SUPPRESS, 
                        help=f'Random seed (default: {t_cfg.seed})')
    parser.add_argument('--clip-grad', type=float, default=argparse.SUPPRESS, 
                        help=f'Gradient clipping (default: {t_cfg.clip_grad})')
    parser.add_argument('--clip-mode', type=str, default=argparse.SUPPRESS, 
                        help=f'Clipping mode (default: {t_cfg.clip_mode})')
    parser.add_argument('--amp', action='store_true', default=argparse.SUPPRESS, 
                        help=f'Enable AMP (default: {t_cfg.amp})')

    # ============================================================================
    # EMA & Eval & Output
    # ============================================================================
    parser.add_argument('--model-ema', action='store_true', default=argparse.SUPPRESS, 
                        help=f'Enable Model EMA (default: {t_cfg.model_ema})')
    parser.add_argument('--model-ema-decay', type=float, default=argparse.SUPPRESS, 
                        help=f'EMA decay (default: {t_cfg.model_ema_decay})')
    parser.add_argument('--model-ema-start-epoch', type=int, default=argparse.SUPPRESS, 
                        help=f'EMA start epoch (default: {t_cfg.model_ema_start_epoch})')

    parser.add_argument('--eval-metric', default=argparse.SUPPRESS, type=str, 
                        help=f'Eval metric (default: {t_cfg.eval_metric})')
    parser.add_argument('--save-visualizations', '--save-images', action='store_true', default=argparse.SUPPRESS, 
                        help=f'Save images (default: {t_cfg.save_visualizations})')
    parser.add_argument('--log-interval', type=int, default=argparse.SUPPRESS, 
                        help=f'Log interval (default: {t_cfg.log_interval})')

    parser.add_argument('--output', default=argparse.SUPPRESS, type=str, 
                        help=f'Output dir (default: {t_cfg.output})')
    parser.add_argument('--exp-name', type=str, default=argparse.SUPPRESS, 
                        help=f'Experiment name (default: {t_cfg.exp_name})')
    parser.add_argument('--resume', type=str, default=argparse.SUPPRESS, 
                        help=f'Resume path (default: {t_cfg.resume})')
    parser.add_argument('--checkpoint-top-k', type=int, default=argparse.SUPPRESS, 
                        help=f'Checkpoint top K (default: {t_cfg.checkpoint_top_k})')

    # ============================================================================
    # Distributed
    # ============================================================================
    parser.add_argument('--num-gpus', type=int, default=argparse.SUPPRESS, 
                        help=f'Number of GPUs (default: {t_cfg.num_gpus})')
    parser.add_argument('--num-nodes', type=int, default=argparse.SUPPRESS, 
                        help=f'Number of nodes (default: {t_cfg.num_nodes})')
    parser.add_argument('--strategy', type=str, default=argparse.SUPPRESS, 
                        help=f'Strategy (default: {t_cfg.strategy})')
    
    # OmegaConf generic override
    parser.add_argument('overrides', nargs='*', 
                        help="Any key=value arguments for OmegaConf (e.g., training_config.optimizer=sgd)")

    return parser


def load_file_to_omegaconf(config_path: str) -> DictConfig:
    """
    Load YAML or TOML into OmegaConf.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if path.suffix == '.toml':
        with open(config_path, 'r') as f:
            toml_dict = toml.load(f)
        return OmegaConf.create(toml_dict)
    elif path.suffix in ['.yaml', '.yml']:
        return OmegaConf.load(config_path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or TOML file.

    Args:
        config_path: Path to YAML or TOML config file

    Returns:
        Dictionary containing configuration
    """
    path = Path(config_path)
    
    if path.suffix == '.toml':
        with open(config_path, 'r') as f:
            config = toml.load(f)
    elif path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(
            f"Unsupported config file format: {path.suffix}. "
            f"Please use .yaml, .yml, or .toml"
        )
    
    return config


def merge_configs(args: argparse.Namespace) -> DictConfig:
    """
    Merge logic: System Defaults < Config File < CLI Arguments
    """
    # 1. Base Layer: System Defaults
    cfg = get_system_defaults()

    # 2. Middle Layer: Config File (If provided)
    if args.config:
        print(f"Loading config from: {args.config}")
        file_cfg = load_file_to_omegaconf(args.config)
        cfg = OmegaConf.merge(cfg, file_cfg)
    # 3. Top Layer: CLI Arguments
    cli_args = vars(args)
    
    override_list = []

    # --- Dataset Mapping ---
    if 'train_anno_path' in cli_args:
        override_list.append(f"dataset_config.train_config.anno_path={cli_args['train_anno_path']}")
    if 'val_anno_path' in cli_args:
        override_list.append(f"dataset_config.val_config.anno_path={cli_args['val_anno_path']}")
    if 'train_dataroot' in cli_args:
        override_list.append(f"dataset_config.train_config.data_root={cli_args['train_dataroot']}")
    if 'test_dataroot' in cli_args:
        override_list.append(f"dataset_config.val_config.data_root={cli_args['test_dataroot']}")
    
    dataset_map = {
        'frames_per_side': 'dataset_config.frames_per_side',
        'downsample': 'dataset_config.downsample',
        'min_change_dur': 'dataset_config.min_change_dur',
        'video_backend': 'dataset_config.video_backend',
        'processor_name_or_path': 'dataset_config.processor_name_or_path',
        'use_cache': 'dataset_config.use_cache',
        'batch_size': 'dataset_config.batch_size',
        'num_workers': 'dataset_config.workers'
    }
    for cli_key, cfg_key in dataset_map.items():
        if cli_key in cli_args:
            override_list.append(f"{cfg_key}={cli_args[cli_key]}")

    if 'img_size' in cli_args:
        override_list.append(f"dataset_config.resolution={cli_args['img_size']}")
        override_list.append(f"model_config.img_size={cli_args['img_size']}")
    
    if 'num_classes' in cli_args:
        override_list.append(f"dataset_config.num_classes={cli_args['num_classes']}")
        override_list.append(f"model_config.num_classes={cli_args['num_classes']}")

    if 'seed' in cli_args:
        override_list.append(f"dataset_config.seed={cli_args['seed']}")
        override_list.append(f"training_config.seed={cli_args['seed']}")

    # --- Model Mapping ---
    model_map = {
        'model': 'model_config.model_name',
        'backbone': 'model_config.backbone',
        'pretrained': 'model_config.pretrained',
        'freeze_backbone': 'model_config.freeze_backbone',
    }
    for cli_key, cfg_key in model_map.items():
        if cli_key in cli_args:
            override_list.append(f"{cfg_key}={cli_args[cli_key]}")

    # --- Training Mapping ---
    training_keys = [
        'optimizer', 'learning_rate', 'weight_decay', 'momentum', 'opt_eps', 'opt_betas',
        'scheduler', 'warmup_epochs', 'warmup_lr', 'decay_epochs', 'decay_rate',
        'min_lr', 'patience_epochs', 'epochs', 'eval_freq', 'clip_grad', 'clip_mode',
        'amp', 'model_ema', 'model_ema_decay', 'model_ema_start_epoch',
        'eval_metric', 'save_visualizations', 'log_interval', 'output', 'exp_name',
        'resume', 'checkpoint_top_k', 'num_gpus', 'num_nodes', 'strategy'
    ]

    for key in training_keys:
        if key in cli_args:
            override_list.append(f"training_config.{key}={cli_args[key]}")
    if override_list:
        cli_cfg = OmegaConf.from_dotlist(override_list)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # 4. Handle Generic Overrides
    if args.overrides:
        gen_cli_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, gen_cli_cfg)

    return cfg


def validate_config(config: DictConfig) -> None:
    """
    Validate configuration to ensure all required fields are set.
    """
    dataset_cfg = config.dataset_config
    
    train_anno = dataset_cfg.train_config.get('anno_path')
    val_anno = dataset_cfg.val_config.get('anno_path')
    train_root = dataset_cfg.train_config.get('data_root')
    val_root = dataset_cfg.val_config.get('data_root')

    if not train_anno or not str(train_anno).strip():
        raise ValueError("Training annotation path not set! Use config or --train-anno-path")
    if not val_anno or not str(val_anno).strip():
        raise ValueError("Validation annotation path not set! Use config or --val-anno-path")
    if not train_root or not str(train_root).strip():
        raise ValueError("Training data root not set! Use config or --dataroot")
    if not val_root or not str(val_root).strip():
        raise ValueError("Validation data root not set! Use config or --dataroot")


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save the config file
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _print_augmentation_config(stage_name, aug_cfg):
    """Print augmentation pipeline summary for a given stage (Train/Val)."""
    entries = []
    if aug_cfg.get("RandomResize", {}).get("enabled", False):
        rr = aug_cfg["RandomResize"]
        interps = ", ".join(rr.get("interpolation", []))
        entries.append(f"RandomResize(interpolation=[{interps}], antialias_prob={rr.get('antialias_prob', 0.5)})")
    else:
        entries.append("Resize(bilinear)")
    entries.append("ToDtype(float32)")

    if aug_cfg.get("ColorJitter", {}).get("enabled", False):
        cj = aug_cfg["ColorJitter"]
        entries.append(
            f"ColorJitter(brightness={cj.get('brightness', 0)}, contrast={cj.get('contrast', 0)}, "
            f"saturation={cj.get('saturation', 0)}, hue={cj.get('hue', 0)})"
        )
    if aug_cfg.get("GaussianBlur", {}).get("enabled", False):
        gb = aug_cfg["GaussianBlur"]
        entries.append(
            f"GaussianBlur(p={gb.get('apply_prob', 0.5)}, kernel={gb.get('kernel_size', 3)}, "
            f"sigma={gb.get('sigma', [0.1, 0.5])})"
        )
    entries.append("Normalize(Default)")

    print(f"\n  [{stage_name} Augmentation Pipeline]")
    for i, entry in enumerate(entries):
        connector = "└─" if i == len(entries) - 1 else "├─"
        print(f"    {connector} {entry}")


def print_config(config: DictConfig) -> None:
    """
    Pretty print configuration.
    """
    print("\n" + "="*80)
    print("Configuration Summary")
    print("="*80)

    ds_cfg = config.dataset_config
    print("\n[Dataset Configuration]")
    print(f"  Dataset: {ds_cfg.dataset}")
    print(f"  Resolution: {ds_cfg.resolution}")
    print(f"  Batch size: {ds_cfg.batch_size}")
    print(f"  Workers: {ds_cfg.workers}")
    print(f"  Frames/side: {ds_cfg.frames_per_side}")
    print(f"  Backend: {ds_cfg.video_backend}")
    if ds_cfg.processor_name_or_path:
        print(f"  Processor: {ds_cfg.processor_name_or_path}")

    print("\n  [Data Paths]")
    print(f"    Train Anno: {ds_cfg.train_config.anno_path}")
    print(f"    Train Root: {ds_cfg.train_config.data_root}")
    print(f"    Val Anno:   {ds_cfg.val_config.anno_path}")
    print(f"    Val Root:   {ds_cfg.val_config.data_root}")

    _print_augmentation_config("Train", OmegaConf.to_container(ds_cfg.train_config, resolve=True).get("augmentation", {}))
    _print_augmentation_config("Val", OmegaConf.to_container(ds_cfg.val_config, resolve=True).get("augmentation", {}))

    md_cfg = config.model_config
    print("\n[Model Configuration]")
    print(f"  Model: {md_cfg.model_name}")
    print(f"  Backbone: {md_cfg.backbone}")
    print(f"  Pretrained: {md_cfg.pretrained}")
    print(f"  Freeze BB: {md_cfg.freeze_backbone}")
    print(f"  Classes: {md_cfg.num_classes}")

    tr_cfg = config.training_config
    print("\n[Training Configuration]")
    print(f"  Epochs: {tr_cfg.epochs}")
    print(f"  Optimizer: {tr_cfg.optimizer} (lr={tr_cfg.learning_rate})")
    print(f"  Scheduler: {tr_cfg.scheduler}")
    print(f"  Mixed Precision: {tr_cfg.amp}")
    print(f"  Hardware: {tr_cfg.num_gpus} GPUs, {tr_cfg.num_nodes} Nodes")
    print(f"  Output: {tr_cfg.output}")

    print("\n" + "="*80 + "\n")


def get_config_with_cli() -> Tuple[Dict[str, Any], argparse.Namespace]:
    """
    Main entry point.
    """
    parser = get_parser()
    args = parser.parse_args()

    cfg = merge_configs(args)

    validate_config(cfg)
    print_config(cfg)

    OmegaConf.resolve(cfg)
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    return config_dict, args
