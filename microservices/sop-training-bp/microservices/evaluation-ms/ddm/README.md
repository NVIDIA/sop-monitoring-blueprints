# DDM-Net: Enhanced Training Pipeline with PyTorch Lightning

This repository extends DDM-Net (see the original README [here](https://github.com/MCG-NJU/DDM)) by rewriting the training code using PyTorch Lightning for better modularity and scalability. The new implementation is easier to customize and supports distributed/multi-GPU training.

## Installation

### Option 1: Conda Environment
```bash
conda create -n ddm python=3.10
conda activate ddm
pip install -r requirements.txt
# this is for nvdinov2 backbone required cuda extension from tao
cd DDM-Net
pip install -e .
```

### Option 2: Docker Container
If you have successfully built our training BP service, you can easily set up the environment for DDM-Net training using Docker.
```bash
docker run -it --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name ddm-ms-test \
  -v /path/to/microservices/ddm-training-ms/ddm:/workspace/sop-ddm-ftms/ddm \
  -w /workspace/sop-ddm-ftms/ddm \
  ddm_ftms:latest \
  /bin/bash
```

## Dataset Preparation

Please refer to the `dataset/sample/` directory for dataset organization and preparation instructions.

## Training

This repository provides refactored pytorch lightning implementations:

### PyTorch Lightning Training

#### Quick Start

```bash
# Multi-GPU training (You can train on single GPU by simply adjust num-gpus to 1)
bash tools/train_multi_lightning.sh
```

#### Key Features

- ✅ **Lightning 2.x** - Latest PyTorch Lightning framework
- ✅ **Automatic DDP** - No manual distributed setup needed, only need to specify the --num-gpus argument
- ✅ **TOML/YAML Configuration** - Easy experiment management, also can override with CLI arguments
- ✅ **EMA Support** - Exponential Moving Average, and delayed start to have update quicker at the beginning
- ✅ **Auto Checkpointing** - Save best models automatically

#### Key Arguments for CLI

| Argument | YAML Key | Description | Default |
|----------|----------|-------------|---------|
| `--config` | - | Path to YAML config file | None |
| `--backbone` | `model_config.backbone` | Backbone: `resnet50`, `nvdinov2_large`, `nvdinov2_huge` | resnet50 |
| `--pretrained` | `model_config.pretrained` | Path to pretrained weights | None |
| `--freeze-backbone` | `model_config.freeze_backbone` | Freeze backbone weights | False |
| `--train-anno-path` | `dataset_config.train_config.anno_path` | Training annotations (JSON) | Required |
| `--val-anno-path` | `dataset_config.val_config.anno_path` | Validation annotations (JSON) | Required |
| `--dataroot` | `dataset_config.train/val_config.data_root` | Video directory | Required |
| `--batch-size` | `dataset_config.batch_size` | Batch size | 2 |
| `--learning-rate` (or `--lr`) | `training_config.learning_rate` | Learning rate | 0.01 |
| `--optimizer` (or `--opt`) | `training_config.optimizer` | Optimizer: adamw, adam, sgd | adamw |
| `--scheduler` (or `--sched`) | `training_config.scheduler` | LR scheduler: step, cosine, plateau | step |
| `--epochs` | `training_config.epochs` | Training epochs | 100 |
| `--num-gpus` | `training_config.num_gpus` | Number of GPUs | 1 |
| `--amp` | `training_config.amp` | Enable mixed precision | False |
| `--model-ema` | `training_config.model_ema` | Enable EMA | False |
| `--save-visualizations` (or `--save-images`) | `training_config.save_visualizations` | Save visualization images | False |

**Note**: 
* CLI arguments always override YAML settings.
* You can also use TOML configuration files instead of YAML if preferred.
* If you have trouble using TensorBoard to view your training logs, it is recommended to run the following command:
```bash
pip install --upgrade setuptools
```

For complete list of parameters, see:
- `DDM-Net/config/full_config_example.yaml` - Full configuration example
- `DDM-Net/config/config.py` - All parameter definitions
- Run `python DDM-Net/train_sop_lightning.py --help` for CLI help
- We provide further explanation for configs in our training pipeline [here](DDM-Net/config/config_guide.md).