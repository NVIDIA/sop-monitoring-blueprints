# Alternative initialization strategies for resnetGEBD

import torch
import torch.nn as nn
import math
from timm.layers import trunc_normal_

def init_weights_xavier(m):
    """Xavier/Glorot initialization - good for tanh activation"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_weights_kaiming(m):
    """Kaiming/He initialization - good for ReLU activation"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_weights_scaled(m, scale=1.0):
    """Scaled initialization for fine-tuning scenarios"""
    if isinstance(m, nn.Linear):
        std = 0.02 * scale  # Smaller scale for fine-tuning
        trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        std = math.sqrt(2.0 / fan_out) * scale
        m.weight.data.normal_(0, std)
        if m.bias is not None:
            m.bias.data.zero_()

def init_weights_lora_style(m, r=4):
    """LoRA-style initialization for adapter modules"""
    if isinstance(m, nn.Linear):
        # Initialize with smaller magnitude for adapter layers
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def selective_init_by_module_name(model, init_fn_map):
    """
    Apply different initialization based on module names

    Example:
        init_fn_map = {
            'ddm_encoder': init_weights_kaiming,
            'transformer': init_weights_xavier,
            'proj': init_weights_scaled,
            'default': init_weights_standard
        }
    """
    for name, module in model.named_modules():
        # Skip backbone
        if 'backbone' in name:
            continue

        # Find matching initialization function
        init_fn = init_fn_map.get('default', init_weights_standard)
        for key, fn in init_fn_map.items():
            if key in name and key != 'default':
                init_fn = fn
                break

        # Apply initialization to leaf modules only
        if len(list(module.children())) == 0:
            init_fn(module)

def init_weights_standard(m):
    """Standard initialization (current implementation)"""
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
