# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os
import sys
from typing import Optional

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.append("../../")
sys.path.append(os.getcwd())

import json
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack

DDM_MODEL_PATH = os.getenv("DDM_MODEL_PATH", "/models/gbed_models/ddm/checkpoint.pth.tar")
FRAMES_PER_SIDE = int(os.getenv("FRAMES_PER_SIDE", "5"))

# sliding windows size = 2 * FRAMES_PER_SIDE + SEQUENCE_BATCH
SEQUENCE_BATCH = int(os.getenv("SEQUENCE_BATCH", "8"))

DDM_MAX_BATCH_SIZE = int(os.getenv("DDM_MAX_BATCH_SIZE", "8"))

# 11 = 5 * 2 + 1
SINGLE_BATCH_SEQUENCE_FRAME_NUMBER = 2 * FRAMES_PER_SIDE + 1

# 18 = 5 * 2 + 8
LONG_WINDOW_SIZE = 2 * FRAMES_PER_SIDE + SEQUENCE_BATCH

try:
    import nvds_action_detector.ds_logger as ds_logger

    logger = ds_logger.get_logger(__name__)
except ImportError:
    logger.debug("######## loading ds_logger failed, Using standard Python logger")
    logger = logging.getLogger(__name__)


class TritonPythonModel:

    def initialize(self, args):
        """
        This function allows
        the model to initialize any state associated with this model.
        """
        start_time = time.time()
        self._model_config = json.loads(args["model_config"])
        logger.info(f"######## self._model_config: {self._model_config}")
        # params = self._model_config["parameters"]
        self._device_id = int(json.loads(args["model_instance_device_id"]))
        self._cuda_device = f"cuda:{self._device_id}"
        self._preprocess_pipeline = None
        # T.Compose(
        #     [
        #         T.Resize((224, 224), antialias=False),
        #         T.ToDtype(torch.float32, scale=True),
        #         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ]
        # )
        self._frames_per_side = FRAMES_PER_SIDE
        from ddm_net import load_ddm_net_model

        self._ddm_net = load_ddm_net_model(
            checkpoint=DDM_MODEL_PATH, frames_per_side=self._frames_per_side, gpu_id=self._device_id
        )
        self.dummy_tensor = torch.randint(
            0, 256, (1, SINGLE_BATCH_SEQUENCE_FRAME_NUMBER, 3, 224, 224), dtype=torch.float32, device=self._cuda_device
        )
        assert self._ddm_net is not None, "######## DDM model initialization failed"
        logger.info(
            f"######## DDM model initialized successfully, initialization time: {time.time() - start_time:.3f}s"
        )

    def execute(self, requests):
        # logger = pb_utils.Logger
        torch.set_default_device(self._cuda_device)
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        logger.debug(f"######## requests numbers: {len(requests)}")
        if len(requests) > 1:
            logger.info(f"######## requests numbers: {len(requests)} more than 1")
        batched_inputs, batch_nums = [], []
        for request in requests:
            # input_0: 1, 3, 11, 224, 224
            input_0 = pb_utils.get_input_tensor_by_name(request, "input_0")
            tensors = from_dlpack(input_0.to_dlpack())
            logger.debug(f"######## original input_0.shape: {tensors.shape}, input_0.device: {tensors.device}")
            # permute the tensor to (1, 11, 3, 224, 224), or (1, 18, 3, 224, 224)
            tensors = torch.permute(tensors, (0, 2, 1, 3, 4))
            if self._preprocess_pipeline is not None:
                with torch.no_grad():
                    preprocessed_tensors = self._preprocess_pipeline(tensors)
                    logger.debug(
                        f"######## preprocessed input_0.shape: {tensors.shape}, input_0.device: {tensors.device}"
                    )
            else:
                preprocessed_tensors = tensors
            # inputs.append(tensors)

            long_window_size = tensors.shape[1]
            assert (
                long_window_size > SINGLE_BATCH_SEQUENCE_FRAME_NUMBER
            ), f"Long window size{long_window_size} less than single batch sequence frame number{SINGLE_BATCH_SEQUENCE_FRAME_NUMBER}"
            seq_batch_size = long_window_size - SINGLE_BATCH_SEQUENCE_FRAME_NUMBER + 1
            for i in range(seq_batch_size):
                window = preprocessed_tensors[:, i : i + SINGLE_BATCH_SEQUENCE_FRAME_NUMBER]
                batched_inputs.append(window)
            batch_nums.append(seq_batch_size)

        # add dummy tensor as workaround batch_size = 1
        if len(batched_inputs) == 1:
            batched_inputs.append(self.dummy_tensor)
            batch_nums.append(1)

        # inputs = torch.stack(inputs, dim=0)
        batched_inputs = torch.cat(batched_inputs, dim=0)

        # Make tensor contiguous if needed (this is necessary after permute)
        if not batched_inputs.is_contiguous():
            batched_inputs = batched_inputs.contiguous()

        # Validate tensor properties
        if not batched_inputs.is_contiguous():
            raise RuntimeError(
                f"Tensor is not contiguous after operations. Shape: {batched_inputs.shape}, Device: {batched_inputs.device}"
            )

        logger.debug(
            f"######## batched_inputs shape: {batched_inputs.shape}, device: {batched_inputs.device}, contiguous: {batched_inputs.is_contiguous()}"
        )

        with torch.no_grad():
            # logger.debug(f"######## _tensor_generate before preprocess tensors.shape: {tensors.shape}")
            # batched_tensors = self._preprocess_pipeline(batched_inputs)
            batched_outputs, _, _ = self._ddm_net(batched_inputs)
            # logger.debug(f"######## ddm_net output: batched_outputs: {batched_outputs}")
            if isinstance(batched_outputs, (tuple, list)):
                batched_outputs = batched_outputs[-1]
            # logger.debug(f"######## the last output of ddm_net, shape: {batched_outputs.shape}, data: {batched_outputs}")
            # window_scores shape: (seq_batch_size,), (8)
            window_scores = F.softmax(batched_outputs, dim=1)[:, 1]
            logger.debug(f"######## window_scores, shape: {window_scores.shape}, data: {window_scores}")
        # window_scores shape: [8] -> we need to keep it as 2D tensor [seq_batch_size, 1] for proper de-batching
        window_scores = window_scores.unsqueeze(1)  # Shape: [seq_batch_size2, 1]
        logger.debug(f"######## window_scores after unsqueeze, shape: {window_scores.shape}, data: {window_scores}")

        # Split into individual outputs for each request
        # len(outputs) = len(batch_nums)
        # outputs = [tensor.unsqueeze(0) for tensor in torch.unbind(window_scores, dim=0)]  # Each becomes [seq_batch_size, 1]
        outputs = [x for x in torch.split(window_scores, batch_nums)]
        logger.debug(f"######## ddm decoupled outputs size: {len(outputs)}")
        responses = []

        # extra dummy tensor not counted in requests
        for i, request in enumerate(requests):
            output = outputs[i]
            seq_batch_size = batch_nums[i]
            output = output.reshape(-1, seq_batch_size)
            logger.debug(f"######## output_0: shape: {output.shape}, data: {output}")
            out_tensor = pb_utils.Tensor("output_0", output.cpu().numpy())
            # out_tensor = pb_utils.Tensor.from_dlpack("output_0", to_dlpack(output))
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
        return responses

    def finalize(self):
        logger.info("Cleaning up ddm model...")
