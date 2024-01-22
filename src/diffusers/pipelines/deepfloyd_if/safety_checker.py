import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPConfig, CLIPVisionModelWithProjection, PreTrainedModel

from ...utils import logging


logger = logging.get_logger(__name__)


class IFSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModelWithProjection(config.vision_config)

        self.p_head = nn.Linear(config.vision_config.projection_dim, 1)
        self.w_head = nn.Linear(config.vision_config.projection_dim, 1)

    @torch.no_grad()
    def forward(self, clip_input, images, p_threshold=0.5, w_threshold=0.5):
        logger.info("Safety checker is off")
        return images, False, False
