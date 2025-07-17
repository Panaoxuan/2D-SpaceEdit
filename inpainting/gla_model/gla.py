import warnings
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.cache_utils import Cache, DynamicCache
from diffusers.utils import logging

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
from fla.models.gla.modeling_gla import GLAMLP, GLABlock, GatedLinearAttention, GLAConfig
from fla.modules import FusedCrossEntropyLoss, RMSNorm, ShortConvolution, FusedRMSNormSwishGate

import torch._dynamo
torch._dynamo.config.suppress_errors = True

logger = logging.get_logger(__name__)


class GatedLinearAttention_ForwardWrapper(GatedLinearAttention):
    def forward(self,*args,**kwargs):
        return super(GatedLinearAttention_ForwardWrapper, self).forward(*args,**kwargs)[0]


DEFAULT_GLA_CONFIG = dict(
    mode = 'chunk',
    hidden_size = 2048,
    expand_k = 0.5,
    expand_v = 1.0,
    num_heads = 4,
    num_kv_heads = None,
    feature_map = None,
    use_short_conv = False,
    conv_size = 4,
    conv_bias = False,
    use_output_gate = True,
    gate_fn = 'swish',
    elementwise_affine = True,
    norm_eps = 1e-6,
    gate_logit_normalizer = 16,
    gate_low_rank_dim = 16,
    clamp_min = None,
    fuse_norm = True,
    layer_idx = None,
)


