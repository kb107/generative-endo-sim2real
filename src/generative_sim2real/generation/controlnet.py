# Adapted from Hugging Face Diffusers (Apache-2.0), revision
# b800f828f3951c4d206e154411bcbfa693cd322e,
# examples/research_projects/pixart/controlnet_pixart_alpha.py.
# Modifications: configurable tiny-test dimensions, correct train() recursion,
# clean-base copying, explicit per-call control scale (no mutable state), validation.
# Gradient checkpointing was removed (upstream did not support it either).
# See UPSTREAM_LICENSE.txt and docs/generator-integration.md.
from typing import Any, Dict, Optional

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import PixArtTransformer2DModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin


class PixArtControlNetAdapterBlock(nn.Module):
    def __init__(
        self,
        block_index,
        # taken from PixArtTransformer2DModel
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = 1152,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_single",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        attention_type: str | None = "default",
    ):
        super().__init__()

        self.block_index = block_index
        self.inner_dim = num_attention_heads * attention_head_dim

        # the first block has a zero before layer
        if self.block_index == 0:
            self.before_proj = nn.Linear(self.inner_dim, self.inner_dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

        self.transformer_block = BasicTransformerBlock(
            self.inner_dim,
            num_attention_heads,
            attention_head_dim,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            attention_bias=attention_bias,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            attention_type=attention_type,
        )

        self.after_proj = nn.Linear(self.inner_dim, self.inner_dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        if self.block_index == 0:
            controlnet_states = self.before_proj(controlnet_states)
            controlnet_states = hidden_states + controlnet_states

        controlnet_states_down = self.transformer_block(
            hidden_states=controlnet_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            added_cond_kwargs=added_cond_kwargs,
            cross_attention_kwargs=cross_attention_kwargs,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            class_labels=None,
        )

        controlnet_states_left = self.after_proj(controlnet_states_down)

        return controlnet_states_left, controlnet_states_down


class PixArtControlNetAdapterModel(ModelMixin, ConfigMixin):
    # N=13, as specified in the paper https://arxiv.org/html/2401.05252v1/#S4 ControlNet-Transformer
    @register_to_config
    def __init__(self, num_layers=13, block_config=None):
        super().__init__()
        self.num_layers = num_layers
        self.controlnet_blocks = nn.ModuleList([
            PixArtControlNetAdapterBlock(block_index=i, **(block_config or {}))
            for i in range(num_layers)
        ])

    @classmethod
    def from_transformer(cls, transformer, num_layers=13):
        if not 0 < num_layers < len(transformer.transformer_blocks):
            raise ValueError("Controller needs 1..(backbone layers - 1) blocks")
        if any("lora_" in n for n, _ in transformer.named_parameters()):
            raise ValueError("Initialize controller from clean base before adding adapters")
        keys = ("num_attention_heads", "attention_head_dim", "dropout",
                "cross_attention_dim", "attention_bias", "activation_fn",
                "num_embeds_ada_norm", "upcast_attention", "norm_type",
                "norm_elementwise_affine", "norm_eps", "attention_type")
        config = {k: transformer.config[k] for k in keys if k in transformer.config}
        control = cls(num_layers=num_layers, block_config=config)
        for i, block in enumerate(control.controlnet_blocks):
            block.transformer_block.load_state_dict(
                transformer.transformer_blocks[i].state_dict(), strict=True)
        return control


class PixArtControlNetTransformerModel(ModelMixin, ConfigMixin):
    def __init__(
        self,
        transformer: PixArtTransformer2DModel,
        controlnet: PixArtControlNetAdapterModel,
        blocks_num=None,
    ):
        super().__init__()

        self.blocks_num = controlnet.num_layers if blocks_num is None else blocks_num
        self.register_to_config(**transformer.config)

        self.transformer = transformer
        self.controlnet = controlnet

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        controlnet_cond: Optional[torch.Tensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        conditioning_scale: float = 1.0,
    ):
        if self.transformer.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError("`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`.")

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size = hidden_states.shape[0]
        height, width = (
            hidden_states.shape[-2] // self.transformer.config.patch_size,
            hidden_states.shape[-1] // self.transformer.config.patch_size,
        )
        hidden_states = self.transformer.pos_embed(hidden_states)

        timestep, embedded_timestep = self.transformer.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        if self.transformer.caption_projection is not None:
            encoder_hidden_states = self.transformer.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        controlnet_states_down = None
        if controlnet_cond is not None:
            controlnet_states_down = self.transformer.pos_embed(controlnet_cond)

        # 2. Blocks
        for block_index, block in enumerate(self.transformer.transformer_blocks):
            if (0 < block_index <= self.blocks_num and controlnet_states_down is not None
                    and conditioning_scale != 0):
                left, controlnet_states_down = self.controlnet.controlnet_blocks[block_index - 1](
                    hidden_states, controlnet_states_down,
                    encoder_hidden_states=encoder_hidden_states, timestep=timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask, encoder_attention_mask=encoder_attention_mask)
                hidden_states = hidden_states + conditioning_scale * left
            hidden_states = block(
                hidden_states, attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask, timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs, class_labels=None)

        # 3. Output
        shift, scale = (
            self.transformer.scale_shift_table[None]
            + embedded_timestep[:, None].to(self.transformer.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.transformer.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
        hidden_states = self.transformer.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(
                -1,
                height,
                width,
                self.transformer.config.patch_size,
                self.transformer.config.patch_size,
                self.transformer.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                -1,
                self.transformer.out_channels,
                height * self.transformer.config.patch_size,
                width * self.transformer.config.patch_size,
            )
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
