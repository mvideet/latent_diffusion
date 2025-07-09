"""
FiLM-enhanced LLaDA model for latent conditioning
"""
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union
import sys
import os
import math

# Ensure llada_8b can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from llada_8b.modeling_llada import (
    LLaDAModel, LLaDABlock, LLaDASequentialBlock, LLaDALlamaBlock,
    LLaDABlockGroup, LLaDAOutput, BufferCache, Dropout, LayerNormBase
)
from llada_8b.configuration_llada import LLaDAConfig, ModelConfig
from transformers import PreTrainedModel

class FiLMAdapter(nn.Module):
    """
    Feature-wise Linear Modulation adapter for conditioning on latents
    """
    def __init__(self, feature_dim: int, latent_dim: int, hidden_dim=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        
        if hidden_dim is None:
            hidden_dim = latent_dim // 2
            
        # Network to generate gamma (scale) and beta (shift) from latents
        self.film_generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim * 2)  # *2 for gamma and beta
        )
        
        # Initialize to identity transformation
        # This initialization sets the FiLM adapter to perform an identity transformation at the start,
        # so the model's initial behavior is unchanged by the FiLM layer. This does not guarantee
        # that the model will always do better, but it does ensure that, at initialization,
        # the FiLM adapters do not harm performance. The model must still learn to use the FiLM parameters
        # to improve over the base model.
        with torch.no_grad():
            # Initialize gamma to 1, beta to 0
            self.film_generator[-1].weight.fill_(0.0)
            self.film_generator[-1].bias[:feature_dim].fill_(1.0)  # gamma = 1
            self.film_generator[-1].bias[feature_dim:].fill_(0.0)  # beta = 0
    
    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning
        Args:
            x: Input features [batch, seq_len, feature_dim]
            latent: Conditioning latent [batch, latent_dim]
        """
        batch_size = x.size(0)
        
        # Convert latent to match x dtype for mixed precision compatibility
        latent = latent.to(x.dtype)
        
        # Generate gamma and beta from latent
        film_params = self.film_generator(latent)  # [batch, feature_dim * 2]
        gamma, beta = film_params.chunk(2, dim=-1)  # Each [batch, feature_dim]
        
        # Reshape for broadcasting: [batch, 1, feature_dim]
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        # Apply FiLM: gamma * x + beta
        return gamma * x + beta


class FiLMSequentialBlock(LLaDASequentialBlock):
    """
    Sequential block with FiLM adapters - properly inheriting from LLaDASequentialBlock
    """
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache, latent_dim: int):
        # Call parent constructor first - this sets up all the base components
        super().__init__(layer_id, config, cache)
        
        # Add FiLM adapters after attention and MLP
        self.film_post_attn = FiLMAdapter(config.d_model, latent_dim)
        self.film_post_mlp = FiLMAdapter(config.d_model, latent_dim)
        
        # Store latent dim for later use
        self.latent_dim = latent_dim

    def reset_parameters(self):
        # Call parent reset_parameters first
        super().reset_parameters()
        
        # FiLM adapters are already initialized to identity in their __init__
        # so no additional reset needed

    def forward(
        self,
        x: torch.Tensor,
        latent: torch.Tensor,  # Added latent parameter
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with FiLM conditioning
        """
                 # Get query, key, value projections (same as parent)
        if self._activation_checkpoint_fn is not None:
             q, k, v = self.att_proj(self._activation_checkpoint_fn(self.attn_norm, x)).split(
                 self.fused_dims, dim=-1
             )
        else:
             q, k, v = self.att_proj(self.attn_norm(x)).split(self.fused_dims, dim=-1)

        # Get attention scores (same as parent)
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

        # Apply FiLM conditioning AFTER attention
        att = self.film_post_attn(att, latent)
        
        # Add attention scores (same as parent)
        x = x + self.dropout(att)

        # MLP computation (same as parent)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x = self.ff_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = self.ff_out(x)
        
        # Apply FiLM conditioning AFTER MLP
        x = self.film_post_mlp(x, latent)
        
        # Final dropout and residual (same as parent)
        x = self.dropout(x)
        x = og_x + x

        return x, cache


class FiLMLlamaBlock(LLaDALlamaBlock):
    """
    Llama block with FiLM adapters - properly inheriting from LLaDALlamaBlock
    """
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache, latent_dim: int):
        # Call parent constructor first
        super().__init__(layer_id, config, cache)
        
        # Add FiLM adapters
        self.film_post_attn = FiLMAdapter(config.d_model, latent_dim)
        self.film_post_mlp = FiLMAdapter(config.d_model, latent_dim)
        
        # Store latent dim
        self.latent_dim = latent_dim

    def reset_parameters(self):
        # Call parent reset_parameters first
        super().reset_parameters()
        
        # FiLM adapters already initialized to identity

    def forward(
        self,
        x: torch.Tensor,
        latent: torch.Tensor,  # Added latent parameter
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with FiLM conditioning
        """
        # Attention computation (same as parent)
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        # Get attention scores (same as parent)
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

        # Apply FiLM conditioning AFTER attention
        att = self.film_post_attn(att, latent)
        
        # Add attention scores (same as parent)
        x = x + self.dropout(att)

        # MLP computation (same as parent)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        
        x, x_up = self.ff_proj(x), self.up_proj(x)  # Llama-style SwiGLU
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = x * x_up  # SwiGLU gating
        x = self.ff_out(x)
        
        # Apply FiLM conditioning AFTER MLP
        x = self.film_post_mlp(x, latent)
        
        # Final dropout and residual (same as parent)
        x = self.dropout(x)
        x = og_x + x

        return x, cache


class FiLMBlockGroup(nn.ModuleList):
    """
    Block group with FiLM support - inheriting from nn.ModuleList like LLaDABlockGroup
    """
    def __init__(self, config: ModelConfig, layer_offset: int, modules=None):
        # Initialize the parent nn.ModuleList with the provided modules (blocks)
        super().__init__(modules)
        # Store the model configuration
        self.config = config
        # Store the offset for this group of layers (used for global layer indexing)
        self.layer_offset = layer_offset
        # Store the activation checkpointing strategy (None by default)
        self.activation_checkpointing_strategy = None
        # Import the activation checkpoint function factory
        from llada_8b.modeling_llada import activation_checkpoint_function
        # Create the activation checkpoint function for this config
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config)

    def forward(self, x, latent, attention_bias=None, layers_past=None, use_cache=False):
        # If use_cache is True, prepare a list to collect attention key/values; else None
        attn_key_values = [] if use_cache else None
        # Iterate over each block in this group
        for block_idx, block in enumerate(self):
            # Get the past key/value for this block if provided, else None
            layer_past = None if layers_past is None else layers_past[block_idx]
            # Adjust block_idx to be global (add group offset)
            block_idx += self.layer_offset
            
            # Determine if this block should use activation checkpointing
            if (
                (self.activation_checkpointing_strategy == "whole_layer")
                or (self.activation_checkpointing_strategy == "one_in_two" and block_idx % 2 == 0)
                # Add other checkpointing strategies as needed
            ):
                # If checkpointing, call the checkpointed function for this block
                x, cache = self._activation_checkpoint_fn(  # type: ignore
                    block, x, latent, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache
                )
            else:
                # Otherwise, call the block directly (no checkpointing)
                x, cache = block(x, latent, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
            
            # If collecting attention key/values, append the cache for this block
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
        # Return the final output tensor and the list of attention key/values (if requested)
        return x, attn_key_values

    def reset_parameters(self):
        # Call reset_parameters on each block in the group
        for block in self:
            block.reset_parameters()

    def set_activation_checkpointing(self, strategy):
        # Set the activation checkpointing strategy for this group
        self.activation_checkpointing_strategy = strategy
        # Propagate the strategy to all blocks in the group
        for block in self:
            block.set_activation_checkpointing(strategy)


class LLaDAModelWithFiLM(LLaDAModel):
    """
    LLaDA model with FiLM adapters for latent conditioning
    Properly inherits from LLaDAModel
    """
    def __init__(self, config: ModelConfig, latent_dim: int, init_params: bool = True):
        # Store latent_dim before calling parent constructor
        self.latent_dim = latent_dim
        
        # Call parent constructor but don't initialize blocks yet
        nn.Module.__init__(self)  # Skip LLaDAModel.__init__ to customize block creation
        
        self.config = config
        self.__cache = BufferCache()

        # Copy all the validation and setup from parent __init__
        if self.config.alibi and self.config.flash_attention:
            raise Exception("ALiBi is currently not supported with FlashAttention")

        if self.config.alibi and self.config.rope:
            raise Exception("ALiBi and RoPE are mutually exclusive")

        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise Exception("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings
                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        self.activation_checkpointing_strategy = None
        from llada_8b.modeling_llada import activation_checkpoint_function
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config)

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise Exception("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

        # Initialize transformer components (same as parent)
        from llada_8b.modeling_llada import LayerNorm
        # "emb_drop" is embedding dropout: a Dropout layer applied to the token embeddings after lookup,
        # before they are passed into the transformer blocks. This helps regularize the model and prevent
        # overfitting by randomly zeroing out elements of the embedding vectors during training.
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                "emb_drop": Dropout(config.embedding_dropout),  # embedding dropout
                "ln_f": LayerNorm.build(config),
            }
        )

        # Create FiLM-enabled blocks instead of regular blocks
        blocks = []
        for i in range(config.n_layers):
            if config.block_type == "sequential":
                block = FiLMSequentialBlock(i, config, self.__cache, latent_dim)
            elif config.block_type == "llama":
                block = FiLMLlamaBlock(i, config, self.__cache, latent_dim)
            else:
                raise NotImplementedError(f"FiLM not implemented for block type: {config.block_type}")
            blocks.append(block)

        if self.config.block_group_size > 1:
            block_groups = [
                FiLMBlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        # Add positional embeddings if needed (same as parent)
        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )
        
        # Add output projection if not weight tying (same as parent)
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )

        # Initialize parameters if requested
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops = None

        # Warm up cache (same as parent)
        if self.config.alibi:
            from llada_8b.modeling_llada import get_causal_attention_bias, _non_meta_init_device
            get_causal_attention_bias(self.__cache, config.max_sequence_length, _non_meta_init_device(config))
            self.get_alibi_attention_bias(config.max_sequence_length, _non_meta_init_device(config))

    def forward(
        self,
        input_ids: torch.LongTensor,
        latent: torch.Tensor,  # New required parameter for conditioning
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
    ) -> LLaDAOutput:
        """
        Forward pass with latent conditioning
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            latent: Conditioning latent [batch, latent_dim]
            ... (other args same as parent)
        """
        # Same validation as parent
        assert not self.config.alibi, "Alibi length extrapolation is not supported for MDM."
        assert self.config.rope, "Rope must be used in Llama-Encoder for MDM."
        assert (past_key_values is None and not use_cache), "The kvcache is not suppotred for MDM."
        
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        
        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        # Get embeddings (same as parent)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings

        if self.config.input_emb_norm:
            x = x * (self.config.d_model**0.5)

        if not (self.config.alibi or self.config.rope):
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            pos_emb = self.transformer.wpe(pos)
            x = pos_emb + x

        x = self.transformer.emb_drop(x)

        # Handle attention masks and bias (same as parent logic)
        if attention_mask is not None and 0.0 in attention_mask:
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
        else:
            attention_mask = None

        if (
            attention_bias is not None
            or attention_mask is not None
            or self.config.alibi
            or past_key_values is not None
        ):
            if attention_bias is None and self.config.alibi:
                from llada_8b.modeling_llada import get_causal_attention_bias
                attention_bias = get_causal_attention_bias(
                    self.__cache, past_length + seq_len, x.device
                ) + self.get_alibi_attention_bias(past_length + seq_len, x.device)
            elif attention_bias is None:
                from llada_8b.modeling_llada import get_causal_attention_bias
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                from llada_8b.modeling_llada import ensure_finite_
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values = [] if use_cache else None
        all_hidden_states = []

        # Apply FiLM blocks with latent conditioning
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                
                # Pass latent to each block
                if (
                    hasattr(self, 'activation_checkpointing_strategy') and
                    self.activation_checkpointing_strategy is not None
                ):
                    # Handle checkpointing with latent
                    x, cache = self._activation_checkpoint_fn(
                        block, x, latent, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache
                    )
                else:
                    x, cache = block(x, latent, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
                
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    all_hidden_states.append(x)
                
                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                    ]
                )
                x, cache = block_group(x, latent, attention_bias=attention_bias, layers_past=layers_past, use_cache=use_cache)
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)

        if last_logits_only:
            x = x[:, -1, :].unsqueeze(1)

        x = self.transformer.ln_f(x)
        if output_hidden_states:
            all_hidden_states.append(x)

        # Get logits (same as parent)
        if self.config.weight_tying:
            logits = nn.functional.linear(x, self.transformer.wte.weight, None)
        else:
            logits = self.transformer.ff_out(x)
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        return LLaDAOutput(
            logits=logits,  # type: ignore
            attn_key_values=attn_key_values, 
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None  # type: ignore
        )


class LLaDAModelLMWithFiLM(PreTrainedModel):
    """
    HuggingFace wrapper for FiLM-enabled LLaDA model
    """
    config_class = LLaDAConfig
    base_model_prefix = "model"
    _no_split_modules = ["FiLMSequentialBlock", "FiLMLlamaBlock"]
    
    def __init__(self, config: LLaDAConfig, latent_dim: int, init_params: bool = False):
        super().__init__(config)
        self.latent_dim = latent_dim
        
        # Convert LLaDAConfig to ModelConfig
        from llada_8b.modeling_llada import create_model_config_from_pretrained_config
        model_config = create_model_config_from_pretrained_config(config)
        model_config.init_device = "cpu"  # Start on CPU like parent
        
        self.model = LLaDAModelWithFiLM(model_config, latent_dim, init_params=init_params)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        latent: Optional[torch.FloatTensor] = None,  # Required conditioning latent
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position = None,
    ):
        if latent is None:
            raise ValueError("latent conditioning vector is required for FiLM model")
            
        if use_cache is None:
            use_cache = self.config.use_cache
            
        if output_attentions:
            raise ValueError("output_attentions is not yet supported in LLaDA")
            
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model.forward(
            input_ids=input_ids,
            latent=latent,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,  # type: ignore
            use_cache=use_cache or False,
            output_hidden_states=output_hidden_states,
        )
        
        logits = outputs.logits
        hidden_states = outputs.hidden_states
        
        loss = None
        if labels is not None:
            import warnings
            warnings.warn("Loss calculation not implemented for FiLM model", UserWarning)
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
            
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.attn_key_values,  # type: ignore
            hidden_states=hidden_states,  # type: ignore
        )
    
    def get_input_embeddings(self):
        return self.model.transformer.wte
    
    def set_input_embeddings(self, value):
        self.model.transformer.wte = value


# Helper function for loading
def load_film_model(base_model_path: str, latent_dim: int, device: str = 'cuda'):
    """
    Loads a LLaDA model with FiLM adapters, initializing it with weights from a base model.
    This is needed to:
      - Reuse the pretrained weights frosm a base LLaDA model for all layers that are compatible.
      - Initialize new FiLM adapter parameters (which are not present in the base model).
      - Ensure the resulting model is ready for inference or further fine-tuning with FiLM conditioning.
    """
    from transformers import AutoModel
    # Load the configuration for the base model (must be LLaDAConfig, not generic PretrainedConfig)
    config = LLaDAConfig.from_pretrained(base_model_path)  # type: ignore

    # Create a new FiLM-augmented model (uninitialized weights)
    print("Creating FiLM model...")
    film_model = LLaDAModelLMWithFiLM(config, latent_dim=latent_dim, init_params=False)

    # Load the original base model to extract its pretrained weights
    print("Loading original model...")
    original_model = AutoModel.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # Transfer all compatible weights from the base model to the FiLM model
    print("Transferring weights...")
    original_state_dict = original_model.state_dict()
    film_state_dict = film_model.state_dict()

    transferred = 0
    skipped = 0

    for name, param in original_state_dict.items():
        if name in film_state_dict:
            if film_state_dict[name].shape == param.shape:
                film_state_dict[name].copy_(param)
                transferred += 1
            else:
                print(f"Shape mismatch for {name}: {film_state_dict[name].shape} vs {param.shape}")
                skipped += 1
        else:
            print(f"Key not found in FiLM model: {name}")
            skipped += 1

    # Count the number of parameters that are unique to the FiLM adapters
    film_only_params = 0
    for name, param in film_state_dict.items():
        if 'film_' in name:
            film_only_params += param.numel()

    print(f"Transferred: {transferred} layers")
    print(f"Skipped: {skipped} layers")
    print(f"New FiLM parameters: {film_only_params:,}")

    # Load the updated state dict into the FiLM model
    film_model.load_state_dict(film_state_dict, strict=True)

    # Move the model to the specified device, convert to bfloat16 for memory efficiency, and set to eval mode
    print("Converting model to bfloat16 and moving to device...")
    return film_model.to(torch.device(device)).to(torch.bfloat16).eval()