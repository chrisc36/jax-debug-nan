# Copyright 2022 The T5X Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""T5.1.1 Transformer model."""
from typing import Any, Callable, Iterable, Optional, Sequence

import jax

from t5x.losses import cross_entropy_with_logits

from t5x import layers

PyTreeDef = type(jax.tree_util.tree_structure(None))
from flax import linen as nn, traverse_util, struct
import jax.numpy as jnp


Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]

Initializer = Callable[[PRNGKey, Shape, DType], Array]


@struct.dataclass
class T5Config:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  # Activation dtypes.
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  head_dim: int = 64
  mlp_dim: int = 2048
  # Activation functions are retrieved from Flax.
  mlp_activations: Sequence[str] = ('relu',)
  dropout_rate: float = 0.0
  # the embedding weights are used in the decoder output layer.
  logits_via_embedding: bool = False
  # Whether to accumulate attention logits in float32 regardless of dtype.
  float32_attention_logits: bool = False
  num_seg_emb: int = 2
  zero_masked_embedding: bool = False
  encoder_max_length = 576 + 256
  decoder_max_length = 512


class EncoderLayer(nn.Module):
  """Transformer encoder layer."""
  config: T5Config

  @nn.compact
  def __call__(self, inputs, encoder_bias, abs_pos_bias, encoder_mask=None, deterministic=False):
    cfg = self.config

    # Attention block.
    assert inputs.ndim == 3
    x = layers.LayerNorm(
      dtype=cfg.dtype, name='pre_attention_layer_norm')(
      inputs)
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    x = layers.MultiHeadDotProductAttention(
      num_heads=cfg.num_heads,
      dtype=cfg.dtype,
      head_dim=cfg.head_dim,
      dropout_rate=cfg.dropout_rate,
      float32_logits=cfg.float32_attention_logits,
      name='attention')(
      x, x, encoder_mask, encoder_bias, abs_pos_bias, deterministic=deterministic)

    x = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      x, deterministic=deterministic)

    x = x + inputs

    # MLP block.
    y = layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(x)
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    y = layers.MlpBlock(
      intermediate_dim=cfg.mlp_dim,
      activations=cfg.mlp_activations,
      intermediate_dropout_rate=cfg.dropout_rate,
      dtype=cfg.dtype,
      name='mlp',
    )(y, deterministic=deterministic)

    y = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      y, deterministic=deterministic)
    y = y + x
    return y


class DecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""
  config: T5Config

  @nn.compact
  def __call__(self,
               inputs,
               encoded,
               self_abs_pos_bias,
               cross_abs_pos_bias,
               decoder_mask=None,
               hidden_layer_mask=None,
               encoder_decoder_mask=None,
               deterministic=False,
               decode=False,
               decoder_bias=None,
               debug=False):
    to_print = {}
    cfg = self.config

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    x = layers.LayerNorm(
      dtype=cfg.dtype, name='pre_self_attention_layer_norm')(
      inputs)
    # Self-attention block
    # *0 fixes
    # (+1, * (x == rare)
    to_print["dot-in"] = x.max()
    x = layers.MultiHeadDotProductAttention(
      num_heads=cfg.num_heads,
      dtype=cfg.dtype,
      head_dim=cfg.head_dim,
      dropout_rate=cfg.dropout_rate,
      float32_logits=cfg.float32_attention_logits,
      name='self_attention')(
      x,
      x,
      decoder_mask,
      decoder_bias,
      self_abs_pos_bias,
      deterministic=deterministic,
      decode=decode,
      debug=debug)
    if debug:
      x, dbg_info = x
      to_print.update({"dot/"+k: v for k, v in dbg_info.items()})
    to_print["x1"] = x.max()

    x = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      x, deterministic=deterministic)

    x = x + inputs
    to_print["x2"] = x.max()

    # Encoder-Decoder block.
    y = layers.LayerNorm(
      dtype=cfg.dtype, name='pre_cross_attention_layer_norm')(
      x)
    if hidden_layer_mask is not None:
      y = y * hidden_layer_mask

    to_print["y"] = x.max()

    to_print["cdot-in"] = y.max()
    y = layers.MultiHeadDotProductAttention(
      num_heads=cfg.num_heads,
      dtype=cfg.dtype,
      head_dim=cfg.head_dim,
      dropout_rate=cfg.dropout_rate,
      float32_logits=cfg.float32_attention_logits,
      name='encoder_decoder_attention')(
      y,
      encoded,
      encoder_decoder_mask,
      None,
      cross_abs_pos_bias,
      deterministic=deterministic,
      debug=False)
    to_print["cdot-out"] = y.max()

    y = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      y, deterministic=deterministic)

    y = y + x

    # MLP block.
    z = layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(y)
    z = layers.MlpBlock(
      intermediate_dim=cfg.mlp_dim,
      activations=cfg.mlp_activations,
      intermediate_dropout_rate=cfg.dropout_rate,
      dtype=cfg.dtype,
      name='mlp',
    )(z, deterministic=deterministic)
    z = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      z, deterministic=deterministic)
    z = z + y

    if not debug:
      to_print.clear()
    return z, to_print


class Encoder(nn.Module):
  """A stack of encoder layers."""
  config: T5Config
  shared_embedding: nn.Module

  def setup(self):
    cfg = self.config
    self.segment_embedding = layers.Embed(
      num_embeddings=cfg.num_seg_emb,
      features=cfg.emb_dim,
      dtype=cfg.dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=nn.initializers.normal(stddev=1.0),
      one_hot=True,
      name='segment_embedding')

    self.positon_embedding = layers.Embed(
      num_embeddings=cfg.encoder_max_length,
      features=cfg.emb_dim,
      dtype=cfg.dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=nn.initializers.normal(stddev=1.0),
      one_hot=True,
      name='position_embedding')

  @nn.compact
  def __call__(
      self,
      embed,
      position_embed,
      mask,
      rel_attention,
      segment_ids=None,
      deterministic=False
  ):
    cfg = self.config
    embed = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      embed, deterministic=deterministic)
    embed = embed.astype(cfg.dtype)

    position_embedding = layers.LayerNorm(
      dtype=cfg.dtype, name='pe_pre_ln')(position_embed)

    # get absolute position bias.
    pos_q = layers.DenseGeneral(
      features=(cfg.num_heads, cfg.head_dim),
      dtype=cfg.dtype,
      kernel_axes=('embed', 'joined_kv'),
      name='position_q_linear',
    )(position_embedding)

    pos_k = layers.DenseGeneral(
      features=(cfg.num_heads, cfg.head_dim),
      dtype=cfg.dtype,
      kernel_axes=('embed', 'joined_kv'),
      name='position_k_linear',
    )(position_embedding)

    mask = layers.make_attention_mask(mask, mask, dtype=cfg.dtype)
    if segment_ids is not None:
      # Only attend between items belonging to the same segment
      mask = mask * jnp.expand_dims(segment_ids == segment_ids[:, None, :], 1)

    pos_scaling = float(cfg.emb_dim / cfg.num_heads) ** -0.5
    abs_pos_bias = jnp.einsum('bqhd,bkhd->bhqk', pos_q, pos_k) * pos_scaling

    for lyr in range(cfg.num_encoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      embed = EncoderLayer(
        config=cfg,
        name=f'layers_{lyr}')(embed, rel_attention, abs_pos_bias, mask, deterministic)

    embed = layers.LayerNorm(dtype=cfg.dtype, name='encoder_norm')(embed)
    embed = nn.Dropout(rate=cfg.dropout_rate)(embed, deterministic=deterministic)
    return embed, position_embedding


class Decoder(nn.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""
  config: T5Config
  shared_embedding: nn.Module

  @nn.compact
  def __call__(self,
               encoded,
               decoder_inputs,
               decoder_positions=None,
               decoder_segments=None,
               decoder_attn_mask=None,
               encoder_decoder_mask=None,
               deterministic=False,
               decode=False,
               decoder_bias=None,
               text_decoder_positions=None,
               cur_index=None):
    cfg = self.config
    to_print = {}

    assert decoder_inputs.ndim == 2  # [batch, len]
    encoded, encoder_position_embedding = encoded

    # [batch, length] -> [batch, length, emb_dim]
    y = self.shared_embedding(decoder_inputs.astype('int32'))

    position_embedding = layers.Embed(
      num_embeddings=cfg.decoder_max_length,
      features=cfg.emb_dim,
      dtype=cfg.dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=nn.initializers.normal(stddev=1.0),
      one_hot=True,
      name='position_embedding')(decoder_positions)

    if cur_index is None:
      y += position_embedding
    else:
      y += position_embedding[:,cur_index][:,None,:]

    y += layers.Embed(
      num_embeddings=cfg.num_seg_emb,
      features=cfg.emb_dim,
      dtype=cfg.dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=nn.initializers.normal(stddev=1.0),
      one_hot=True,
      name='segments_embedding')(decoder_segments)

    y = layers.LayerNorm(dtype=cfg.dtype, name='pre_ln')(y)

    position_embedding = layers.LayerNorm(
      dtype=cfg.dtype, name='pe_pre_ln')(position_embedding)

    # get absolute position bias.
    self_pos_q = layers.DenseGeneral(
      features=(cfg.num_heads, cfg.head_dim),
      dtype=cfg.dtype,
      kernel_axes=('embed', 'joined_kv'),
      name='self_position_q_linear',
    )(position_embedding)

    self_pos_k = layers.DenseGeneral(
      features=(cfg.num_heads, cfg.head_dim),
      dtype=cfg.dtype,
      kernel_axes=('embed', 'joined_kv'),
      name='self_position_k_linear',
    )(position_embedding)

    pos_scaling = float(cfg.emb_dim / cfg.num_heads) ** -0.5
    self_abs_pos_bias = jnp.einsum('bqhd,bkhd->bhqk', self_pos_q, self_pos_k) * pos_scaling

    # get absolute position bias.
    cross_pos_q = layers.DenseGeneral(
      features=(cfg.num_heads, cfg.head_dim),
      dtype=cfg.dtype,
      kernel_axes=('embed', 'joined_kv'),
      name='cross_position_q_linear',
    )(position_embedding)

    cross_pos_k = layers.DenseGeneral(
      features=(cfg.num_heads, cfg.head_dim),
      dtype=cfg.dtype,
      kernel_axes=('embed', 'joined_kv'),
      name='cross_position_k_linear',
    )(encoder_position_embedding)

    cross_abs_pos_bias = jnp.einsum('bqhd,bkhd->bhqk', cross_pos_q, cross_pos_k) * pos_scaling

    y = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    masked_tokens = jnp.any(decoder_attn_mask == 1, axis=(3, 1))
    masked_tokens = jnp.expand_dims(masked_tokens, 2)

    if self.config.zero_masked_embedding:
      y = y * masked_tokens

    for lyr in range(cfg.num_decoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      y, layer_out = DecoderLayer(
        config=cfg,
        name=f'layers_{lyr}')(
        y,
        encoded,
        self_abs_pos_bias,
        cross_abs_pos_bias,
        hidden_layer_mask=masked_tokens,
        decoder_mask=decoder_attn_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        deterministic=deterministic,
        decode=decode,
        decoder_bias=decoder_bias,
        debug=lyr >= 20
      )
      for k, v in layer_out.items():
        to_print[f"L{lyr}/{k}"] = v

      if self.config.zero_masked_embedding:
        y = y * masked_tokens

    y = layers.LayerNorm(dtype=cfg.dtype, name='decoder_norm')(y)
    y = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      y, deterministic=deterministic)

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.shared_embedding.attend(y)
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = layers.DenseGeneral(
        cfg.vocab_size,
        dtype=jnp.float32,  # Use float32 for stabiliity.
        kernel_axes=('embed', 'vocab'),
        name='logits_dense')(y)
    return logits, to_print


class Transformer(nn.Module):
  config: T5Config

  def setup(self):
    cfg = self.config

    self.shared_embedding = layers.Embed(
      num_embeddings=cfg.vocab_size,
      features=cfg.emb_dim,
      dtype=cfg.dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=nn.initializers.normal(stddev=1.0),
      one_hot=True,
      name='token_embedder')

    self.encoder = Encoder(
      config=cfg,
      shared_embedding=self.shared_embedding,
    )
    self.decoder = Decoder(
      config=cfg,
      shared_embedding=self.shared_embedding)

  @nn.compact
  def __call__(
      self,
      features,
      input_seq_len: int=512,
      enable_dropout: bool = True,
  ):
    cfg = self.config
    features = traverse_util.unflatten_dict(features, sep="/")
    seq_len = input_seq_len

    dim = self.config.emb_dim

    # Input sequence doesn't seem to matter, so just use empty values
    batch = features["target_input_tokens"].shape[0]
    input_mask = jnp.ones((batch, seq_len), dtype=jnp.int32)
    embed, position = self.encoder(
      jnp.zeros((batch, seq_len, dim), dtype=self.config.dtype),
      jnp.zeros((batch, seq_len, dim), dtype=self.config.dtype),
      input_mask,
      jnp.zeros((batch, self.config.num_heads, seq_len, seq_len), dtype=self.config.dtype),
      deterministic=not enable_dropout
    )

    seq_len = features["target_input_tokens"].shape[1]
    target_mask = features["target_mask"]

    encoder_decoder_mask = layers.make_attention_mask(target_mask, input_mask, dtype=cfg.dtype)
    decoder_attn_mask = layers.make_decoder_mask(
      decoder_target_tokens=target_mask,
      dtype=cfg.dtype)

    # Do the decoding
    # Target
    logits, dbg_info = self.decoder(
      (embed, position),
      decoder_positions=features["target_position_id"],
      decoder_segments=features["target_modality_id"],
      decoder_inputs=features["target_input_tokens"],
      decoder_attn_mask=decoder_attn_mask,
      encoder_decoder_mask=encoder_decoder_mask,
      deterministic=not enable_dropout,
      decode=False,
      decoder_bias=jnp.zeros((batch, self.config.num_heads, seq_len, seq_len), dtype=self.config.dtype)
    )

    vocab_size = logits.shape[-1]
    target = jnp.expand_dims(features["target_target_tokens"], 2) == jnp.arange(vocab_size).reshape((1, 1, -1))
    total_loss, z_loss = cross_entropy_with_logits(logits, target, z_loss=0.001)
    total_loss = total_loss.mean()
    dbg_info["total_loss"] = total_loss
    dbg_info["z_loss"] = z_loss.mean()
    return total_loss, dbg_info
