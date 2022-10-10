import argparse
import functools
import pickle
import time
from typing import Any
import numpy as np
import jax
from flax import struct, traverse_util
from jax import random
import jax.numpy as jnp
from t5x import train_state as train_state_lib

from t5x import partitioning
from t5x.checkpoints import Checkpointer, PartitionSpec
from absl import logging

from t5x.model import T5Config, Transformer


XL_CONFIG = T5Config(
  vocab_size=33152 + 16384,
  dtype='bfloat16',
  emb_dim=2048,
  num_heads=32,
  num_encoder_layers=24,
  num_decoder_layers=24,
  head_dim=64,
  mlp_dim=5120,
  mlp_activations=('gelu', 'linear'),
  dropout_rate=00.,
  logits_via_embedding=True,
  num_seg_emb=8,
  float32_attention_logits=False,
  zero_masked_embedding=False
)


class DummyTrainState(struct.PyTreeNode):
  params: Any = struct.field(pytree_node=True)
  other: Any = struct.field(pytree_node=True)


def train_with_lr(
    train_state,
    batch,
    dropout_rng,
    model,
    input_seq_len,
):
  """Main training function with LR schedule."""
  rngs = {'dropout': dropout_rng} if dropout_rng is not None else None
  grad_fn = jax.value_and_grad(model.apply, has_aux=True)
  (_, dbg_info), grad_accum = grad_fn(
    {'params': train_state.params}, batch, input_seq_len,
    enable_dropout=dropout_rng is not None, rngs=rngs)
  grads = traverse_util.flatten_dict(grad_accum).values()
  dbg_info["mean_grad"] = jnp.stack([x.mean() for x in grads]).mean()
  return train_state, dbg_info


def table_string(table) -> str:
  if len(table) == 0:
    return ""
  col_lens = [0] * len(table[0])
  for row in table:
    for i, cell in enumerate(row):
      col_lens[i] = max(len(cell), col_lens[i])

  formats = ["{0:<%d}" % x for x in col_lens]
  out = []
  for row in table:
    out.append(" ".join(formats[i].format(row[i]) for i in range(len(row))))
  return "\n".join(out)


def run(model, checkpoint: str, allocate: str, partitioner, input_seq_len=500):
  rng = random.PRNGKey(3452)

  # Load the input batch
  with open("features.pkl", "rb") as f:
    batch = pickle.load(f)

  # Loading the model checkpoint using T5X Checkpointer
  seq_len = input_seq_len
  init_or_restore_tick = time.time()

  optimizer_def = None
  def initialize_train_state(rng):
    initial_variables = model.init(rng, batch, enable_dropout=False)
    if optimizer_def:
      return train_state_lib.FlaxOptimTrainState.create(optimizer_def, initial_variables)
    return train_state_lib.InferenceState.create(initial_variables)
  global_train_state_shape = jax.eval_shape(initialize_train_state, rng=jax.random.PRNGKey(0))

  checkpointer = Checkpointer(
    train_state=global_train_state_shape,
    partitioner=partitioner,
    checkpoints_dir='',  # unused for restore
    dataset_iterator=None,
    restore_dtype=jnp.float32,
    use_gda=False
  )
  partitioner._params_on_devices = False
  train_state = checkpointer.restore(path=checkpoint)
  logging.info('Moving params to devices.')
  train_state_axes = partitioner.get_mesh_axes(train_state)
  train_state = partitioner.move_params_to_devices(
    train_state, train_state_axes)

  # train_state_axes = partitioner.get_mesh_axes(global_train_state_shape)
  # p_initialize_train_state_fn = partitioner.partition(
  #   initialize_train_state,
  #   in_axis_resources=None,
  #   out_axis_resources=train_state_axes)
  # train_state = p_initialize_train_state_fn(rng)

  # Allocate some extra arrays to pass into the function
  donated = []
  donated_axis = []
  if allocate == "small":
    for i in range(20):
      donated.append(jnp.zeros((2048, 2048)))
      donated_axis.append(PartitionSpec(None, "model"))
  elif allocate is None or allocate == "none":
    pass
  elif allocate == "nan":
    # Store some shapes/dtype/axis that were used when I originally trained the model,
    # this specific set of allocation seems to cause NaN loss
    with open("meta.pkl", "rb") as f:
      meta = pickle.load(f)
    for sh, dtype, ax in meta[:1500]:
      donated.append(jnp.zeros(sh, dtype=dtype))
      donated_axis.append(ax)
  elif allocate == "partial-nan":
    # Fewer arrays cause issues but don't cause the loss to be NaN
    with open("meta.pkl", "rb") as f:
      meta = pickle.load(f)
    for sh, dtype, ax in meta[:1400]:
      donated.append(jnp.zeros(sh, dtype=dtype))
      donated_axis.append(ax)
  elif allocate == "dbg":
    with open("meta.pkl", "rb") as f:
      meta = pickle.load(f)
    for sh, dtype, ax in meta[:1500]:
      donated.append(jnp.zeros(sh, dtype=dtype))
      donated_axis.append(ax)
  else:
    raise NotImplementedError(allocate)
  donated = partitioner.move_params_to_devices(donated, donated_axis)
  logging.info(f"Allocated {len(donated)} array, total entries={sum(np.prod(x.shape) for x in donated)}")

  init_or_restore_secs = time.time() - init_or_restore_tick
  logging.info('Initialize/restore complete (%.2f seconds).',
               init_or_restore_secs)

  logging.info(f"Donating {len(donated_axis)} vals")
  train_state_axes = DummyTrainState(train_state_axes.params, donated_axis)
  train_state = DummyTrainState(train_state.params, donated)
  del checkpointer, donated, donated_axis

  # Compiling a training step following T5x
  logging.info('Compiling train loop.')
  logging.flush()
  tick = time.time()

  train_step = functools.partial(
    train_with_lr,
    dropout_rng=None,
    model=model,
    input_seq_len=seq_len
  )
  train_step = partitioner.partition(
    train_step,
    in_axis_resources=(train_state_axes,
                       partitioner.data_partition_spec),
    out_axis_resources=(train_state_axes, None),
    donate_argnums=(),
    static_argnums=(),
  )
  train_step = partitioner.compile(train_step, train_state, batch)
  tock = time.time()
  logging.info(f"Compiling took {tock-tick:0.1f} seconds")

  logging.info('Start step')
  train_state, dbg_info = train_step(train_state, batch)
  logging.info('Done')

  dbg_info["x_diff (should <= 0)"] = dbg_info["L23/dot/diff-max"]
  dbg_info["mean_grad (should be finite)"] = dbg_info["mean_grad"]
  dbg_info["loss (should be finite)"] = dbg_info["total_loss"]

  table = []
  for k, v in dbg_info.items():
    if isinstance(v, float):
      table.append([k, "%0.4g" % v])
    else:
      table.append([k, str(v)])
  print(table_string(table))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint")
  parser.add_argument("--allocate", "-a", type=str)
  args = parser.parse_args()

  logging.set_verbosity(logging.INFO)
  logging.info(f"Args={args}")

  if sum(d.platform == "tpu" for d in jax.devices()) == 0:
    raise ValueError("No TPU found!")

  num_partitions = 4
  config = XL_CONFIG
  model = Transformer(config)
  partitioner = partitioning.PjitPartitioner(num_partitions, None)
  run(model, args.checkpoint, args.allocate, partitioner)


if __name__ == '__main__':
  main()