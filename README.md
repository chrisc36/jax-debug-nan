# Jax NaN

This repo contains code to reproduce a tricky NaN I found while training a neural network. Its contains
some lightly modified code from [T5X](https://github.com/google-research/t5x) and a script to reproduce
the issue.

# Setup
On a TPU v3-8, install following T5x:

```
python3 -m pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Download the checkpoint from gs://chrisc-public/debug-checkpoint:

```
gsutil -m cp -r gs://chrisc-public/debug-checkpoint .
```

# Run

``python3 main.py debug-checkpoint -a none``

Results are finite

``python3 main.py debug-checkpoint -a partial-nan``

Gradient becomes NaN and intermediate values x-diff becomes > 0, but loss is finite

``python3 main.py debug-checkpoint -a nan``

Loss becomes NaN

The only thing that changes with these calls is the number of entirely unused arrays 
passed through the train_step function.