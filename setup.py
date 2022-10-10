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

"""Install T5X."""

import os
import sys
import setuptools

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 't5x')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

_LONG_DESCRIPTION = ""

_jax_version = '0.3.21'
_jaxlib_version = '0.3.20'

setuptools.setup(
  name='t5x',
  version=__version__,
  description='T5-eXtended in JAX',
  long_description=_LONG_DESCRIPTION,
  long_description_content_type='text/markdown',
  author='Google Inc.',
  license='Apache 2.0',
  packages=setuptools.find_packages(),
  package_data={
    '': ['**/*.gin'],  # not all subdirectories may have __init__.py.
  },
  scripts=[],
  install_requires=[
    'absl-py',
    'flax @ git+https://github.com/google/flax#egg=flax',
    f'jax >= {_jax_version}',
    f'jaxlib >= {_jaxlib_version}',
    'numpy',
    'orbax==0.0.9',
    't5',
    'tensorflow',
    'tensorstore >= 0.1.20',
    'einops'
  ],
  extras_require={
    'gcp': [
      'gevent', 'google-api-python-client', 'google-compute-engine',
      'google-cloud-storage', 'oauth2client'
    ],
    'test': ['pytest'],

    'data': ['ffmpeg-python', 'scikit-video', 'librosa', 'scikit-image', 'pafy', 'youtube_dl==2020.12.2', 'tensorflow_io', 'pydub'],

    # Cloud TPU requirements.
    'tpu': [f'jax[tpu] >= {_jax_version}'],
  },
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
  ],
  keywords='text nlp machinelearning',
)