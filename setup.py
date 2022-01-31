# Copyright 2022 The jax3d Authors.
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

"""Setup."""

from __future__ import annotations

import pathlib

import setuptools

readme = pathlib.Path('README.md').read_text('utf-8')


def _load_requirements(path: str) -> list[str]:
  reqs = pathlib.Path(path).read_text().splitlines()
  reqs = [r.strip() for r in reqs]
  reqs = [r for r in reqs if r and not r.startswith('#')]
  return reqs


requirements = _load_requirements('requirements.txt')
requirements_dev = _load_requirements('requirements-dev.txt')

setuptools.setup(
    # Package metadata
    name='jax3d',
    version='0.1.0',
    author='Jax3d team',
    author_email='jax3d+dev@google.com',
    license='Apache 2.0',
    url='https://github.com/google-research/jax3d',
    description='Neural Rendering in Jax.',
    long_description=readme,
    long_description_content_type='text/markdown',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Multimedia :: Graphics :: 3D Rendering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='jax deep machine learning neural rendering nerf',

    # Package data, dependencies
    packages=setuptools.find_packages(),
    package_data={
        # Extra files included inside the `jax3d` package.
        'jax3d': ['*.gin'],
    },
    install_requires=requirements,
    extras_require={
        'dev': requirements_dev,
    },
    python_requires='>=3.7',
    entry_points={},
)
