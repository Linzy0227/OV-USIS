# Installation

The supplied environment uses Python 3.10 and PyTorch 2.7.0 with CUDA 12.8.
Use a compatible NVIDIA driver and a CUDA toolkit matching your PyTorch build
when compiling the native extensions. The environment file includes the
additional runtime dependencies listed in [requirements.txt](requirements.txt).

Run these commands from the project directory:

```bash
conda env create -f environment_ovs.yml
conda activate ovs
python -m pip install --no-build-isolation -e detectron2
```

Use the bundled, modified Detectron2 for this benchmark's evaluation output.
For an existing compatible PyTorch environment, install the additional runtime
dependencies with `python -m pip install -r requirements.txt`, then install
the bundled Detectron2 as above. This does not replace the other dependencies
specified in `environment_ovs.yml`.

`--no-build-isolation` lets Detectron2's build use the PyTorch already installed
in the active environment; its setup script imports PyTorch during the build.

## MSDeformAttn extension

`CUDA_HOME` must point to the installed CUDA toolkit.

```bash
cd ovusis/modeling/S3B/ops
sh make.sh
cd ../../../..
```

When building on a machine without an accessible GPU, specify the architecture
of the target GPU explicitly, for example:

```bash
cd ovusis/modeling/S3B/ops
TORCH_CUDA_ARCH_LIST='8.0' FORCE_CUDA=1 python setup.py build install
cd ../../../..
```

## Pretrained weights and data

```bash
sh download.sh
```

See [GETTING_STARTED.md](GETTING_STARTED.md) for the dataset layout, A/B
settings, and training and evaluation commands.

The default OpenCV dependency is the headless package. It supports training
and writing visualizations to files. If you need OpenCV display windows,
replace `opencv-python-headless` with the corresponding `opencv-python`
version in your environment; install only one of the two packages.
