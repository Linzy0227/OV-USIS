# Getting Started with OV-USIS

Follow [INSTALL.md](INSTALL.md) to create the `ovs` environment and build the
bundled Detectron2 and MSDeformAttn extension. Run `sh download.sh` to download
the pretrained weights into this project's `pretrained/` directory.

## Dataset

Annotations are included in this repository. Download the images linked in
[README.md](README.md), then use:

```text
dataset/
├── train/
├── val/
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

Place the dataset in the project's `dataset/` directory as shown above.

| Setting | Training images / classes | Evaluation images / classes | Shared | Unseen |
| --- | --- | --- | --- | --- |
| A | train / 83 | val / 115 | 40 | 75 |
| B | val / 115 | train / 83 | 40 | 43 |

B reverses the same image splits. Category names and IDs are read from the
annotation files. For a different training dataset, set
`MODEL.SEM_SEG_HEAD.NUM_CLASSES` in the YAML config to its training class count.

## Training and evaluation

| Model / setting | Config | Output directory |
| --- | --- | --- |
| ConvNeXt-Base / A | configs/convnext_base_ovusis_a.yaml | output/convnext_b_a |
| ConvNeXt-Base / B | configs/convnext_base_ovusis_b.yaml | output/convnext_b_b |
| ConvNeXt-Large / A | configs/convnext_large_ovusis_a.yaml | output/convnext_l_a |
| ConvNeXt-Large / B | configs/convnext_large_ovusis_b.yaml | output/convnext_l_b |

```bash
# Train using the settings written in train.sh.
sh train.sh

# Evaluate using the settings written in test.sh.
sh test.sh
```

Edit the command inside `train.sh` or `test.sh` before running it:

- `CUDA_VISIBLE_DEVICES`: physical GPU IDs to use.
- `--num-gpus`: number of selected GPUs.
- `--config-file`: one of the configurations listed above.
- `MODEL.WEIGHTS` in `test.sh`: checkpoint matching the selected experiment.

The supplied scripts use ConvNeXt-Large, setting A. Training uses GPUs 0 and 1;
evaluation uses GPU 1. Adjust batch size and learning rate in the YAML config
when changing the GPU count. To resume training, add `--resume` to the Python
command in `train.sh`. The scripts change to the project directory before
launching Python, so their relative paths refer to this project.

## Metrics

Copy one complete per-category **segmentation** AP table from the evaluation
log into a text file. Do not combine bounding-box and segmentation tables or
multiple evaluation runs. The expected columns are:

```text
| category | AP | AP50 | AP75 |
| Diver    | 60 | 80   | 70   |
```

This snippet illustrates the format; the script requires a row for every
category in your evaluation annotation file, with AP values on the 0–100 scale.

```bash
python calculate_metrics.py --split a --results results_a.txt
python calculate_metrics.py --split b --results results_b.txt

# Custom training/evaluation annotations and JSON output.
python calculate_metrics.py --results /path/to/results.txt \
  --train-json /path/to/train.json --test-json /path/to/test.json \
  --output /path/to/summary.json
```

Reports AP, AP50, and AP75 for shared, unseen, and all evaluation categories.
