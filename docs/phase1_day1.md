# Phase 1 Day 1: CUB Dataset and ResNet Baseline Shell

## Goal

Today we build the minimum baseline infrastructure: parse CUB metadata, produce image tensors, and define a ResNet-18 classifier that can later be trained.

## What

A PyTorch `Dataset` is an object that maps an integer index to one training example.

For CUB classification, one example is:

```text
image, label, metadata
```

After transform and batching:

```text
image: [3, 224, 224]
batch images: [B, 3, 224, 224]
labels: [B]
logits: [B, 200]
embedding: [B, 512]
```

## Why

Fine-grained visual recognition depends heavily on clean data handling. If labels, splits, or transforms are wrong, later model improvements become meaningless because experiments compare noise.

## Where

This module sits at the start of the pipeline:

```text
CUB metadata files
-> CUB200Dataset
-> DataLoader
-> ResNet-18
-> logits / embedding
```

## How

CUB provides metadata text files:

| File | Role |
| --- | --- |
| `images.txt` | maps image id to relative image path |
| `image_class_labels.txt` | maps image id to class id |
| `train_test_split.txt` | maps image id to train/test split |
| `classes.txt` | maps class id to class name |

The dataset converts CUB's 1-based class ids into 0-based labels because PyTorch `CrossEntropyLoss` expects target labels from `0` to `num_classes - 1`.

## Trade-off

This first dataset class is deliberately simple. It does not yet use bounding boxes or part annotations, even though CUB provides them. That keeps the first baseline focused on classification and makes the later explainability phase easier to interpret.

## Research Value

This is the first reproducibility boundary. Once data parsing is stable, we can compare ResNet, Swin-Tiny, contrastive loss, and retrieval metrics without wondering whether the train/test split changed accidentally.

## Next Step

After this smoke test passes with either fake data or the real CUB folder, the next lesson is the actual training loop:

```text
batch -> forward -> loss -> backward -> optimizer.step -> validation metric
```
