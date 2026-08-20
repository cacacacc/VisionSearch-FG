# Mission: VisionSearch-FG

## Why

Build a research-grade fine-grained visual recognition and image retrieval project with PyTorch, using CUB-200-2011 as the main dataset. The practical goal is to learn how to move from a reliable baseline to representation learning, retrieval evaluation, and explainability analysis.

## Success looks like

- Train and evaluate a ResNet-18 fine-grained classification baseline.
- Extract visual embeddings and use them for image retrieval.
- Compare CNN and modern lightweight Transformer backbones under limited hardware.
- Explain model behavior with Grad-CAM, attention visualization, and t-SNE.
- Write experiment notes that include motivation, hypothesis, method, metric, and analysis.

## Constraints

- Hardware is a lightweight laptop, so the project should favor transfer learning and pretrained lightweight models.
- Avoid large-scale training and high-memory models until the baseline is stable.
- Teach concepts before code: purpose, pipeline position, input/output, and tensor shapes should be explicit.

## Out of scope

- Chasing state-of-the-art accuracy as the first goal.
- Training large Transformer models from scratch.
- Adding complex research ideas before the baseline is reproducible.
