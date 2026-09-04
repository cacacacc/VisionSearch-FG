"""Backbones, classification heads, and embedding models."""

from visionsearch_fg.models.angular_margin import (
    AngularMarginClassifier,
    AngularMarginHead,
    AngularMarginOutput,
    default_margin_for_type,
    degrees_from_radians,
)
from visionsearch_fg.models.resnet import (
    FineTuneMode,
    ModelOutput,
    ResNet18Classifier,
    build_resnet18_classifier,
)
from visionsearch_fg.models.swin import SwinTinyClassifier, build_swin_tiny_classifier
from visionsearch_fg.models.swin_head_selector import (
    HeadSelectorOutput,
    SwinHeadAwareClassifier,
    build_swin_head_aware_classifier,
)
from visionsearch_fg.models.contrastive import (
    ContrastiveClassifier,
    ContrastiveOutput,
    build_contrastive_classifier,
)

__all__ = [
    "ContrastiveClassifier",
    "ContrastiveOutput",
    "AngularMarginClassifier",
    "AngularMarginHead",
    "AngularMarginOutput",
    "FineTuneMode",
    "HeadSelectorOutput",
    "ModelOutput",
    "ResNet18Classifier",
    "SwinHeadAwareClassifier",
    "SwinTinyClassifier",
    "build_contrastive_classifier",
    "default_margin_for_type",
    "degrees_from_radians",
    "build_resnet18_classifier",
    "build_swin_head_aware_classifier",
    "build_swin_tiny_classifier",
]
