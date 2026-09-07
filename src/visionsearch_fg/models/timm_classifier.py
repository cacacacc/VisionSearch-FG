from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from visionsearch_fg.models.resnet import ModelOutput

FineTuneMode = Literal["frozen", "partial", "full"]


class TimmClassifier(nn.Module):
    """暴露池化视觉嵌入的通用 timm 图像分类器。

    ``timm`` 提供大量主干网络，且输出约定各不相同。该封装器将它们标准化为
    统一接口，返回 ``logits`` 和二维嵌入，同时支持通过配置中的模块路径
    执行部分微调。
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 200,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        fine_tune_mode: FineTuneMode | None = None,
        trainable_backbone_layers: list[str] | None = None,
        model_kwargs: dict | None = None,
    ) -> None:
        super().__init__()

        try:
            import timm
        except ImportError as error:
            raise ImportError(
                "timm is required for model.backbone values that start with 'timm:'."
            ) from error

        timm_kwargs = dict(model_kwargs or {})
        # 先移除封装层的池化配置，再把其余模型专属选项传给 timm.create_model。
        global_pool = timm_kwargs.pop("global_pool", "avg")
        backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=global_pool,
            **timm_kwargs,
        )
        embedding_dim = int(getattr(backbone, "num_features", 0))
        if embedding_dim <= 0:
            raise ValueError(f"Unable to infer embedding dimension for timm model: {model_name}")

        self.model_name = model_name
        self.backbone = backbone
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        mode = fine_tune_mode or ("frozen" if freeze_backbone else "full")
        self.set_fine_tune_mode(
            mode=mode,
            trainable_backbone_layers=trainable_backbone_layers,
        )

    def forward(self, images: torch.Tensor) -> ModelOutput:
        """运行所选 timm encoder，并统一其特征形状。"""
        embedding = self.backbone(images)
        if embedding.ndim == 4:
            # 某些 CNN 主干网络即使设置 global_pool=none 也会返回 [B, C, H, W]。
            embedding = embedding.mean(dim=(-2, -1))
        elif embedding.ndim > 2:
            # 基于 token 的主干网络可能返回额外的空间或 token 维度。
            embedding = torch.flatten(embedding, start_dim=1)
        logits = self.classifier(embedding)
        return ModelOutput(logits=logits, embedding=embedding)

    def set_fine_tune_mode(
        self,
        mode: FineTuneMode,
        trainable_backbone_layers: list[str] | None = None,
    ) -> None:
        """应用冻结、全量微调或显式指定的部分微调。"""
        if mode == "full":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True
            return

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        if mode == "frozen":
            return

        if mode == "partial":
            layers = trainable_backbone_layers or []
            if not layers:
                raise ValueError("partial timm fine_tune_mode requires trainable_backbone_layers")
            for layer_name in layers:
                layer = _resolve_module(self.backbone, layer_name)
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            return

        raise ValueError(f"Unsupported timm fine_tune_mode: {mode}")


def build_timm_classifier(
    model_name: str,
    num_classes: int = 200,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    fine_tune_mode: FineTuneMode | None = None,
    trainable_backbone_layers: list[str] | None = None,
    model_kwargs: dict | None = None,
) -> TimmClassifier:
    return TimmClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        fine_tune_mode=fine_tune_mode,
        trainable_backbone_layers=trainable_backbone_layers,
        model_kwargs=model_kwargs,
    )


def _resolve_module(root: nn.Module, module_path: str) -> nn.Module:
    """解析类似 ``stages.3.blocks.0`` 的点分属性/索引路径。"""
    module: nn.Module = root
    for part in module_path.split("."):
        if part.isdigit() and isinstance(module, (nn.Sequential, nn.ModuleList)):
            module = module[int(part)]
            continue
        child = getattr(module, part, None)
        if child is None:
            raise ValueError(f"Unknown timm backbone layer: {module_path}")
        module = child
    return module
