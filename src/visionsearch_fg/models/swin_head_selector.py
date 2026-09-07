from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import Swin_T_Weights, swin_t

FineTuneMode = Literal["frozen", "full"]


@dataclass(frozen=True)
class HeadSelectorOutput:
    """暴露训练、检索和 selector 分析所需的全部特征。

    ``logits`` 用于分类损失。``embedding`` 是最终用于检索的全局加局部
    表征。其余字段保留中间值，使脚本可以直接检查 selector 关注的位置，而不用
    从模型内部重新构造。
    """

    logits: torch.Tensor
    embedding: torch.Tensor
    global_embedding: torch.Tensor
    local_embedding: torch.Tensor
    selector_logits: torch.Tensor
    selector_weights: torch.Tensor


class SwinHeadAwareClassifier(nn.Module):
    """在最终阶段 token 上学习软局部选择器的 Swin-Tiny。

    普通 Swin 分类器会通过全局平均池化将最终空间特征图压成一个向量。
    该模型保留空间 token，并为每个 token 学习一个标量分数。softmax 将这些分数
    转为权重，因此局部特征是可微的加权平均，而不是硬 Top-K 选择。
    这会保留选择器和主干网络上的梯度流。
    """

    def __init__(
        self,
        num_classes: int = 200,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        fine_tune_mode: FineTuneMode | None = None,
        selector_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if selector_temperature <= 0:
            raise ValueError("selector_temperature must be positive")

        # 移除 Swin 原始 ImageNet 分类头，由主干网络提供最终 token 特征。
        # 本模块负责接入使用全局/局部融合表征的新分类器。
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        backbone = swin_t(weights=weights)
        embedding_dim = backbone.head.in_features
        backbone.head = nn.Identity()

        self.backbone = backbone
        # 每个 token 得到一个标量证据分数，forward() 中的 softmax 会把这些
        # 分数独立转换成每张图像的空间位置概率分布。
        self.selector = nn.Linear(embedding_dim, 1)
        # 全局和局部特征会拼接，因此分类器输入维度是 Swin token 维度的两倍。
        self.classifier = nn.Linear(embedding_dim * 2, num_classes)
        self.embedding_dim = embedding_dim * 2
        self.token_dim = embedding_dim
        self.num_classes = num_classes
        self.selector_temperature = selector_temperature

        mode = fine_tune_mode or ("frozen" if freeze_backbone else "full")
        self.set_fine_tune_mode(mode)

    def forward(self, images: torch.Tensor) -> HeadSelectorOutput:
        """分类图像，并同时返回检索特征和证据图。

        对标准 224x224 输入，``tokens`` 的形状为 ``[B, 49, 768]``：49 对应 7x7
        最终空间网格，768 是通道维度。选择器沿 49 个 token 位置工作，
        而不是沿通道工作。
        """

        tokens = self.extract_tokens(images)

        # 对所有最终阶段 token 求平均，得到常规全局 Swin 表征。
        global_embedding = tokens.mean(dim=1)

        # 为每个 token 生成一个未归一化证据分数。只压缩最后一维，
        # 将 [B, N, 1] 变为 [B, N]。温度系数控制分布集中度。
        selector_logits = self.selector(tokens).squeeze(-1)
        selector_weights = torch.softmax(selector_logits / self.selector_temperature, dim=1)

        # 加权池化生成可微局部表征。这里的 einsum 表示：对每个批内样本，
        # 将 token 特征乘以对应标量权重后求和。
        local_embedding = torch.einsum("bn,bnc->bc", selector_weights, tokens)

        # 分别归一化两个分支，避免局部分支只因特征模长更大而主导结果。
        # 拼接后再归一化一次，因为最终嵌入会暴露给检索和对比学习目标。
        embedding = F.normalize(
            torch.cat(
                [
                    F.normalize(global_embedding, p=2, dim=1),
                    F.normalize(local_embedding, p=2, dim=1),
                ],
                dim=1,
            ),
            p=2,
            dim=1,
        )
        logits = self.classifier(embedding)
        return HeadSelectorOutput(
            logits=logits,
            embedding=embedding,
            global_embedding=global_embedding,
            local_embedding=local_embedding,
            selector_logits=selector_logits,
            selector_weights=selector_weights,
        )

    def extract_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """以 ``[B, N, C]`` 形式返回归一化后的最终阶段 Swin token。

        ``torchvision`` 将 Swin 特征提取器暴露为 ``features``。根据
        torchvision 版本不同，输出可能是通道后置 ``[B, H, W, C]``，也可能是
        通道前置 ``[B, C, H, W]``。下方辅助函数会在 Swin 最终归一化前统一布局。
        """

        spatial_features = self.backbone.features(images)
        tokens = flatten_swin_spatial_features(spatial_features)
        return self.backbone.norm(tokens)

    def set_fine_tune_mode(self, mode: FineTuneMode) -> None:
        """控制梯度是否更新 Swin backbone。

        选择器和分类器在这里不会被修改。因此冻结实验仍会训练新增分类头，
        同时把预训练 Swin 特征作为固定编码器使用。
        """

        if mode == "full":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True
            return
        if mode == "frozen":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            return
        raise ValueError(f"Unsupported Swin head-aware fine_tune_mode: {mode}")


def flatten_swin_spatial_features(spatial_features: torch.Tensor) -> torch.Tensor:
    """将 Swin 特征图展平为 ``[batch, tokens, channels]``。

    选择器需要统一布局。形状检查可以避免意外张量被静默展平；布局分支则兼容
    不同 torchvision 版本使用的特征约定。对于 7x7 特征图，``H * W`` 会变成
    49 个 token。
    """

    if spatial_features.ndim != 4:
        raise ValueError(f"Expected Swin spatial features with shape [B,H,W,C], got {tuple(spatial_features.shape)}")
    if spatial_features.shape[-1] >= spatial_features.shape[1]:
        batch_size, height, width, channels = spatial_features.shape
        return spatial_features.reshape(batch_size, height * width, channels)
    batch_size, channels, height, width = spatial_features.shape
    return spatial_features.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)


def build_swin_head_aware_classifier(
    num_classes: int = 200,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    fine_tune_mode: FineTuneMode | None = None,
    selector_temperature: float = 1.0,
) -> SwinHeadAwareClassifier:
    return SwinHeadAwareClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        fine_tune_mode=fine_tune_mode,
        selector_temperature=selector_temperature,
    )
