from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

try:
    from ..formats import xyxy_prediction_to_friendy
except ImportError:
    from formats import xyxy_prediction_to_friendy


@dataclass
class RetinaNetAdapter:
    model: torch.nn.Module
    num_classes: int
    name: str = "retinanet"

    def to(self, device):
        self.model.to(device)
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def training_step(self, images, targets):
        self.model.train()
        return self._loss_forward(images, targets)

    def validation_step(self, images, targets):
        was_training = self.model.training
        self.model.train()
        _set_batch_norm_eval(self.model)
        try:
            return self._loss_forward(images, targets)
        finally:
            self.model.train(was_training)

    def _loss_forward(self, images, targets):
        losses = self.model(images, targets)
        return sum(loss for loss in losses.values()), losses

    @torch.no_grad()
    def predict(self, images):
        self.model.eval()
        predictions = self.model(images)
        return [
            retinanet_prediction_to_friendy(prediction, image)
            for prediction, image in zip(predictions, images)
        ]


def build_retinanet(
    num_classes: int,
    weights: Optional[str] = None,
    weights_backbone: Optional[str] = None,
    trainable_backbone_layers: Optional[int] = None,
    variant: str = "resnet50_fpn_v2",
    **kwargs: Any,
) -> RetinaNetAdapter:
    if variant not in {"resnet50_fpn", "resnet50_fpn_v2"}:
        raise ValueError(f"Unsupported RetinaNet variant: {variant}")

    from torchvision.models.detection import (
        RetinaNet_ResNet50_FPN_V2_Weights,
        RetinaNet_ResNet50_FPN_Weights,
        retinanet_resnet50_fpn,
        retinanet_resnet50_fpn_v2,
    )
    from torchvision.models import ResNet50_Weights

    if variant == "resnet50_fpn_v2":
        builder = retinanet_resnet50_fpn_v2
        weight_enum = RetinaNet_ResNet50_FPN_V2_Weights
    else:
        builder = retinanet_resnet50_fpn
        weight_enum = RetinaNet_ResNet50_FPN_Weights

    model_weights = _resolve_weights(weight_enum, weights)
    backbone_weights = (
        None
        if model_weights is not None
        else _resolve_weights(ResNet50_Weights, weights_backbone)
    )

    try:
        model = builder(
            weights=model_weights,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
            trainable_backbone_layers=trainable_backbone_layers,
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        if model_weights is None and backbone_weights is None:
            raise  # no weights were requested, so this is a real build error
        # Fall back to random init (not a crash) if the pretrained weights can't
        # be downloaded — e.g. no network or a blocked torchvision host.
        print(
            f"[retinanet] Pretrained weights unavailable ({exc}); "
            f"training from scratch (random init)."
        )
        model = builder(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            trainable_backbone_layers=trainable_backbone_layers,
            **kwargs,
        )
    return RetinaNetAdapter(model=model, num_classes=num_classes)


def retinanet_prediction_to_friendy(
    prediction: Dict[str, torch.Tensor], image: torch.Tensor
) -> torch.Tensor:
    image_height, image_width = image.shape[-2:]
    return xyxy_prediction_to_friendy(
        prediction["boxes"],
        prediction["scores"],
        prediction["labels"],
        image_width=image_width,
        image_height=image_height,
    )


def _resolve_weights(enum_cls, value):
    if value is None:
        return None

    if value is True:
        value = "DEFAULT"
    elif value is False:
        return None

    return enum_cls.verify(value)


def _set_batch_norm_eval(module: torch.nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()
