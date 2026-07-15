from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

try:
    from ..formats import xyxy_prediction_to_friendy
except ImportError:
    from formats import xyxy_prediction_to_friendy


@dataclass
class FasterRCNNAdapter:
    model: torch.nn.Module
    num_classes: int
    name: str = "fasterrcnn"

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
            fasterrcnn_prediction_to_friendy(prediction, image)
            for prediction, image in zip(predictions, images)
        ]


# variant -> ImageNet backbone family, so the right backbone weight enum is picked
# when the caller supplies `weights_backbone` instead of the full COCO `weights`.
_VARIANT_BACKBONES = {
    "resnet50_fpn": "resnet50",
    "resnet50_fpn_v2": "resnet50",
    "mobilenet_v3_large_fpn": "mobilenet_v3_large",
    "mobilenet_v3_large_320_fpn": "mobilenet_v3_large",
}


def build_fasterrcnn(
    num_classes: int,
    weights: Optional[str] = None,
    weights_backbone: Optional[str] = None,
    trainable_backbone_layers: Optional[int] = None,
    variant: str = "resnet50_fpn_v2",
    box_score_thresh: Optional[float] = None,
    box_nms_thresh: Optional[float] = None,
    box_detections_per_img: Optional[int] = None,
    rpn_pre_nms_top_n_test: Optional[int] = None,
    rpn_post_nms_top_n_test: Optional[int] = None,
    rpn_nms_thresh: Optional[float] = None,
    rpn_score_thresh: Optional[float] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    **kwargs: Any,
) -> FasterRCNNAdapter:
    if variant not in _VARIANT_BACKBONES:
        raise ValueError(f"Unsupported Faster R-CNN variant: {variant}")

    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        FasterRCNN_ResNet50_FPN_V2_Weights,
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
        fasterrcnn_mobilenet_v3_large_fpn,
        fasterrcnn_resnet50_fpn,
        fasterrcnn_resnet50_fpn_v2,
    )
    from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights

    builder_by_variant = {
        "resnet50_fpn": (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights),
        "resnet50_fpn_v2": (fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights),
        "mobilenet_v3_large_fpn": (
            fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        ),
        "mobilenet_v3_large_320_fpn": (
            fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        ),
    }
    backbone_weight_enum_by_family = {
        "resnet50": ResNet50_Weights,
        "mobilenet_v3_large": MobileNet_V3_Large_Weights,
    }

    builder, weight_enum = builder_by_variant[variant]
    backbone_weight_enum = backbone_weight_enum_by_family[_VARIANT_BACKBONES[variant]]

    model_weights = _resolve_weights(weight_enum, weights)
    backbone_weights = (
        None
        if model_weights is not None
        else _resolve_weights(backbone_weight_enum, weights_backbone)
    )

    # Only forward RoI-head / RPN knobs the caller actually set — torchvision's own
    # defaults (box_score_thresh=0.05, rpn_post_nms_top_n_test=1000, ...) already
    # apply when omitted; passing an explicit None would stomp them with None.
    head_kwargs = {
        "box_score_thresh": box_score_thresh,
        "box_nms_thresh": box_nms_thresh,
        "box_detections_per_img": box_detections_per_img,
        "rpn_pre_nms_top_n_test": rpn_pre_nms_top_n_test,
        "rpn_post_nms_top_n_test": rpn_post_nms_top_n_test,
        "rpn_nms_thresh": rpn_nms_thresh,
        "rpn_score_thresh": rpn_score_thresh,
        "min_size": min_size,
        "max_size": max_size,
    }
    head_kwargs = {k: v for k, v in head_kwargs.items() if v is not None}

    try:
        model = builder(
            weights=model_weights,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
            trainable_backbone_layers=trainable_backbone_layers,
            **head_kwargs,
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        if model_weights is None and backbone_weights is None:
            raise  # no weights were requested, so this is a real build error
        # Fall back to random init (not a crash) if the pretrained weights can't
        # be downloaded — e.g. no network or a blocked torchvision host.
        print(
            f"[fasterrcnn] Pretrained weights unavailable ({exc}); "
            f"training from scratch (random init)."
        )
        model = builder(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            trainable_backbone_layers=trainable_backbone_layers,
            **head_kwargs,
            **kwargs,
        )
    return FasterRCNNAdapter(model=model, num_classes=num_classes)


def fasterrcnn_prediction_to_friendy(
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
