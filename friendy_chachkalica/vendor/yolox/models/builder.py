from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX


YOLOX_VARIANTS = {
    "yolox-nano": {"depth": 0.33, "width": 0.25, "depthwise": True},
    "yolox-tiny": {"depth": 0.33, "width": 0.375, "depthwise": False},
    "yolox-s": {"depth": 0.33, "width": 0.50, "depthwise": False},
    "yolox-m": {"depth": 0.67, "width": 0.75, "depthwise": False},
    "yolox-l": {"depth": 1.0, "width": 1.0, "depthwise": False},
    "yolox-x": {"depth": 1.33, "width": 1.25, "depthwise": False},
}


def build_yolox_model(num_classes, variant="yolox-s", act="silu"):
    try:
        cfg = YOLOX_VARIANTS[variant]
    except KeyError as exc:
        available = ", ".join(sorted(YOLOX_VARIANTS))
        raise ValueError(f"Unsupported YOLOX variant '{variant}'. Available: {available}") from exc

    depth = cfg["depth"]
    width = cfg["width"]
    depthwise = cfg["depthwise"]
    in_channels = [256, 512, 1024]

    backbone = YOLOPAFPN(
        depth=depth,
        width=width,
        in_channels=in_channels,
        depthwise=depthwise,
        act=act,
    )
    head = YOLOXHead(
        num_classes=num_classes,
        width=width,
        in_channels=in_channels,
        depthwise=depthwise,
        act=act,
    )
    head.initialize_biases(1e-2)
    return YOLOX(backbone=backbone, head=head)
