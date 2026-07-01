try:
    from .adapters.retinanet import build_retinanet
    from .adapters.rfdetr import build_rfdetr
    from .adapters.rtdetr import build_rtdetr
    from .adapters.yolox import build_yolox
except ImportError:
    from adapters.retinanet import build_retinanet
    from adapters.rfdetr import build_rfdetr
    from adapters.rtdetr import build_rtdetr
    from adapters.yolox import build_yolox


MODEL_REGISTRY = {
    "retinanet": build_retinanet,
    "rfdetr": build_rfdetr,
    "rtdetr": build_rtdetr,
    "yolox": build_yolox,
}


def build_model(name, **kwargs):
    """Build a registered detector adapter.

    Args:
        name: Registered model name, for example "retinanet".
        **kwargs: Model-specific builder options such as num_classes, weights,
            score_threshold, weights_backbone, trainable_backbone_layers, and
            variant.

    Weight examples:
        build_model("retinanet", num_classes=3)
        build_model("retinanet", num_classes=3, weights_backbone="DEFAULT")
        build_model("retinanet", num_classes=91, weights="DEFAULT")
        build_model("rtdetr", num_classes=3)
        build_model("rtdetr", num_classes=3, weights="PekingU/rtdetr_r50vd")
        build_model("rfdetr", num_classes=3)
        build_model("rfdetr", num_classes=3, variant="base")
        build_model("yolox", num_classes=3)
        build_model("yolox", num_classes=3, variant="yolox-s")
    """
    try:
        builder = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}") from exc

    return builder(**kwargs)
