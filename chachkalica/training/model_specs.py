"""Per-architecture builder-option specs — the config UI's mirror of the adapters.

The trainer (``friendy_chachkalica``) is model-specific only inside its ``adapters/``:
every ``build_<arch>(num_classes, **params)`` signature defines the knobs that
architecture accepts. This module re-declares those knobs so the admin can offer
a real form field per option (a size dropdown, thresholds, backbone weights …)
instead of hand-typed JSON, while the DB stays model-agnostic — every value here
is still stored in :attr:`ExperimentModel.params` and spread back into the model
YAML entry by :mod:`training.services.config_gen`.

Keep this in sync with ``/home/luka/workspace/chachkalica/friendy_chachkalica/adapters/*.py``:
one entry per selectable ``build_<arch>`` kwarg. ``key`` is the ``params`` key
(and the YAML kwarg name); ``kind`` picks the widget; ``default`` is the adapter
default shown as guidance (a blank field means "use the adapter default").
"""

# RT-DETR has no ``variant`` kwarg: its backbone size is chosen by loading the
# matching pretrained checkpoint, so the "size" dropdown writes ``weights`` (the
# HuggingFace repo id) directly. This overrides the ``pretrained`` checkbox.
RTDETR_SIZE_CHOICES = [
    ("PekingU/rtdetr_r18vd", "r18vd (smallest)"),
    ("PekingU/rtdetr_r34vd", "r34vd"),
    ("PekingU/rtdetr_r50vd", "r50vd (default)"),
    ("PekingU/rtdetr_r101vd", "r101vd (largest)"),
]


# Each spec: {key, label, kind, choices?, default?, help?}
#   kind in {"choice", "int", "float", "bool", "str"}
#   choices: list of "value" or ("value", "label")
ARCH_FIELD_SPECS: dict[str, list[dict]] = {
    "retinanet": [
        {
            "key": "variant", "label": "Size / variant", "kind": "choice",
            "choices": ["resnet50_fpn", "resnet50_fpn_v2"],
            "default": "resnet50_fpn_v2",
            "help": "RetinaNet backbone/FPN variant.",
        },
        {
            "key": "weights_backbone", "label": "Backbone weights", "kind": "str",
            "help": "ImageNet backbone weights, e.g. DEFAULT. Ignored when "
                    "'pretrained' loads the full COCO weights.",
        },
        {
            "key": "trainable_backbone_layers", "label": "Trainable backbone layers",
            "kind": "int",
            "help": "How many backbone stages to fine-tune (0–5). Blank = torchvision default.",
        },
    ],
    "rfdetr": [
        {
            "key": "variant", "label": "Size / variant", "kind": "choice",
            "choices": ["nano", "small", "medium", "base", "large"],
            "default": "base",
            "help": "RF-DETR size (all Apache-2.0). xlarge/2xlarge are non-free and rejected.",
        },
        {
            "key": "resolution", "label": "Input resolution", "kind": "int",
            "help": "Square input size. Must be divisible by 56 (the DINOv2 backbone's "
                    "patch stride) — e.g. 560, 616, 672, 728, 784, 840, 896. A value that "
                    "isn't will crash on epoch 1. Blank = the variant's native resolution.",
        },
        {
            "key": "score_threshold", "label": "Score threshold", "kind": "float",
            "default": 0.5,
            "help": "Default confidence cutoff used at prediction time.",
        },
    ],
    "rtdetr": [
        {
            "key": "weights", "label": "Size (pretrained backbone)", "kind": "choice",
            "choices": RTDETR_SIZE_CHOICES,
            "default": "PekingU/rtdetr_r50vd",
            "help": "RT-DETR backbone size, selected via its pretrained checkpoint repo. "
                    "Overrides the 'pretrained' checkbox.",
        },
        {
            "key": "score_threshold", "label": "Score threshold", "kind": "float",
            "default": 0.5, "help": "Default confidence cutoff used at prediction time.",
        },
        {
            "key": "input_max_size", "label": "Input max size", "kind": "int",
            "default": 640, "help": "Longest-side cap; larger inputs are downscaled.",
        },
        {
            "key": "input_size_multiple", "label": "Input size multiple", "kind": "int",
            "default": 32, "help": "Pad each side up to this multiple.",
        },
        {
            "key": "ignore_mismatched_sizes", "label": "Ignore mismatched sizes",
            "kind": "bool", "default": True,
            "help": "Re-init the head when the pretrained class count differs.",
        },
    ],
    "yolox": [
        {
            "key": "variant", "label": "Size / variant", "kind": "choice",
            "choices": ["yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x"],
            "default": "yolox-s",
            "help": "YOLOX size (all Apache-2.0).",
        },
        {
            "key": "score_threshold", "label": "Score threshold", "kind": "float",
            "default": 0.3, "help": "Default confidence cutoff used at prediction time.",
        },
        {
            "key": "nms_threshold", "label": "NMS threshold", "kind": "float",
            "default": 0.45, "help": "IoU threshold for non-maximum suppression.",
        },
    ],
}

# Every params key any arch's form owns. On save we strip these from params
# before re-applying the selected arch's values, so switching arch never leaves
# a stale kwarg (e.g. yolox's nms_threshold) that a different adapter would reject.
ALL_SPEC_KEYS = {spec["key"] for specs in ARCH_FIELD_SPECS.values() for spec in specs}

FIELD_PREFIX = "xm_"  # form-field namespace: xm_<arch>_<key>


def field_name(arch: str, key: str) -> str:
    """Form-field name for an (arch, param-key) spec — unique across archs."""
    return f"{FIELD_PREFIX}{arch}_{key}"


def spec_field_names() -> list[str]:
    """Every builder-option form-field name, in arch/spec declaration order.

    Shared by the form (which declares the fields) and the admin inline (which
    lays them out) so the two never drift.
    """
    return [
        field_name(arch, spec["key"])
        for arch, specs in ARCH_FIELD_SPECS.items()
        for spec in specs
    ]


def normalized_choices(spec: dict) -> list[tuple[str, str]]:
    """Spec choices as (value, label) pairs (a bare string becomes (s, s))."""
    out = []
    for choice in spec.get("choices", []):
        if isinstance(choice, (list, tuple)):
            out.append((str(choice[0]), str(choice[1])))
        else:
            out.append((str(choice), str(choice)))
    return out
