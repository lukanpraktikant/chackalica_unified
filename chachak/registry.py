"""Pipeline registry and builder, mirroring ``friendy_chachkalica/registry.py``."""

try:
    from .pipeline import (
        BatchDetectPipeline,
        BatchPeoplePipeline,
        ChainedPipeline,
        PeopleDetectFirstPipeline,
    )
except ImportError:  # run as a flat script
    from pipeline import (
        BatchDetectPipeline,
        BatchPeoplePipeline,
        ChainedPipeline,
        PeopleDetectFirstPipeline,
    )


PIPELINE_REGISTRY = {
    "batch_detect": BatchDetectPipeline,
    "people_detect_first": PeopleDetectFirstPipeline,
    "batch_people": BatchPeoplePipeline,
    "chain": ChainedPipeline,
}


def _build_named(name, config, model_adapter, device, detector):
    try:
        cls = PIPELINE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(PIPELINE_REGISTRY))
        raise ValueError(
            f"Unknown pipeline '{name}'. Available pipelines: {available}"
        ) from exc
    return cls(model_adapter, device, config, detector=detector)


def build_pipeline(config, model_adapter, device, detector=None):
    """Build the pipeline described by ``config`` (recursively for 'chain')."""
    if config.pipeline == "chain":
        children = [
            _build_named(child, config, model_adapter, device, detector)
            for child in config.chain
        ]
        return ChainedPipeline(
            model_adapter, device, config, detector=detector, pipelines=children
        )
    return _build_named(config.pipeline, config, model_adapter, device, detector)
