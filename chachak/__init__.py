"""chachak — stackable inference/eval pipelines over the trained detector.

Public API:
    load_pipeline_config, build_pipeline, run_pipeline, and the pipeline classes.
"""

try:
    from .config import PipelineConfig, load_pipeline_config
    from .registry import PIPELINE_REGISTRY, build_pipeline
    from .pipeline import (
        BatchDetectPipeline,
        BatchPeoplePipeline,
        ChainedPipeline,
        PeopleDetectFirstPipeline,
        Pipeline,
    )
    from .run import run_pipeline
except ImportError:  # run as a flat script
    from config import PipelineConfig, load_pipeline_config
    from registry import PIPELINE_REGISTRY, build_pipeline
    from pipeline import (
        BatchDetectPipeline,
        BatchPeoplePipeline,
        ChainedPipeline,
        PeopleDetectFirstPipeline,
        Pipeline,
    )
    from run import run_pipeline

__all__ = [
    "PipelineConfig",
    "load_pipeline_config",
    "PIPELINE_REGISTRY",
    "build_pipeline",
    "run_pipeline",
    "Pipeline",
    "BatchDetectPipeline",
    "PeopleDetectFirstPipeline",
    "BatchPeoplePipeline",
    "ChainedPipeline",
]
