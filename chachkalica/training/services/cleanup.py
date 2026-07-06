"""Delete a run/eval's on-disk artifacts when its database row is deleted.

Wired via ``post_delete`` signals (see :mod:`training.signals`) so it fires for
admin single deletes, bulk "delete selected", and cascades (deleting an
Experiment cascades to its runs) alike.

Guarded: only paths *under* the configured ``runs_root`` / ``configs_root`` are
ever removed, so a blank, relative, or rogue ``output_dir`` can never escalate
into deleting the project root or the whole filesystem.
"""

import logging
import shutil
from pathlib import Path

from training.models import TrainingSettings
from training.services.config_gen import _resolve

logger = logging.getLogger(__name__)


def _within(root: Path, path: Path) -> bool:
    """True if ``path`` is strictly inside ``root`` (never ``root`` itself)."""
    root, path = root.resolve(), path.resolve()
    if path == root:
        return False
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _rm_dir(path_str: str, root_str: str) -> None:
    if not path_str:
        return
    path, root = Path(path_str), _resolve(root_str)
    if not _within(root, path):
        logger.warning("cleanup: refusing to delete %s — not under %s", path, root)
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        logger.info("cleanup: removed %s", path)


def _rm_file(path_str: str, root_str: str) -> None:
    if not path_str:
        return
    path, root = Path(path_str), _resolve(root_str)
    if not _within(root, path):
        logger.warning("cleanup: refusing to delete %s — not under %s", path, root)
        return
    if path.is_file():
        path.unlink(missing_ok=True)
        logger.info("cleanup: removed %s", path)


def remove_run_artifacts(run) -> None:
    """Delete a training run's output dir and generated config YAML."""
    ts = TrainingSettings.load()
    _rm_dir(run.output_dir, ts.runs_root)
    _rm_file(run.config_yaml_path, ts.configs_root)


def remove_eval_artifacts(eval_run) -> None:
    """Delete an eval run's output dir and generated request YAML."""
    ts = TrainingSettings.load()
    _rm_dir(eval_run.output_dir, ts.runs_root)
    _rm_file(eval_run.request_yaml_path, ts.configs_root)
