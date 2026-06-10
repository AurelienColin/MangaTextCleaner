"""Smoke tests for MangaTextCleaner — import-level, guarded heavy deps."""
import importlib
import pytest

# MangaTextCleaner.py loads keras model at import time — skip if keras absent
keras_spec = importlib.util.find_spec("keras")
skip_if_no_keras = pytest.mark.skipif(
    keras_spec is None, reason="keras not installed — skipping model-loading module"
)


def test_import_deprecation_warnings():
    """deprecation_warnings sets env vars + configures warnings — no model load."""
    # Import only the filter_warnings function, don't call it (needs TF)
    import importlib as _il
    spec = _il.util.find_spec("deprecation_warnings")
    assert spec is not None, "deprecation_warnings module not found"


def test_pillow_available():
    pil = pytest.importorskip("PIL", reason="Pillow not installed")
    from PIL import Image
    assert hasattr(Image, "open")


def test_numpy_available():
    np = pytest.importorskip("numpy", reason="numpy not installed")
    assert np.__version__


@skip_if_no_keras
def test_import_manga_text_cleaner():
    """Full module import — only runs when keras is present."""
    import MangaTextCleaner  # noqa: F401
