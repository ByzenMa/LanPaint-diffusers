"""Dedicated adapter for Z-Image + ControlNet LanPaint pipeline.

This adapter intentionally lives in its own module (physical isolation)
so the existing ``ZImageAdapter`` and ``z-image`` pipeline are unaffected.
"""

from lanpaint_pipeline.adapters.z_image import ZImageAdapter


class ZImageControlNetAdapter(ZImageAdapter):
    """Isolated adapter for the z-image-controlnet route."""

    pass
