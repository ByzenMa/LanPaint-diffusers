"""Mask/PolyEdge generation utilities based on refine.py workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np
from PIL import Image

from refine import (
    fill_skeleton_region,
    generate_edge_skeleton,
    get_segmentation_mask,
    get_subject_and_background,
    get_subject_and_background_edge,
)


@dataclass
class PolyEdgeArtifacts:
    mask_path: str
    outline_edge_path: str
    segmentation_mask_fine_path: str
    subject_path: str
    background_path: str
    detail_edge_path: str
    background_edge_path: str
    polyedge_path: str


class PolyEdgeMaskTool:
    """Generate polyedge + intermediate images from an input source image."""

    def __init__(self, predictor):
        self.predictor = predictor

    def generate(self, source_image_path: str, output_dir: str) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)

        bgr = cv2.imread(source_image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read source image: {source_image_path}")
        source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        combined_mask = get_segmentation_mask(self.predictor, source)
        outline_edge = generate_edge_skeleton(combined_mask)
        segmentation_mask_fine = fill_skeleton_region(outline_edge)

        subject, background = get_subject_and_background(source, segmentation_mask_fine)
        detail_edge, background_edge = get_subject_and_background_edge(subject, background, outline_edge)

        polyedge = Image.merge(
            "RGB",
            (
                Image.fromarray(outline_edge),
                Image.fromarray(detail_edge),
                Image.fromarray(background_edge),
            ),
        )

        artifacts = PolyEdgeArtifacts(
            mask_path=os.path.join(output_dir, "mask.png"),
            outline_edge_path=os.path.join(output_dir, "outline_edge.png"),
            segmentation_mask_fine_path=os.path.join(output_dir, "segmentation_mask_fine.png"),
            subject_path=os.path.join(output_dir, "subject.png"),
            background_path=os.path.join(output_dir, "background.png"),
            detail_edge_path=os.path.join(output_dir, "detail_edge.png"),
            background_edge_path=os.path.join(output_dir, "background_edge.png"),
            polyedge_path=os.path.join(output_dir, "polyEdge.png"),
        )

        Image.fromarray(combined_mask).save(artifacts.mask_path)
        Image.fromarray(outline_edge).save(artifacts.outline_edge_path)
        Image.fromarray(segmentation_mask_fine).save(artifacts.segmentation_mask_fine_path)
        Image.fromarray(subject).save(artifacts.subject_path)
        Image.fromarray(background).save(artifacts.background_path)
        Image.fromarray(detail_edge).save(artifacts.detail_edge_path)
        Image.fromarray(background_edge).save(artifacts.background_edge_path)
        polyedge.save(artifacts.polyedge_path)

        return {
            "mask": artifacts.mask_path,
            "outline_edge": artifacts.outline_edge_path,
            "segmentation_mask_fine": artifacts.segmentation_mask_fine_path,
            "subject": artifacts.subject_path,
            "background": artifacts.background_path,
            "detail_edge": artifacts.detail_edge_path,
            "background_edge": artifacts.background_edge_path,
            "polyedge": artifacts.polyedge_path,
        }

    def generate_polyedge_array(self, source_image_path: str, output_dir: str) -> np.ndarray:
        paths = self.generate(source_image_path, output_dir)
        bgr = cv2.imread(paths["polyedge"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read generated polyedge: {paths['polyedge']}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
