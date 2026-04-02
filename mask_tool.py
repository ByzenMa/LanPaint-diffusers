#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""mask_tool.py

基于 refine.py 中的 get_polyedege 流程，提供可复用工具类：
输入原图 -> 生成 polyEdge，并保存中间过程图（subject/detail_edge 等）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

from refine import (
    fill_skeleton_region,
    generate_edge_skeleton,
    get_segmentation_mask,
    get_subject_and_background,
    get_subject_and_background_edge,
)


@dataclass
class MaskToolOutput:
    mask: str
    outline_edge: str
    segmentation_mask_fine: str
    subject: str
    background: str
    detail_edge: str
    background_edge: str
    polyedge: str


class PolyEdgeMaskTool:
    """根据输入原图生成 polyEdge，并保存全流程中间图。"""

    def __init__(self, predictor: SamPredictor):
        self.predictor = predictor

    def run(self, source_image_path: str, output_dir: str) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)

        bgr = cv2.imread(source_image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {source_image_path}")
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

        output = MaskToolOutput(
            mask=os.path.join(output_dir, "mask.png"),
            outline_edge=os.path.join(output_dir, "outline_edge.png"),
            segmentation_mask_fine=os.path.join(output_dir, "segmentation_mask_fine.png"),
            subject=os.path.join(output_dir, "subject.png"),
            background=os.path.join(output_dir, "background.png"),
            detail_edge=os.path.join(output_dir, "detail_edge.png"),
            background_edge=os.path.join(output_dir, "background_edge.png"),
            polyedge=os.path.join(output_dir, "polyEdge.png"),
        )

        Image.fromarray(combined_mask).save(output.mask)
        Image.fromarray(outline_edge).save(output.outline_edge)
        Image.fromarray(segmentation_mask_fine).save(output.segmentation_mask_fine)
        Image.fromarray(subject).save(output.subject)
        Image.fromarray(background).save(output.background)
        Image.fromarray(detail_edge).save(output.detail_edge)
        Image.fromarray(background_edge).save(output.background_edge)
        polyedge.save(output.polyedge)

        return output.__dict__

    def run_and_return_polyedge_array(self, source_image_path: str, output_dir: str) -> np.ndarray:
        outputs = self.run(source_image_path, output_dir)
        bgr = cv2.imread(outputs["polyedge"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read generated polyedge: {outputs['polyedge']}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_sam_predictor(sam_checkpoint: str, model_type: str = "vit_l") -> SamPredictor:
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    sam.to(device=device)
    return SamPredictor(sam)
