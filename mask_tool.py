#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standalone mask/polyedge extraction tool.

参考 refine.py 中的 get_polyedege 处理逻辑：
输入原图后生成并保存：
- mask.png
- outline_edge.png
- segmentation_mask_fine.png
- subject.png
- background.png
- detail_edge.png
- background_edge.png
- polyEdge.png
"""

from __future__ import annotations

import argparse
import os

import cv2
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

from refine import (
    fill_skeleton_region,
    generate_edge_skeleton,
    get_segmentation_mask,
    get_subject_and_background,
    get_subject_and_background_edge,
)


def build_predictor(sam_checkpoint: str, model_type: str) -> SamPredictor:
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    sam.to(device=device)
    return SamPredictor(sam)


def extract_mask_and_polyedge(source_image_path: str, output_dir: str, predictor: SamPredictor) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    bgr = cv2.imread(source_image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read source image: {source_image_path}")
    source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    combined_mask = get_segmentation_mask(predictor, source)
    # 按照 Masked_Load_Me_in_Loader 的习惯，调换黑白语义（0 <-> 255）
    # 注意这里只调整导出的 mask 可视化，不影响后续 polyedge 生成逻辑。
    combined_mask_swapped = cv2.bitwise_not(combined_mask)
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

    paths = {
        "mask": os.path.join(output_dir, "mask.png"),
        "outline_edge": os.path.join(output_dir, "outline_edge.png"),
        "segmentation_mask_fine": os.path.join(output_dir, "segmentation_mask_fine.png"),
        "subject": os.path.join(output_dir, "subject.png"),
        "background": os.path.join(output_dir, "background.png"),
        "detail_edge": os.path.join(output_dir, "detail_edge.png"),
        "background_edge": os.path.join(output_dir, "background_edge.png"),
        "polyedge": os.path.join(output_dir, "polyEdge.png"),
    }

    Image.fromarray(combined_mask_swapped).save(paths["mask"])
    Image.fromarray(outline_edge).save(paths["outline_edge"])
    Image.fromarray(segmentation_mask_fine).save(paths["segmentation_mask_fine"])
    Image.fromarray(subject).save(paths["subject"])
    Image.fromarray(background).save(paths["background"])
    Image.fromarray(detail_edge).save(paths["detail_edge"])
    Image.fromarray(background_edge).save(paths["background_edge"])
    polyedge.save(paths["polyedge"])

    return paths


def parse_args():
    parser = argparse.ArgumentParser(description="Extract mask/polyedge artifacts from a source image")
    parser.add_argument("--image", required=True, help="Input source image path")
    parser.add_argument("--output-dir", required=True, help="Output directory for generated artifacts")
    parser.add_argument("--sam-checkpoint", required=True, help="SAM checkpoint path")
    parser.add_argument("--sam-model-type", default="vit_l", help="SAM model type, e.g. vit_l / vit_h / vit_b")
    return parser.parse_args()


def main():
    args = parse_args()
    predictor = build_predictor(args.sam_checkpoint, args.sam_model_type)
    outputs = extract_mask_and_polyedge(args.image, args.output_dir, predictor)
    print("Generated files:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
