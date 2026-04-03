"""LanPaint + Z-Image + multi-channel ControlNet inpaint pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps

from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler

from lanpaint_pipeline.pipeline import LanPaintInpaintPipeline


@dataclass
class ControlInjectConfig:
    r_range: Tuple[float, float] = (0.0, 1.0)
    g_range: Tuple[float, float] = (0.3, 0.6)
    b_range: Tuple[float, float] = (0.0, 0.3)


class LanPaintZImageControlNetPipeline:
    """Two-stage inpaint: LanPaint(z-image) -> ControlNet(polyedge) refine."""

    def __init__(
        self,
        lanpaint_pipe: LanPaintInpaintPipeline,
        control_pipe: StableDiffusionControlNetInpaintPipeline,
        inject_cfg: Optional[ControlInjectConfig] = None,
    ):
        self.lanpaint_pipe = lanpaint_pipe
        self.control_pipe = control_pipe
        self.inject_cfg = inject_cfg or ControlInjectConfig()
        self.device = lanpaint_pipe.adapter.device

    @classmethod
    def from_components(
        cls,
        lanpaint_pipe: LanPaintInpaintPipeline,
        *,
        sd15_model_id: str,
        controlnet_model_id: str,
        torch_dtype: torch.dtype = torch.float16,
    ) -> "LanPaintZImageControlNetPipeline":
        # Build 3 controlnets (R/G/B) so channel windows can be controlled independently.
        controlnet = [
            ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch_dtype),
            ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch_dtype),
            ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch_dtype),
        ]
        control_pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            sd15_model_id,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch_dtype,
        )
        control_pipe.scheduler = UniPCMultistepScheduler.from_config(control_pipe.scheduler.config)
        control_pipe.to(lanpaint_pipe.adapter.device)
        return cls(lanpaint_pipe=lanpaint_pipe, control_pipe=control_pipe)

    @staticmethod
    def _to_pil_image(data: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(data, Image.Image):
            return data
        if isinstance(data, str):
            return Image.open(data).convert("RGB")
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                return Image.fromarray(data).convert("RGB")
            return Image.fromarray(data)
        raise TypeError(f"Unsupported image type: {type(data)}")

    @staticmethod
    def _load_mask(mask_data: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(mask_data, str):
            return Image.open(mask_data).convert("L")
        if isinstance(mask_data, Image.Image):
            return mask_data.convert("L")
        if isinstance(mask_data, np.ndarray):
            if mask_data.ndim == 3:
                mask_data = mask_data[..., 0]
            return Image.fromarray(mask_data).convert("L")
        raise TypeError(f"Unsupported mask type: {type(mask_data)}")

    @staticmethod
    def _compute_scale(step_i: int, total_steps: int, rng: Tuple[float, float]) -> float:
        if total_steps <= 1:
            return 1.0
        ratio = step_i / float(total_steps - 1)
        return 1.0 if (rng[0] <= ratio < rng[1]) else 0.0

    def __call__(
        self,
        *,
        prompt: str,
        image: Union[str, Image.Image, np.ndarray],
        mask_image: Union[str, Image.Image, np.ndarray],
        polyedge_image: Union[str, Image.Image, np.ndarray],
        negative_prompt: str = "",
        guidance_scale: float = 5.0,
        num_inference_steps: int = 20,
        seed: int = 0,
        lanpaint_kwargs: Optional[dict] = None,
        save_visualize_dir: Optional[str] = None,
    ) -> Image.Image:
        lanpaint_kwargs = lanpaint_kwargs or {}

        # Stage-1: LanPaint + Z-Image
        lp_out = self.lanpaint_pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            **lanpaint_kwargs,
        )
        stage1_image = lp_out.images[0]
        if save_visualize_dir:
            os.makedirs(save_visualize_dir, exist_ok=True)
            stage1_image.save(os.path.join(save_visualize_dir, "stage1_lanpaint_zimage.png"))

        # Stage-2: SD1.5 ControlNet multi-channel polyedge refinement.
        polyedge = self._to_pil_image(polyedge_image).convert("RGB")
        mask_stage2 = ImageOps.invert(self._load_mask(mask_image))
        target_size = stage1_image.size

        # Keep stage-2 inputs spatially aligned with stage-1 output.
        # LanPaint preprocess may resize image/mask, so polyedge and mask must follow.
        if polyedge.size != target_size:
            polyedge = polyedge.resize(target_size, Image.BICUBIC)
        if mask_stage2.size != target_size:
            mask_stage2 = mask_stage2.resize(target_size, Image.NEAREST)

        r, g, b = polyedge.split()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        control_guidance_start = [
            self.inject_cfg.r_range[0],
            self.inject_cfg.g_range[0],
            self.inject_cfg.b_range[0],
        ]
        control_guidance_end = [
            self.inject_cfg.r_range[1],
            self.inject_cfg.g_range[1],
            self.inject_cfg.b_range[1],
        ]

        result = self.control_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=stage1_image,
            mask_image=mask_stage2,
            control_image=[r.convert("RGB"), g.convert("RGB"), b.convert("RGB")],
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            controlnet_conditioning_scale=[1.0, 1.0, 1.0],
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
        if save_visualize_dir:
            os.makedirs(save_visualize_dir, exist_ok=True)
            result.save(os.path.join(save_visualize_dir, "stage2_controlnet_refine.png"))

        return result


def build_polyedge_with_refine(source_image_path: str, image_name: str, predictor):
    """Bridge helper: delegate polyedge generation to refine.py pipeline."""
    from refine import get_polyedege

    return get_polyedege(source_image_path, image_name, predictor)
