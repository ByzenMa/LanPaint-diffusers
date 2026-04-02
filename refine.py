#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import json
import torch
import numpy as np
import cv2
from typing import Optional, Union
from PIL import Image
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte
from segment_anything import sam_model_registry, SamPredictor

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

class ConfigManager:
    def __init__(self, config_path: str = "./config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    @property
    def sd1_5_path(self):
        return self.config["model"]["sd1_5_path"]

    @property
    def controlnet_canny_path(self):
        return self.config["model"]["controlnet_canny_path"]

    # partial injection ranges
    @property
    def r_channel_start(self):
        return float(self.config["control_injection"]["r_channel_start"])

    @property
    def r_channel_end(self):
        return float(self.config["control_injection"]["r_channel_end"])

    @property
    def g_channel_start(self):
        return float(self.config["control_injection"]["g_channel_start"])

    @property
    def g_channel_end(self):
        return float(self.config["control_injection"]["g_channel_end"])

    @property
    def b_channel_start(self):
        return float(self.config["control_injection"]["b_channel_start"])

    @property
    def b_channel_end(self):
        return float(self.config["control_injection"]["b_channel_end"])

    # inference
    @property
    def num_inference_steps(self):
        return int(self.config["inference"]["num_inference_steps"])

    @property
    def guidance_scale(self):
        return float(self.config["inference"]["guidance_scale"])

    @property
    def seed(self):
        return int(self.config["inference"]["seed"])


def get_segmentation_mask(predictor, image_array):
    predictor.set_image(image_array)
    input_points = np.array([[image_array.shape[1] // 2, image_array.shape[0] // 2]])
    input_labels = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True  # 输出多个掩码
    )

    combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255  # 将True/False转换为0/255
    return combined_mask


def generate_edge_skeleton(mask, refinement_iterations=3):
    if mask is None:
        raise ValueError("无法读取掩码图像。")

    # 使用Canny边缘检测
    edges = cv2.Canny(mask, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = edges
    for _ in range(refinement_iterations):
        dilated_edges = cv2.dilate(dilated_edges, kernel, iterations=1)

        dilated_edges_binary = dilated_edges > 0
        skeleton = skeletonize(dilated_edges_binary)
        skeleton = img_as_ubyte(skeleton)

        dilated_edges = skeleton

    return skeleton


def fill_skeleton_region(skeleton):
    _, binary_skeleton = cv2.threshold(skeleton, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(skeleton)
    cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
    return mask


def get_subject_and_background(image, mask):
    basic_background = np.zeros_like(image)
    subject = np.where(mask[:, :, None], image, basic_background)
    inverted_mask = 255 - mask
    background = np.where(inverted_mask[:, :, None], image, basic_background)

    return subject, background


def get_subject_and_background_edge(subject, background, outline_edge):
    # 使用Canny边缘检测提取主体和背景的直接边缘
    basic_detail_edge = cv2.Canny(subject, threshold1=50, threshold2=150)
    basic_background_edge = cv2.Canny(background, threshold1=50, threshold2=150)

    # 将轮廓边缘图膨胀
    kernel_size = (5, 5)
    kernel = np.ones(kernel_size, dtype=np.uint8)
    dilated_outline_edge = cv2.dilate(outline_edge, kernel, iterations=1)

    # 将主体和背景的直接边缘减去膨胀后的轮廓边缘
    detail_edge = cv2.subtract(basic_detail_edge, dilated_outline_edge)
    background_edge = cv2.subtract(basic_background_edge, dilated_outline_edge)

    return detail_edge, background_edge


class MultiCannyPolyEdgePipeline:
    """
    使用一个 ControlNet，通过 R/G/B 三个不同的 channel 做 partial injection。
    例如:
      - R 全程 (0.0 ~ 1.0)
      - G 在某个区间 (0.3 ~ 0.6)
      - B 在某个区间 (0.0 ~ 0.3)
    """

    def __init__(
            self,
            sd_path: str,
            cn_path: str,
            r_range: tuple,
            g_range: tuple,
            b_range: tuple,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.r_range = r_range
        self.g_range = g_range
        self.b_range = b_range

        print(f"[INFO] Loading single ControlNet from: {cn_path}")
        controlnet = ControlNetModel.from_pretrained(
            cn_path, torch_dtype=torch.float16
        ).to(self.device)

        print(f"[INFO] Loading SD1.5 from: {sd_path}")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_path,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("[INFO] xFormers memory efficient attention activated.")
        except Exception as e:
            print("[WARNING] xFormers not available:", e)

    def _split_channels(self, poly_edge: np.ndarray, outname: str = "example"):
        if poly_edge.ndim != 3 or poly_edge.shape[2] != 3:
            raise ValueError("poly_edge must be (H,W,3).")

        r = poly_edge[:, :, 0]
        g = poly_edge[:, :, 1]
        b = poly_edge[:, :, 2]

        # 分别保存到 refine/r, refine/g, refine/b 以便可视化
        os.makedirs("refine/r", exist_ok=True)
        os.makedirs("refine/g", exist_ok=True)
        os.makedirs("refine/b", exist_ok=True)

        # 注意：OpenCV保存单通道图时，会看起来是灰度，为了可视化区分，
        # 也可以再转换成伪彩色。这里为了简单，直接保存灰度即可。
        cv2.imwrite(f"refine/r/{outname}", r)
        cv2.imwrite(f"refine/g/{outname}", g)
        cv2.imwrite(f"refine/b/{outname}", b)

        # 转为 PIL
        r_pil = Image.fromarray(r)
        g_pil = Image.fromarray(g)
        b_pil = Image.fromarray(b)

        return r_pil, g_pil, b_pil

    def _prep_control_img(self, pil_img: Image.Image) -> torch.Tensor:
        """
        (H,W) single‐channel => (1,3,H,W) scaled to [0,1], float16.
        ControlNet 的输入需要是三通道彩色图，但实际上三通道都是一样的灰度即可。
        同时，保证宽高是 8 的倍数。
        """
        w, h = pil_img.size
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        new_w = max(new_w, 8)
        new_h = max(new_h, 8)
        if (new_w != w) or (new_h != h):
            pil_img = pil_img.resize((new_w, new_h), Image.BICUBIC)

        arr = np.array(pil_img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.shape[2] == 1:
            arr = np.concatenate([arr, arr, arr], axis=2)
        if arr.max() > 1.0:
            arr = arr / 255.0

        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device, torch.float16)
        return tensor

    def _compute_scale(self, step_i: int, total_steps: int, rng: tuple):
        """
        用于判断在第 i 步时，某个通道是否要启用控制。
        如果 ratio = i / (total_steps - 1) 落在 [rng[0], rng[1]) 内，则返回1.0，否则返回0.0
        """
        ratio = step_i / float(total_steps - 1)
        return 1.0 if (rng[0] <= ratio < rng[1]) else 0.0

    @torch.no_grad()
    def __call__(
            self,
            poly_edge_image: np.ndarray,
            prompt: str,
            num_inference_steps: int,
            guidance_scale: float,
            seed: Optional[int] = None,
            outname: str = "example.png",
    ) -> Image.Image:
        # 1) Split R/G/B
        r_pil, g_pil, b_pil = self._split_channels(poly_edge_image, outname=outname)

        # 2) 准备 ControlNet 的输入
        c_r = self._prep_control_img(r_pil)
        c_g = self._prep_control_img(g_pil)
        c_b = self._prep_control_img(b_pil)

        # 3) 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        do_cfg = guidance_scale > 1.0
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        # 4) 编码 prompt
        text_embeds = self.pipe._encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt="",
        )

        # 5) 初始化 latent
        bs, _, H, W = c_r.shape
        in_channels = self.pipe.unet.config.in_channels
        latents = torch.randn(
            (bs, in_channels, H // 8, W // 8),
            generator=torch.Generator(device=self.device),
            device=self.device,
            dtype=text_embeds.dtype,
        )
        latents *= self.pipe.scheduler.init_noise_sigma

        # 6) 确定通道注入区间
        rr = (self.r_range[0], self.r_range[1])
        gr = (self.g_range[0], self.g_range[1])
        br = (self.b_range[0], self.b_range[1])

        # 7) 逐步去噪
        for i, t in enumerate(timesteps):
            scale_r = self._compute_scale(i, num_inference_steps, rr)
            scale_g = self._compute_scale(i, num_inference_steps, gr)
            scale_b = self._compute_scale(i, num_inference_steps, br)

            # expand if CFG
            if do_cfg:
                lat_in = torch.cat([latents, latents], dim=0)
            else:
                lat_in = latents

            lat_in = self.pipe.scheduler.scale_model_input(lat_in, t)
            noise_agg = torch.zeros_like(lat_in, device=self.device)

            def forward_channel(cond_img: torch.Tensor, sc: float):
                if sc < 1e-7:
                    return None
                ctrl_out = self.pipe.controlnet(
                    lat_in,
                    t,
                    encoder_hidden_states=text_embeds,
                    controlnet_cond=cond_img,
                    return_dict=False,
                )
                down_res, mid_res = ctrl_out
                unet_out_tuple = self.pipe.unet(
                    lat_in,
                    t,
                    encoder_hidden_states=text_embeds,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                    return_dict=False,
                )
                return unet_out_tuple[0] * sc

            # R
            r_pred = forward_channel(c_r, scale_r)
            if r_pred is not None:
                noise_agg += r_pred

            # G
            g_pred = forward_channel(c_g, scale_g)
            if g_pred is not None:
                noise_agg += g_pred

            # B
            b_pred = forward_channel(c_b, scale_b)
            if b_pred is not None:
                noise_agg += b_pred

            # 如果本 step 所有通道都没参与，则 fallback 到 Unet 默认前向
            if torch.all(noise_agg.abs() < 1e-8):
                unet_def = self.pipe.unet(
                    lat_in, t, encoder_hidden_states=text_embeds, return_dict=False
                )
                noise_agg = unet_def[0]

            # CFG
            if do_cfg:
                half = noise_agg.shape[0] // 2
                noise_uncond, noise_text = noise_agg[:half], noise_agg[half:]
                noise_agg = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            latents = self.pipe.scheduler.step(
                model_output=noise_agg, timestep=t, sample=latents, return_dict=False
            )[0]

        # 8) 解码 latents => final image
        vae_out = self.pipe.vae.decode(latents / 0.18215, return_dict=False)[0]
        vae_out = (vae_out / 2 + 0.5).clamp(0, 1)
        arr = vae_out.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
        arr = (arr * 255).astype(np.uint8)
        out_img = Image.fromarray(arr)
        return out_img


def get_polyedege(image_dir, image_name, predictor):
    source_image_path = os.path.join(image_dir, image_name)
    bgr = cv2.imread(source_image_path, cv2.IMREAD_COLOR)
    source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    combined_mask = get_segmentation_mask(predictor, source)
    combined_mask_pil = Image.fromarray(combined_mask)
    combined_mask_pil.save(os.path.join(image_dir, f"mask.png"))

    outline_edge = generate_edge_skeleton(combined_mask)
    outline_edge_pil = Image.fromarray(outline_edge)
    outline_edge_pil.save(os.path.join(image_dir, f"outline_edge.png"))

    segmentation_mask_fine = fill_skeleton_region(outline_edge)
    segmentation_mask_fine_pil = Image.fromarray(segmentation_mask_fine)
    segmentation_mask_fine_pil.save(os.path.join(image_dir, f"segmentation_mask_fine.png"))


    subject, background = get_subject_and_background(source, segmentation_mask_fine)
    subject_pil = Image.fromarray(subject)
    subject_pil.save(os.path.join(image_dir, f"subject.png"))
    background_pil = Image.fromarray(background)
    background_pil.save(os.path.join(image_dir, f"background.png"))

    detail_edge, background_edge = get_subject_and_background_edge(subject, background, outline_edge)
    detail_edge_pil = Image.fromarray(detail_edge)
    detail_edge_pil.save(os.path.join(image_dir, f"detail_edge.png"))
    background_edge_pil = Image.fromarray(background_edge)
    background_edge_pil.save(os.path.join(image_dir, f"background_edge.png"))

    polyEdge = Image.merge('RGB', (Image.fromarray(outline_edge), Image.fromarray(detail_edge),
                                   Image.fromarray(background_edge)))
    polyEdge_path = os.path.join(image_dir, f"polyEdge.png")
    polyEdge.save(polyEdge_path)

    bgr = cv2.imread(polyEdge_path, cv2.IMREAD_COLOR)
    if bgr is None:
        assert False, f"[ERROR] Failed to read image: {polyEdge_path}"
    poly_edge = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return poly_edge


if __name__ == "__main__":
    sam_path = "model/sam-vit/sam_vit_l.pth"
    sam = sam_model_registry['vit_l'](checkpoint=sam_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择设备
    sam.to(device=device)
    predictor = SamPredictor(sam)

    get_polyedege('refine', 'source.jpg', predictor)
    polyEdge_path = os.path.join('refine', f"polyEdge.png")
    bgr = cv2.imread(polyEdge_path, cv2.IMREAD_COLOR)
    if bgr is None:
        assert False, f"[ERROR] Failed to read image: {polyEdge_path}"
    poly_edge = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    prompt_str = "a abyssinian cat"
    out_name = "result.png"

    cfg = ConfigManager("./config.yaml")
    pipeline = MultiCannyPolyEdgePipeline(
        sd_path=cfg.sd1_5_path,
        cn_path=cfg.controlnet_canny_path,
        r_range=(cfg.r_channel_start, cfg.r_channel_end),
        g_range=(cfg.g_channel_start, cfg.g_channel_end),
        b_range=(cfg.b_channel_start, cfg.b_channel_end),
    )

    result_img = pipeline(
        poly_edge_image=poly_edge,
        prompt=prompt_str,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        seed=cfg.seed,
        outname=out_name,
    )

    out_final_path = os.path.join("refine", out_name)
    result_img.save(out_final_path)
    print(f"[INFO] Done! Saved final => {out_final_path}")