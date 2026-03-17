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

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)


################################################
# 1) Config
################################################
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



################################################
# 2) MultiCannyPolyEdgePipeline
#    (One ControlNet, partial injection)
################################################
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
        """Return PIL single‐channel images for R/G/B from (H,W,3).
           并另存分离后的r/g/b图像到 results/r, results/g, results.b
        """
        if poly_edge.ndim != 3 or poly_edge.shape[2] != 3:
            raise ValueError("poly_edge must be (H,W,3).")



        r = poly_edge[:, :, 0]
        g = poly_edge[:, :, 1]
        b = poly_edge[:, :, 2]

        # 分别保存到 results/r, results/g, results/b 以便可视化
        os.makedirs("results/r", exist_ok=True)
        os.makedirs("results/g", exist_ok=True)
        os.makedirs("results/b", exist_ok=True)

        # 注意：OpenCV保存单通道图时，会看起来是灰度，为了可视化区分，
        # 也可以再转换成伪彩色。这里为了简单，直接保存灰度即可。
        cv2.imwrite(f"results/r/{outname}", r)
        cv2.imwrite(f"results/g/{outname}", g)
        cv2.imwrite(f"results/b/{outname}", b)

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


################################################
# 3) Main
################################################
def main():
    # 1) 加载配置
    cfg = ConfigManager("./config.yaml")

    # 2) 初始化 pipeline
    pipeline = MultiCannyPolyEdgePipeline(
        sd_path=cfg.sd1_5_path,
        cn_path=cfg.controlnet_canny_path,
        r_range=(cfg.r_channel_start, cfg.r_channel_end),
        g_range=(cfg.g_channel_start, cfg.g_channel_end),
        b_range=(cfg.b_channel_start, cfg.b_channel_end),
    )

    # 3) 读取 testData/prompt.json
    prompt_json = os.path.join("testData", "prompt.json")
    if not os.path.exists(prompt_json):
        raise FileNotFoundError(f"prompt.json not found at: {prompt_json}")

    with open(prompt_json, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # 4) 遍历 prompt.json 中的每个样本
    for idx, item in enumerate(data_list):
        source_rel = item["source"]  # e.g. "source/3.png"
        prompt_str = item["prompt"]  # e.g. "a tiger in the grass"

        # 构造绝对路径
        source_path = os.path.join("testData", source_rel)
        if not os.path.exists(source_path):
            print(f"[WARNING] {source_path} not found, skip...")
            continue

        # 读取 polyEdge (三通道)
        bgr = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[ERROR] Failed to read image: {source_path}")
            continue

        # OpenCV 默认读取是 BGR，我们需要转为 RGB
        poly_edge = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 生成 outname 与当前 source 文件同名，以区分结果
        outname = os.path.basename(source_path)  # 比如 "3.png"

        # 5) 执行推理
        print(f"\n[INFO] Inference idx={idx}, source={source_path}, prompt='{prompt_str}'")
        result_img = pipeline(
            poly_edge_image=poly_edge,
            prompt=prompt_str,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            seed=cfg.seed,
            outname=outname,
        )

        # 6) 保存最终合成结果
        os.makedirs("results/results", exist_ok=True)
        out_final_path = os.path.join("results/results", outname)
        result_img.save(out_final_path)
        print(f"[INFO] Done! Saved final => {out_final_path}")


if __name__ == "__main__":
    main()
