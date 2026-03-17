#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import json
import torch
import numpy as np
import cv2
from typing import Optional, Union, List, Tuple
from PIL import Image
import matplotlib.pyplot as plt

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)


################################################
# 1) Config
################################################
class ConfigManager:
    def __init__(self, config_path: str = "./inference/config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
    # model
    @property
    def sd1_5_path(self):
        return self.config["model"]["sd1_5_path"]

    @property
    def controlnet_canny_path(self):
        return self.config["model"]["controlnet_canny_path"]

    # control injection
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
        """Return seed value. -1 means using random seed"""
        seed = int(self.config["inference"]["seed"])
        if seed == -1:
            return None  # 返回None表示使用随机种子
        return seed

    # data
    @property
    def input_path(self):
        return self.config["data"]["input_path"]
    # output
    @property
    def save_path(self):
        return self.config["output"]["save_path"]

    @property
    def debug_shape_mismatch(self):
        return bool(self.config["output"]["debug_shape_mismatch"])


################################################
# 2) Channel Processing
################################################
class ChannelProcessor:
    """处理RGB通道的组合与分段."""

    @staticmethod
    def simulate_channel_process(
            r_range: Tuple[float, float],
            g_range: Tuple[float, float],
            b_range: Tuple[float, float]
    ) -> Tuple[List[str], List[List[float]]]:
        """模拟通道处理过程，返回每个时间段的活跃通道组合."""
        points = sorted(set([0, 1,
                             r_range[0], r_range[1],
                             g_range[0], g_range[1],
                             b_range[0], b_range[1]]))

        channel_list = []
        channel_start_end_list = [[], []]
        prev_point = points[0]

        for current_point in points[1:]:
            active_channels = []
            if r_range[0] <= prev_point < r_range[1]:
                active_channels.append('r')
            if g_range[0] <= prev_point < g_range[1]:
                active_channels.append('g')
            if b_range[0] <= prev_point < b_range[1]:
                active_channels.append('b')

            if not active_channels:
                active_channels.append('none')

            active_channels_sorted = ''.join(sorted(active_channels))
            channel_list.append(active_channels_sorted)
            channel_start_end_list[0].append(prev_point)
            channel_start_end_list[1].append(current_point)
            prev_point = current_point

        return channel_list, channel_start_end_list

    @staticmethod
    def process_image_channels(
            rgb_image: np.ndarray,
            r_range: Tuple[float, float],
            g_range: Tuple[float, float],
            b_range: Tuple[float, float]
    ) -> Tuple[List[np.ndarray], List[List[float]]]:
        """处理图像通道，返回组合后的图像列表和对应的时间区间."""
        r, g, b = cv2.split(rgb_image)
        channel_map = {'r': r, 'g': g, 'b': b}

        channel_list_str, channel_start_end_list = ChannelProcessor.simulate_channel_process(
            r_range, g_range, b_range
        )

        channel_list_img = []
        for channels in channel_list_str:
            if channels == 'none':
                combined = np.zeros_like(r)
            else:
                combined = np.zeros_like(r, dtype=np.float32)
                for ch in channels:
                    combined += channel_map[ch].astype(np.float32)
                combined = np.clip(combined, 0, 255).astype(np.uint8)

            channel_list_img.append(combined)

        return channel_list_img, channel_start_end_list


################################################
# 3) MultiControlNetPipeline3Chan
################################################
class MultiControlNetPipeline3Chan:
    def __init__(
            self,
            sd_path: str,
            cn_path: str,
            r_range: tuple,
            g_range: tuple,
            b_range: tuple,
            debug_shape: bool = True,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.r_range = r_range
        self.g_range = g_range
        self.b_range = b_range
        self.debug_shape = debug_shape

        # 初始化通道处理器
        self.channel_processor = ChannelProcessor()

        # 加载ControlNet (只需要一个实例)
        print(f"[INFO] Loading ControlNet from: {cn_path}")
        self.controlnet = ControlNetModel.from_pretrained(
            cn_path, torch_dtype=torch.float16
        ).to(self.device)

        # 加载SD1.5
        print(f"[INFO] Loading SD1.5 from: {sd_path}")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_path,
            controlnet=self.controlnet,
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

    def _prep_control_img(self, img_array: np.ndarray) -> torch.Tensor:
        """准备控制图像."""
        if img_array.ndim == 2:
            img_array = img_array[..., None]
        if img_array.shape[2] == 1:
            img_array = np.concatenate([img_array] * 3, axis=2)

        h, w = img_array.shape[:2]
        new_h = ((h - 1) // 8 + 1) * 8
        new_w = ((w - 1) // 8 + 1) * 8

        if (new_h != h) or (new_w != w):
            if self.debug_shape:
                print(f"[DEBUG] resizing from {w}x{h} => {new_w}x{new_h}")
            img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if img_array.max() > 1.0:
            img_array = img_array / 255.0

        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device, torch.float16)

    def _get_active_scale(self, step_i: int, total_steps: int, ranges: List[List[float]]) -> float:
        """获取当前步骤的控制强度."""
        ratio = step_i / float(total_steps - 1)
        for start, end in zip(ranges[0], ranges[1]):
            if start <= ratio < end:
                return 1.0
        return 0.0

    @torch.no_grad()
    def __call__(
            self,
            poly_edge_image: np.ndarray,
            prompt: str,
            num_inference_steps: int,
            guidance_scale: float,
            seed: Optional[int] = None,
    ) -> Image.Image:
        # 1) 处理通道组合
        combined_channels, time_ranges = self.channel_processor.process_image_channels(
            poly_edge_image,
            (self.r_range[0], self.r_range[1]),
            (self.g_range[0], self.g_range[1]),
            (self.b_range[0], self.b_range[1])
        )

        # 2) 准备控制图像
        control_images = [self._prep_control_img(img) for img in combined_channels]

        # 3) 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        do_cfg = guidance_scale > 1.0
        pipe = self.pipe
        pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = pipe.scheduler.timesteps

        # 4) 编码prompt
        text_embeds = pipe._encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt="",
        )

        # 5) 初始化latents
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=pipe.unet.config.in_channels,
            height=poly_edge_image.shape[0],
            width=poly_edge_image.shape[1],
            dtype=text_embeds.dtype,
            device=self.device,
            generator=generator,
        )

        # 6) 逐步去噪
        for i, t in enumerate(timesteps):
            # 计算当前步骤的控制强度
            scale = self._get_active_scale(i, num_inference_steps, time_ranges)

            if do_cfg:
                lat_in = torch.cat([latents] * 2)
            else:
                lat_in = latents

            lat_in = pipe.scheduler.scale_model_input(lat_in, t)

            # 确定当前活跃的控制图像
            active_idx = None
            for idx, (start, end) in enumerate(zip(time_ranges[0], time_ranges[1])):
                ratio = i / float(num_inference_steps - 1)
                if start <= ratio < end:
                    active_idx = idx
                    break

            if active_idx is not None and scale > 0:
                # 使用当前活跃的控制图像
                control_img = control_images[active_idx]

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    lat_in,
                    t,
                    encoder_hidden_states=text_embeds,
                    controlnet_cond=control_img,
                    return_dict=False,
                )

                # 应用控制信号
                noise_pred = pipe.unet(
                    lat_in,
                    t,
                    encoder_hidden_states=text_embeds,
                    down_block_additional_residuals=[d * scale for d in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample * scale,
                ).sample
            else:
                # 不使用控制信号
                noise_pred = pipe.unet(
                    lat_in,
                    t,
                    encoder_hidden_states=text_embeds,
                ).sample

            # CFG
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # 更新latents
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # 7) 解码得到最终图像
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]

        return image


################################################
# 4) Main
################################################
def main():
    cfg = ConfigManager()

    pipeline = MultiControlNetPipeline3Chan(
        sd_path=cfg.sd1_5_path,
        cn_path=cfg.controlnet_canny_path,
        r_range=(cfg.r_channel_start, cfg.r_channel_end),
        g_range=(cfg.g_channel_start, cfg.g_channel_end),
        b_range=(cfg.b_channel_start, cfg.b_channel_end),
        debug_shape=cfg.debug_shape_mismatch,
    )

    prompt_json = os.path.join(cfg.input_path, "prompt.json")
    if not os.path.exists(prompt_json):
        raise FileNotFoundError(f"prompt.json not found: {prompt_json}")

    with open(prompt_json, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    for idx, item in enumerate(data_list):
        source_rel = item["source"]
        target_rel = item["target"]
        prompt_str = item["prompt"]

        source_path = os.path.join(cfg.input_path, source_rel)
        if not os.path.exists(source_path):
            print(f"[WARNING] {source_path} not found, skip...")
            continue

        target_path = os.path.join(cfg.input_path, target_rel)
        if not os.path.exists(target_path):
            print(f"[WARNING] {target_path} not found, skip...")
            continue

        bgr_source = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if bgr_source is None:
            print(f"[ERROR] Failed to read image: {source_path}")
            continue
        poly_edge = cv2.cvtColor(bgr_source, cv2.COLOR_BGR2RGB)

        bgr_target = cv2.imread(target_path, cv2.IMREAD_COLOR)
        if bgr_target is None:
            print(f"[ERROR] Failed to read image: {target_path}")
            continue
        target = cv2.cvtColor(bgr_target, cv2.COLOR_BGR2RGB)

        print(f"\n[INFO] Inference idx={idx}, source={source_path}, prompt='{prompt_str}'")
        result_img = pipeline(
            poly_edge_image=poly_edge,
            prompt=prompt_str,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            seed=cfg.seed,
        )
        resultDir = os.path.join(cfg.save_path, "result")
        # resultDir = "./results2/G(" + str(cfg.g_channel_start) + "->"+ str(cfg.g_channel_end) + "),B(" + str(cfg.b_channel_start) + "->" + str(cfg.b_channel_end) + ")result"
        os.makedirs(resultDir, exist_ok=True)
        save_path = os.path.join(resultDir, os.path.basename(source_path))
        result_img.save(save_path)
        print(f"[INFO] Done! Saved generated image => {save_path}")

        # 创建一个包含三张图的布局
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # 显示每张图并添加标题
        axes[0].imshow(poly_edge)
        axes[0].set_title('PolyEdge')
        axes[0].axis('off')
        axes[1].imshow(target)
        axes[1].set_title('GroundTruth')
        axes[1].axis('off')
        axes[2].imshow(result_img)
        axes[2].set_title('Result')
        axes[2].axis('off')
        # 调整子图之间的间距
        plt.tight_layout()
        fig.text(0.5, -0.05, "prompt:" + prompt_str, ha='center', fontsize=18)

        # 保存图像
        mergeDir = os.path.join(cfg.input_path, "merge")
        # mergeDir = "./results2/G(" + str(cfg.g_channel_start) + "->"+ str(cfg.g_channel_end) + "),B(" + str(cfg.b_channel_start) + "->" + str(cfg.b_channel_end) + ")merge"
        os.makedirs(mergeDir, exist_ok=True)
        save_path = os.path.join(mergeDir, os.path.basename(source_path))
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2, facecolor='white')
        print(f"[INFO] Done! Saved merged image => {save_path}")

def runAtime(g_channel_start, g_channel_end, b_channel_start, b_channel_end):
    cfg = ConfigManager()
    pipeline = MultiControlNetPipeline3Chan(
        sd_path=cfg.sd1_5_path,
        cn_path=cfg.controlnet_canny_path,
        r_range=(cfg.r_channel_start, cfg.r_channel_end),
        g_range=(g_channel_start, g_channel_end),
        b_range=(b_channel_start, b_channel_end),
        debug_shape=cfg.debug_shape_mismatch,
    )

    prompt_json = os.path.join(cfg.input_path, "prompt.json")
    if not os.path.exists(prompt_json):
        raise FileNotFoundError(f"prompt.json not found: {prompt_json}")

    with open(prompt_json, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    for idx, item in enumerate(data_list):
        source_rel = item["source"]
        target_rel = item["target"]
        prompt_str = item["prompt"]

        source_path = os.path.join(cfg.input_path, source_rel)
        if not os.path.exists(source_path):
            print(f"[WARNING] {source_path} not found, skip...")
            continue

        target_path = os.path.join(cfg.input_path, target_rel)
        if not os.path.exists(target_path):
            print(f"[WARNING] {target_path} not found, skip...")
            continue

        bgr_source = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if bgr_source is None:
            print(f"[ERROR] Failed to read image: {source_path}")
            continue
        poly_edge = cv2.cvtColor(bgr_source, cv2.COLOR_BGR2RGB)

        bgr_target = cv2.imread(target_path, cv2.IMREAD_COLOR)
        if bgr_target is None:
            print(f"[ERROR] Failed to read image: {target_path}")
            continue
        target = cv2.cvtColor(bgr_target, cv2.COLOR_BGR2RGB)

        print(f"\n[INFO] Inference idx={idx}, source={source_path}, prompt='{prompt_str}'")
        result_img = pipeline(
            poly_edge_image=poly_edge,
            prompt=prompt_str,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            seed=cfg.seed,
        )

        resultDir = "./results2/G(" + str(g_channel_start) + "->"+ str(g_channel_end) + "),B(" + str(b_channel_start) + "->" + str(b_channel_end) + ")result"
        os.makedirs(resultDir, exist_ok=True)
        save_path = os.path.join(resultDir, os.path.basename(source_path))
        result_img.save(save_path)
        print(f"[INFO] Done! Saved generated image => {save_path}")

        # 创建一个包含三张图的布局
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # 显示每张图并添加标题
        axes[0].imshow(poly_edge)
        axes[0].set_title('PolyEdge')
        axes[0].axis('off')
        axes[1].imshow(target)
        axes[1].set_title('GroundTruth')
        axes[1].axis('off')
        axes[2].imshow(result_img)
        axes[2].set_title('Result')
        axes[2].axis('off')
        # 调整子图之间的间距
        plt.tight_layout()
        fig.text(0.5, -0.05, "prompt:" + prompt_str, ha='center', fontsize=18)

        # 保存图像
        mergeDir = "./results2/G(" + str(g_channel_start) + "->"+ str(g_channel_end) + "),B(" + str(b_channel_start) + "->" + str(b_channel_end) + ")merge"
        os.makedirs(mergeDir, exist_ok=True)
        save_path = os.path.join(mergeDir, os.path.basename(source_path))
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2, facecolor='white')
        print(f"[INFO] Done! Saved merged image => {save_path}")

if __name__ == "__main__":
    main()
    # for g_channel_start in range(0, 11):
    #     for g_channel_end in range(g_channel_start, 11):
    #         for b_channel_start in range(0, 10):
    #             for b_channel_end in range(b_channel_start, 11):
    #                 print(g_channel_start,g_channel_end,b_channel_start,b_channel_end)
    #                 runAtime(g_channel_start/10.0, g_channel_end/10.0, b_channel_start/10.0, b_channel_end/10.0)