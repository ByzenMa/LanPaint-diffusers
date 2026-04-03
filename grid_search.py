#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parameter grid-search runner for run_lanpaint.py.

Supports:
- model: z-image / z-image-controlnet
- parameters:
  - lp-n-steps
  - lp-friction
  - lp-lambda
  - guidance-scale
  - num-steps

Each combination is saved to output dir with filename containing the parameter values.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import shlex
import subprocess
import time
from typing import List, Sequence, Tuple

def _build_range(start: int, end: int, step: int) -> List[int]:
    return list(range(start, end + 1, step))


def _uniform_sample_grid(
    grid: Sequence[Tuple[int, float, float, float, int]],
    sample_ratio: float,
) -> List[Tuple[int, float, float, float, int]]:
    if not (0 < sample_ratio <= 1):
        raise ValueError("--sample-ratio must be in (0, 1].")

    total = len(grid)
    target = max(1, math.ceil(total * sample_ratio))
    if target >= total:
        return list(grid)

    if target == 1:
        return [grid[0]]

    # Uniformly pick combinations over the whole ordered grid.
    idxs = [round(i * (total - 1) / (target - 1)) for i in range(target)]
    # De-duplicate in rare rounding collisions while preserving order.
    unique_idxs = list(dict.fromkeys(idxs))
    return [grid[i] for i in unique_idxs]


def _fmt_float(v: float) -> str:
    text = f"{v:.6g}"
    return text.replace("-", "m").replace(".", "p")


def build_filename(model: str, lp_n_steps: int, lp_friction: float, lp_lambda: float, guidance_scale: float, num_steps: int) -> str:
    return (
        f"{model}"
        f"_lpn{lp_n_steps}"
        f"_fr{_fmt_float(lp_friction)}"
        f"_lam{_fmt_float(lp_lambda)}"
        f"_cfg{_fmt_float(guidance_scale)}"
        f"_steps{num_steps}.png"
    )


def build_filename_with_metric(
    model: str,
    lp_n_steps: int,
    lp_friction: float,
    lp_lambda: float,
    guidance_scale: float,
    num_steps: int,
    metric_name: str,
    metric_value: float,
) -> str:
    return (
        f"{model}"
        f"_lpn{lp_n_steps}"
        f"_fr{_fmt_float(lp_friction)}"
        f"_lam{_fmt_float(lp_lambda)}"
        f"_cfg{_fmt_float(guidance_scale)}"
        f"_steps{num_steps}"
        f"_{metric_name}{_fmt_float(metric_value)}.png"
    )


def compute_psnr(reference_path: str, output_path: str) -> float:
    from PIL import Image, ImageChops

    ref = Image.open(reference_path).convert("RGB")
    out = Image.open(output_path).convert("RGB")

    if out.size != ref.size:
        out = out.resize(ref.size, Image.BICUBIC)

    diff = ImageChops.difference(ref, out)
    hist = diff.histogram()
    sq_err = 0.0
    for channel in range(3):
        offset = channel * 256
        sq_err += sum((value * value) * hist[offset + value] for value in range(256))

    mse = sq_err / float(ref.size[0] * ref.size[1] * 3)
    if mse == 0:
        return 100.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def run_grid_search(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Fixed ranges requested by user:
    # 1) lp-n-steps: [1, 20], step 2
    # 2) lp-friction: [1, 20], step 5
    # 3) lp-lambda: [1, 20], step 3
    # 4) guidance-scale: [1, 15], step 1
    # 5) num-steps: [1, 30], step 2
    lp_n_steps_list: List[int] = _build_range(1, 20, 2)
    lp_friction_list: List[float] = [float(v) for v in _build_range(1, 20, 5)]
    lp_lambda_list: List[float] = [float(v) for v in _build_range(1, 20, 3)]
    guidance_scale_list: List[float] = [float(v) for v in _build_range(1, 15, 1)]
    num_steps_list: List[int] = _build_range(1, 30, 2)

    grid = list(itertools.product(
        lp_n_steps_list,
        lp_friction_list,
        lp_lambda_list,
        guidance_scale_list,
        num_steps_list,
    ))
    sampled_grid = _uniform_sample_grid(grid, args.sample_ratio)

    total_points = len(grid)
    sampled_points = len(sampled_grid)
    print("=" * 80)
    print(f"Grid points (total): {total_points}")
    print(f"Sample points (uniform): {sampled_points}")
    print(f"Sample ratio: {args.sample_ratio}")
    print(f"Eval metric: {args.eval_metric}")
    print("=" * 80)

    t0 = time.time()
    best_metric = float("-inf")
    best_file = ""
    best_params = ""
    for idx, (lp_n_steps, lp_friction, lp_lambda, guidance_scale, num_steps) in enumerate(sampled_grid, start=1):
        filename = build_filename(
            model=args.model,
            lp_n_steps=lp_n_steps,
            lp_friction=lp_friction,
            lp_lambda=lp_lambda,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
        )
        out_path = os.path.join(args.output_dir, filename)

        cmd = [
            "python",
            "run_lanpaint.py",
            "--model", args.model,
            "--prompt", args.prompt,
            "--image", args.image,
            "--mask", args.mask,
            "--lp-n-steps", str(lp_n_steps),
            "--lp-friction", str(lp_friction),
            "--lp-lambda", str(lp_lambda),
            "--guidance-scale", str(guidance_scale),
            "--num-steps", str(num_steps),
            "--seed", str(args.seed),
            "--output", out_path,
        ]

        if args.model == "z-image-controlnet":
            if not args.polyedge:
                raise ValueError("--polyedge is required when --model z-image-controlnet")
            cmd.extend(["--polyedge", args.polyedge])
            if args.controlnet_model_id:
                cmd.extend(["--controlnet-model-id", args.controlnet_model_id])
            if args.sd15_model_id:
                cmd.extend(["--sd15-model-id", args.sd15_model_id])
            cmd.extend([
                "--control-r-start", str(args.control_r_start),
                "--control-r-end", str(args.control_r_end),
                "--control-g-start", str(args.control_g_start),
                "--control-g-end", str(args.control_g_end),
                "--control-b-start", str(args.control_b_start),
                "--control-b-end", str(args.control_b_end),
            ])

        progress = 100.0 * idx / sampled_points
        elapsed = time.time() - t0
        print(f"[{idx}/{sampled_points}] ({progress:6.2f}%) elapsed={elapsed:8.1f}s")
        print(f"Running: {' '.join(shlex.quote(c) for c in cmd)}")
        subprocess.run(cmd, check=True)

        if args.eval_metric == "psnr":
            metric_value = compute_psnr(args.image, out_path)
        else:
            raise ValueError(f"Unsupported --eval-metric: {args.eval_metric}")

        metric_filename = build_filename_with_metric(
            model=args.model,
            lp_n_steps=lp_n_steps,
            lp_friction=lp_friction,
            lp_lambda=lp_lambda,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            metric_name=args.eval_metric,
            metric_value=metric_value,
        )
        metric_out_path = os.path.join(args.output_dir, metric_filename)
        os.replace(out_path, metric_out_path)

        params_text = (
            f"lpn={lp_n_steps}, fr={lp_friction}, lam={lp_lambda}, "
            f"cfg={guidance_scale}, steps={num_steps}"
        )

        if metric_value > best_metric:
            best_metric = metric_value
            best_file = metric_filename
            best_params = params_text

        print(
            f"Current metric: {args.eval_metric.upper()}={metric_value:.4f} | "
            f"Best: {args.eval_metric.upper()}={best_metric:.4f} | "
            f"Best file: {best_file} | Best params: {best_params}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search launcher for LanPaint runs")
    parser.add_argument("--model", choices=["z-image", "z-image-controlnet"], required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--polyedge", default=None, help="Required for z-image-controlnet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Uniform sampling ratio over full grid in (0,1], e.g. 0.2 means 20%% of combinations.",
    )
    parser.add_argument(
        "--eval-metric",
        choices=["psnr"],
        default="psnr",
        help="Evaluation metric against input image. Higher is better.",
    )

    parser.add_argument("--controlnet-model-id", default=None)
    parser.add_argument("--sd15-model-id", default=None)
    parser.add_argument("--control-r-start", type=float, default=0.0)
    parser.add_argument("--control-r-end", type=float, default=1.0)
    parser.add_argument("--control-g-start", type=float, default=0.3)
    parser.add_argument("--control-g-end", type=float, default=0.6)
    parser.add_argument("--control-b-start", type=float, default=0.0)
    parser.add_argument("--control-b-end", type=float, default=0.3)

    return parser.parse_args()


def main():
    args = parse_args()
    run_grid_search(args)


if __name__ == "__main__":
    main()
