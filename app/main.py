import os
import cv2
import json
from manager import ConfigManager
from multi_canny_poly_edge import MultiCannyPolyEdgePipeline


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
