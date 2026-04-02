#!/usr/bin/env bash
set -euo pipefail

# LanPaint + Z-Image + ControlNet 示例脚本（放在仓库根目录）
# 说明：脚本在根目录；运行结果保存到 results/new/ 目录。
# 使用前请将以下路径替换为你的本地文件。

IMAGE_PATH="path/to/original.png"
MASK_PATH="path/to/mask.png"
POLYEDGE_PATH="path/to/polyedge.png"

mkdir -p results/new

python run_lanpaint.py \
  --model z-image-controlnet \
  --prompt "restore the removed subject region with natural background continuation" \
  --image "${IMAGE_PATH}" \
  --mask "${MASK_PATH}" \
  --polyedge "${POLYEDGE_PATH}" \
  --controlnet-model-id "lllyasviel/sd-controlnet-canny" \
  --sd15-model-id "runwayml/stable-diffusion-v1-5" \
  --control-r-start 0.0 \
  --control-r-end 1.0 \
  --control-g-start 0.3 \
  --control-g-end 0.6 \
  --control-b-start 0.0 \
  --control-b-end 0.3 \
  --lp-n-steps 5 \
  --lp-friction 15.0 \
  --lp-lambda 16.0 \
  --guidance-scale 5.0 \
  --num-steps 20 \
  --seed 0 \
  --output results/new/lanpaint_zimage_controlnet_output.png

printf "Done. Output saved to results/new/lanpaint_zimage_controlnet_output.png\n"
# Z-Image Turbo Inpaint
# python run_lanpaint.py --model z-image \
#     --lp-n-steps 5 \
#     --lp-friction 15.0 \
#     --lp-lambda 16 \
#     --seed 0 \
#     --guidance-scale 1.0 \
#     --num-steps 9 \
#     --prompt "Latina female with thick wavy hair, white shirt, harbor boats and pastel houses behind. Breezy seaside light, warm tones, cinematic close-up." \
#     --image "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_21/Original_No_Mask.png" \
#     --mask "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_21/Masked_Load_Me_in_Loader.png"

# Z-Image Turbo Outpaint
# python run_lanpaint.py --model z-image \
#     --lp-n-steps 5 \
#     --lp-friction 15.0 \
#     --lp-lambda 16 \
#     --seed 42 \
#     --guidance-scale 1.0 \
#     --num-steps 15 \
#     --prompt "Latina female with thick wavy hair, white shirt, harbor boats and pastel houses behind. Breezy seaside light, warm tones, cinematic close-up." \
#     --image "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_22/Original_No_Mask.png" \
#     --outpaint-pad "l200r200t200b200" \

# Qwen Image Edit Inpaint
python run_lanpaint.py --model qwen \
    --prompt "change the girl's cloth to red evening gown" \
    --image "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_14/Original_No_Mask.png" \
    --mask "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_14/Masked_Load_Me_in_Loader.png" \
    --seed 0 \
    --num-steps 20 \
    --guidance-scale 2.5 \
    --lp-n-steps 5
