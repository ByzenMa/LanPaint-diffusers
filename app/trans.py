import numpy as np
import cv2  # 确保您的环境中已安装OpenCV


def simulate_channel_process(r_channel_start, r_channel_end,
                             g_channel_start, g_channel_end,
                             b_channel_start, b_channel_end):
    # 收集所有时间点并排序，去重
    points = sorted(set([0, 1, r_channel_start, r_channel_end,
                         g_channel_start, g_channel_end,
                         b_channel_start, b_channel_end]))

    channel_list = []
    channel_start_end_list = [[], []]
    prev_point = points[0]

    for current_point in points[1:]:
        active_channels = []
        if r_channel_start <= prev_point < r_channel_end:
            active_channels.append('r')
        if g_channel_start <= prev_point < g_channel_end:
            active_channels.append('g')
        if b_channel_start <= prev_point < b_channel_end:
            active_channels.append('b')

        if not active_channels:
            active_channels.append('none')  # 使用 'none' 表示无通道参与

        # 按字母顺序排序通道名称，确保一致性
        active_channels_sorted = ''.join(sorted(active_channels))
        channel_list.append(active_channels_sorted)
        channel_start_end_list[0].append(prev_point)
        channel_start_end_list[1].append(current_point)
        prev_point = current_point

    return channel_list, channel_start_end_list


def process_image_channels(RGB, r_channel_start, r_channel_end,
                           g_channel_start, g_channel_end,
                           b_channel_start, b_channel_end):
    r, g, b = cv2.split(RGB)

    # 使用字典映射通道名称到图像
    channel_map = {
        'r': r,
        'g': g,
        'b': b
    }

    channel_list_str, channel_start_end_list = simulate_channel_process(
        r_channel_start, r_channel_end,
        g_channel_start, g_channel_end,
        b_channel_start, b_channel_end
    )

    channel_list_img = []
    for channels in channel_list_str:
        if channels == 'none':
            combined = np.zeros_like(r)  # 创建空白图像
        else:
            # 逐个通道相加
            combined = np.zeros_like(r, dtype=np.float32)
            for ch in channels:
                combined += channel_map[ch].astype(np.float32)
            # 防止溢出并转换为无符号8位整数
            combined = np.clip(combined, 0, 255).astype(np.uint8)

        channel_list_img.append(combined)

    return channel_list_img, channel_start_end_list

# Given input parameters and an example RGB image (for testing purposes)
r_channel_start, r_channel_end = 0, 1
g_channel_start, g_channel_end = 0, 0.3
b_channel_start, b_channel_end = 0.3, 0.6

# Create a sample RGB image filled with random values for demonstration purposes
RGB = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Process the image channels and print the output
channel_list_img, channel_start_end_list = process_image_channels(
    RGB, r_channel_start, r_channel_end,
    g_channel_start, g_channel_end,
    b_channel_start, b_channel_end
)

# Output the results
print("Output Results:")
print("channel_list (as images):", ["Image" for _ in channel_list_img])  # Placeholder for actual images
print("channel_start_end_list:", channel_start_end_list)

# Optionally save or display the processed images
# For example, to save each channel image to file:
for idx, img in enumerate(channel_list_img, start=1):
    cv2.imwrite(f'channel_{idx}.png', img)

# Or to display them using OpenCV's imshow function:
# for img in channel_list_img:
#     cv2.imshow('Processed Channel', img)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()