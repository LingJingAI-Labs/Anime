import cv2
import numpy as np
import argparse
import os
import glob

def white_text_based_removal(image_path, output_path, y_position=1320, height=120):
    """
    基于白色文本检测的字幕擦除方法
    
    参数:
    image_path: 输入图像路径
    output_path: 输出图像路径
    y_position: 字幕区域的起始y坐标（从上往下）
    height: 字幕区域的高度
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像：{image_path}")
        return
    
    img_height, img_width = img.shape[:2]
    
    if y_position >= img_height:
        print(f"警告：指定y_position ({y_position}) 超出图像高度 ({img_height})，已跳过处理：{image_path}")
        return
    
    actual_y_end = min(y_position + height, img_height)
    actual_height = actual_y_end - y_position

    if actual_height <= 0:
        print(f"警告：计算得到的字幕区域高度为零或负 ({actual_height})，已跳过处理：{image_path}")
        return

    original_img = img.copy()
    
    roi = img[y_position:actual_y_end, :]
    if roi.size == 0:
        print(f"警告：提取的ROI为空，检查y_position和height。跳过：{image_path}")
        return

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 185]) 
    upper_white = np.array([180, 70, 255])
    white_mask_roi = cv2.inRange(hsv_roi, lower_white, upper_white)
    combined_mask_roi = white_mask_roi

    opening_kernel_size = 3
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size))
    opened_mask_roi = cv2.morphologyEx(combined_mask_roi, cv2.MORPH_OPEN, opening_kernel, iterations=1)

    # 第四步：扩展白色区域 (增强覆盖)
    dilate_kernel_size = 9      # 从 7 增大到 9
    dilate_iterations = 2       # 保持 2 次迭代
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
    dilated_mask_roi = cv2.dilate(opened_mask_roi, dilate_kernel, iterations=dilate_iterations)
    
    # 第五步：创建用于修复的蒙版 (final_inpaint_mask_full) 和其模糊版本
    final_inpaint_mask_full = np.zeros(img.shape[:2], dtype=np.uint8)
    final_inpaint_mask_full[y_position:actual_y_end, :] = dilated_mask_roi
    
    text_mask_blur_ksize = 15  # 从 11 增大到 15 (必须是奇数)
    if text_mask_blur_ksize % 2 == 0: text_mask_blur_ksize +=1 
    final_inpaint_mask_full_blurred = cv2.GaussianBlur(final_inpaint_mask_full, (text_mask_blur_ksize, text_mask_blur_ksize), 0)
    
    # 第六步：对白色文本区域应用修复算法
    inpaint_radius = 10 # 保持 10
    inpainted_img = cv2.inpaint(img, final_inpaint_mask_full_blurred, inpaint_radius, cv2.INPAINT_NS)
    
    # 第七步：创建平滑过渡区域 - Blending
    alpha_blur_ksize = 15 # 配合修复蒙版模糊的调整 (必须是奇数)
    if alpha_blur_ksize % 2 == 0: alpha_blur_ksize +=1
    
    blending_alpha_base = cv2.GaussianBlur(final_inpaint_mask_full, (alpha_blur_ksize, alpha_blur_ksize), 0)
    alpha_blend_mask = blending_alpha_base.astype(np.float32) / 255.0
    
    edge_transition_height = 50
    roi_band_sharp_mask = np.zeros(img.shape[:2], dtype=np.float32)
    roi_band_sharp_mask[y_position:actual_y_end, :] = 1.0
    
    vt_blur_ksize = 2 * (edge_transition_height // 2 * 2) + 1 
    if vt_blur_ksize < 5 : vt_blur_ksize = 5
    vertical_transition_mask = cv2.GaussianBlur(roi_band_sharp_mask, (vt_blur_ksize, vt_blur_ksize), 0)

    final_blend_alpha = alpha_blend_mask * vertical_transition_mask
    
    blended_img = original_img.copy()
    for c in range(3):
        blended_img[:, :, c] = (final_blend_alpha * inpainted_img[:, :, c] +
                               (1 - final_blend_alpha) * original_img[:, :, c]).astype(np.uint8)
    
    img = blended_img

    # 第八步：应用最终的色彩平衡优化
    region_expand = 20 
    search_start_y = max(0, y_position - region_expand)
    search_end_y = min(img_height, actual_y_end + region_expand)
    
    for y_coord in range(search_start_y, search_end_y):
        if y_coord <= 0 or y_coord >= img_height - 1:
            continue
        current_row = img[y_coord, :].astype(np.float32)
        prev_row = img[y_coord-1, :].astype(np.float32)
        next_row = img[y_coord+1, :].astype(np.float32)
        diff_prev = np.mean(np.abs(current_row - prev_row), axis=1)
        diff_next = np.mean(np.abs(current_row - next_row), axis=1)
        
        color_diff_threshold = 15.0
        
        problematic_x_indices = np.where((diff_prev > color_diff_threshold) & (diff_next > color_diff_threshold))[0]
        if len(problematic_x_indices) > 0:
            for x_coord in problematic_x_indices:
                if x_coord > 0 and x_coord < img_width - 1:
                    val_top = img[y_coord-1, x_coord].astype(np.float32)
                    val_bottom = img[y_coord+1, x_coord].astype(np.float32)
                    val_left = img[y_coord, x_coord-1].astype(np.float32)
                    val_right = img[y_coord, x_coord+1].astype(np.float32)
                    averaged_color = (val_top + val_bottom + val_left + val_right) / 4.0
                    img[y_coord, x_coord] = averaged_color.astype(np.uint8)
    
    # --- 可选: 调试时保存中间蒙版图像 ---
    debug_output_dir = os.path.join(os.path.dirname(output_path), "debug_masks")
    os.makedirs(debug_output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    # cv2.imwrite(os.path.join(debug_output_dir, f"{base_filename}_01_opened_mask_roi.png"), opened_mask_roi)
    # cv2.imwrite(os.path.join(debug_output_dir, f"{base_filename}_02_dilated_mask_roi.png"), dilated_mask_roi) # 关注这个
    # cv2.imwrite(os.path.join(debug_output_dir, f"{base_filename}_03_final_inpaint_mask_full_unblurred_for_alpha.png"), final_inpaint_mask_full)
    # cv2.imwrite(os.path.join(debug_output_dir, f"{base_filename}_04_final_inpaint_mask_full_blurred_for_inpaint.png"), final_inpaint_mask_full_blurred) # 关注这个
    # cv2.imwrite(os.path.join(debug_output_dir, f"{base_filename}_05_inpainted_img_before_blend.png"), inpainted_img) # 关注这个
    # cv2.imwrite(os.path.join(debug_output_dir, f"{base_filename}_06_alpha_blend_mask.png"), (alpha_blend_mask * 255).astype(np.uint8)) # 关注这个
    # cv2.imwrite(os.path.join(debug_output_dir, f"{base_filename}_07_final_blend_alpha.png"), (final_blend_alpha * 255).astype(np.uint8))
    # --- 调试结束 ---

    cv2.imwrite(output_path, img)
    print(f"处理完成，结果保存至：{output_path}")

# process_directory 和 if __name__ == "__main__": 部分与之前版本相同，此处省略
def process_directory(input_dir, output_dir, y_position=1320, height=120):
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        print(f"在 {input_dir} 目录中没有找到图像文件 (支持的格式: {', '.join(image_extensions)})")
        return
    print(f"找到 {len(image_files)} 个图像文件")
    
    for i, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        base, ext_dot = os.path.splitext(filename) 
        output_filename = f"{base}_removed{ext_dot}"
        output_path = os.path.join(output_dir, output_filename)
        print(f"处理 [{i+1}/{len(image_files)}]: {filename} -> {output_filename}")
        white_text_based_removal(image_path, output_path, y_position, height)
    print("所有图像处理完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='基于白色文本检测的字幕擦除')
    parser.add_argument('--input', default='data/tmp/', help='输入图像或目录路径')
    parser.add_argument('--output', default='data/tmp-opt/', help='输出图像或目录路径')
    parser.add_argument('--y_position', type=int, default=1320, help='字幕区域的起始y坐标（从上往下）')
    parser.add_argument('--height', type=int, default=150, help='字幕区域的高度') 
    
    args = parser.parse_args()

    if os.path.isdir(args.input):
        process_directory(args.input, args.output, args.y_position, args.height)
    elif os.path.isfile(args.input):
        os.makedirs(args.output, exist_ok=True) 
        filename = os.path.basename(args.input)
        base, ext_dot = os.path.splitext(filename)
        output_filename = f"{base}_removed{ext_dot}"
        output_path = os.path.join(args.output, output_filename)
        print(f"处理单个文件: {args.input} -> {output_path}")
        white_text_based_removal(args.input, output_path, args.y_position, args.height)
        print("单个文件处理完成")
    else:
        print(f"错误：输入路径 {args.input} 不是有效的文件或目录")