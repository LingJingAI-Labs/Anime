import cv2
import numpy as np
import matplotlib.pyplot as plt
import av
import time
from matplotlib.gridspec import GridSpec
import threading
import queue
import os
import shutil

def extract_frames(video_path, sensitivity=5.0, min_keyframes=3, max_keyframes=12, show_progress=True, output_dir="data/initial_frames"):
    """
    使用单一敏感度参数控制关键帧提取的主函数，并保存结果到输出目录
    每次运行前清空输出目录
    """
    # 清空并重建输出目录
    if os.path.exists(output_dir):
        # 删除目录中的所有文件
        for file_name in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"清空文件夹时出错: {e}")
        print(f"已清空输出目录: {output_dir}")
    else:
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    keyframes, keyframe_images, frame_diffs, flow_magnitudes, fps, initial_keyframes = extract_keyframes_with_sensitivity(
        video_path, sensitivity, min_keyframes, max_keyframes, show_progress)

    saved_image_paths = []
    if not keyframes: # 如果没有提取到任何关键帧
        print("未能提取到任何关键帧。")
        return [], [], None, []

    for i, (frame_idx, img) in enumerate(zip(keyframes, keyframe_images)):
        total_seconds = frame_idx / fps if fps > 0 else 0 # 避免fps为0的情况
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds - int(total_seconds)) * 1000)
        time_str = f"{minutes:02d}m{seconds:02d}s{milliseconds:03d}ms"
        
        img_filename = f"{video_name}_keyframe_{time_str}_{i+1:02d}.jpg"
        img_path = os.path.join(output_dir, img_filename)
        cv2.imwrite(img_path, img)
        saved_image_paths.append(img_path)
        print(f"保存关键帧 {i+1}/{len(keyframes)}: {img_path}")
    
    output_chart_path = None # 暂时禁用可视化图表生成
    # chart_filename = f"visualize_{video_name}_keyframes_sensitivity_{sensitivity:.1f}.jpg"
    # output_chart_path = os.path.join(output_dir, chart_filename)
    # if keyframe_images: # 确保有图像可以可视化
    #     visualize_results(keyframes, keyframe_images, frame_diffs, flow_magnitudes,
    #                     initial_keyframes=initial_keyframes, fps=fps, output_path=output_chart_path)
    #     print(f"关键帧可视化已保存至: {output_chart_path}")
    # else:
    #     print("没有关键帧图像可供可视化。")
        
    print(f"总共提取并保存了 {len(keyframes)} 个关键帧")

    return keyframes, keyframe_images, output_chart_path, saved_image_paths

def extract_keyframes_with_sensitivity(video_path, sensitivity=5.0, min_keyframes=6, max_keyframes=12, show_progress=True):
    """
    使用单一敏感度参数提取视频关键帧，并控制最大和最小数量
    """
    sensitivity = max(1.0, min(10.0, float(sensitivity)))
    
    diff_threshold = 0.5 - (sensitivity * 0.04)
    flow_threshold = 8.0 - (sensitivity * 0.6)
    min_frame_distance = 2.0 - sensitivity * 0.15
    
    print(f"使用敏感度: {sensitivity}/10")
    print(f"映射到直方图差异阈值: {diff_threshold:.3f}")
    print(f"映射到光流阈值: {flow_threshold:.3f}")
    print(f"最小关键帧间距: {min_frame_distance:.1f}秒")
    
    keyframes, keyframe_images, frame_diffs, flow_magnitudes, fps, actual_total_frames = extract_keyframes_raw(
        video_path, diff_threshold, flow_threshold, show_progress)
    
    initial_keyframes = keyframes.copy() # 保存原始提取的关键帧
    
    if not keyframes: # 如果raw提取未能返回任何帧（除了可能的首帧）
        print("警告: 初始关键帧提取未能找到显著帧。")
        # 即使如此，我们仍应尝试满足 min_keyframes (如果大于0)
        # 确保至少有第0帧
        if 0 not in keyframes:
             keyframes = [0]
             # 需要读取第0帧图像
             container_temp = av.open(video_path)
             temp_stream = container_temp.streams.video[0]
             if temp_stream.frames > 0:
                for frame_obj in container_temp.decode(video=0):
                    keyframe_images = [frame_obj.to_ndarray(format='bgr24')]
                    break
             container_temp.close()
        initial_keyframes = keyframes.copy()


    # 应用最小帧间距过滤
    min_frames_gap = int(min_frame_distance * fps) if fps > 0 else 0
    
    if keyframes: # 确保keyframes不是空的
        filtered_keyframes = [keyframes[0]] 
        filtered_images = [keyframe_images[0]]
        for i in range(1, len(keyframes)):
            if keyframes[i] - filtered_keyframes[-1] >= min_frames_gap:
                filtered_keyframes.append(keyframes[i])
                filtered_images.append(keyframe_images[i])
    else: # 如果 keyframes 为空，则 filtered_keyframes 也为空
        filtered_keyframes = []
        filtered_images = []

    # 如果关键帧数量少于最小值，尝试添加帧
    if len(filtered_keyframes) < min_keyframes:
        print(f"检测到的关键帧数量({len(filtered_keyframes)})低于最小要求({min_keyframes})，将添加额外帧")
        
        # 获取视频总帧数 (使用 extract_keyframes_raw 返回的 actual_total_frames)
        # 如果 actual_total_frames 为0，尝试再次获取
        if actual_total_frames == 0:
            try:
                container_temp = av.open(video_path)
                video_stream_temp = next(s for s in container_temp.streams if s.type == 'video')
                actual_total_frames = video_stream_temp.frames
                container_temp.close()
                if actual_total_frames == 0 and fps > 0 and video_stream_temp.duration is not None and video_stream_temp.time_base is not None:
                    actual_total_frames = int(video_stream_temp.duration * video_stream_temp.time_base * fps)
                print(f"重新获取视频总帧数: {actual_total_frames}")
            except Exception as e:
                print(f"尝试获取总帧数失败: {e}, 无法添加额外帧。")
                actual_total_frames = 0 # 确保为0，避免后续错误

        if actual_total_frames > 0:
            frames_to_add_count = min_keyframes - len(filtered_keyframes)
            additional_frames_indices = []
            
            if frames_to_add_count > 0:
                existing_kf_set = set(filtered_keyframes)
                
                # 均匀选择候选帧索引
                # 我们目标是总共 min_keyframes 个帧。
                # 如果 filtered_keyframes 为空，我们就选 min_keyframes 个。
                # 如果 filtered_keyframes 有N个，我们就选 min_keyframes-N 个新的。
                
                # 为了更均匀，我们考虑整个视频长度来选择 min_keyframes 个点，然后去掉已有的
                potential_indices = np.linspace(0, actual_total_frames - 1, min_keyframes, dtype=int).tolist()

                for frame_idx_candidate in potential_indices:
                    if len(additional_frames_indices) >= frames_to_add_count:
                        break
                    if frame_idx_candidate not in existing_kf_set:
                        # 简单检查是否与现有帧过于接近 (更严格的检查在后面)
                        is_too_close = False
                        for existing_kf in filtered_keyframes:
                            if abs(frame_idx_candidate - existing_kf) < (min_frames_gap / 2 if min_frames_gap > 0 else fps / 2):
                                is_too_close = True
                                break
                        if not is_too_close:
                            additional_frames_indices.append(frame_idx_candidate)
                            existing_kf_set.add(frame_idx_candidate) # 临时添加到集合中，避免重复选择

                # 如果通过linspace选取的帧不足，可能是因为它们都太接近已有的帧
                # 此时，可以尝试在最大的空隙中添加帧
                if len(additional_frames_indices) < frames_to_add_count and len(filtered_keyframes) > 0:
                    sorted_existing_kf = sorted(list(existing_kf_set))
                    gaps = [] # (start_frame, end_frame, duration)
                    if sorted_existing_kf[0] > 0:
                        gaps.append((-1, sorted_existing_kf[0], sorted_existing_kf[0]))
                    for i in range(len(sorted_existing_kf) - 1):
                        gaps.append((sorted_existing_kf[i], sorted_existing_kf[i+1], sorted_existing_kf[i+1] - sorted_existing_kf[i]))
                    if sorted_existing_kf[-1] < actual_total_frames -1:
                        gaps.append((sorted_existing_kf[-1], actual_total_frames -1, actual_total_frames -1 - sorted_existing_kf[-1]))
                    
                    gaps.sort(key=lambda x: x[2], reverse=True) # 按间隙大小排序

                    for start_kf, end_kf, _ in gaps:
                        if len(additional_frames_indices) >= frames_to_add_count:
                            break
                        # 在这个间隙中间选一个点
                        if end_kf - start_kf > (min_frames_gap if min_frames_gap > 0 else fps): # 确保间隙足够大
                            candidate = start_kf + (end_kf - start_kf) // 2
                            if candidate not in existing_kf_set:
                                additional_frames_indices.append(candidate)
                                existing_kf_set.add(candidate)
                
                if additional_frames_indices:
                    print(f"将补充以下帧索引: {additional_frames_indices}")
                    additional_images = extract_specific_frames(video_path, additional_frames_indices)
                    
                    for i, frame_idx_new in enumerate(additional_frames_indices):
                        # 再次检查，以防万一
                        is_present = False
                        for kf_idx_existing in filtered_keyframes:
                            if kf_idx_existing == frame_idx_new:
                                is_present = True
                                break
                        if not is_present:
                            filtered_keyframes.append(frame_idx_new)
                            filtered_images.append(additional_images[i])
                    
                    combined = sorted(list(zip(filtered_keyframes, filtered_images)), key=lambda x: x[0])
                    if combined:
                        filtered_keyframes, filtered_images = zip(*combined)
                        filtered_keyframes = list(filtered_keyframes)
                        filtered_images = list(filtered_images)
            print(f"补充帧后，关键帧数量: {len(filtered_keyframes)}")


    # 如果关键帧数量仍然过多，选择最重要的帧
    if len(filtered_keyframes) > max_keyframes:
        print(f"关键帧数量({len(filtered_keyframes)})超出最大限制({max_keyframes})，将进行筛选")
        frame_scores = []

        # 确保 frame_diffs 和 flow_magnitudes 至少与最大可能的帧索引一样长
        # （理论上 initial_keyframes 里的帧索引应该在这些数组的范围内）
        max_idx_in_kf = 0
        if initial_keyframes: # 使用 initial_keyframes 来确定原始差异的范围
             max_idx_in_kf = max(initial_keyframes) if initial_keyframes else 0
        
        # 安全地访问 frame_diffs 和 flow_magnitudes
        safe_frame_diffs = frame_diffs if frame_diffs else [0.0] * (max_idx_in_kf + 1)
        safe_flow_magnitudes = flow_magnitudes if flow_magnitudes else [0.0] * (max_idx_in_kf + 1)


        for i, kf_idx_val in enumerate(filtered_keyframes):
            if i == 0 and kf_idx_val == 0:  # 总是保留第一帧 (frame 0)
                frame_scores.append((i, float('inf'))) # 最高优先级
                continue
            
            # 查找 kf_idx_val 在 initial_keyframes 中的位置以获取其原始 diff/flow
            # 如果 kf_idx_val 是补充的帧，它可能不在 initial_keyframes 中
            score = 0.1 # 默认低分给补充帧
            if kf_idx_val in initial_keyframes:
                # 帧的索引是 kf_idx_val。对应的差异值在 frame_diffs[kf_idx_val-1]
                # (因为 frame_diffs[0] 是 frame 1 和 frame 0 的差异)
                diff_idx = kf_idx_val - 1
                if diff_idx >= 0:
                    hist_s = safe_frame_diffs[diff_idx] if diff_idx < len(safe_frame_diffs) else 0
                    flow_s = safe_flow_magnitudes[diff_idx] if diff_idx < len(safe_flow_magnitudes) else 0
                    score = hist_s + flow_s * 0.1 
            frame_scores.append((i, score))
        
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择分数最高的帧，直到达到max_keyframes
        # 确保第一帧(索引0)总是在内，如果它存在于filtered_keyframes中
        selected_indices_in_filtered = []
        if 0 in [fs[0] for fs in frame_scores if filtered_keyframes[fs[0]]==0]: # 如果第0帧在其中
             selected_indices_in_filtered.append([fs[0] for fs in frame_scores if filtered_keyframes[fs[0]]==0][0])

        for original_idx, _ in frame_scores:
            if len(selected_indices_in_filtered) >= max_keyframes:
                break
            if original_idx not in selected_indices_in_filtered: # 避免重复添加（比如第0帧）
                selected_indices_in_filtered.append(original_idx)
        
        selected_indices_in_filtered.sort() # 按原始顺序排序
        
        final_keyframes = [filtered_keyframes[i] for i in selected_indices_in_filtered]
        final_images = [filtered_images[i] for i in selected_indices_in_filtered]
    else:
        final_keyframes = filtered_keyframes
        final_images = filtered_images
    
    print(f"初始提取(raw): {len(initial_keyframes)} 帧") # 使用 initial_keyframes 的长度
    print(f"间距过滤后: {len(filtered_keyframes)} 帧")
    print(f"最终选择: {len(final_keyframes)} 帧 (最小要求: {min_keyframes}, 最大限制: {max_keyframes})")
    
    # 确保 frame_diffs 和 flow_magnitudes 长度与 actual_total_frames 匹配，如果可视化需要
    # (当前可视化已注释掉)
    # if actual_total_frames > 0 :
    #     if len(frame_diffs) < actual_total_frames -1 :
    #         frame_diffs.extend([0.0] * (actual_total_frames - 1 - len(frame_diffs)))
    #     if len(flow_magnitudes) < actual_total_frames -1:
    #         flow_magnitudes.extend([0.0] * (actual_total_frames - 1 - len(flow_magnitudes)))


    return final_keyframes, final_images, frame_diffs, flow_magnitudes, fps, initial_keyframes


def extract_specific_frames(video_path, frame_indices):
    """并行提取特定帧索引的图像"""
    if not frame_indices:
        return []
    images = [None] * len(frame_indices)
    # 创建一个 (原始索引, 帧号) 对的列表，并按帧号排序以优化读取
    indexed_frame_indices = sorted(list(enumerate(frame_indices)), key=lambda x: x[1])
    
    container = None
    try:
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        
        # PyAV 的 seek 效率不高，对于已排序的帧索引，顺序解码可能更快
        # 但如果帧索引跨度很大，seek还是必要的
        # 这里我们仍然使用 seek，因为帧索引可能不连续
        
        current_frame_pos = -1 # PyAV解码器内部有时会跟踪位置
        for original_list_idx, target_frame_idx in indexed_frame_indices:
            # 检查视频总帧数，避免请求不存在的帧
            if target_frame_idx >= video_stream.frames and video_stream.frames > 0 : # video_stream.frames > 0 确保元数据有效
                print(f"警告: 请求的帧索引 {target_frame_idx} 超出视频总帧数 {video_stream.frames}。跳过此帧。")
                images[original_list_idx] = np.zeros((100,100,3), dtype=np.uint8) # 放一个占位符
                continue

            # 对于非常接近的帧，可能不需要重新seek
            # 但为了简单和可靠性，每次都seek
            # PyAV的seek是按时间戳，需要转换为pts
            # target_pts = int(target_frame_idx * video_stream.duration * video_stream.time_base / video_stream.frames) # 粗略估计
            # container.seek(target_pts, stream=video_stream, backward=True, any_frame=False)
            
            # 更可靠的是直接 seek 到帧号（如果解码器支持良好）
            # PyAV seek takes timestamp in stream time_base.
            # We need to find the timestamp for target_frame_idx.
            # A common way is to seek to a timestamp slightly before the expected frame.
            # For simplicity here, we'll iterate if the target_frame_idx is close, otherwise seek.
            # However, the most robust frame access for arbitrary frames is often to decode sequentially
            # or use a library with better random access if performance is critical for many sparse frames.

            # The provided code for extract_specific_frames had a simple seek and decode 1 frame. Let's stick to that pattern.
            # container.seek(offset=target_frame_idx, stream=video_stream, any_frame=False) # offset is frame number for some containers
            
            # Try to seek to a time_base representation
            # This is tricky with PyAV as `seek` expects a timestamp in `stream.time_base` units.
            # If average_rate is reliable:
            if video_stream.average_rate:
                seek_time_seconds = target_frame_idx / float(video_stream.average_rate)
                seek_pts = int(seek_time_seconds / video_stream.time_base)
                container.seek(seek_pts, stream=video_stream, backward=True, any_frame=False)
            else: # Fallback if average_rate is not available, less accurate
                 # Approximate seek, then decode until target
                if video_stream.frames > 0: # only if total frames metadata is somewhat reliable
                    seek_pts = int(target_frame_idx * video_stream.duration * video_stream.time_base / video_stream.frames)
                    container.seek(seek_pts, stream=video_stream, backward=True, any_frame=False)


            frame_found = False
            for frame in container.decode(video=0): # Decode from seek point
                if frame.pts is None: # Some frames might not have pts initially
                    # Heuristic: assume frames are sequential after seek
                    # This part is tricky without reliable pts or frame numbers from the decoded frame object itself.
                    # For simplicity, we take the first frame after seek if it's for the correct index,
                    # but this isn't robust for all videos/seeks.
                    # A better way would be to count frames from a known point or use frame.index if available and reliable.
                    images[original_list_idx] = frame.to_ndarray(format='bgr24')
                    frame_found = True
                    break # Assuming the seek got us to the right frame or close enough for the first one.

                # If we have pts, try to match more accurately
                # current_decoded_frame_idx = int(frame.pts * video_stream.time_base * float(video_stream.average_rate)) # Estimate
                # if current_decoded_frame_idx >= target_frame_idx:
                
                # Simplified: take the first frame after seek for this example
                images[original_list_idx] = frame.to_ndarray(format='bgr24')
                frame_found = True
                break # Take the first frame available after seek
            
            if not frame_found:
                print(f"警告: 未能为索引 {target_frame_idx} 提取到特定帧。")
                images[original_list_idx] = np.zeros((100,100,3), dtype=np.uint8) # Placeholder for missing frame

    except Exception as e:
        print(f"提取特定帧时发生错误: {e}")
        # Fill remaining images with placeholders if error occurs mid-process
        for i in range(len(images)):
            if images[i] is None:
                images[i] = np.zeros((100,100,3), dtype=np.uint8)
    finally:
        if container:
            container.close()
            
    return images


class FrameProcessor:
    def __init__(self, scale_factor=0.25):
        self.scale_factor = scale_factor
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_cuda:
            print("使用CUDA进行GPU加速 (注意: 当前实现未完全启用CUDA光流的持续流模式)")
            
    def process_frame_batch(self, frames_batch, prev_hist, prev_gray_small):
        results = []
        # These are the features of the frame *before* the current batch starts
        batch_internal_last_hist = prev_hist.copy()
        batch_internal_last_gray_small = prev_gray_small.copy()
        
        for curr_frame in frames_batch:
            curr_hist = cv2.calcHist([curr_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()
            
            # Compare current frame with the previous frame in the sequence (batch_internal_last_hist)
            hist_diff = cv2.compareHist(batch_internal_last_hist, curr_hist, cv2.HISTCMP_CHISQR_ALT)
            # Potentially sensitive scaling factor, ensure it doesn't make all diffs too small
            # hist_diff = min(1.0, hist_diff / 100.0) # Original, keeping for now
            # Consider making the divisor configurable or adapting it based on typical CHISQR_ALT values
            # For now, let's try a less aggressive scaling for testing:
            hist_diff = min(1.0, hist_diff / 50.0) # TEST: less aggressive scaling

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            curr_gray_small = cv2.resize(curr_gray, (0,0), fx=self.scale_factor, fy=self.scale_factor)
            
            flow_magnitude = 0.0
            if batch_internal_last_gray_small is not None and batch_internal_last_gray_small.shape == curr_gray_small.shape:
                if self.use_cuda:
                    # For CUDA, need to ensure GpuMat lifecycle is managed or use a persistent flow object
                    # This is a simplified version, may not be most efficient for CUDA
                    prev_gpu = cv2.cuda_GpuMat()
                    prev_gpu.upload(batch_internal_last_gray_small)
                    curr_gpu = cv2.cuda_GpuMat()
                    curr_gpu.upload(curr_gray_small)
                    
                    # Create flow object each time for simplicity, though not ideal for performance
                    flow_calculator_gpu = cv2.cuda.FarnebackOpticalFlow.create(5, 0.5, False, 15, 3, 5, 1.2, 0)
                    flow_gpu = flow_calculator_gpu.calc(prev_gpu, curr_gpu, None)
                    flow = flow_gpu.download()
                else:
                    flow = cv2.calcOpticalFlowFarneback(
                        batch_internal_last_gray_small, curr_gray_small, None, 
                        0.5, 3, 15, 3, 5, 1.2, 0)
                    
                if flow is not None:
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    flow_magnitude = np.mean(mag)
            
            results.append({
                'frame': curr_frame.copy(), # Store a copy of the frame
                # 'hist': curr_hist, # Not strictly needed in result if not used later
                # 'gray_small': curr_gray_small, # Not strictly needed
                'hist_diff': hist_diff,
                'flow_magnitude': flow_magnitude
            })
            
            batch_internal_last_hist = curr_hist.copy()
            batch_internal_last_gray_small = curr_gray_small.copy()
            
        # Return results, and the hist/gray of the *last frame processed in this batch*
        return results, batch_internal_last_hist, batch_internal_last_gray_small

def extract_keyframes_raw(video_path, diff_threshold, flow_threshold, show_progress=True, scale_factor=0.3):
    start_time = time.time()
    num_threads = min(os.cpu_count(), 4)
    print(f"使用{num_threads}个线程进行处理 (extract_keyframes_raw)")
    
    actual_decoded_frame_count = 0 # Dynamically count decoded frames

    container = None
    try:
        container = av.open(video_path)
        video_stream = next(s for s in container.streams if s.type == 'video')
        metadata_frame_count = video_stream.frames # Can be unreliable
        fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0 # Fallback FPS
        
        if metadata_frame_count == 0 and video_stream.duration is not None and video_stream.time_base is not None and fps > 0:
            metadata_frame_count = int(video_stream.duration * video_stream.time_base * fps)
            print(f"元数据帧数为0，根据时长估算为: {metadata_frame_count} 帧")
        else:
            print(f"视频元数据包含 {metadata_frame_count} 帧，FPS: {fps:.2f}")

    except Exception as e:
        print(f"打开视频或读取元数据失败: {e}")
        return [], [], [], [], 0, 0

    keyframes = [0] # Always include the first frame
    keyframe_images = []
    frame_diffs = [] # To store diff between frame i and i-1, so index matches frame i-1
    flow_magnitudes = []
    
    processor = FrameProcessor(scale_factor=scale_factor)
    
    # Read the first frame
    first_frame_data = None
    try:
        # container.seek(0) # Already at start after open
        for frame_obj in container.decode(video=0):
            first_frame_data = frame_obj.to_ndarray(format='bgr24')
            keyframe_images.append(first_frame_data.copy())
            actual_decoded_frame_count = 1
            break
    except Exception as e:
        print(f"读取第一帧失败: {e}")
        container.close()
        return [], [], [], [], fps, 0
        
    if first_frame_data is None:
        print("未能读取到第一帧。")
        container.close()
        return [], [], [], [], fps, 0

    first_gray = cv2.cvtColor(first_frame_data, cv2.COLOR_BGR2GRAY)
    first_gray_small = cv2.resize(first_gray, (0,0), fx=scale_factor, fy=scale_factor)
    first_hist = cv2.calcHist([first_frame_data], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    first_hist = cv2.normalize(first_hist, first_hist).flatten()
    
    # Re-open or reset container is not strictly needed if we decode sequentially from here
    # For producer, it will start a new decode loop.
    # container.close()
    # container = av.open(video_path) # Producer will handle its own container instance

    frame_queue = queue.Queue(maxsize=num_threads * 2) # Batches of (original_index, frame_array)
    result_queue = queue.Queue() # Batches of (original_index, processed_result_dict)
    end_signal = object()
    
    # Frame Producer
    # Needs its own container instance to avoid conflicts if main thread uses one too
    producer_container_ref = av.open(video_path) # Producer's own container
    def frame_producer():
        nonlocal actual_decoded_frame_count
        frame_idx_counter = 0 # 0 is first_frame_data, already handled
        try:
            # Skip the first frame as it's handled
            first_skipped = False
            for frame_obj in producer_container_ref.decode(video=0):
                if not first_skipped:
                    first_skipped = True
                    frame_idx_counter += 1 # Corresponds to frame 0
                    continue # Skip actual frame 0 data, it's processed

                frame_data = frame_obj.to_ndarray(format='bgr24')
                frame_queue.put((frame_idx_counter, frame_data))
                frame_idx_counter += 1
            
            actual_decoded_frame_count = frame_idx_counter # Final count of frames including frame 0
            print(f"生产者完成，共解码 {actual_decoded_frame_count} 帧 (包括第0帧)")

        except Exception as e:
            print(f"帧生产者错误: {e}")
        finally:
            for _ in range(num_threads):
                frame_queue.put((None, end_signal))
            producer_container_ref.close()
            print("生产者线程已关闭其视频容器并发送所有结束信号")

    # Frame Consumer
    def frame_consumer(thread_id, initial_prev_hist_for_thread, initial_prev_gray_small_for_thread):
        batch_size = 5 
        current_batch_frames_to_process = [] # list of (idx, frame_data)
        
        # These track the hist/gray of the frame *before* the current batch starts for this thread
        # They are updated after each batch is processed by this thread.
        hist_before_this_batch = initial_prev_hist_for_thread.copy()
        gray_small_before_this_batch = initial_prev_gray_small_for_thread.copy()
        
        processed_by_thread_count = 0
        try:
            while True:
                try:
                    idx, frame_data_item = frame_queue.get(timeout=1.0) 
                    if frame_data_item is end_signal:
                        if current_batch_frames_to_process: # Process any remaining frames in batch
                            batch_raw_frames = [item[1] for item in current_batch_frames_to_process]
                            processed_results, hist_after_batch, gray_small_after_batch = processor.process_frame_batch(
                                batch_raw_frames, hist_before_this_batch, gray_small_before_this_batch)
                            
                            for i, res_dict in enumerate(processed_results):
                                original_frame_index = current_batch_frames_to_process[i][0]
                                result_queue.put((original_frame_index, res_dict))
                            # No update to hist_before_this_batch needed as thread is ending
                        current_batch_frames_to_process = []
                        print(f"线程{thread_id}接收到结束信号 (处理了 {processed_by_thread_count} 帧)")
                        frame_queue.task_done() # Mark end_signal as processed
                        break # Exit the while loop
                    
                    current_batch_frames_to_process.append((idx, frame_data_item))
                    
                    if len(current_batch_frames_to_process) >= batch_size:
                        batch_raw_frames = [item[1] for item in current_batch_frames_to_process]
                        processed_results, hist_after_batch, gray_small_after_batch = processor.process_frame_batch(
                            batch_raw_frames, hist_before_this_batch, gray_small_before_this_batch)
                        
                        for i, res_dict in enumerate(processed_results):
                            original_frame_index = current_batch_frames_to_process[i][0]
                            result_queue.put((original_frame_index, res_dict))
                        
                        processed_by_thread_count += len(current_batch_frames_to_process)
                        hist_before_this_batch = hist_after_batch # Update for next batch for this thread
                        gray_small_before_this_batch = gray_small_after_batch
                        current_batch_frames_to_process = []
                        
                    frame_queue.task_done() # Mark data item as processed

                except queue.Empty: # Timeout from frame_queue.get()
                    # If queue is empty, check if producer might be done (though end_signal is better)
                    # This mainly keeps the thread alive if producer is slow.
                    continue 
        except Exception as e:
            print(f"帧处理器{thread_id}错误: {e}")
        finally:
            result_queue.put((None, None)) # Signal main thread this consumer is done
            print(f"线程{thread_id}发送完成信号到结果队列")

    producer_thread = threading.Thread(target=frame_producer)
    producer_thread.daemon = True
    producer_thread.start()
    
    consumer_threads = []
    for i in range(num_threads):
        # Each consumer starts by comparing its first batch's first frame
        # to the video's actual first frame (first_hist, first_gray_small)
        t = threading.Thread(target=frame_consumer, args=(i, first_hist, first_gray_small))
        t.daemon = True
        t.start()
        consumer_threads.append(t)
    
    processed_frames_results = {} # Store results indexed by frame number
    finished_consumer_threads = 0
    max_processed_idx_overall = 0

    print("开始收集处理结果...")
    while finished_consumer_threads < num_threads:
        try:
            idx, result_dict = result_queue.get(timeout=5.0) # Increased timeout for results
            
            if idx is None and result_dict is None: # Consumer finished
                finished_consumer_threads += 1
                print(f"一个消费者线程完成, 总完成数: {finished_consumer_threads}/{num_threads}")
                result_queue.task_done()
                continue
            
            if idx is not None:
                 processed_frames_results[idx] = result_dict
                 max_processed_idx_overall = max(max_processed_idx_overall, idx)

                 # Store diffs and flows. result_dict contains diff of idx vs idx-1
                 # So frame_diffs[idx-1] = result_dict['hist_diff']
                 # Ensure lists are long enough
                 diff_storage_idx = idx -1 
                 if diff_storage_idx >= 0: # Only for frames > 0
                    while len(frame_diffs) <= diff_storage_idx:
                        frame_diffs.append(0.0)
                    while len(flow_magnitudes) <= diff_storage_idx:
                        flow_magnitudes.append(0.0)
                    
                    frame_diffs[diff_storage_idx] = result_dict['hist_diff']
                    flow_magnitudes[diff_storage_idx] = result_dict['flow_magnitude']

                    if result_dict['hist_diff'] > diff_threshold or result_dict['flow_magnitude'] > flow_threshold:
                        if idx not in keyframes: # Avoid duplicates if somehow re-processed
                            keyframes.append(idx)
                            keyframe_images.append(result_dict['frame']) # Frame was stored in result_dict

            result_queue.task_done()
            
            if show_progress and len(processed_frames_results) % 100 == 0 and len(processed_frames_results) > 0:
                current_progress_frames = len(processed_frames_results)
                elapsed_progress = time.time() - start_time
                fps_progress = current_progress_frames / elapsed_progress if elapsed_progress > 0 else 0
                # Use metadata_frame_count for ETA if available and seems reasonable
                total_for_eta = metadata_frame_count if metadata_frame_count > actual_decoded_frame_count else actual_decoded_frame_count
                if total_for_eta > 0 and fps_progress > 0:
                    eta = (total_for_eta - current_progress_frames) / fps_progress
                    print(f"总进度: {current_progress_frames}/{total_for_eta if total_for_eta > 0 else '??'} ({current_progress_frames/total_for_eta*100 if total_for_eta > 0 else 0:.1f}%) - "
                        f"速度: {fps_progress:.1f} fps - ETA: {eta:.1f}秒")
                else:
                     print(f"总进度: {current_progress_frames} 帧 - 速度: {fps_progress:.1f} fps")


        except queue.Empty: # Timeout from result_queue.get()
            print(f"结果队列超时。已完成消费者: {finished_consumer_threads}/{num_threads}. "
                  f"生产者是否仍在运行: {producer_thread.is_alive()}")
            if not producer_thread.is_alive() and finished_consumer_threads == num_threads:
                print("所有线程似乎已完成，但结果队列中可能仍有项目或超时。")
                break # Exit if all consumers are done and producer is also done
            if not producer_thread.is_alive() and all(not t.is_alive() for t in consumer_threads):
                 print("所有生产者和消费者线程已结束。强制退出结果收集。")
                 break


    print("等待生产者线程完成...")
    producer_thread.join(timeout=10)
    if producer_thread.is_alive():
        print("警告: 生产者线程超时未能加入。")

    print("等待所有消费者线程完成...")
    for i, t in enumerate(consumer_threads):
        t.join(timeout=10)
        if t.is_alive():
            print(f"警告: 消费者线程 {i} 超时未能加入。")
    
    if container: # Close the main thread's container if it was used beyond metadata
        container.close()

    # Sort keyframes and corresponding images by frame index
    if keyframes:
        # Ensure frame 0 is unique if it was added multiple times due to logic paths
        # Create a dictionary to keep only the first occurrence of each frame index
        unique_kf_map = {}
        for kf_idx, kf_img in zip(keyframes, keyframe_images):
            if kf_idx not in unique_kf_map:
                unique_kf_map[kf_idx] = kf_img
        
        sorted_unique_indices = sorted(unique_kf_map.keys())
        keyframes = sorted_unique_indices
        keyframe_images = [unique_kf_map[idx] for idx in sorted_unique_indices]


    elapsed = time.time() - start_time
    # Use actual_decoded_frame_count from producer for FPS calculation
    # This count includes frame 0.
    final_fps_calc = (actual_decoded_frame_count / elapsed) if elapsed > 0 and actual_decoded_frame_count > 0 else 0.0
    
    print(f"共提取 {len(keyframes)} 个原始关键帧，总耗时 {elapsed:.2f} 秒，处理了 {actual_decoded_frame_count} 帧，平均处理速度 {final_fps_calc:.1f} fps")
    return keyframes, keyframe_images, frame_diffs, flow_magnitudes, fps, actual_decoded_frame_count

# visualize_results function remains unchanged, but it's commented out in extract_frames
def visualize_results(keyframes, keyframe_images, frame_diffs, flow_magnitudes, 
                    initial_keyframes=None, fps=30.0, output_path="keyframes_result.jpg"):
    """改进版关键帧可视化函数 - 在单一图表上展示所有信息"""
    if not keyframes or not keyframe_images:
        print("没有关键帧数据可供可视化。")
        return None
    if len(frame_diffs) == 0 or len(flow_magnitudes) == 0 : # Ensure these have data
        print("帧差异或光流数据为空，无法生成完整图表。")
        # Attempt to make a simpler plot if only images are available
        # For now, just return if critical data is missing for the full plot
        if len(keyframes) != len(keyframe_images):
             print(f"关键帧数量 ({len(keyframes)}) 与图像数量 ({len(keyframe_images)}) 不匹配。")
             return None


    plt.style.use('ggplot')
    fig = plt.figure(figsize=(24, 12), dpi=60, facecolor='white')
    gs = GridSpec(2, 6, height_ratios=[1.8, 1], hspace=0.4, wspace=0.2)
    
    window_size = 5
    
    # Ensure data has enough points for convolution, pad if necessary
    min_len_for_conv = window_size
    
    if len(frame_diffs) >= min_len_for_conv:
        scaled_diffs = [diff * 100 for diff in frame_diffs]
        smoothed_diffs = np.convolve(scaled_diffs, np.ones(window_size)/window_size, mode='same')
    elif len(frame_diffs) > 0: # Not enough for full conv, use raw or shorter window
        scaled_diffs = [diff * 100 for diff in frame_diffs]
        smoothed_diffs = np.array(scaled_diffs) # Use raw if too short
    else: # No data
        smoothed_diffs = np.array([0])


    if len(flow_magnitudes) >= min_len_for_conv:
        smoothed_flow = np.convolve(flow_magnitudes, np.ones(window_size)/window_size, mode='same')
    elif len(flow_magnitudes) > 0:
        smoothed_flow = np.array(flow_magnitudes)
    else:
        smoothed_flow = np.array([0])

    # Log transform, handle cases where smoothed_diffs might be all zeros or negative
    log_diffs = []
    if np.any(smoothed_diffs > 0): # Check if there are any positive values
        for diff_val in smoothed_diffs:
            if diff_val <= 1.0: # Adjusted threshold for log10
                log_diffs.append(0) 
            else:
                log_diffs.append(np.log10(diff_val))
    else: # All non-positive
        log_diffs = np.zeros_like(smoothed_diffs)
    
    ax_main = fig.add_subplot(gs[0, :])
    ax_flow = ax_main.twinx()
    ax_main.set_facecolor('#f8f9fa')
    ax_main.set_ylim(bottom=-0.05)
    
    diff_line = ax_main.plot(log_diffs, label='Histogram Difference Score (log10)', 
                            color='#3498db', linewidth=2.0, alpha=0.8)
    ax_main.set_ylabel('Histogram Difference Score (log10)', 
                    color='#3498db', fontsize=12, fontweight='bold')
    ax_main.tick_params(axis='y', labelcolor='#3498db')
    
    flow_line = ax_flow.plot(smoothed_flow, label='Optical Flow (smoothed)', 
                            color='#e74c3c', linewidth=2.0, alpha=0.8)
    ax_flow.set_ylabel('Optical Flow Magnitude', 
                    color='#e74c3c', fontsize=12, fontweight='bold')
    ax_flow.tick_params(axis='y', labelcolor='#e74c3c')
    
    ax_main.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
    ax_main.set_title('Result of Keyframe Extraction', fontsize=16, pad=15, fontweight='bold')
    ax_main.set_xlabel('Frame # (relative to diff/flow arrays)', fontsize=12, fontweight='bold')
    
    # Dummy thresholds for plot if not available from params (should be passed)
    # These are just for visualization, actual thresholds are in extract_keyframes_with_sensitivity
    plot_diff_thresh_log = np.log10(0.4*100) if 0.4*100 > 1 else 0 # Example, assuming 0.4 is scaled diff
    plot_flow_thresh = 5.0 
    ax_main.axhline(y=plot_diff_thresh_log, color='#3498db', linestyle='--', alpha=0.5, 
                    label='Approx. Hist Threshold (log)')
    ax_flow.axhline(y=plot_flow_thresh, color='#e74c3c', linestyle='--', alpha=0.5,
                label='Approx. Flow Threshold')
    
    max_len_data = len(log_diffs)

    if initial_keyframes is not None and max_len_data > 0:
        valid_initial_frames = [kf for kf in initial_keyframes if kf < max_len_data and kf-1 >=0] # kf-1 for index in diff array
        initial_y_values = [log_diffs[kf-1] for kf in valid_initial_frames] # kf-1 as diffs are for kf vs kf-1
        ax_main.scatter([kf-1 for kf in valid_initial_frames], initial_y_values, 
                        marker='o', s=30, color='#8e44ad', alpha=0.7, 
                        label='Initial Keyframes (pos in diff array)', zorder=4)
    
    marker_colors = ['#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#d35400', '#2980b9']
    
    for idx, kf_val in enumerate(keyframes): # kf_val is actual frame number
        # For plotting on diff/flow graph, use kf_val-1 as index if kf_val > 0
        plot_idx = kf_val -1 
        if plot_idx < 0 or plot_idx >= max_len_data :
            if kf_val == 0 and max_len_data > 0 : # Special case for frame 0, plot at start of graph
                plot_idx_eff = 0
                y_data = log_diffs[plot_idx_eff] if plot_idx_eff < len(log_diffs) else 0
            else:
                continue # Cannot plot this keyframe on the graph
        else:
            plot_idx_eff = plot_idx
            y_data = log_diffs[plot_idx_eff]
            
        color = marker_colors[idx % len(marker_colors)]
        highlight_width = max_len_data * 0.01 if max_len_data > 0 else 1
        ax_main.axvspan(plot_idx_eff - highlight_width/2, plot_idx_eff + highlight_width/2, 
                        color=color, alpha=0.2, zorder=1)
        ax_main.axvline(x=plot_idx_eff, color=color, linestyle='-', alpha=0.7, linewidth=1.5)
        
        ax_main.plot(plot_idx_eff, y_data, 
                    marker='*', markersize=14, color=color, 
                    markeredgewidth=1.5, markeredgecolor='white', zorder=5,
                    label='Selected Keyframe (pos in diff array)' if idx == 0 else "")
            
        time_sec = kf_val / fps if fps > 0 else 0
        y_max = ax_main.get_ylim()[1]
        y_offset = y_max * (0.15 if idx % 2 == 0 else -0.15)
        y_pos = min(max(y_data + y_offset, y_max * 0.1), y_max * 0.9)
        x_pos = plot_idx_eff
        ha = 'center'
        
        ax_main.annotate(f"Frame: {kf_val}\n({time_sec:.2f}s)",
                        xy=(plot_idx_eff, y_data), xytext=(x_pos, y_pos),
                        fontsize=9, ha=ha, va='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2" if idx % 2 == 0 else "arc3,rad=-0.2",
                                        color=color, lw=1.5))
    
    lines, labels = ax_main.get_legend_handles_labels()
    if ax_flow.get_legend_handles_labels()[1]: # Add flow lines/labels if they exist
        flow_lines, flow_labels = ax_flow.get_legend_handles_labels()
        lines.extend(flow_lines)
        labels.extend(flow_labels)

    # Add custom legend items if they weren't auto-added by scatter/plot with label
    if initial_keyframes is not None and not any('Initial Keyframes' in lab for lab in labels):
        lines.append(plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor='#8e44ad', markersize=8))
        labels.append('Initial Keyframes (pos in diff array)')
    
    if not any('Selected Keyframe' in lab for lab in labels) and keyframes:
         lines.append(plt.Line2D([0], [0], marker='*', color='w', 
                            markerfacecolor=marker_colors[0], markersize=12))
         labels.append('Selected Keyframe (pos in diff array)')

    leg = ax_main.legend(lines, labels, loc='upper right', frameon=True, 
                        fancybox=True, framealpha=0.8, shadow=True, fontsize=10)
    if leg: leg.get_frame().set_facecolor('#ffffff')
    
    # Keyframe display part
    # Ensure keyframe_images matches keyframes after all filtering
    if len(keyframes) != len(keyframe_images):
        print(f"可视化警告: 最终关键帧列表长度 ({len(keyframes)}) 与图像列表长度 ({len(keyframe_images)}) 不匹配。跳过图像显示。")
    else:
        # Calculate scores for display selection (up to 6 images)
        display_kf_indices = list(range(len(keyframes))) # Default to all if few
        if len(keyframes) > 6:
            # This scoring is a bit redundant if already done in sensitivity func, but good for display selection
            kf_display_scores = []
            max_diff_val = max(log_diffs) if np.any(log_diffs > 0) else 1.0
            max_flow_val = max(smoothed_flow) if np.any(smoothed_flow > 0) else 1.0

            for i, kf_val in enumerate(keyframes):
                score = 0.0
                if kf_val == 0: score = float('inf') # Prioritize first frame
                else:
                    diff_idx = kf_val - 1
                    if diff_idx >=0 and diff_idx < len(log_diffs) and diff_idx < len(smoothed_flow) :
                        norm_log_diff = log_diffs[diff_idx] / max_diff_val if max_diff_val != 0 else 0
                        norm_flow = smoothed_flow[diff_idx] / max_flow_val if max_flow_val != 0 else 0
                        score = norm_log_diff + norm_flow
                kf_display_scores.append((i, score))
            
            kf_display_scores.sort(key=lambda x: x[1], reverse=True)
            display_kf_indices = sorted([x[0] for x in kf_display_scores[:6]])


        for display_order_idx, original_kf_list_idx in enumerate(display_kf_indices):
            if display_order_idx >= 6: break # Max 6 images in plot

            ax = fig.add_subplot(gs[1, display_order_idx])
            img_to_show = keyframe_images[original_kf_list_idx]
            ax.imshow(cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB))
            
            frame_time = keyframes[original_kf_list_idx] / fps if fps > 0 else 0
            
            current_color = marker_colors[display_order_idx % len(marker_colors)]
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(current_color)
                spine.set_linewidth(3)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Frame: {keyframes[original_kf_list_idx]}', 
                        color=current_color, fontweight='bold', pad=8, fontsize=11)
            ax.text(0.5, -0.08, f'{frame_time:.2f} seconds', 
                    fontsize=10, ha='center', transform=ax.transAxes)
            
            # Small circle index in corner (adjust positions if needed)
            # Get image dimensions for text placement
            img_h, img_w = img_to_show.shape[:2]
            circle_radius_px = min(img_h, img_w) * 0.08 # Relative radius
            text_x_px = circle_radius_px
            text_y_px = circle_radius_px

            # Draw circle using Axes coordinates for simplicity if transform is complex
            # For pixel coords, more complex transform is needed. This is an approximation.
            # circle = plt.Circle((text_x_px/img_w, 1 - text_y_px/img_h), radius = circle_radius_px/img_w , color=current_color, transform=ax.transAxes, clip_on=False)
            # ax.add_patch(circle)
            # ax.text(text_x_px/img_w, 1- text_y_px/img_h, f"{display_order_idx+1}", fontsize=14, weight='bold', 
            #         color='white', ha='center', va='center', transform=ax.transAxes)
            # Simpler text label
            ax.text(0.05, 0.95, f"#{display_order_idx+1}", fontsize=12, weight='bold', color='white',
                    ha='left', va='top', transform=ax.transAxes,
                    bbox=dict(boxstyle="circle,pad=0.3", fc=current_color, ec="none"))


    plt.subplots_adjust(wspace=0.25, hspace=0.45, top=0.95, bottom=0.05, left=0.05, right=0.95)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white') # Lower DPI for speed if needed
    return fig


if __name__ == "__main__":
    # 请将 "input/videoplayback.mp4" 替换为您的视频文件路径
    video_path = "input/0519-3.mp4" 
    if not os.path.exists(video_path):
        print(f"错误: 视频文件未找到于路径 {video_path}")
        print("请确保视频文件存在，或修改 video_path 变量。")
        # 创建一个虚拟的input文件夹和空的mp4文件，以便代码至少可以尝试运行而不会因路径不存在而崩溃
        os.makedirs("input", exist_ok=True)
        with open(video_path, "w") as f:
            f.write("") # Create an empty file if it doesn't exist
        print(f"已创建空的占位视频文件 {video_path}。请替换为真实视频。")
        # exit() # 可以取消注释以在文件不存在时退出

    # 指定最小关键帧数量为6，最大为8，并保存到output文件夹
    keyframes, images, chart_path, image_paths = extract_frames(video_path, 
                                                sensitivity=7, # 敏感度可以调整 (1-10)
                                                min_keyframes=70,  # 调整为您期望的最小数量
                                                max_keyframes=80, # 调整为您期望的最大数量
                                                output_dir="output_frames") # 修改输出目录名
    
    if chart_path:
        print(f"生成的关键帧可视化已保存至: {chart_path}")
    if image_paths:
        print(f"关键帧图像已保存至: {', '.join(image_paths)}")
    if keyframes:
        print(f"关键帧位置 (帧号): {keyframes}")
    
    # 添加这一行来保持图像窗口打开 (如果 visualize_results 被调用并生成了图表)
    # 如果 visualize_results 被注释掉了，plt.show()可能不会显示任何内容
    # if chart_path:
    #    plt.show() # 通常在脚本末尾调用，以显示所有打开的matplotlib图表