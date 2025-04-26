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
    
    # 获取视频文件名（不含路径和扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 1. 提取关键帧
    keyframes, keyframe_images, frame_diffs, flow_magnitudes, fps, initial_keyframes = extract_keyframes_with_sensitivity(
        video_path, sensitivity, min_keyframes, max_keyframes, show_progress)

    # 2. 保存关键帧图像到输出目录
    saved_image_paths = []
    for i, (frame_idx, img) in enumerate(zip(keyframes, keyframe_images)):
        # 计算时间戳
        total_seconds = frame_idx / fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds - int(total_seconds)) * 1000)
        time_str = f"{minutes:02d}m{seconds:02d}s{milliseconds:03d}ms"
        
        # 使用新的命名规则
        img_filename = f"{video_name}_keyframe_{time_str}_{i+1:02d}.jpg"
        img_path = os.path.join(output_dir, img_filename)
        cv2.imwrite(img_path, img)
        saved_image_paths.append(img_path)
        print(f"保存关键帧 {i+1}/{len(keyframes)}: {img_path}")
    
    # # 3. 可视化结果并保存到输出目录
    # chart_filename = f"visualize_{video_name}_keyframes_sensitivity_{sensitivity:.1f}.jpg"
    # output_chart_path = os.path.join(output_dir, chart_filename)
    
    # visualize_results(keyframes, keyframe_images, frame_diffs, flow_magnitudes,
    #                 initial_keyframes=initial_keyframes, fps=fps, output_path=output_chart_path)
    
    # print(f"关键帧可视化已保存至: {output_chart_path}")
    # print(f"总共提取并保存了 {len(keyframes)} 个关键帧")

    # return keyframes, keyframe_images, output_chart_path, saved_image_paths
    output_chart_path = None
    return keyframes, keyframe_images, output_chart_path, saved_image_paths

def extract_keyframes_with_sensitivity(video_path, sensitivity=5.0, min_keyframes=6, max_keyframes=12, show_progress=True):
    """
    使用单一敏感度参数提取视频关键帧，并控制最大和最小数量
    """
    # 验证敏感度参数
    sensitivity = max(1.0, min(10.0, float(sensitivity)))
    
    # 调整映射公式，使阈值整体提高
    diff_threshold = 0.8 - (sensitivity * 0.05)  # 范围从0.3到0.75
    flow_threshold = 10.0 - (sensitivity * 0.7)  # 范围从3.0到9.3
    
    # 最小帧间距（秒），随敏感度调整
    min_frame_distance = 3.0 - sensitivity * 0.2  # 范围从1.0到2.8秒
    
    print(f"使用敏感度: {sensitivity}/10")
    print(f"映射到直方图差异阈值: {diff_threshold:.3f}")
    print(f"映射到光流阈值: {flow_threshold:.3f}")
    print(f"最小关键帧间距: {min_frame_distance:.1f}秒")
    
    # 提取初始关键帧集合
    keyframes, keyframe_images, frame_diffs, flow_magnitudes, fps = extract_keyframes_raw(
        video_path, diff_threshold, flow_threshold, show_progress)
    
    # 保存初始关键帧供后续可视化使用
    initial_keyframes = keyframes.copy()
    
    # 应用最小帧间距过滤
    min_frames_gap = int(min_frame_distance * fps)
    filtered_keyframes = [0]  # 始终保留第一帧
    filtered_images = [keyframe_images[0]]
    
    # 应用最小帧间距过滤
    min_frames_gap = int(min_frame_distance * fps)
    filtered_keyframes = [0]  # 始终保留第一帧
    filtered_images = [keyframe_images[0]]
    
    for i in range(1, len(keyframes)):
        # 检查与上一个保留的关键帧的距离
        if keyframes[i] - filtered_keyframes[-1] >= min_frames_gap:
            filtered_keyframes.append(keyframes[i])
            filtered_images.append(keyframe_images[i])
    
    # 如果关键帧数量少于最小值，尝试降低阈值或添加等间隔帧
    if len(filtered_keyframes) < min_keyframes:
        print(f"检测到的关键帧数量({len(filtered_keyframes)})低于最小要求({min_keyframes})，将添加额外帧")
        
        # 策略1: 如果已经存在一些关键帧，在它们之间均匀添加帧
        if len(filtered_keyframes) > 1:
            # 获取视频总帧数
            container = av.open(video_path)
            video_stream = next(s for s in container.streams if s.type == 'video')
            total_frames = video_stream.frames
            container.close()
            
            # 计算需要添加的帧数
            frames_to_add = min_keyframes - len(filtered_keyframes)
            
            # 均匀添加帧
            if frames_to_add > 0:
                # 使用线性间隔选择帧
                additional_frames = []
                
                # 将视频分成大致均匀的部分
                sections = min_keyframes - 1
                frames_per_section = total_frames // sections
                
                for i in range(1, sections):
                    frame_pos = i * frames_per_section
                    # 检查是否与现有关键帧过近
                    is_too_close = False
                    for existing in filtered_keyframes:
                        if abs(frame_pos - existing) < min_frames_gap // 2:
                            is_too_close = True
                            break
                    
                    if not is_too_close and frame_pos not in filtered_keyframes:
                        additional_frames.append(frame_pos)
                        if len(additional_frames) + len(filtered_keyframes) >= min_keyframes:
                            break
                
                # 读取这些帧并添加到关键帧集合
                if additional_frames:
                    # 并行提取额外帧
                    additional_images = extract_specific_frames(video_path, additional_frames)
                    
                    # 合并帧
                    for i, frame_idx in enumerate(sorted(additional_frames)):
                        filtered_keyframes.append(frame_idx)
                        filtered_images.append(additional_images[i])
                    
                    # 按帧顺序重新排序
                    combined = list(zip(filtered_keyframes, filtered_images))
                    combined.sort(key=lambda x: x[0])
                    filtered_keyframes, filtered_images = zip(*combined)
                    filtered_keyframes = list(filtered_keyframes)
                    filtered_images = list(filtered_images)
    
    # 如果关键帧数量仍然过多，选择最重要的帧
    if len(filtered_keyframes) > max_keyframes:
        # 计算每个关键帧的重要性分数
        frame_scores = []
        for i, kf in enumerate(filtered_keyframes):
            if i == 0:  # 保留第一帧
                frame_scores.append((i, 0))
                continue
            
            # 计算帧的重要性（基于直方图变化和光流）
            kf_idx = keyframes.index(kf) if kf in keyframes else -1
            if kf_idx >= 0 and kf < len(frame_diffs):
                hist_score = frame_diffs[kf-1] if kf > 0 and kf-1 < len(frame_diffs) else 0
                flow_score = flow_magnitudes[kf-1] if kf > 0 and kf-1 < len(flow_magnitudes) else 0
                score = hist_score + flow_score * 0.1
                frame_scores.append((i, score))
            else:
                # 对于额外添加的帧，给予较低的分数
                frame_scores.append((i, 0.1))
        
        # 按重要性排序
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择重要帧索引
        selected_indices = [0]  # 始终保留第一帧
        selected_indices.extend([x[0] for x in frame_scores[:(max_keyframes-1)]])
        selected_indices.sort()  # 按时间顺序排序
        
        # 提取选中的帧
        final_keyframes = [filtered_keyframes[i] for i in selected_indices]
        final_images = [filtered_images[i] for i in selected_indices]
    else:
        final_keyframes = filtered_keyframes
        final_images = filtered_images
    
    print(f"初始提取: {len(keyframes)} 帧")
    print(f"间距过滤后: {len(filtered_keyframes)} 帧")
    print(f"最终选择: {len(final_keyframes)} 帧 (最小要求: {min_keyframes}, 最大限制: {max_keyframes})")
    
    return final_keyframes, final_images, frame_diffs, flow_magnitudes, fps, initial_keyframes

def extract_specific_frames(video_path, frame_indices):
    """并行提取特定帧索引的图像"""
    images = [None] * len(frame_indices)
    indices_sorted = sorted(enumerate(frame_indices), key=lambda x: x[1])
    
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    
    for orig_idx, frame_idx in indices_sorted:
        container.seek(frame_idx, stream=video_stream)
        for frame in container.decode(video=0):
            images[orig_idx] = frame.to_ndarray(format='bgr24')
            break
            
    container.close()
    return images

# 定义帧处理器类处理批量帧
class FrameProcessor:
    def __init__(self, scale_factor=0.25):
        self.scale_factor = scale_factor
        # 检查是否可用CUDA加速
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_cuda:
            print("使用CUDA进行GPU加速")
            
    def process_frame_batch(self, frames_batch, prev_hist, prev_gray_small):
        results = []
        
        for curr_frame in frames_batch:
            # 计算直方图
            curr_hist = cv2.calcHist([curr_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()
            
            # 计算直方图差异
            hist_diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CHISQR_ALT)
            hist_diff = min(1.0, hist_diff / 100.0)  # 限制最大值为1.0
            
            # 计算光流
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            curr_gray_small = cv2.resize(curr_gray, (0,0), fx=self.scale_factor, fy=self.scale_factor)
            
            # 计算光流 - 可能的GPU加速
            if self.use_cuda:
                prev_gpu = cv2.cuda_GpuMat(prev_gray_small)
                curr_gpu = cv2.cuda_GpuMat(curr_gray_small)
                flow_gpu = cv2.cuda.FarnebackOpticalFlow.create(
                    5, 0.5, False, 15, 3, 5, 1.2, 0)
                flow = flow_gpu.calc(prev_gpu, curr_gpu, None)
                flow = flow.download()
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray_small, curr_gray_small, None, 
                    0.5, 3, 15, 3, 5, 1.2, 0)
                
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitude = np.mean(mag)
            
            results.append({
                'frame': curr_frame.copy(),
                'hist': curr_hist,
                'gray_small': curr_gray_small,
                'hist_diff': hist_diff,
                'flow_magnitude': flow_magnitude
            })
            
            # 更新前一帧信息用于下一帧处理
            prev_hist = curr_hist
            prev_gray_small = curr_gray_small
            
        return results, prev_hist, prev_gray_small

def extract_keyframes_raw(video_path, diff_threshold, flow_threshold, show_progress=True, scale_factor=0.3):
    """多线程优化版本的关键帧提取功能"""
    start_time = time.time()
    
    # 获取CPU核心数，为线程数量做准备
    num_threads = min(os.cpu_count(), 4)  # 使用最多4个线程，避免过多线程带来的开销
    print(f"使用{num_threads}个线程进行处理")
    
    # 使用PyAV快速获取视频信息
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == 'video')
    frame_count = video_stream.frames
    fps = float(video_stream.average_rate)
    print(f"视频包含 {frame_count} 帧，FPS: {fps}")
    
    # 初始化结果容器
    keyframes = [0]
    keyframe_images = []
    frame_diffs = []
    flow_magnitudes = []
    
    # 创建帧处理器实例
    processor = FrameProcessor(scale_factor=scale_factor)
    
    # 开始读取第一帧
    container.seek(0)
    for frame in container.decode(video=0):
        first_frame = frame.to_ndarray(format='bgr24')
        keyframe_images.append(first_frame.copy())
        
        # 准备第一帧的直方图和灰度图
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        first_gray_small = cv2.resize(first_gray, (0,0), fx=scale_factor, fy=scale_factor)
        first_hist = cv2.calcHist([first_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        first_hist = cv2.normalize(first_hist, first_hist).flatten()
        break
    
    # 重新打开容器以避免seek问题
    container.close()
    container = av.open(video_path)
    
    # 创建队列用于存储解码后的帧
    frame_queue = queue.Queue(maxsize=num_threads*2)
    result_queue = queue.Queue()
    
    # 结束信号
    end_signal = object()
    
    # 定义帧生产者函数
    def frame_producer():
        frame_idx = 0
        try:
            for frame in container.decode(video=0):
                if frame_idx > 0:  # 跳过第一帧，已经处理过了
                    frame_queue.put((frame_idx, frame.to_ndarray(format='bgr24')))
                frame_idx += 1
            
            # 发送结束信号
            for _ in range(num_threads):
                frame_queue.put((None, end_signal))
        except Exception as e:
            print(f"帧生产者错误: {e}")
    
    # 定义帧处理函数
    def frame_consumer(thread_id, prev_hist, prev_gray_small):
        batch_size = 5  # 每次处理的批量大小
        batch = []
        last_idx = 0
        
        try:
            while True:
                try:
                    idx, frame = frame_queue.get(timeout=2.0)  # 增加超时时间
                    if frame is end_signal:
                        # 处理剩余的批次
                        if batch:
                            batch_results, prev_hist, prev_gray_small = processor.process_frame_batch(
                                [b[1] for b in batch], prev_hist.copy(), prev_gray_small.copy())
                            
                            for i, res in enumerate(batch_results):
                                result_queue.put((batch[i][0], res))
                        print(f"线程{thread_id}接收到结束信号")
                        break
                    
                    batch.append((idx, frame))
                    last_idx = idx
                    
                    # 当批次填满时处理
                    if len(batch) >= batch_size:
                        batch_results, prev_hist, prev_gray_small = processor.process_frame_batch(
                            [b[1] for b in batch], prev_hist.copy(), prev_gray_small.copy())
                        
                        for i, res in enumerate(batch_results):
                            result_queue.put((batch[i][0], res))
                        
                        # 清理大对象
                        del batch_results
                        batch = []
                        
                    # 更新进度
                    if show_progress and idx % 50 == 0:
                        elapsed = time.time() - start_time
                        frames_per_second = idx / elapsed if elapsed > 0 else 0
                        print(f"线程{thread_id}处理到帧: {idx}, 速度: {frames_per_second:.1f} fps")
                    
                    frame_queue.task_done()
                except queue.Empty:
                    continue
                    
            # 在退出前发送明确的结束信号
            print(f"线程{thread_id}发送完成信号")
            result_queue.put((None, None))
            frame_queue.task_done()  # 标记结束信号已处理
        except Exception as e:
            print(f"帧处理器{thread_id}错误: {e}")
            # 确保即使出错也发送完成信号
            result_queue.put((None, None))
    
    # 启动帧生产者线程
    producer_thread = threading.Thread(target=frame_producer)
    producer_thread.daemon = True
    producer_thread.start()
    
    # 启动帧处理线程
    consumer_threads = []
    for i in range(num_threads):
        # 各线程使用同一个初始直方图和灰度图的副本
        t = threading.Thread(
            target=frame_consumer, 
            args=(i, first_hist.copy(), first_gray_small.copy())
        )
        t.daemon = True
        t.start()
        consumer_threads.append(t)
    
    # 收集处理结果的线程
    processed_frames = {}
    finished_threads = 0
    last_progress = 0

    # 增加超时保护机制
    max_wait_time = 5  # 最长等待时间(秒)
    last_activity_time = time.time()

    print("开始收集处理结果...")
    while finished_threads < num_threads and len(processed_frames) < frame_count - 1:
        try:
            idx, result = result_queue.get(timeout=1.0)
            last_activity_time = time.time()  # 重置活动时间
            
            if idx is None and result is None:
                finished_threads += 1
                print(f"收到线程完成信号: {finished_threads}/{num_threads}")
                result_queue.task_done()
                continue
                
            processed_frames[idx] = result
            
            # 检查是否为关键帧
            hist_diff = result['hist_diff']
            flow_magnitude = result['flow_magnitude']
            
            # 确保数组索引不越界
            while len(frame_diffs) <= idx-1:
                frame_diffs.append(0.0)
            while len(flow_magnitudes) <= idx-1:
                flow_magnitudes.append(0.0)
                
            if idx > 0:  # 避免第一帧(idx=0)的问题
                frame_diffs[idx-1] = hist_diff
                flow_magnitudes[idx-1] = flow_magnitude
            
            is_keyframe = (hist_diff > diff_threshold) or (flow_magnitude > flow_threshold)
            
            if is_keyframe:
                keyframes.append(idx)
                keyframe_images.append(result['frame'])
                
            result_queue.task_done()
            
            # 更新总体进度
            if show_progress and len(processed_frames) % 100 == 0 and len(processed_frames) > last_progress:
                last_progress = len(processed_frames)
                elapsed = time.time() - start_time
                frames_processed = len(processed_frames)
                fps_processing = frames_processed / elapsed if elapsed > 0 else 0
                eta = (frame_count - frames_processed) / fps_processing if fps_processing > 0 else 0
                print(f"总进度: {frames_processed}/{frame_count} ({frames_processed/frame_count*100:.1f}%) - "
                    f"速度: {fps_processing:.1f} fps - ETA: {eta:.1f}秒")
                    
        except queue.Empty:
            # 如果太久没有活动，可能是卡住了
            if time.time() - last_activity_time > max_wait_time:
                print(f"警告: 超过{max_wait_time}秒无活动，可能线程已卡住。检测到{finished_threads}/{num_threads}线程完成。")
                print(f"已处理{len(processed_frames)}/{frame_count}帧，假设剩余帧无需处理。")
                break  # 跳出循环，继续后续处理
            continue
    
    # 等待所有线程完成
    for t in consumer_threads:
        t.join()
    producer_thread.join()
    
    # 关闭容器
    container.close()
    
    # 确保帧差异和光流列表长度正确
    if len(frame_diffs) < frame_count - 1:
        padding = [0.0] * (frame_count - 1 - len(frame_diffs))
        frame_diffs.extend(padding)
        flow_magnitudes.extend(padding)
    
    elapsed = time.time() - start_time
    print(f"共提取 {len(keyframes)} 个关键帧，总耗时 {elapsed:.2f} 秒，平均处理速度 {frame_count/elapsed:.1f} fps")
    return keyframes, keyframe_images, frame_diffs, flow_magnitudes, fps

def visualize_results(keyframes, keyframe_images, frame_diffs, flow_magnitudes, 
                    initial_keyframes=None, fps=30.0, output_path="keyframes_result.jpg"):
    """改进版关键帧可视化函数 - 在单一图表上展示所有信息"""
    # 设置高质量的风格
    plt.style.use('ggplot')
    
    # 创建具有良好比例的图形 - 使用两行布局
    fig = plt.figure(figsize=(24, 12), dpi=60, facecolor='white')
    
    # 定义布局 - 回到两行布局
    gs = GridSpec(2, 6, height_ratios=[1.8, 1], hspace=0.4, wspace=0.2)
    
    # ------ 数据预处理和平滑化 ------
    # 应用移动平均来平滑数据
    window_size = 5
    
    # 平滑处理直方图差异
    scaled_diffs = [diff * 100 for diff in frame_diffs]  # 放大100倍
    smoothed_diffs = np.convolve(scaled_diffs, np.ones(window_size)/window_size, mode='same')
    
    # 平滑处理光流数据
    smoothed_flow = np.convolve(flow_magnitudes, np.ones(window_size)/window_size, mode='same')
    
    # 转换为对数值（避免负值）
    log_diffs = []
    for diff in smoothed_diffs:
        if diff <= 0:
            log_diffs.append(0)
        else:
            # 提高下限至1.0，确保不会出现负对数值
            log_diff = np.log10(max(1.0, diff))
            log_diffs.append(log_diff)
    
    # ------ 绘制主图表 ------
    ax_main = fig.add_subplot(gs[0, :])
    ax_flow = ax_main.twinx()
    
    # 设置背景
    ax_main.set_facecolor('#f8f9fa')
    
    # 确保Y轴不显示负值
    ax_main.set_ylim(bottom=-0.05)
    
    # 绘制直方图差异曲线（使用平滑数据）
    diff_line = ax_main.plot(log_diffs, label='Histogram Difference Score', 
                            color='#3498db', linewidth=2.0, alpha=0.8)
    
    ax_main.set_ylabel('Histogram Difference Score (log10)', 
                    color='#3498db', fontsize=12, fontweight='bold')
    ax_main.tick_params(axis='y', labelcolor='#3498db')
    
    # 绘制光流值曲线（使用平滑数据）
    flow_line = ax_flow.plot(smoothed_flow, label='Optical Flow (smoothed)', 
                            color='#e74c3c', linewidth=2.0, alpha=0.8)
    
    ax_flow.set_ylabel('Optical Flow Magnitude', 
                    color='#e74c3c', fontsize=12, fontweight='bold')
    ax_flow.tick_params(axis='y', labelcolor='#e74c3c')
    
    # 设置网格线样式
    ax_main.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
    
    # 标题和X轴
    ax_main.set_title('Result of Keyframe Extraction', fontsize=16, pad=15, fontweight='bold')
    ax_main.set_xlabel('Frame #', fontsize=12, fontweight='bold')
    
    # 绘制阈值水平线 - 使用虚线标记阈值水平
    diff_threshold = 0.4  # 假设的差异阈值
    flow_threshold = 5.0  # 假设的光流阈值
    ax_main.axhline(y=diff_threshold, color='#3498db', linestyle='--', alpha=0.5, 
                    label='Histogram Threshold')
    ax_flow.axhline(y=flow_threshold, color='#e74c3c', linestyle='--', alpha=0.5,
                label='Flow Threshold')
    
    # 标记初始提取的关键帧点 - 在图表上直接标记
    if initial_keyframes is not None:
        valid_initial_frames = [kf for kf in initial_keyframes if kf < len(log_diffs)]
        initial_y_values = [log_diffs[kf] if kf < len(log_diffs) else 0 for kf in valid_initial_frames]
        
        ax_main.scatter(valid_initial_frames, initial_y_values, 
                        marker='o', s=30, color='#8e44ad', alpha=0.7, 
                        label='Initial Keyframes', zorder=4)
    
    # 绘制和标记最终选择的关键帧
    marker_colors = ['#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#d35400', '#2980b9']
    
    for idx, kf in enumerate(keyframes):
        if kf >= len(log_diffs):
            continue
            
        color = marker_colors[idx % len(marker_colors)]
        
        # 添加垂直高亮区域
        highlight_width = len(log_diffs) * 0.01  # 宽度是总帧数的1%
        ax_main.axvspan(kf - highlight_width/2, kf + highlight_width/2, 
                        color=color, alpha=0.2, zorder=1)
        
        # 添加垂直线标记关键帧位置
        ax_main.axvline(x=kf, color=color, linestyle='-', alpha=0.7, linewidth=1.5)
        
        # 添加星形标记在图表中标记关键帧
        if kf < len(log_diffs):
            ax_main.plot(kf, log_diffs[kf], 
                        marker='*', markersize=14, color=color, 
                        markeredgewidth=1.5, markeredgecolor='white', zorder=5,
                        label='Selected Keyframe' if idx == 0 else "")
            
        # 添加帧号和时间信息标签
        time_sec = kf / fps
        
        # 计算标签位置，避免重叠
        y_data = log_diffs[kf] if kf < len(log_diffs) else 0
        y_max = ax_main.get_ylim()[1]
        
        # 交替在上方和下方放置标签
        y_offset = y_max * (0.15 if idx % 2 == 0 else -0.15)
        y_pos = min(max(y_data + y_offset, y_max * 0.1), y_max * 0.9)
        
        # 确定水平位置，避免靠近图表边界
        x_pos = kf
        ha = 'center'
        
        # 添加带背景的文本标签
        ax_main.annotate(f"Frame: {kf}\n({time_sec:.2f}s)",
                        xy=(kf, y_data), xytext=(x_pos, y_pos),
                        fontsize=9, ha=ha, va='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2" if idx % 2 == 0 else "arc3,rad=-0.2",
                                        color=color, lw=1.5))
    
    # 合并图例
    lines = diff_line + flow_line
    labels = [l.get_label() for l in lines]
    
    # 添加初始关键帧和选定关键帧到图例
    if initial_keyframes is not None:
        lines.append(plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor='#8e44ad', markersize=8))
        labels.append('Initial Keyframes')
    
    lines.append(plt.Line2D([0], [0], marker='*', color='w', 
                            markerfacecolor=marker_colors[0], markersize=12))
    labels.append('Selected Keyframes')
    
    # 添加阈值线到图例
    lines.append(plt.Line2D([0], [0], linestyle='--', color='#3498db', alpha=0.5))
    labels.append('Histogram Threshold')
    lines.append(plt.Line2D([0], [0], linestyle='--', color='#e74c3c', alpha=0.5))
    labels.append('Flow Threshold')
    
    # 放置图例在右上角
    leg = ax_main.legend(lines, labels, loc='upper right', frameon=True, 
                        fancybox=True, framealpha=0.8, shadow=True, fontsize=10)
    leg.get_frame().set_facecolor('#ffffff')
    
    # ------ 下半部分：显示关键帧 ------
    # 计算关键帧分数并选择显示的帧
    keyframe_scores = []
    for i, kf in enumerate(keyframes):
        if kf < len(frame_diffs):
            score = 0
            if i > 0:
                log_diff_idx = min(kf-1, len(log_diffs)-1)
                flow_idx = min(kf-1, len(flow_magnitudes)-1)
                
                log_diff = log_diffs[log_diff_idx]
                flow = smoothed_flow[flow_idx]
                
                # 归一化
                norm_log_diff = log_diff / max(log_diffs) if max(log_diffs) > 0 else 0
                norm_flow = flow / max(smoothed_flow) if max(smoothed_flow) > 0 else 0
                
                score = norm_log_diff + norm_flow
            keyframe_scores.append((i, score))
    
    keyframe_scores.sort(key=lambda x: x[1], reverse=True)
    total_kf = min(len(keyframe_images), 6)
    top_kf_indices = [x[0] for x in keyframe_scores[:total_kf]]
    top_kf_indices = sorted(top_kf_indices)  # 按时间顺序排序
    
    for idx, i in enumerate(top_kf_indices):
        if idx >= total_kf:
            continue
            
        # 创建子图
        ax = fig.add_subplot(gs[1, idx])
        
        # 显示图像
        ax.imshow(cv2.cvtColor(keyframe_images[i], cv2.COLOR_BGR2RGB))
        
        # 计算秒数
        frame_time = keyframes[i] / fps
        
        # 设置美观的边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(marker_colors[idx % len(marker_colors)])
            spine.set_linewidth(3)
        
        # 移除刻度但保留边框
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加标题，使用与上方标记相同的颜色
        ax.set_title(f'Frame: {keyframes[i]}', 
                    color=marker_colors[idx % len(marker_colors)],
                    fontweight='bold', pad=8, fontsize=11)
        
        # 添加秒数标签
        ax.text(0.5, -0.08, f'{frame_time:.2f} seconds', 
                fontsize=10, ha='center', transform=ax.transAxes)
        
        # 添加索引标签（小角标）
        circle = plt.Circle((0, 0), 15, color=marker_colors[idx % len(marker_colors)])
        ax.add_patch(circle)
        ax.text(60, 80, f"{idx+1}", fontsize=14, weight='bold', 
                color='white', ha='center', va='center')
    
    # 调整布局
    plt.subplots_adjust(wspace=0.25, hspace=0.45, top=0.95, bottom=0.05, left=0.05, right=0.95)
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # 显示图像但不阻塞
    plt.show(block=False)
    return fig

if __name__ == "__main__":
    video_path = "input/Episode-1-1.mp4"
    
    # 指定最小关键帧数量为6，最大为8，并保存到output文件夹
    keyframes, images, chart_path, image_paths = extract_frames(video_path, 
                                                sensitivity=2.5, 
                                                min_keyframes=6, 
                                                max_keyframes=12,
                                                output_dir="output")
    
    print(f"生成的关键帧可视化已保存至: {chart_path}")
    print(f"关键帧图像已保存至: {', '.join(image_paths)}")
    print(f"关键帧位置: {keyframes}")
    
    # 添加这一行来保持图像窗口打开
    plt.show()