import argparse
import os
import sys
import shutil
from yolo_detector import image_inference  # 导入 YOLO 检测函数

# 假设 comfyui_redrawer.py 在 src 目录下
# 将 src 目录添加到 Python 路径，以便导入
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = script_dir # 如果 main.py 在 src 里面
if src_dir not in sys.path:
    sys.path.append(src_dir)

from frame_extractor import extract_frames # 假设你的提取函数在这里
from comfyui_redrawer import ComfyUITester, SERVER_ADDRESS, WORKFLOW_FILE, OUTPUT_FOLDER # 导入所需内容

def clear_directory(dir_path):
    """清空指定目录下的所有文件和子目录"""
    if not os.path.isdir(dir_path):
        print(f"目录不存在，无需清空: {dir_path}")
        return
    print(f"开始清空目录: {dir_path}")
    for item_name in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item_name)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
                # print(f"  已删除文件: {item_name}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                # print(f"  已删除子目录: {item_name}")
        except Exception as e:
            print(f"  清空时出错 ({item_name}): {e}")
    print(f"目录已清空: {dir_path}")

def main():
    parser = argparse.ArgumentParser(description="视频关键帧提取并使用 ComfyUI 转绘")
    # --- 修改开始 ---
    # 将 video_path 改为可选参数 --video 或 -v
    parser.add_argument("-v", "--video_path", default="input/test-05.mp4",
                        help="输入视频文件的路径 (默认为 'input/test-05.mp4')")
    # --- 修改结束 ---
    parser.add_argument("-s", "--sensitivity", type=float, default=2.5,
                        help="关键帧提取敏感度 (1.0 - 10.0, 越高提取越多)")
    parser.add_argument("--min_keyframes", type=int, default=40,
                        help="最少提取的关键帧数量")
    parser.add_argument("--max_keyframes", type=int, default=60,
                        help="最多提取的关键帧数量")
    parser.add_argument("-o", "--output_dir", default="data/initial_frames",
                        help="关键帧提取的输出目录")
    parser.add_argument("--redraw_output_dir", default="data/redraw_results",
                        help="ComfyUI 转绘结果的输出目录")
    parser.add_argument("--no_progress", action="store_true",
                        help="不显示帧提取进度")
    # <--- 新增: ComfyUI 相关参数 (可选，如果不想硬编码)
    parser.add_argument("--comfy_server", default=SERVER_ADDRESS, help="ComfyUI 服务器地址")
    parser.add_argument("--comfy_workflow", default=WORKFLOW_FILE, help="ComfyUI 工作流文件路径")
    # 添加 YOLO 模型路径参数
    parser.add_argument("--yolo_model", default="models/best.pt", 
                        help="YOLO 模型权重路径 (用于角色检测)")
    
    # 修改: 将 use_yolo 改为 no_yolo，默认使用 YOLO
    parser.add_argument("--no_yolo", action="store_true",
                        help="禁用 YOLO 检测器自动识别角色类型")
    
    # 添加手动指定角色类型的参数
    parser.add_argument("--char", choices=["nanzhu", "nvzhu", "nanpei", "nvpei"], 
                        default=None, help="手动指定角色类型 (覆盖 YOLO 检测)")

    args = parser.parse_args()

    # 确保输入视频文件存在 (现在检查 args.video_path)
    project_root = os.path.dirname(script_dir) # 获取项目根目录
    absolute_video_path = os.path.join(project_root, args.video_path)

    if not os.path.exists(absolute_video_path):
        print(f"错误: 输入视频文件未找到: {absolute_video_path} (原始输入: {args.video_path})")
        # 尝试直接使用 args.video_path (如果用户提供了绝对路径)
        if not os.path.exists(args.video_path):
             print(f"也尝试了直接路径: {args.video_path}，同样未找到。")
             return
        else:
             # 如果直接路径存在，则使用它
             absolute_video_path = args.video_path
             print(f"使用了用户提供的路径: {absolute_video_path}")

    # 确保帧提取输出目录存在并清空
    absolute_output_dir = os.path.join(project_root, args.output_dir)
    clear_directory(absolute_output_dir)
    if not os.path.exists(absolute_output_dir):
        os.makedirs(absolute_output_dir)
        print(f"创建帧提取输出目录: {absolute_output_dir}")

    # 确保转绘结果输出目录存在并清空
    absolute_redraw_output_dir = os.path.join(project_root, args.redraw_output_dir)
    clear_directory(absolute_redraw_output_dir)
    if not os.path.exists(absolute_redraw_output_dir):
        os.makedirs(absolute_redraw_output_dir)
        print(f"创建转绘结果输出目录: {absolute_redraw_output_dir}")

    print(f"开始处理视频: {absolute_video_path}")
    print(f"帧提取参数: sensitivity={args.sensitivity}, min={args.min_keyframes}, max={args.max_keyframes}")
    print(f"帧提取输出目录: {absolute_output_dir}")
    print(f"ComfyUI 服务器: {args.comfy_server}")
    print(f"ComfyUI 工作流: {args.comfy_workflow}")
    print(f"转绘结果输出目录: {absolute_redraw_output_dir}")
    
    # 显示 YOLO 检测状态
    if args.no_yolo:
        print("YOLO 检测已禁用")
    else:
        print("YOLO 检测已启用")

    try:
        # 1. 提取关键帧
        print("\n--- 开始提取关键帧 ---")
        keyframes, keyframe_images, output_chart_path, saved_image_paths = extract_frames(
            video_path=absolute_video_path,
            sensitivity=args.sensitivity,
            min_keyframes=args.min_keyframes,
            max_keyframes=args.max_keyframes,
            show_progress=not args.no_progress,
            output_dir=absolute_output_dir
        )

        print("\n关键帧提取完成!")
        print(f"总共提取了 {len(keyframes)} 个关键帧。")
        print(f"关键帧图像保存在: {absolute_output_dir}")
        # if output_chart_path:
        #      print(f"可视化图表保存在: {output_chart_path}")

        if not saved_image_paths:
            print("没有提取到关键帧图像，无法进行转绘。")
            return

        # 2. 确保工作流文件路径正确
        print("\n--- 初始化 ComfyUI Redrawer ---")
        absolute_workflow_path = os.path.join(project_root, args.comfy_workflow)
        if not os.path.exists(absolute_workflow_path):
            # 尝试直接使用原始路径
            if not os.path.exists(args.comfy_workflow):
                print(f"错误: ComfyUI 工作流文件未找到: {absolute_workflow_path} 或 {args.comfy_workflow}")
                # 使用默认值
                absolute_workflow_path = args.comfy_workflow
                print(f"将尝试使用默认工作流路径: {absolute_workflow_path}")
            else:
                absolute_workflow_path = args.comfy_workflow
                print(f"使用了用户提供的工作流路径: {absolute_workflow_path}")

        # 3. 初始化 ComfyUI Redrawer
        redrawer = ComfyUITester(
            server_address=args.comfy_server,
            workflow_file=absolute_workflow_path,
            output_folder=absolute_redraw_output_dir
        )

        # 准备 YOLO 模型路径 (如果需要)
        absolute_model_path = None
        # 修改逻辑：默认使用 YOLO，除非 --no_yolo 被指定
        if not args.no_yolo and not args.char:
            absolute_model_path = os.path.join(project_root, args.yolo_model)
            if not os.path.exists(absolute_model_path):
                # 尝试直接使用路径
                if not os.path.exists(args.yolo_model):
                    # 尝试在脚本目录查找
                    script_dir_model_path = os.path.join(script_dir, args.yolo_model)
                    if os.path.exists(script_dir_model_path):
                        absolute_model_path = script_dir_model_path
                        print(f"在脚本目录找到 YOLO 模型: {absolute_model_path}")
                    else:
                        print(f"警告: YOLO 模型文件未找到: {absolute_model_path} 或 {args.yolo_model}")
                        print("将使用默认工作流")
                        absolute_model_path = None
                else:
                    absolute_model_path = args.yolo_model
                    print(f"将使用 YOLO 模型: {absolute_model_path}")
            else:
                print(f"将使用 YOLO 模型: {absolute_model_path}")

        # 4. 逐帧处理
        print("\n--- 开始逐帧转绘 ---")
        all_redraw_results = []
        for i, image_path in enumerate(saved_image_paths):
            print(f"\n[帧 {i+1}/{len(saved_image_paths)}]")
            if not os.path.exists(image_path):
                print(f"警告: 图像文件不存在，跳过: {image_path}")
                continue

            # 检测角色类型
            char_type = args.char  # 优先使用命令行指定的角色
            
            # 如果没有指定角色并且启用了 YOLO 检测
            if not char_type and not args.no_yolo and absolute_model_path:
                print(f"使用 YOLO 检测角色类型...")
                char_type = image_inference(image_path, absolute_model_path)
                if char_type:
                    print(f"检测到角色类型: {char_type}")
                else:
                    print(f"未检测到角色类型，将使用默认工作流")
                    
            # 如果检测到或指定了角色，更新工作流
            if char_type:
                redrawer.set_workflow_for_char(char_type)
                
            # 处理图像
            result_files = redrawer.process_image(image_path)
            
            if result_files:
                all_redraw_results.extend(result_files)
            else:
                print(f"处理图像失败: {os.path.basename(image_path)}")
                # 可以选择在这里停止或继续处理下一张

        print("\n--- 所有帧处理完毕 ---")
        if all_redraw_results:
            print("成功生成的转绘文件:")
            # for f_path in all_redraw_results:
            #     print(f"- {f_path}")
        else:
            print("未能成功生成任何转绘文件。")

    except Exception as e:
        import traceback
        print(f"\n处理过程中发生严重错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()