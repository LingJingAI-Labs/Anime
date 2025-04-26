import torch
import cv2
from ultralytics import YOLO
import argparse
import glob
import os

def image_inference(image_path, model_path):
    """
    对单张图片进行YOLO推理，返回占比最大的标签
    
    参数:
    image_path (str): 输入图片的路径
    model_path (str): YOLO模型权重的路径
    
    返回:
    str: 推理结果标签（"nanzhu"、"nvzhu"、"nanpei"、"nvpei"中的一个）
    """
    # 有效标签列表
    valid_labels = ["nanzhu", "nvzhu", "nanpei", "nvpei"]
    DEFAULT_LABEL = "nanzhu"  # 设置默认标签为nanzhu
    
    # 设备检查
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"将使用的设备: {device}")
    
    # 加载模型
    try:
        model = YOLO(model_path)
        model.to(device)
        print(f"成功加载模型并转移到设备: {device}")
    except Exception as e:
        print(f"加载模型或将其转移到设备时出错: {e}")
        return DEFAULT_LABEL  # 发生错误时返回默认标签
    
    # 读取图片
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误：无法读取图片 {image_path}")
            return DEFAULT_LABEL  # 读取失败时返回默认标签
        print(f"成功读取图片: {image_path}")
    except Exception as e:
        print(f"读取图片时出错: {e}")
        return DEFAULT_LABEL  # 发生异常时返回默认标签
    
    # 进行推理
    results = model(image, conf=0.25)
    
    # 处理结果
    if not results or len(results[0].boxes) == 0:
        print("未检测到任何目标，默认返回nanzhu")
        return DEFAULT_LABEL  # 无检测结果时返回默认标签
    
    # 找出最大面积的检测框
    max_area = 0
    max_label = None
    
    for box in results[0].boxes:
        # 获取边界框
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        area = (x2 - x1) * (y2 - y1)
        
        # 获取类别
        class_id = int(box.cls[0].item())
        class_name = results[0].names[class_id]
        
        # 只考虑有效标签
        if class_name in valid_labels:
            # 比较面积
            if area > max_area:
                max_area = area
                max_label = class_name
    
    if max_label:
        print(f"检测到的最大目标: {max_label}")
    else:
        print("未检测到有效标签，默认返回nanzhu")
        return DEFAULT_LABEL  # 未找到有效标签时返回默认标签
    
    return max_label

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="YOLO 角色检测测试工具")
    parser.add_argument("--model", default="models/best.pt", help="YOLO 模型路径")
    parser.add_argument("--dir", default="data/initial_frames", help="要检测的图片目录")
    parser.add_argument("--image", default=None, help="单张图片路径 (如果提供，则只检测该图片)")
    args = parser.parse_args()
    
    # 确保模型文件存在
    if not os.path.exists(args.model):
        # 尝试在当前脚本目录下查找
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_in_script_dir = os.path.join(script_dir, args.model)
        if os.path.exists(model_in_script_dir):
            args.model = model_in_script_dir
        else:
            print(f"错误: 模型文件不存在: {args.model}")
            exit(1)
    
    print(f"使用模型: {args.model}")
    
    # 处理单张图片或目录
    if args.image:
        if os.path.exists(args.image):
            print(f"\n处理单张图片: {args.image}")
            result = image_inference(args.image, args.model)
            print(f"结果: {result}")
        else:
            print(f"错误: 图片不存在: {args.image}")
    else:
        # 确保目录存在
        image_dir = args.dir
        if not os.path.exists(image_dir):
            # 尝试在当前脚本目录下查找
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dir_in_script_dir = os.path.join(script_dir, args.dir)
            if os.path.exists(dir_in_script_dir):
                image_dir = dir_in_script_dir
            else:
                # 尝试在项目根目录下查找
                project_root = os.path.dirname(script_dir)  # 假设脚本在 src 目录下
                dir_in_project_root = os.path.join(project_root, args.dir)
                if os.path.exists(dir_in_project_root):
                    image_dir = dir_in_project_root
                else:
                    print(f"错误: 图片目录不存在: {args.dir}")
                    exit(1)
        
        # 查找所有图片文件
        image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                     glob.glob(os.path.join(image_dir, "*.jpeg")) + \
                     glob.glob(os.path.join(image_dir, "*.png"))
        
        if not image_files:
            print(f"在目录 {image_dir} 中未找到任何图片")
            exit(1)
        
        print(f"找到 {len(image_files)} 张图片在目录 {image_dir} 中")
        
        # 统计各类别数量
        stats = {"nanzhu": 0, "nvzhu": 0, "nanpei": 0, "nvpei": 0}
        
        # 处理每张图片
        for i, img_path in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] 处理图片: {os.path.basename(img_path)}")
            result = image_inference(img_path, args.model)
            # 由于所有未检测到的情况现在都返回"nanzhu"，不再需要专门的"未检测到"类别
            stats[result] += 1
                
            # 每10张图片显示一次统计
            if (i+1) % 10 == 0 or i+1 == len(image_files):
                print("\n当前检测统计:")
                for category, count in stats.items():
                    print(f"  {category}: {count} 张")
                print()
        
        # 显示最终统计
        print("\n===== 检测完成 =====")
        print("最终统计结果:")
        for category, count in stats.items():
            percentage = (count / len(image_files)) * 100
            print(f"  {category}: {count} 张 ({percentage:.1f}%)")