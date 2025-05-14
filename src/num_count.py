import os

def count_image_files(directory_path):
    """
    计算指定目录及其子目录中图像文件的总数。

    参数:
    directory_path (str): 要搜索的目录路径。

    返回:
    int: 找到的图像文件总数。
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic', '.heif'}
    image_count = 0

    # 检查目录是否存在
    if not os.path.isdir(directory_path):
        print(f"错误：目录 '{directory_path}' 不存在或不是一个有效的目录。")
        return 0

    print(f"正在扫描目录: {directory_path}")

    for root, _, files in os.walk(directory_path):
        # print(f"  正在检查: {root}") # 如果需要，可以取消注释这行来查看正在扫描的子目录
        for filename in files:
            # 获取文件扩展名并转换为小写
            extension = os.path.splitext(filename)[1].lower()
            if extension in image_extensions:
                image_count += 1
                # print(f"    找到图像: {os.path.join(root, filename)}") # 如果需要，可以取消注释这行来查看找到的每个图像文件

    return image_count

if __name__ == "__main__":
    # **************************************************
    # 请将下面的路径替换为您要扫描的实际路径
    # **************************************************
    target_directory = "/Users/snychng/Downloads/lora"

    total_images = count_image_files(target_directory)

    if os.path.isdir(target_directory): # 再次检查，以确保只有在目录有效时才打印最终计数
        print(f"\n在目录 '{target_directory}' 及其子目录中总共找到 {total_images} 个图像文件。")