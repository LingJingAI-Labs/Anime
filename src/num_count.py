import os
import shutil
import re

def organize_images_by_episode(source_folder_path):
    """
    整理指定文件夹中的图片到按集数命名的子文件夹中。

    参数:
    source_folder_path (str): 包含图片的源文件夹路径。
    """
    # 1. 确定新的基础目标文件夹路径
    # 它将是源文件夹的同级目录，并在源文件夹名称后附加 "_分类后"
    source_folder_name = os.path.basename(source_folder_path.rstrip('/\\')) # 获取源文件夹名，如 "opt_auto"
    parent_directory = os.path.dirname(source_folder_path.rstrip('/\\'))   # 获取源文件夹的父目录
    
    # 新建的文件夹名称
    destination_base_folder_name = f"{source_folder_name}_分集"
    destination_base_folder_path = os.path.join(parent_directory, destination_base_folder_name)

    # 2. 创建基础目标文件夹（如果它还不存在）
    os.makedirs(destination_base_folder_path, exist_ok=True)
    print(f"文件将整理到: {destination_base_folder_path}")

    # 3. 遍历源文件夹中的所有文件
    files_moved_count = 0
    files_skipped_count = 0

    for filename in os.listdir(source_folder_path):
        source_file_path = os.path.join(source_folder_path, filename)

        # 4. 确保处理的是文件，而不是子目录
        if not os.path.isfile(source_file_path):
            continue

        # 5. 使用正则表达式从文件名中提取集数编号
        # 匹配以 "E" (不区分大小写) 开头，后跟三位数字的模式
        # 例如: E050C027-1-iter10... -> 提取 "050"
        match = re.match(r"E(\d{3})", filename, re.IGNORECASE)

        if match:
            episode_digits = match.group(1)  # 这是字符串，如 "050"
            episode_number = int(episode_digits)  # 转换为整数，如 50
            
            # 6. 创建目标子文件夹的名称 (例如 "50集")
            target_subfolder_name = f"{episode_number}集"
            target_subfolder_path = os.path.join(destination_base_folder_path, target_subfolder_name)

            # 7. 创建目标子文件夹 (如果它还不存在)
            os.makedirs(target_subfolder_path, exist_ok=True)

            # 8. 构建文件的完整目标路径
            destination_file_path = os.path.join(target_subfolder_path, filename)

            # 9. 移动文件
            try:
                shutil.move(source_file_path, destination_file_path)
                print(f"已移动: {filename} -> {target_subfolder_name}")
                files_moved_count += 1
            except Exception as e:
                print(f"错误: 移动文件 {filename} 失败: {e}")
                files_skipped_count += 1
        else:
            # 如果文件名不符合 "E***" 格式，则打印消息并跳过
            print(f"跳过 (格式不匹配 E***): {filename}")
            files_skipped_count += 1
    
    print(f"\n处理完成！")
    print(f"总共移动了 {files_moved_count} 个文件。")
    print(f"总共跳过了 {files_skipped_count} 个文件（包括格式不匹配或移动失败的）。")

if __name__ == "__main__":
    # **********************************************************************
    # * 请在这里设置你的源文件夹路径                                         *
    # **********************************************************************
    source_directory = "/Users/snychng/Work/code/Anime/data/250511/opt_auto"
    
    if not os.path.isdir(source_directory):
        print(f"错误: 源文件夹 '{source_directory}' 不存在或不是一个有效的目录。请检查路径。")
    else:
        organize_images_by_episode(source_directory)