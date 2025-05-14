import os
import shutil
import re

def organize_images_by_episode(source_folder):
    """
    Organizes images in the source_folder into subfolders based on episode number.
    E.g., E014****.png goes into a subfolder named "14集".
    """
    if not os.path.isdir(source_folder):
        print(f"错误：源文件夹 '{source_folder}' 不存在。")
        return

    print(f"正在处理文件夹: {source_folder}")

    for filename in os.listdir(source_folder):
        source_file_path = os.path.join(source_folder, filename)

        # 确保是文件，并且以 'E' 开头，以 '.png' 结尾
        if os.path.isfile(source_file_path) and \
           filename.startswith("E") and \
           filename.endswith(".png"):

            # 尝试从文件名提取集数 (E后面的三位数字)
            # 例如 E014C039... -> 014
            if len(filename) >= 4 and filename[1:4].isdigit():
                episode_str_padded = filename[1:4] # 如 "014", "010"
                try:
                    episode_num = int(episode_str_padded) # 如 14, 10
                except ValueError:
                    print(f"警告：无法从 '{filename}' 提取有效的集数编号，跳过。")
                    continue

                # 构建目标文件夹名称，例如 "14集", "10集"
                target_folder_name = f"{episode_num}集"
                target_dir_path = os.path.join(source_folder, target_folder_name)

                # 如果目标文件夹不存在，则创建它
                if not os.path.exists(target_dir_path):
                    try:
                        os.makedirs(target_dir_path)
                        print(f"已创建文件夹: {target_dir_path}")
                    except OSError as e:
                        print(f"错误：创建文件夹 '{target_dir_path}' 失败: {e}")
                        continue # 如果创建失败，跳过此文件

                # 构建目标文件完整路径
                destination_file_path = os.path.join(target_dir_path, filename)

                # 移动文件
                try:
                    shutil.move(source_file_path, destination_file_path)
                    print(f"已移动: {filename}  ->  {target_folder_name}/")
                except Exception as e:
                    print(f"错误：移动文件 '{filename}' 到 '{destination_file_path}' 失败: {e}")
            else:
                print(f"警告：文件名 '{filename}' 不符合 E*** 格式或长度不足，跳过。")
        # else:
        #     if os.path.isfile(source_file_path): # 如果需要，可以取消注释以查看跳过了哪些文件
        #         print(f"跳过不匹配的文件: {filename}")

    print("处理完成。")

if __name__ == "__main__":
    # ******************************************************************
    # ** 请将下面的路径修改为你的实际图片文件夹路径 **
    # ******************************************************************
    folder_to_process = "/Users/snychng/Work/code/Anime/data/250513/opt_auto"
    
    organize_images_by_episode(folder_to_process)