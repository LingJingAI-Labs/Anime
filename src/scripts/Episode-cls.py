import os
import re
import shutil

def classify_images_by_episode(source_folder):
    """
    根据图片文件名中的 E<num> 将图片移动到对应的 <num>集 文件夹中。

    参数:
    source_folder (str): 包含图片的源文件夹路径。
    """
    print(f"开始处理文件夹：{source_folder}")

    if not os.path.isdir(source_folder):
        print(f"错误：文件夹 {source_folder} 不存在。")
        return

    # 正则表达式用于匹配文件名中的 E<num> 部分
    # E 后跟一个或多个数字 (\d+)，然后是 C
    pattern = re.compile(r"^E(\d+)C.*", re.IGNORECASE) # re.IGNORECASE 使匹配不区分大小写，尽管示例中都是大写E

    processed_files = 0
    created_folders = set()

    for filename in os.listdir(source_folder):
        source_file_path = os.path.join(source_folder, filename)

        # 确保处理的是文件而不是文件夹
        if os.path.isfile(source_file_path):
            match = pattern.match(filename)
            if match:
                # 提取集数编号，例如 "009" 或 "011"
                episode_number_str = match.group(1)
                # 将 "009" 转换为整数 9，再转为字符串 "9" 用于文件夹名
                episode_number = int(episode_number_str)

                # 构建目标文件夹名，例如 "9集"
                destination_subdir_name = f"{episode_number}集"
                destination_folder_path = os.path.join(source_folder, destination_subdir_name)

                # 如果目标文件夹不存在，则创建它
                if not os.path.exists(destination_folder_path):
                    try:
                        os.makedirs(destination_folder_path)
                        print(f"已创建文件夹：{destination_folder_path}")
                        created_folders.add(destination_subdir_name)
                    except OSError as e:
                        print(f"创建文件夹 {destination_folder_path} 失败：{e}")
                        continue # 跳过此文件

                # 构建目标文件完整路径
                destination_file_path = os.path.join(destination_folder_path, filename)

                # 移动文件
                try:
                    shutil.move(source_file_path, destination_file_path)
                    print(f"已移动 '{filename}' 到 '{destination_subdir_name}'")
                    processed_files += 1
                except Exception as e:
                    print(f"移动文件 '{filename}' 失败：{e}")
            else:
                print(f"文件名 '{filename}' 不符合 E<num>C... 格式，已跳过。")
        # else:
        #     print(f"跳过子目录或非文件项：{filename}") # 如果需要，可以取消注释此行

    print("\n处理完成！")
    print(f"总共处理并移动了 {processed_files} 个文件。")
    if created_folders:
        print(f"创建的新文件夹有：{', '.join(sorted(list(created_folders)))}")
    else:
        print("没有创建新的文件夹。")

# --- 使用示例 ---
if __name__ == "__main__":
    # 请将这里的路径替换为你的实际图片文件夹路径
    target_directory = "data/250514-wuji/opt_auto/"

    # 在运行脚本前，强烈建议先备份你的数据！
    # input("请按 Enter 键开始执行文件分类，或按 Ctrl+C 取消...") # 可选：添加一个确认步骤

    classify_images_by_episode(target_directory)