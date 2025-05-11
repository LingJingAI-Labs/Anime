import os

# --- 配置路径 ---
# 源图片文件夹 (图片1类型)
source_opt_auto_dir = "/Users/snychng/Work/code/Anime/data/250508/opt_auto"

# 参考图片文件夹基础路径 (图片2类型的基础路径)
reference_base_dir = "/Users/snychng/Work/code/Anime/data/250508"

# (目标文件夹基础路径不再需要，因为我们只检查不复制)
# destination_base_dir = "/Users/snychng/Work/code/Anime/data/250508-opt"

# --- 主逻辑 ---
def check_unmatched_images():
    print(f"开始检查未匹配的图片...")
    print(f"源 (opt_auto) 文件夹: {source_opt_auto_dir}")
    print(f"参考文件夹基础路径: {reference_base_dir}")
    print("-" * 30)

    if not os.path.isdir(source_opt_auto_dir):
        print(f"错误: 源文件夹 {source_opt_auto_dir} 不存在！")
        return

    if not os.path.isdir(reference_base_dir):
        print(f"错误: 参考文件夹基础路径 {reference_base_dir} 不存在！")
        return

    # 用于记录所有从 source_opt_auto_dir 中被匹配到的文件名
    matched_source_files = set()
    total_source_files_checked = 0
    total_matches_found = 0

    # 获取源文件夹中的所有文件名，以便后续检查
    source_file_list = []
    if os.path.isdir(source_opt_auto_dir):
        for f_name in os.listdir(source_opt_auto_dir):
            if os.path.isfile(os.path.join(source_opt_auto_dir, f_name)):
                source_file_list.append(f_name)
    total_source_files_checked = len(source_file_list)
    print(f"在源文件夹 {source_opt_auto_dir} 中找到 {total_source_files_checked} 个文件待检查。")


    # 遍历场景1到场景9
    for i in range(1, 10):
        scene_name = f"场景{i}"
        current_reference_subdir = os.path.join(reference_base_dir, scene_name, "01")

        print(f"\n正在检查 {scene_name} 的匹配情况...")
        print(f"  参考子文件夹: {current_reference_subdir}")

        if not os.path.isdir(current_reference_subdir):
            print(f"  警告: 参考子文件夹 {current_reference_subdir} 不存在，跳过此场景的匹配检查。")
            continue

        # 1. 获取当前场景 "01" 文件夹中所有图片文件名的前8个字符
        reference_prefixes = set()
        for ref_filename in os.listdir(current_reference_subdir):
            ref_filepath = os.path.join(current_reference_subdir, ref_filename)
            if os.path.isfile(ref_filepath) and len(ref_filename) >= 8: # 确保文件名至少有8个字符
                prefix = ref_filename[:8]
                reference_prefixes.add(prefix)

        if not reference_prefixes:
            print(f"  注意: 在 {current_reference_subdir} 中没有找到任何参考图片文件。")
            continue
        
        # print(f"  {scene_name} 的参考前缀: {reference_prefixes}") # 用于调试

        # 2. 遍历 source_opt_auto_dir 中的图片 (使用之前获取的列表)
        matches_for_this_scene = 0
        for src_filename in source_file_list: # 遍历所有源文件
            if len(src_filename) >= 8: # 确保源文件名至少有8个字符
                src_prefix = src_filename[:8]

                # 3. 如果前缀匹配
                if src_prefix in reference_prefixes:
                    if src_filename not in matched_source_files: # 避免重复计数同一个源文件多次匹配到不同场景（虽然不太可能基于当前逻辑）
                        print(f"  匹配发现: 源文件 '{src_filename}' (前缀 {src_prefix}) 可以匹配到 {scene_name}")
                        matched_source_files.add(src_filename)
                        matches_for_this_scene +=1
                        total_matches_found +=1 # 这个计数器会计算总的匹配次数，一个源文件如果能匹配多个场景会被多次计入这里
                                                # matched_source_files.add 会保证每个源文件只被记录一次为“已匹配”

        if matches_for_this_scene > 0:
            print(f"  {scene_name}: 为 {matches_for_this_scene} 个源文件找到了匹配参考。")
        else:
            print(f"  {scene_name}: 没有为任何新的源文件找到匹配参考。")


    print("-" * 30)
    print(f"检查完成。总共检查了 {total_source_files_checked} 个源文件。")
    print(f"共找到 {len(matched_source_files)} 个不同的源文件有至少一个匹配参考。")


    # --- 列出未被匹配的文件 ---
    print("\n--- 未匹配的源文件列表 ---")
    unmatched_files_list = []
    for filename in source_file_list:
        if filename not in matched_source_files:
            unmatched_files_list.append(filename)

    if unmatched_files_list:
        print(f"在源文件夹 '{source_opt_auto_dir}' 中，以下 {len(unmatched_files_list)} 个 (共 {total_source_files_checked} 个) 文件没有匹配到任何场景参考：")
        for f_name in unmatched_files_list:
            print(f"  - {f_name}")
    else:
        if total_source_files_checked > 0 :
            print(f"恭喜！源文件夹 '{source_opt_auto_dir}' 中的所有 {total_source_files_checked} 个文件都已成功找到匹配参考。")
        else:
            print(f"源文件夹 '{source_opt_auto_dir}' 中没有文件可供检查。")


if __name__ == "__main__":
    # 在运行前，请再次确认路径是否正确！
    check_unmatched_images()
    # print("脚本已准备好。请取消上面一行 'check_unmatched_images()' 的注释以运行。")