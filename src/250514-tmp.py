import subprocess
import sys
import os

# 脚本文件名列表，按执行顺序列出
scripts_to_run = [
    "comfyui_redrawer_0514-chmr.py",
    "comfyui_redrawer_0514-wuji.py"
]

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

print("=============================================")
print("自动化脚本执行器 (允许失败并继续)")
print("=============================================\n")

all_scripts_succeeded = True # 标志，用于跟踪是否有任何脚本失败
failed_scripts_count = 0

for script_file in scripts_to_run:
    script_path = os.path.join(current_dir, script_file)

    if not os.path.exists(script_path):
        print(f"警告: 脚本 '{script_path}' 未找到。跳过执行。\n")
        all_scripts_succeeded = False # 标记为整体不成功
        failed_scripts_count += 1
        continue

    print(f"--- [开始] 执行脚本: {script_file} ---")
    try:
        # sys.executable 是当前运行此脚本的 Python 解释器路径
        # 将 check 设置为 False (或移除它)，这样即使脚本失败也不会抛出 CalledProcessError
        # 我们将手动检查 returncode
        process = subprocess.run(
            [sys.executable, script_path],
            # check=False, # 显式设置为 False 或直接移除
            text=True,
            # cwd=current_dir # 如果脚本依赖于特定的工作目录，可以设置它
        )

        if process.returncode == 0:
            print(f"--- [成功] 脚本: {script_file} 执行完毕。退出码: {process.returncode} ---\n")
        else:
            print(f"--- [注意] 脚本: {script_file} 执行完毕，但返回了非零退出码。---")
            print(f"退出码: {process.returncode}")
            # 子脚本的错误输出应该已经直接显示在控制台了。
            print(f"将继续执行下一个脚本...\n")
            all_scripts_succeeded = False # 标记为整体不成功
            failed_scripts_count +=1

    except FileNotFoundError:
        print(f"错误: 无法找到 Python 解释器 '{sys.executable}' 或脚本 '{script_path}' 无法被执行。")
        print(f"将继续执行下一个脚本...\n")
        all_scripts_succeeded = False
        failed_scripts_count += 1
    except Exception as e:
        # 捕获其他在 subprocess.run 期间可能发生的意外错误
        print(f"执行脚本 {script_file} 时发生未知错误: {e}")
        print(f"将继续执行下一个脚本...\n")
        all_scripts_succeeded = False
        failed_scripts_count += 1

print("=============================================")
if all_scripts_succeeded:
    print("所有脚本均已尝试执行，并且所有已执行的脚本都成功完成。")
elif failed_scripts_count == len(scripts_to_run):
    print("所有脚本都尝试执行，但均失败或未找到。")
else:
    print(f"所有脚本均已尝试执行。其中 {len(scripts_to_run) - failed_scripts_count} 个脚本成功，{failed_scripts_count} 个脚本失败或未找到。")
print("=============================================")

if not all_scripts_succeeded:
    print("\n请检查上面日志中的错误信息和非零退出码的脚本。")