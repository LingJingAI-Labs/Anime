import os
import getpass
import paramiko
import time

# --- 配置 ---
REMOTE_HOST = "36.143.229.162"
REMOTE_USER = "root"
# 密码将在运行时输入

# 本地 Pixel 文件夹路径 (请修改为你的实际路径)
# 例如: "D:/Lingjing/Snychng/Models/Pixel" 或 "/mnt/d/Lingjing/Snychng/Models/Pixel"
LOCAL_PIXEL_MODELS_PATH = "D:/Snychng/Models/tmp0519" # <--- !!! 修改这里 !!!

# 远程目标路径
REMOTE_LORA_BASE_PATH = "/data/comfyui/models/Lora/wuji/char"
REMOTE_LORA_PIXEL_SUBDIR = "叶慕白" # 新建的子文件夹名

# --- 函数定义 ---
def sftp_upload_file(sftp_client, local_path, remote_path):
    """上传单个文件到SFTP服务器"""
    try:
        print(f"  正在上传: {os.path.basename(local_path)} -> {remote_path} ...")

        # Get file size
        file_size = os.path.getsize(local_path)

        # Start time for speed calculation
        global start_time
        start_time = time.time()

        # Upload file with callback for progress
        sftp_client.put(local_path, remote_path, callback=lambda transferred, total: display_progress(transferred, file_size))

        print(f"\n  上传成功: {os.path.basename(local_path)}")
    except Exception as e:
        print(f"  上传失败: {os.path.basename(local_path)}. 错误: {e}")

def get_model_files(folder_path):
    """获取文件夹内所有 .safetensors 文件"""
    files = []
    for f_name in os.listdir(folder_path):
        if f_name.lower().endswith(".safetensors"):
            full_path = os.path.join(folder_path, f_name)
            if os.path.isfile(full_path):
                files.append(full_path)
    return files

# Function to calculate upload speed and display progress
def display_progress(transferred, total):
    progress_percentage = (transferred / total) * 100
    print(f"\r上传进度: {progress_percentage:.2f}%", end="")

    # Calculate speed
    speed = transferred / (time.time() - start_time)
    print(f" 速度: {speed / 1024:.2f} KB/s", end="")

def create_remote_directory_if_not_exists(sftp_client, remote_dir_path):
    """在SFTP服务器上创建目录（如果不存在）"""
    try:
        sftp_client.stat(remote_dir_path)
        print(f"远程目录 '{remote_dir_path}' 已存在。")
    except FileNotFoundError: # paramiko raises FileNotFoundError if dir/file doesn't exist
        print(f"远程目录 '{remote_dir_path}' 不存在，正在创建...")
        try:
            sftp_client.mkdir(remote_dir_path)
            print(f"远程目录 '{remote_dir_path}' 创建成功。")
        except Exception as e:
            print(f"创建远程目录 '{remote_dir_path}' 失败: {e}")
            return False # Indicate failure
    except Exception as e:
        print(f"检查远程目录 '{remote_dir_path}' 时出错: {e}")
        return False
    return True

# --- 主逻辑 ---
if __name__ == "__main__":
    if not os.path.isdir(LOCAL_PIXEL_MODELS_PATH):
        print(f"错误：本地文件夹 '{LOCAL_PIXEL_MODELS_PATH}' 不存在或不是一个目录。请检查路径。")
        exit(1)

    print(f"本地模型文件夹: {LOCAL_PIXEL_MODELS_PATH}")
    
    model_files = get_model_files(LOCAL_PIXEL_MODELS_PATH)

    if not model_files:
        print("在指定的本地文件夹中没有找到 .safetensors 文件。")
        exit(0)
    
    print("\n将要上传的文件:")
    for p in model_files:
        print(f"  - {os.path.basename(p)}")
    
    remote_password = getpass.getpass(f"请输入服务器 {REMOTE_HOST} 用户 {REMOTE_USER} 的密码: ")

    ssh_client = None
    sftp = None

    try:
        print(f"\n正在连接到 {REMOTE_HOST}...")
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) # 自动添加主机密钥
        ssh_client.connect(REMOTE_HOST, username=REMOTE_USER, password=remote_password)
        sftp = ssh_client.open_sftp()
        print("连接成功，SFTP会话已打开。")

        # 上传所有模型到 Lora/pixel 文件夹
        remote_lora_pixel_path = f"{REMOTE_LORA_BASE_PATH}/{REMOTE_LORA_PIXEL_SUBDIR}"
        print(f"\n[任务] 上传所有模型到 Lora/{REMOTE_LORA_PIXEL_SUBDIR} 文件夹:")
        
        # 确保远程 Lora/pixel 目录存在
        if not create_remote_directory_if_not_exists(sftp, remote_lora_pixel_path):
            print(f"无法创建或访问远程目录 {remote_lora_pixel_path}。终止上传。")
            exit(1)

        for local_model_path in model_files:
            remote_model_path = f"{remote_lora_pixel_path}/{os.path.basename(local_model_path)}"
            sftp_upload_file(sftp, local_model_path, remote_model_path)
            
        print("\n所有操作完成。")

    except paramiko.AuthenticationException:
        print("认证失败！请检查用户名或密码。")
    except paramiko.SSHException as sshException:
        print(f"SSH 连接错误: {sshException}")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if sftp:
            sftp.close()
            print("SFTP会话已关闭。")
        if ssh_client:
            ssh_client.close()
            print("SSH连接已关闭。")
