#!/usr/bin/env python3
import os
import re
import paramiko
import sys
from tqdm import tqdm  # 需要安装: pip install tqdm paramiko

# ===== 配置信息 =====
LOCAL_PATH = "/Users/snychng/Downloads/lora"
SERVER_USER = "root"     # 替换为您的服务器用户名
SERVER_IP = "36.143.229.162"      # 替换为您的服务器IP
SERVER_PASSWORD = "Lj666666!" # 替换为您的服务器密码
SERVER_BASE_PATH = "/data/comfyui/models/Lora/wuji/char"
# ===================

def extract_character_name(filename):
    """从文件名中提取角色名称"""
    match = re.match(r'char\d+(.+?)(?:-\d+)?\.safetensors', filename)
    if match:
        return match.group(1)
    return None

def main():
    # 连接到SSH服务器
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        print(f"正在连接到 {SERVER_USER}@{SERVER_IP}...")
        ssh.connect(SERVER_IP, username=SERVER_USER, password=SERVER_PASSWORD)
        print("SSH连接成功！")
        
        # 创建SFTP客户端
        sftp = ssh.open_sftp()
        
        # 创建基础目录
        ssh.exec_command(f'mkdir -p {SERVER_BASE_PATH}')
        
        # 获取本地模型文件列表
        try:
            files = [f for f in os.listdir(LOCAL_PATH) if f.startswith('char') and f.endswith('.safetensors')]
        except FileNotFoundError:
            print(f"错误: 本地路径 {LOCAL_PATH} 不存在")
            return
        
        if not files:
            print(f"警告: 在 {LOCAL_PATH} 中没有找到模型文件")
            return
        
        print(f"找到了 {len(files)} 个模型文件，准备上传...")
        
        # 处理每个文件
        for filename in files:
            char_name = extract_character_name(filename)
            if not char_name:
                print(f"警告: 无法从 {filename} 中提取角色名")
                continue
            
            # 创建角色目录
            remote_char_dir = f"{SERVER_BASE_PATH}/{char_name}"
            stdin, stdout, stderr = ssh.exec_command(f'mkdir -p {remote_char_dir}')
            stderr_content = stderr.read().decode('utf-8')
            if stderr_content:
                print(f"创建目录错误: {stderr_content}")
                continue
            
            # 上传文件
            local_file_path = os.path.join(LOCAL_PATH, filename)
            remote_file_path = f"{remote_char_dir}/{filename}"
            
            # 获取文件大小以显示进度
            file_size = os.path.getsize(local_file_path)
            
            print(f"上传 {filename} 到 {remote_char_dir}/ ({file_size/1024/1024:.2f} MB)")
            
            # 使用tqdm显示上传进度
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename) as pbar:
                sftp.put(local_file_path, remote_file_path, callback=lambda transferred, total: pbar.update(transferred - pbar.n))
            
            print(f"成功: {filename} 已上传到 {remote_char_dir}/")
        
        print("所有文件处理完成！")
        
    except paramiko.AuthenticationException:
        print("认证失败: 用户名或密码错误")
    except paramiko.SSHException as e:
        print(f"SSH连接错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        try:
            sftp.close()
        except:
            pass
        ssh.close()
        print("SSH连接已关闭")

if __name__ == "__main__":
    main()