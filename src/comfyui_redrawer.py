import json
import requests
import time
import matplotlib.pyplot as plt
import threading
import os
import random
import sys
from datetime import datetime
import uuid
import websocket
import paramiko
import re
import glob
import base64 # 需要导入 base64
from prompt_reasoning import generate_anime_prompt

# 配置参数
SERVER_ADDRESS = "http://106.54.35.113:6889"  # ComfyUI服务器地址
WORKFLOW_FILE = "workflow/T2I-nanzhu-prompt.json"  # 默认工作流，可以被覆盖
OUTPUT_FOLDER = "data/redraw_results"
MONITOR_INTERVAL = 0.5  # 监控间隔(秒)
MAX_WAIT_TIME = 300  # 最大等待时间

# SSH连接配置
SSH_CONFIG = {
    "hostname": "106.54.35.113",
    "port": 22,
    "username": "root",
    "password": "Lj666666",
    "timeout": 10
}

IMAGE_INPUT_NODE_ID = "74" # 输入图像节点的ID
PROMPT_NODE_ID = "155"    # <--- 新增: 定义目标提示词节点的ID

class RemoteMonitor:
    """通过SSH连接监控远程服务器资源"""

    def __init__(self):
        """初始化远程监控"""
        self.ssh_client = None
        self.connected = False
        self._connect()

    def _connect(self):
        """建立SSH连接"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            connect_kwargs = {
                "hostname": SSH_CONFIG["hostname"],
                "port": SSH_CONFIG["port"],
                "username": SSH_CONFIG["username"],
                "timeout": SSH_CONFIG["timeout"]
            }

            if "password" in SSH_CONFIG and SSH_CONFIG["password"]:
                connect_kwargs["password"] = SSH_CONFIG["password"]
            elif "key_filename" in SSH_CONFIG and SSH_CONFIG["key_filename"]:
                connect_kwargs["key_filename"] = SSH_CONFIG["key_filename"]

            self.ssh_client.connect(**connect_kwargs)
            self.connected = True
            print("已成功连接到远程服务器以监控资源")
        except Exception as e:
            print(f"SSH连接失败: {e}")
            self.connected = False

    def execute_command(self, command):
        """执行远程命令并返回输出"""
        if not self.connected or not self.ssh_client:
            print("未连接到远程服务器")
            return None

        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(command, timeout=5)
            return stdout.read().decode()
        except Exception as e:
            print(f"执行命令失败: {e}")
            self._connect()
            return None

    def get_cpu_stats(self):
        """获取CPU使用情况"""
        result = self.execute_command("top -bn1 | grep 'Cpu(s)' | awk '{print $2 + $4}'")
        if result:
            try:
                cpu_percent = float(result.strip())
                return cpu_percent
            except: pass
        return 0.0

    def get_memory_stats(self):
        """获取内存使用情况"""
        result = self.execute_command("free -m | grep Mem")
        if result:
            try:
                parts = result.split()
                total = float(parts[1]); used = float(parts[2]); free = float(parts[3])
                total_gb = total / 1024; used_gb = used / 1024; free_gb = free / 1024
                percent = (used / total) * 100
                return {"total_gb": total_gb, "used_gb": used_gb, "free_gb": free_gb, "percent": percent}
            except: pass
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "percent": 0}

    def get_gpu_stats(self):
        """获取GPU使用情况"""
        result = self.execute_command("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits")
        if not result: return []
        gpu_stats = []
        try:
            for line in result.strip().split('\n'):
                if not line.strip(): continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    gpu_idx = int(parts[0]); gpu_name = parts[1]; gpu_util = float(parts[2])
                    mem_used = float(parts[3]) / 1024; mem_total = float(parts[4]) / 1024; temp = float(parts[5])
                    gpu_stats.append({
                        'index': gpu_idx, 'name': gpu_name, 'gpu_util': gpu_util,
                        'memory_used': mem_used, 'memory_total': mem_total,
                        'memory_percent': (mem_used / mem_total) * 100 if mem_total > 0 else 0,
                        'temperature': temp
                    })
        except Exception as e: print(f"解析GPU数据失败: {e}")
        return gpu_stats

    def get_all_stats(self):
        """获取所有资源统计信息"""
        return {"cpu_percent": self.get_cpu_stats(), "memory": self.get_memory_stats(), "gpu": self.get_gpu_stats()}

    def close(self):
        """关闭SSH连接"""
        if self.ssh_client:
            try: self.ssh_client.close(); print("已关闭SSH连接")
            except: pass

class ResourceMonitor:
    """远程服务器资源监控类"""
    # ... (ResourceMonitor 代码保持不变，这里省略以减少篇幅) ...
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.running = False
        self.data = {
            'timestamps': [], 'cpu_percent': [], 'memory_percent': [],
            'memory_used_gb': [], 'memory_available_gb': [], 'time_points': []
        }
        self.remote_monitor = RemoteMonitor()
        gpu_stats = self.remote_monitor.get_gpu_stats()
        self.has_gpu = len(gpu_stats) > 0
        if self.has_gpu:
            print(f"检测到远程服务器上有 {len(gpu_stats)} 个GPU:")
            for gpu in gpu_stats: print(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['memory_total']:.2f} GB)")
            self.data['gpu_data'] = []
        else: print("远程服务器上未检测到GPU")
        self.test_name = None; self.start_time = None; self.monitor_interval = MONITOR_INTERVAL
        os.makedirs(output_folder, exist_ok=True)

    def start_monitoring(self, test_name, output_dir=None):
        self.test_name = test_name; self.start_time = time.time(); self.test_output_dir = output_dir
        self.data = {'timestamps': [], 'cpu_percent': [], 'memory_percent': [], 'memory_used_gb': [], 'memory_available_gb': [], 'time_points': []}
        if self.has_gpu: self.data['gpu_data'] = []
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop); self.monitor_thread.daemon = True; self.monitor_thread.start()
        print(f"开始监控远程服务器 - 测试名称: {test_name}" + (f" 输出目录: {output_dir}" if output_dir else ""))

    def stop_monitoring(self):
        if self.running:
            self.running = False
            if hasattr(self, 'monitor_thread'): self.monitor_thread.join(timeout=2)
            self.remote_monitor.close()
            print("资源监控已停止。")
            return self.data
        return None

    def _generate_report(self, test_output_dir): pass # 禁用报告生成

    def _monitor_loop(self):
        while self.running:
            try:
                stats = self.remote_monitor.get_all_stats()
                elapsed = time.time() - self.start_time; self.data['timestamps'].append(elapsed)
                cpu_percent = stats["cpu_percent"]; self.data['cpu_percent'].append(cpu_percent)
                mem = stats["memory"]; self.data['memory_percent'].append(mem["percent"])
                self.data['memory_used_gb'].append(mem["used_gb"]); self.data['memory_available_gb'].append(mem["free_gb"])
                if self.has_gpu:
                    gpu_stats = stats["gpu"]
                    if gpu_stats: self.data['gpu_data'].append(gpu_stats)
                now = datetime.now().strftime('%H:%M:%S.%f')[:-3]; self.data['time_points'].append(now)
                if int(elapsed * 2) % 10 == 0 and len(self.data['cpu_percent']) % 10 == 0:
                    status_msg = f"远程监控: {elapsed:.1f}秒 | CPU: {cpu_percent:.1f}% | RAM: {mem['percent']:.1f}%"
                    if self.has_gpu and gpu_stats:
                        for gpu in gpu_stats: status_msg += f" | GPU{gpu['index']}: {gpu['gpu_util']}% 内存:{gpu['memory_used']:.1f}GB"
                    print(status_msg)
                time.sleep(self.monitor_interval)
            except Exception as e: print(f"监控错误: {e}"); time.sleep(1)

class ComfyUITester:
    """用于与ComfyUI交互并处理图像的工具"""

    def __init__(self, server_address, workflow_file, output_folder, char=None):
        self.server_address = server_address
        self.api_url = server_address.rstrip('/')
        self.workflow_file = workflow_file
        if char: self.set_workflow_for_char(char)
        self.output_folder = output_folder
        self.client_id = str(uuid.uuid4())
        self.monitor = ResourceMonitor(output_folder)
        print(f"ComfyUI redrawer 初始化 (Client ID: {self.client_id[:8]}...)")
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"输出目录设置为: {os.path.abspath(self.output_folder)}")

    def set_workflow_for_char(self, char):
        """根据角色类型设置相应的工作流文件"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        char_workflows = {
            "nanzhu": "workflow/T2I-nanzhu.json",
            "nvzhu": "workflow/T2I-nvzhu.json",
            "nanpei": "workflow/T2I-nanpei.json"
        }
        new_workflow = char_workflows.get(char)
        if new_workflow:
            candidate_path = os.path.join(script_dir, new_workflow)
            if os.path.exists(candidate_path):
                self.workflow_file = candidate_path
            elif os.path.exists(new_workflow): # Check if it's an absolute path or relative to CWD
                 self.workflow_file = new_workflow
            else:
                print(f"警告: 找不到角色工作流: {new_workflow} 或 {candidate_path}。将使用当前设置的工作流。")
                return self.workflow_file # Keep current
            print(f"已设置工作流为: {self.workflow_file} (角色: {char})")
        else:
            print(f"未知角色类型 '{char}'，使用当前工作流: {self.workflow_file}")
        return self.workflow_file

    def load_workflow(self):
        """加载工作流文件"""
        try:
            # Try loading relative to script first, then absolute/CWD
            script_dir = os.path.dirname(os.path.abspath(__file__))
            potential_path = os.path.join(script_dir, self.workflow_file)

            if os.path.exists(potential_path):
                workflow_path = potential_path
            elif os.path.exists(self.workflow_file):
                 workflow_path = self.workflow_file
            else:
                print(f"错误: 工作流文件未找到: {self.workflow_file} 或 {potential_path}")
                return None

            with open(workflow_path, 'r', encoding='utf-8') as f:
                print(f"成功加载工作流: {workflow_path}")
                return json.load(f)
        except Exception as e:
            print(f"加载工作流失败: {e}")
            return None

    # --- 修改: update_workflow 方法以接受生成的提示词 ---
    def update_workflow(self, workflow, image_filename, generated_prompt: str | None):
        """
        更新工作流中的输入图像节点和目标提示词节点。

        Args:
            workflow: 要修改的工作流字典。
            image_filename: 由 upload_image 返回的服务器上的图像文件名。
            generated_prompt: 由 generate_anime_prompt 生成的提示词字符串，或 None。

        Returns:
            修改后的工作流字典，如果出错则返回 None。
        """
        if not workflow:
            return None

        # 深拷贝工作流，避免修改原始模板
        modified_workflow = json.loads(json.dumps(workflow))

        # 1. 更新图像输入节点
        if IMAGE_INPUT_NODE_ID not in modified_workflow:
            print(f"错误: 在工作流中未找到图像输入节点 ID: {IMAGE_INPUT_NODE_ID}")
            return None
        if "inputs" not in modified_workflow[IMAGE_INPUT_NODE_ID]:
            print(f"错误: 节点 {IMAGE_INPUT_NODE_ID} 缺少 'inputs' 字段")
            return None

        # 使用 upload_image 返回的文件名 (可能包含子文件夹)
        modified_workflow[IMAGE_INPUT_NODE_ID]["inputs"]["image"] = image_filename
        print(f"已更新图像输入节点 {IMAGE_INPUT_NODE_ID} 的图像为: {image_filename}")

        # 2. 更新提示词节点 (如果生成了提示词)
        if generated_prompt:
            if PROMPT_NODE_ID not in modified_workflow:
                print(f"警告: 在工作流中未找到目标提示词节点 ID: {PROMPT_NODE_ID}。无法更新提示词。")
            elif "inputs" not in modified_workflow[PROMPT_NODE_ID]:
                print(f"警告: 目标提示词节点 {PROMPT_NODE_ID} 缺少 'inputs' 字段。无法更新提示词。")
            elif "prompt" not in modified_workflow[PROMPT_NODE_ID]["inputs"]:
                 print(f"警告: 目标提示词节点 {PROMPT_NODE_ID} 的 'inputs' 中缺少 'prompt' 键。无法更新提示词。")
            else:
                modified_workflow[PROMPT_NODE_ID]["inputs"]["prompt"] = generated_prompt
                print(f"已更新提示词节点 {PROMPT_NODE_ID} 的 prompt 为生成的动漫风格提示词。")
        else:
             print(f"未提供生成的提示词，节点 {PROMPT_NODE_ID} 将使用工作流中的默认提示词。")


        # 3. 可选：更新种子以获得不同的结果
        random_seed = random.randint(0, 2**32 - 1)
        for node_id, node in modified_workflow.items():
            if "class_type" in node and "KSampler" in node["class_type"]:
                if "inputs" in node and "seed" in node["inputs"]:
                    node["inputs"]["seed"] = random_seed
                    print(f"已更新采样器节点 {node_id} 的种子为: {random_seed}")
                    break # 假设只有一个 KSampler

        return modified_workflow
    # --- 修改结束 ---

    def upload_image(self, image_path, subfolder=""):
        """Uploads an image file to the ComfyUI server's input directory."""
        if not os.path.exists(image_path):
            print(f"错误: 无法上传，图像文件不存在: {image_path}")
            return None
        filename = os.path.basename(image_path)
        upload_url = f"{self.api_url}/upload/image"
        headers = {'Accept': 'application/json'}
        try:
            with open(image_path, 'rb') as f:
                files = {'image': (filename, f)}
                data = {'overwrite': 'true'}
                if subfolder: data['subfolder'] = subfolder
                print(f"  正在上传图像: {filename} 到服务器...")
                response = requests.post(upload_url, headers=headers, files=files, data=data)
                response.raise_for_status()
                upload_data = response.json()
                server_filename = upload_data.get('name', filename) # 获取服务器上的文件名（可能包含子文件夹）
                print(f"  图像上传成功: {server_filename}")
                return server_filename # 返回服务器确认的文件名
        except requests.exceptions.RequestException as e:
            print(f"  图像上传失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try: print(f"  服务器响应: {e.response.text}")
                except Exception: pass
            return None
        except Exception as e:
            print(f"  处理上传时发生意外错误: {e}")
            return None

    def send_prompt(self, workflow):
        """提交工作流到服务器"""
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        data = json.dumps({'prompt': workflow, 'client_id': self.client_id}) # 添加 client_id
        try:
            response = requests.post(f"{self.api_url}/prompt", headers=headers, data=data)
            response.raise_for_status()
            return response.json()
        except Exception as e: print(f"提交工作流错误: {e}"); return None

    def get_history(self, prompt_id):
        """获取工作流执行结果"""
        headers = {'Accept': 'application/json'}
        try:
            response = requests.get(f"{self.api_url}/history/{prompt_id}", headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e: print(f"获取历史记录错误: {e}"); return None

    def download_output_images(self, history, prompt_id, output_dir):
        """从执行结果中提取并下载生成的图片到指定目录"""
        if not history or prompt_id not in history: print("没有找到执行结果"); return []
        save_dir = output_dir; os.makedirs(save_dir, exist_ok=True)
        outputs = history[prompt_id].get('outputs', {}); downloaded_files = []
        image_nodes = {node_id: node_output['images'] for node_id, node_output in outputs.items() if 'images' in node_output}
        if not image_nodes: print("在输出中未找到图像"); return []
        print(f"准备从服务器下载图像到: {save_dir}")
        for node_id, images in image_nodes.items():
            for image_data in images:
                filename = image_data.get('filename'); subfolder = image_data.get('subfolder'); img_type = image_data.get('type')
                if not filename: continue
                image_url = f"{self.api_url}/view?filename={filename}&subfolder={subfolder}&type={img_type}"
                local_path = os.path.join(save_dir, filename)
                try:
                    print(f"  正在下载: {filename} ...")
                    response = requests.get(image_url, stream=True); response.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
                    print(f"  已保存到: {local_path}")
                    downloaded_files.append(os.path.abspath(local_path))
                except requests.exceptions.RequestException as e: print(f"  下载图像失败 {filename}: {e}")
                except Exception as e: print(f"  保存图像时出错 {filename}: {e}")
        return downloaded_files

    # --- 修改: process_image 以调用 prompt 生成 ---
    def process_image(self, image_path):
        """处理单个图像：生成提示、加载工作流、更新、提交、监控、下载结果"""
        image_name = os.path.basename(image_path)
        print(f"------------------------------")
        print(f"开始处理图像: {image_name}")

        # 1. --- 新增: 生成动漫风格提示词 ---
        print(f"  正在为图像生成动漫风格提示词...")
        # 需要确保设置了 AIHUBMIX_API_KEY 环境变量
        anime_prompt = generate_anime_prompt(image_path)
        if anime_prompt:
            # 打印部分提示词以确认
            print(f"  成功生成提示词 (前100字符): {anime_prompt[:100]}...")
        else:
            print("  未能生成动漫风格提示词，将使用工作流中的默认提示词。")
            # anime_prompt 将为 None

        # 2. 加载基础工作流
        workflow = self.load_workflow()
        if not workflow:
            print("无法加载工作流，跳过处理")
            return []

        # 3. 上传图像
        # 注意：确保 ComfyUI 的输入目录配置正确，或者 image_path 对服务器可见
        uploaded_filename = self.upload_image(image_path)
        if not uploaded_filename:
             print("图像上传失败，无法继续处理")
             return []


        # 4. 更新工作流 (传入生成的提示词和上传后的文件名)
        modified_workflow = self.update_workflow(workflow, uploaded_filename, anime_prompt)
        if not modified_workflow:
            print("更新工作流失败，跳过处理")
            return []

        # 5. 确定输出目录 (直接使用 self.output_folder)
        current_output_dir = self.output_folder
        print(f"本次输出将保存到: {current_output_dir}")

        # 6. 开始资源监控
        monitor_test_name = f"redraw_{image_name}"
        self.monitor.start_monitoring(monitor_test_name, output_dir=current_output_dir)

        # 7. 提交工作流
        prompt_response = self.send_prompt(modified_workflow)
        if not prompt_response or 'prompt_id' not in prompt_response:
            print("提交工作流失败")
            self.monitor.stop_monitoring()
            return []

        prompt_id = prompt_response['prompt_id']
        print(f"工作流已提交, Prompt ID: {prompt_id}")

        # 8. 等待完成 (可以选择使用 WebSocket 或 API 轮询)
        # completed = self.wait_for_completion_with_progress(prompt_id) # 使用 WebSocket
        completed = self.wait_for_completion(prompt_id) # 使用 API 轮询

        # 9. 停止监控
        monitoring_data = self.monitor.stop_monitoring()

        # 10. 处理结果
        if completed:
            # 再次获取最终历史记录以确保获取到输出
            final_history = self.get_history(prompt_id)
            if final_history:
                 status = final_history.get(prompt_id, {}).get('status', {}).get('status_str', 'unknown')
                 if status == 'success': # 检查状态是否真的是成功
                     print("工作流执行成功!")
                     output_images = self.download_output_images(final_history, prompt_id, current_output_dir)
                     print(f"成功下载 {len(output_images)} 张图片到 {current_output_dir}")
                     return output_images
                 else:
                     print(f"工作流执行完成但状态为: {status}")
                     outputs = final_history.get(prompt_id, {}).get('outputs', {})
                     for node_id, node_output in outputs.items():
                         if 'errors' in node_output: print(f"  节点 {node_id} 错误: {node_output['errors']}")
                     return [] # 返回空列表表示虽然完成但未成功
            else:
                print("任务标记为完成，但无法获取最终历史记录。")
                return []
        else:
            # 超时或其他原因导致未完成
            print(f"工作流未能成功完成 (可能超时或出错)。")
            # 尝试获取历史记录看是否有错误信息
            history = self.get_history(prompt_id)
            if history and prompt_id in history:
                outputs = history[prompt_id].get('outputs', {})
                for node_id, node_output in outputs.items():
                    if 'errors' in node_output: print(f"  节点 {node_id} 错误: {node_output['errors']}")
            return [] # 未完成，返回空列表
    # --- 修改结束 ---

    # --- wait_for_completion_with_progress 和 wait_for_completion 方法保持不变 ---
    def wait_for_completion_with_progress(self, prompt_id):
        """等待工作流执行完成，使用WebSocket获取进度"""
        ws_url = self.api_url.replace('http://', 'ws://') + '/ws'
        ws_url = f"{ws_url}?clientId={self.client_id}"
        start_time = time.time(); completed = False; last_progress = None
        try:
            print(f"正在连接WebSocket: {ws_url}"); ws = websocket.create_connection(ws_url, timeout=10)
            print("开始监听进度更新...")
            while time.time() - start_time < MAX_WAIT_TIME:
                try:
                    ws.settimeout(1.0); message = ws.recv(); data = json.loads(message)
                    msg_type = data.get("type"); msg_data = data.get("data", {})
                    message_prompt_id = msg_data.get("prompt_id")
                    if message_prompt_id and message_prompt_id != prompt_id: continue
                    if msg_type == "progress" and msg_data:
                        value = msg_data.get("value", 0); max_val = msg_data.get("max", 1)
                        if max_val > 0 and last_progress != value:
                            progress_percent = value/max_val*100; progress_bar = "#" * int(progress_percent/2) + "-" * (50 - int(progress_percent/2))
                            progress = f"进度: {value}/{max_val} [{progress_bar}] {progress_percent:.1f}%"
                            print(f"\r{progress}", end="", flush=True); last_progress = value
                    elif msg_type == "status" and msg_data:
                        status_data = msg_data.get("status", {}); exec_info = status_data.get("exec_info", {})
                        queue_remaining = exec_info.get("queue_remaining", 0)
                        if queue_remaining > 0: print(f"\n队列中还有 {queue_remaining} 个任务待执行")
                    elif msg_type == "executed" and msg_data.get("prompt_id") == prompt_id:
                        print("\n任务执行完成!"); completed = True; break
                    elif msg_type == "execution_error" and msg_data.get("prompt_id") == prompt_id:
                        print(f"\n执行错误: {msg_data.get('exception_message', '未知错误')}"); break
                except websocket.WebSocketTimeoutException:
                    elapsed = time.time() - start_time
                    if int(elapsed) % 2 == 0: # 每2秒API检查一次
                        history = self.get_history(prompt_id)
                        if history and prompt_id in history and history[prompt_id].get("outputs"):
                             print("\n任务已完成! (通过API检测)"); completed = True; break
                except Exception as e: print(f"\nWebSocket错误: {e}"); break
            if not completed and time.time() - start_time >= MAX_WAIT_TIME: print(f"\n等待超时 ({MAX_WAIT_TIME}秒)")
        except Exception as e:
            print(f"连接WebSocket失败: {e}")
            print("将使用API方式检查任务状态...")
            completed = self.wait_for_completion(prompt_id)
        finally:
            try:
                if 'ws' in locals() and ws.sock and ws.sock.connected: ws.close()
            except: pass
        return completed

    def wait_for_completion(self, prompt_id):
        """等待工作流执行完成 (使用API轮询)"""
        start_time = time.time(); last_check_time = 0
        while time.time() - start_time < MAX_WAIT_TIME:
            current_time = time.time()
            if current_time - last_check_time >= 2: # 每2秒检查一次
                last_check_time = current_time
                history = self.get_history(prompt_id)
                if history and prompt_id in history:
                    # 检查是否有输出，或状态是否为完成/失败
                    outputs = history[prompt_id].get("outputs")
                    status_str = history[prompt_id].get("status", {}).get("status_str")
                    if outputs or status_str in ['success', 'failed', 'error']:
                         elapsed = time.time() - start_time
                         print(f"\n任务已完成或结束! 状态: {status_str}, 耗时: {elapsed:.2f}秒")
                         # 只有当有输出且状态为 success 时才真正算成功完成
                         return bool(outputs) and status_str == 'success'
                    else:
                         elapsed = time.time() - start_time
                         q_rem = history[prompt_id].get("status", {}).get("exec_info", {}).get("queue_remaining", "N/A")
                         print(f"\r等待任务完成，队列剩余: {q_rem}, 已耗时: {elapsed:.2f}秒", end="")
            time.sleep(0.5) # 短暂暂停
        print(f"\n等待超时 ({MAX_WAIT_TIME}秒)")
        return False # 超时返回 False

    def run_test(self, test_image=None):
        """Run a test with an optional test image"""
        print("\n--- Running ComfyUI Redraw ---")
        if test_image:
            if not os.path.exists(test_image):
                print(f"测试图片未找到: {test_image}")
                return False
            print(f"正在处理测试图片: {test_image}")
            # process_image 现在包含了加载、生成提示、更新、提交、监控、下载的完整流程
            results = self.process_image(test_image)
            if results:
                print("\n处理完成!")
                print("生成的图片:")
                for res_path in results: print(f"- {res_path}")
                return True
            else:
                print("\n处理失败 - 未生成输出或发生错误。")
                return False
        else:
            print("未提供测试图片")
            return False

# 主函数部分
if __name__ == "__main__":
    # 创建测试器实例
    tester = ComfyUITester(
        server_address=SERVER_ADDRESS,
        workflow_file=WORKFLOW_FILE,
        output_folder=OUTPUT_FOLDER
    )

    # 定义输入图片目录
    input_image_dir = "data/initial_frames"
    image_files = glob.glob(os.path.join(input_image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(input_image_dir, "*.png")) + \
                  glob.glob(os.path.join(input_image_dir, "*.jpeg")) # 添加jpeg

    if not image_files:
        print(f"在目录 {input_image_dir} 中未找到任何图片文件 (.jpg, .png, .jpeg)")
    else:
        print(f"找到 {len(image_files)} 个图片文件，将逐一处理:")
        # 对找到的每个图片文件运行处理
        successful_runs = 0
        failed_runs = 0
        for image_path in image_files:
            print(f"\n===== 开始处理图片: {os.path.basename(image_path)} =====")
            # 调用 run_test 处理单个图片 (run_test 内部调用 process_image)
            success = tester.run_test(image_path)
            if success:
                successful_runs += 1
                print(f"===== 成功处理图片: {os.path.basename(image_path)} =====")
            else:
                failed_runs += 1
                print(f"===== 处理图片失败: {os.path.basename(image_path)} =====")
            # 可选：在处理每个图像之间添加短暂的暂停
            # time.sleep(2)

        print(f"\n--- 所有处理完成 ---")
        print(f"成功处理: {successful_runs} 张图片")
        print(f"处理失败: {failed_runs} 张图片")


    # 确保监控器在退出前关闭 (如果启动了)
    if hasattr(tester, 'monitor') and tester.monitor.running:
        tester.monitor.stop_monitoring()