import json
import requests
import time
import matplotlib.pyplot as plt
import threading
import os
import random
import sys  # <--- Add this import
from datetime import datetime
import uuid
import websocket
import paramiko
import re
import glob
import json
import os
import sys

# 配置参数
SERVER_ADDRESS = "http://106.54.35.113:6889"  # ComfyUI服务器地址
WORKFLOW_FILE = "workflow/T2I-nanzhu.json"  # 默认工作流，可以被覆盖
OUTPUT_FOLDER = "data/redraw_results"  # <--- 修改: 可以更改输出文件夹名称
MONITOR_INTERVAL = 0.5  # 监控间隔(秒)
MAX_WAIT_TIME = 300  # <--- 修改: 适当增加最大等待时间

# SSH连接配置 - 保持不变或根据需要修改
SSH_CONFIG = {
    "hostname": "106.54.35.113",
    "port": 22,
    "username": "root",
    "password": "Lj666666",
    "timeout": 10
}

# <--- 删除: 不再需要 TEST_MODEL
# # 要测试的模型
# TEST_MODEL = {
#     "name": "CHEYENNE_v20.safetensors",
#     "node_id": "1615",
#     "steps_node_id": "3028"
# }

IMAGE_INPUT_NODE_ID = "74" # <--- 新增: 定义输入图像节点的ID

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
            
            # 使用密码或密钥连接
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
            # 尝试重新连接
            self._connect()
            return None
    
    def get_cpu_stats(self):
        """获取CPU使用情况"""
        result = self.execute_command("top -bn1 | grep 'Cpu(s)' | awk '{print $2 + $4}'")
        if result:
            try:
                cpu_percent = float(result.strip())
                return cpu_percent
            except:
                pass
        return 0.0
    
    def get_memory_stats(self):
        """获取内存使用情况"""
        result = self.execute_command("free -m | grep Mem")
        if result:
            try:
                parts = result.split()
                total = float(parts[1])
                used = float(parts[2])
                free = float(parts[3])
                
                # 转换为GB
                total_gb = total / 1024
                used_gb = used / 1024
                free_gb = free / 1024
                percent = (used / total) * 100
                
                return {
                    "total_gb": total_gb,
                    "used_gb": used_gb,
                    "free_gb": free_gb,
                    "percent": percent
                }
            except:
                pass
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "percent": 0}
    
    def get_gpu_stats(self):
        """获取GPU使用情况"""
        result = self.execute_command("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits")
        if not result:
            return []
        
        gpu_stats = []
        try:
            for line in result.strip().split('\n'):
                if not line.strip():
                    continue
                    
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    gpu_idx = int(parts[0])
                    gpu_name = parts[1]
                    gpu_util = float(parts[2])
                    mem_used = float(parts[3]) / 1024  # MB转GB
                    mem_total = float(parts[4]) / 1024  # MB转GB
                    temp = float(parts[5])
                    
                    gpu_stats.append({
                        'index': gpu_idx,
                        'name': gpu_name,
                        'gpu_util': gpu_util,
                        'memory_used': mem_used,
                        'memory_total': mem_total,
                        'memory_percent': (mem_used / mem_total) * 100 if mem_total > 0 else 0,
                        'temperature': temp
                    })
        except Exception as e:
            print(f"解析GPU数据失败: {e}")
        
        return gpu_stats
    
    def get_all_stats(self):
        """获取所有资源统计信息"""
        return {
            "cpu_percent": self.get_cpu_stats(),
            "memory": self.get_memory_stats(),
            "gpu": self.get_gpu_stats()
        }
    
    def close(self):
        """关闭SSH连接"""
        if self.ssh_client:
            try:
                self.ssh_client.close()
                print("已关闭SSH连接")
            except:
                pass

class ResourceMonitor:
    """远程服务器资源监控类"""
    
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.running = False
        self.data = {
            'timestamps': [],
            'cpu_percent': [],
            'memory_percent': [],
            'memory_used_gb': [],
            'memory_available_gb': [],
            'time_points': []
        }
        
        # 初始化远程监控
        self.remote_monitor = RemoteMonitor()
        
        # 检查是否有GPU
        gpu_stats = self.remote_monitor.get_gpu_stats()
        self.has_gpu = len(gpu_stats) > 0
        if self.has_gpu:
            print(f"检测到远程服务器上有 {len(gpu_stats)} 个GPU:")
            for gpu in gpu_stats:
                print(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['memory_total']:.2f} GB)")
            self.data['gpu_data'] = []
        else:
            print("远程服务器上未检测到GPU")
        
        self.test_name = None
        self.start_time = None
        self.monitor_interval = MONITOR_INTERVAL
        os.makedirs(output_folder, exist_ok=True)
    
    def start_monitoring(self, test_name, output_dir=None):
        """开始监控资源使用情况"""
        self.test_name = test_name
        self.start_time = time.time()
        self.test_output_dir = output_dir  # 保存输出目录
        self.data = {
            'timestamps': [],
            'cpu_percent': [],
            'memory_percent': [],
            'memory_used_gb': [],
            'memory_available_gb': [],
            'time_points': []
        }
        
        # 重置GPU数据
        if self.has_gpu:
            self.data['gpu_data'] = []
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"开始监控远程服务器 - 测试名称: {test_name}" + (f" 输出目录: {output_dir}" if output_dir else ""))
    
    def stop_monitoring(self):
        """停止监控并生成报告"""
        if self.running:
            self.running = False
            if hasattr(self, 'monitor_thread'):
                self.monitor_thread.join(timeout=2)

            # 关闭远程监控连接
            self.remote_monitor.close()

            # --- 修改开始: 禁用报告生成 ---
            # if len(self.data['timestamps']) > 5:
            #     # 检查 test_output_dir 是否设置，如果没有，则使用 self.output_folder
            #     test_output_dir = self.test_output_dir if self.test_output_dir else self.output_folder
            #     if not test_output_dir:
            #         print("错误：无法确定保存监控报告的目录。")
            #         return None # 或者返回空数据

            #     os.makedirs(test_output_dir, exist_ok=True) # 确保目录存在
            #     self._generate_report(test_output_dir) # 传递目录
            #     return self.data
            # else:
            #     print("监控数据不足，无法生成报告")
            # --- 修改结束 ---
            print("资源监控已停止。") # 可以保留一个简单的停止信息
            return self.data # 仍然可以返回收集到的数据，即使不保存
        return None # 如果没有运行，返回 None

    def _generate_report(self, test_output_dir): # <--- 参数 test_output_dir 现在可能不再需要，但保留以防万一
        """生成资源使用情况报告图表"""
        print(f"正在生成资源监控报告 (目标目录: {test_output_dir})...") # 打印信息，但后续不再保存

        # --- 修改开始: 禁用文件保存 ---
        # # 保存数据
        # data_file = os.path.join(test_output_dir, "monitor_data.json")
        # try:
        #     with open(data_file, 'w') as f:
        #         json.dump(self.data, f, indent=2)
        # except Exception as e:
        #      print(f"  保存 monitor_data.json 时出错: {e}")


        # # 图表数量
        # num_plots = 2  # 默认CPU和内存
        # if self.has_gpu and 'gpu_data' in self.data and self.data['gpu_data']:
        #     num_plots += 1  # 添加GPU图表

        # # 生成图表
        # try:
        #     plt.figure(figsize=(12, 5 * num_plots))
        #     # 尝试设置支持中文的字体
        #     try:
        #         plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'WenQuanYi Micro Hei'] # 添加更多备选字体
        #         plt.rcParams['axes.unicode_minus'] = False
        #     except Exception as font_e:
        #          print(f"  设置中文字体时出错 (将使用默认字体): {font_e}")


        #     # CPU图表
        #     plt.subplot(num_plots, 1, 1)
        #     plt.plot(self.data['timestamps'], self.data['cpu_percent'], label='CPU使用率')
        #     plt.xlabel('时间 (秒)')
        #     plt.ylabel('使用率 (%)')
        #     plt.title('远程服务器CPU使用情况')
        #     plt.grid(True)
        #     plt.legend()

        #     # 内存图表
        #     plt.subplot(num_plots, 1, 2)
        #     plt.plot(self.data['timestamps'], self.data['memory_percent'], label='内存使用率')
        #     plt.plot(self.data['timestamps'], self.data['memory_used_gb'], label='已用内存(GB)')
        #     plt.plot(self.data['timestamps'], self.data['memory_available_gb'], label='可用内存(GB)')
        #     plt.xlabel('时间 (秒)')
        #     plt.ylabel('使用率 (%) / 容量 (GB)')
        #     plt.title('远程服务器内存使用情况')
        #     plt.grid(True)
        #     plt.legend()

        #     # GPU图表 (如果有)
        #     if self.has_gpu and 'gpu_data' in self.data and self.data['gpu_data']:
        #         plt.subplot(num_plots, 1, 3)

        #         # 找出有多少个不同的GPU
        #         gpu_indices = set()
        #         for time_point in self.data['gpu_data']:
        #             for gpu in time_point:
        #                 gpu_indices.add(gpu['index'])

        #         # 准备每个GPU的数据
        #         gpu_utils = {idx: [] for idx in gpu_indices}
        #         gpu_mems = {idx: [] for idx in gpu_indices}

        #         # 收集每个时间点的数据
        #         for time_idx, gpu_stats in enumerate(self.data['gpu_data']):
        #             # 为每个已知的GPU索引创建一个条目
        #             for idx in gpu_indices:
        #                 # 查找当前时间点是否有这个GPU的数据
        #                 gpu_found = False
        #                 for gpu in gpu_stats:
        #                     if gpu['index'] == idx:
        #                         gpu_utils[idx].append(gpu['gpu_util'])
        #                         gpu_mems[idx].append(gpu['memory_used'])
        #                         gpu_found = True
        #                         break

        #                 # 如果没有找到，填充0
        #                 if not gpu_found:
        #                     gpu_utils[idx].append(0)
        #                     gpu_mems[idx].append(0)

        #         # 绘制GPU数据
        #         for idx in gpu_indices:
        #             # 确保数据长度匹配时间戳长度
        #             if len(self.data['timestamps']) == len(gpu_utils[idx]):
        #                 plt.plot(self.data['timestamps'], gpu_utils[idx],
        #                         label=f'GPU {idx} 使用率(%)')
        #             if len(self.data['timestamps']) == len(gpu_mems[idx]):
        #                 plt.plot(self.data['timestamps'], gpu_mems[idx],
        #                         label=f'GPU {idx} 内存(GB)')


        #         plt.xlabel('时间 (秒)')
        #         plt.ylabel('使用率 (%) / 内存 (GB)')
        #         plt.title('远程服务器GPU使用情况')
        #         plt.grid(True)
        #         plt.legend()

        #     plt.tight_layout()
        #     report_path = os.path.join(test_output_dir, "resource_usage.png")
        #     plt.savefig(report_path)
        #     plt.close()

        #     print(f"远程资源监控报告图表已生成。") # 不再提保存路径
        # except Exception as e:
        #      print(f"  生成或保存资源监控图表时出错: {e}")
        # --- 修改结束 ---
        pass # 函数体现在为空，或者只包含打印信息

    def _monitor_loop(self):
        """资源监控循环"""
        while self.running:
            try:
                # 获取远程资源统计信息
                stats = self.remote_monitor.get_all_stats()
                
                # 时间戳
                elapsed = time.time() - self.start_time
                self.data['timestamps'].append(elapsed)
                
                # CPU使用率
                cpu_percent = stats["cpu_percent"]
                self.data['cpu_percent'].append(cpu_percent)
                
                # 内存使用情况
                mem = stats["memory"]
                self.data['memory_percent'].append(mem["percent"])
                self.data['memory_used_gb'].append(mem["used_gb"])
                self.data['memory_available_gb'].append(mem["free_gb"])
                
                # GPU使用情况监控
                if self.has_gpu:
                    gpu_stats = stats["gpu"]
                    if gpu_stats:
                        self.data['gpu_data'].append(gpu_stats)
                
                # 当前时间
                now = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                self.data['time_points'].append(now)
                
                # 打印状态(间隔)
                if int(elapsed * 2) % 10 == 0 and len(self.data['cpu_percent']) % 10 == 0:
                    status_msg = f"远程监控: {elapsed:.1f}秒 | CPU: {cpu_percent:.1f}% | RAM: {mem['percent']:.1f}%"
                    
                    # 添加GPU信息
                    if self.has_gpu and gpu_stats:
                        for gpu in gpu_stats:
                            status_msg += f" | GPU{gpu['index']}: {gpu['gpu_util']}% 内存:{gpu['memory_used']:.1f}GB"
                    
                    print(status_msg)
                
                time.sleep(self.monitor_interval)
            except Exception as e:
                print(f"监控错误: {e}")
                time.sleep(1)

class ComfyUITester:
    """用于与ComfyUI交互并处理图像的工具"""

    def __init__(self, server_address, workflow_file, output_folder, char=None):
        self.server_address = server_address
        self.api_url = server_address.rstrip('/')
        
        # 初始设置工作流文件
        self.workflow_file = workflow_file
        
        # 如果提供了角色参数，则更新工作流文件
        if char:
            self.set_workflow_for_char(char)
            
        self.output_folder = output_folder
        self.client_id = str(uuid.uuid4())
        # 监控器现在使用主输出文件夹，但不会在其中创建文件
        self.monitor = ResourceMonitor(output_folder)
        print(f"ComfyUI redrawer 初始化 (Client ID: {self.client_id[:8]}...)")
        # 确保主输出目录存在
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"输出目录设置为: {os.path.abspath(self.output_folder)}") # 打印绝对路径以确认
    
    def set_workflow_for_char(self, char):
        """根据角色类型设置相应的工作流文件"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if char == "nanzhu":
            new_workflow = "workflow/T2I-nanzhu.json"
        elif char == "nvzhu":
            new_workflow = "workflow/T2I-nvzhu.json"
        elif char == "nanpei":
            new_workflow = "workflow/T2I-nanpei.json"
        else:
            # 如果不是已知类型，保持当前工作流
            print(f"未知角色类型 '{char}'，使用默认工作流")
            return self.workflow_file
        
        # 更新工作流路径（相对于脚本目录）
        self.workflow_file = os.path.join(script_dir, new_workflow)
        if not os.path.exists(self.workflow_file):
            # 如果相对路径找不到，尝试直接使用 workflow_file
            self.workflow_file = new_workflow
            if not os.path.exists(self.workflow_file):
                print(f"警告: 找不到角色工作流: {new_workflow}")
                return self.workflow_file
        
        print(f"已设置工作流为: {new_workflow} (角色: {char})")
        return self.workflow_file

    def load_workflow(self):
        """加载工作流文件"""
        try:
            # <--- 修改: 使用绝对路径或相对于脚本位置的路径加载工作流
            script_dir = os.path.dirname(os.path.abspath(__file__))
            workflow_path = os.path.join(script_dir, self.workflow_file)
            if not os.path.exists(workflow_path):
                # 如果相对路径找不到，尝试直接使用 workflow_file (可能是绝对路径)
                workflow_path = self.workflow_file
                if not os.path.exists(workflow_path):
                    print(f"错误: 工作流文件未找到: {self.workflow_file} 或 {os.path.join(script_dir, self.workflow_file)}")
                    return None

            with open(workflow_path, 'r', encoding='utf-8') as f:
                print(f"成功加载工作流: {workflow_path}")
                return json.load(f)
        except Exception as e:
            print(f"加载工作流失败: {e}")
            return None

    # <--- 重写: update_workflow 方法
    def update_workflow(self, workflow, image_path):
        """更新工作流中的输入图像节点"""
        if not workflow:
            return None

        # 深拷贝工作流，避免修改原始模板
        modified_workflow = json.loads(json.dumps(workflow))

        # 检查目标节点是否存在
        if IMAGE_INPUT_NODE_ID not in modified_workflow:
            print(f"错误: 在工作流中未找到节点 ID: {IMAGE_INPUT_NODE_ID}")
            return None

        # 检查节点结构是否符合预期
        if "inputs" not in modified_workflow[IMAGE_INPUT_NODE_ID]:
            print(f"错误: 节点 {IMAGE_INPUT_NODE_ID} 缺少 'inputs' 字段")
            return None

        # 更新图像路径
        # 注意：ComfyUI LoadImage 节点需要的是相对于 ComfyUI 输入目录的文件名，
        # 或者如果 ComfyUI 配置了允许绝对路径，则可以是绝对路径。
        # 这里我们假设 ComfyUI 可以访问这个 image_path。
        # 如果 ComfyUI 运行在远程服务器上，需要确保文件已上传到服务器的 input 目录，
        # 或者使用支持远程路径的自定义节点。
        # 为简单起见，我们先直接使用文件名。
        image_filename = os.path.basename(image_path)
        modified_workflow[IMAGE_INPUT_NODE_ID]["inputs"]["image"] = image_filename

        # 可选：更新种子以获得不同的结果
        random_seed = random.randint(0, 2**32 - 1)
        # 查找 KSampler 节点并更新种子 (如果需要)
        for node_id, node in modified_workflow.items():
            if "class_type" in node and "KSampler" in node["class_type"]:
                if "inputs" in node and "seed" in node["inputs"]:
                    node["inputs"]["seed"] = random_seed
                    print(f"已更新采样器节点 {node_id} 的种子为: {random_seed}")
                    # 假设只有一个 KSampler，找到后可以跳出
                    break

        print(f"已更新节点 {IMAGE_INPUT_NODE_ID} 的图像为: {image_filename}")
        return modified_workflow

    # --- Add the missing upload_image method here ---
    def upload_image(self, image_path, subfolder=""):
        """Uploads an image file to the ComfyUI server's input directory."""
        if not os.path.exists(image_path):
            print(f"错误: 无法上传，图像文件不存在: {image_path}")
            return None

        filename = os.path.basename(image_path)
        upload_url = f"{self.api_url}/upload/image"
        headers = {'Accept': 'application/json'} # Keep headers minimal for file upload

        try:
            with open(image_path, 'rb') as f:
                files = {'image': (filename, f)}
                data = {'overwrite': 'true'} # Overwrite if file exists
                if subfolder:
                    data['subfolder'] = subfolder

                print(f"  正在上传图像: {filename} 到服务器...")
                response = requests.post(upload_url, headers=headers, files=files, data=data)
                response.raise_for_status()
                upload_data = response.json()
                print(f"  图像上传成功: {upload_data.get('name', filename)}")
                # Return the name ComfyUI uses (might include subfolder)
                return upload_data.get('name', filename)
        except requests.exceptions.RequestException as e:
            print(f"  图像上传失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    print(f"  服务器响应: {e.response.text}")
                except Exception:
                    pass # Ignore if response text is not available
            return None
        except Exception as e:
            print(f"  处理上传时发生意外错误: {e}")
            return None
    # --- End of upload_image method ---

    def send_prompt(self, workflow):
        """提交工作流到服务器，使用代码1中的方法"""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Origin': self.api_url,
            'Referer': self.api_url + '/',
        }
        
        data = json.dumps({'prompt': workflow})
        
        try:
            response = requests.post(f"{self.api_url}/prompt", headers=headers, data=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"提交工作流错误: {e}")
            return None
    
    def get_history(self, prompt_id):
        """获取工作流执行结果，使用代码1中的方法"""
        headers = {
            'Accept': 'application/json',
            'Origin': self.api_url,
            'Referer': self.api_url + '/',
        }
        
        try:
            response = requests.get(f"{self.api_url}/history/{prompt_id}", headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"获取历史记录错误: {e}")
            return None
    
    def download_output_images(self, history, prompt_id, output_dir):
        """从执行结果中提取并下载生成的图片到指定目录"""
        if not history or prompt_id not in history:
            print("没有找到执行结果")
            return []

        # 使用传入的目录 (现在应该是 self.output_folder)
        save_dir = output_dir
        os.makedirs(save_dir, exist_ok=True) # 再次确保目录存在

        outputs = history[prompt_id].get('outputs', {})
        downloaded_files = []

        # 查找包含图像输出的节点
        image_nodes = {}
        for node_id, node_output in outputs.items():
            if 'images' in node_output:
                image_nodes[node_id] = node_output['images']

        if not image_nodes:
            print("在输出中未找到图像")
            return []

        print(f"准备从服务器下载图像到: {save_dir}")
        for node_id, images in image_nodes.items():
            for image_data in images:
                filename = image_data.get('filename')
                subfolder = image_data.get('subfolder')
                img_type = image_data.get('type') # 通常是 'output' 或 'temp'

                if not filename:
                    continue

                # 构建图像的URL
                # 注意：这里假设图像可以直接通过 /view API 访问
                image_url = f"{self.api_url}/view?filename={filename}&subfolder={subfolder}&type={img_type}"
                # 构建本地保存路径，直接保存在 output_dir 下
                local_path = os.path.join(save_dir, filename)

                try:
                    print(f"  正在下载: {filename} ...")
                    response = requests.get(image_url, stream=True)
                    response.raise_for_status()

                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"  已保存到: {local_path}")
                    downloaded_files.append(os.path.abspath(local_path)) # 返回绝对路径
                except requests.exceptions.RequestException as e:
                    print(f"  下载图像失败 {filename}: {e}")
                except Exception as e:
                    print(f"  保存图像时出错 {filename}: {e}")

        return downloaded_files

    def process_image(self, image_path):
        """处理单个图像：加载工作流、更新、提交、监控、下载结果"""
        image_name = os.path.basename(image_path)
        print(f"------------------------------")
        print(f"开始处理图像: {image_name}")

        # 1. 加载基础工作流
        workflow = self.load_workflow()
        if not workflow:
            print("无法加载工作流，跳过处理")
            return []

        # 2. 上传图像 (Ensure this call exists)
        uploaded_filename = self.upload_image(image_path)
        if not uploaded_filename:
            print("图像上传失败，无法继续处理")
            return []

        # 3. 更新工作流
        modified_workflow = self.update_workflow(workflow, uploaded_filename) # Use the result from upload
        if not modified_workflow:
            print("更新工作流失败，跳过处理")
            return []

        # --- 修改: 不再创建 session 文件夹 ---
        # 4. 确定输出目录 (直接使用 self.output_folder)
        current_output_dir = self.output_folder
        # os.makedirs(current_output_dir, exist_ok=True) # 在 __init__ 中已创建
        print(f"本次输出将保存到: {current_output_dir}")
        # --- 修改结束 ---

        # 5. 开始资源监控 (输出目录参数现在指向主目录)
        monitor_test_name = f"redraw_{image_name}"
        self.monitor.start_monitoring(monitor_test_name, output_dir=current_output_dir)

        # 6. 提交工作流
        prompt_response = self.send_prompt(modified_workflow)
        if not prompt_response or 'prompt_id' not in prompt_response:
            print("提交工作流失败")
            self.monitor.stop_monitoring()
            return []

        prompt_id = prompt_response['prompt_id']
        print(f"工作流已提交, Prompt ID: {prompt_id}")

        # 7. 轮询等待结果
        start_wait_time = time.time()
        final_history = None
        while time.time() - start_wait_time < MAX_WAIT_TIME:
            history = self.get_history(prompt_id)
            if history and prompt_id in history:
                status = history[prompt_id].get('status', {})
                current_status = status.get('status_str', 'unknown')
                exec_info = status.get('exec_info', {})
                queue_remaining = exec_info.get('queue_remaining', 0)

                progress = history[prompt_id].get('progress')
                progress_str = f", 进度: {progress}%" if progress is not None else ""
                print(f"  状态: {current_status}, 队列剩余: {queue_remaining}{progress_str} (已等待 {time.time() - start_wait_time:.1f}s)")

                if current_status in ['success', 'failed', 'error']:
                    final_history = history
                    break
                if current_status == 'pending' and queue_remaining == 0:
                    pass

            time.sleep(MONITOR_INTERVAL * 2)

        # 8. 停止监控
        monitoring_data = self.monitor.stop_monitoring() # 监控器不再保存文件

        # 9. 处理结果
        if final_history:
            status = final_history[prompt_id].get('status', {}).get('status_str', 'unknown')
            if status == 'success':
                print("工作流执行成功!")
                # --- 修改: 使用主输出目录下载 ---
                output_images = self.download_output_images(final_history, prompt_id, current_output_dir)
                print(f"成功下载 {len(output_images)} 张图片到 {current_output_dir}")
                # --- 修改结束 ---
                return output_images
            else:
                print(f"工作流执行失败或出错。状态: {status}")
                outputs = final_history[prompt_id].get('outputs', {})
                for node_id, node_output in outputs.items():
                    if 'errors' in node_output:
                        print(f"  节点 {node_id} 错误: {node_output['errors']}")
                return []
        else:
            print(f"等待结果超时 ({MAX_WAIT_TIME}秒)")
            return []

    def wait_for_completion_with_progress(self, prompt_id):
        """等待工作流执行完成，使用WebSocket获取进度"""
        # 从HTTP URL构建WebSocket URL
        ws_url = self.api_url.replace('http://', 'ws://') + '/ws'
        
        # 添加client_id参数
        ws_url = f"{ws_url}?clientId={self.client_id}"
        
        start_time = time.time()
        completed = False
        last_progress = None
        
        try:
            # 创建WebSocket连接
            print(f"正在连接WebSocket: {ws_url}")
            ws = websocket.create_connection(ws_url, timeout=10)
            
            print("开始监听进度更新...")
            
            # 等待直到完成或超时
            while time.time() - start_time < MAX_WAIT_TIME:
                try:
                    # 设置较短的接收超时
                    ws.settimeout(1.0)
                    message = ws.recv()
                    data = json.loads(message)
                    
                    msg_type = data.get("type")
                    msg_data = data.get("data", {})
                    
                    # 只处理与当前prompt_id相关的消息
                    message_prompt_id = msg_data.get("prompt_id")
                    if message_prompt_id and message_prompt_id != prompt_id:
                        continue
                    
                    # 处理进度消息
                    if msg_type == "progress" and msg_data:
                        value = msg_data.get("value", 0)
                        max_val = msg_data.get("max", 1)
                        if max_val > 0 and last_progress != value:
                            progress_percent = value/max_val*100
                            progress_bar = "#" * int(progress_percent/2) + "-" * (50 - int(progress_percent/2))
                            progress = f"进度: {value}/{max_val} [{progress_bar}] {progress_percent:.1f}%"
                            print(f"\r{progress}", end="", flush=True)
                            last_progress = value
                    
                    # 处理执行状态消息
                    elif msg_type == "status" and msg_data:
                        status_data = msg_data.get("status", {})
                        if status_data:
                            exec_info = status_data.get("exec_info", {})
                            queue_remaining = exec_info.get("queue_remaining", 0)
                            if queue_remaining > 0:
                                print(f"\n队列中还有 {queue_remaining} 个任务待执行")
                    
                    # 处理执行完成消息
                    elif msg_type == "executed" and msg_data.get("prompt_id") == prompt_id:
                        print("\n任务执行完成!")
                        completed = True
                        break
                    
                    # 处理执行错误消息
                    elif msg_type == "execution_error" and msg_data.get("prompt_id") == prompt_id:
                        print(f"\n执行错误: {msg_data.get('exception_message', '未知错误')}")
                        break
                
                except websocket.WebSocketTimeoutException:
                    # 超时继续循环，并检查是否需要使用API请求检查状态
                    elapsed = time.time() - start_time
                    # 每10秒通过API检查一次状态
                    if int(elapsed) % 2 == 0:
                        history = self.get_history(prompt_id)
                        if history and prompt_id in history:
                            outputs = history[prompt_id].get("outputs", {})
                            if outputs:
                                print("\n任务已完成! (通过API检测)")
                                completed = True
                                break
                except Exception as e:
                    print(f"\nWebSocket错误: {e}")
                    # 发生错误时退出WebSocket循环，转为API检查
                    break
            
            # 检查是否因为超时而结束
            if time.time() - start_time >= MAX_WAIT_TIME:
                print(f"\n等待超时 ({MAX_WAIT_TIME}秒)")
        
        except Exception as e:
            print(f"连接WebSocket失败: {e}")
            print("将使用API方式检查任务状态...")
            # 如果WebSocket连接失败，使用轮询API的方式
            completed = self.wait_for_completion(prompt_id)
        
        finally:
            # 确保WebSocket连接关闭
            try:
                if 'ws' in locals():
                    ws.close()
            except:
                pass
        
        return completed
    
    def wait_for_completion(self, prompt_id):
        """等待工作流执行完成 (使用API轮询，作为WebSocket的备选方案)"""
        start_time = time.time()
        last_check_time = 0
        
        while time.time() - start_time < MAX_WAIT_TIME:
            # 每2秒检查一次
            current_time = time.time()
            if current_time - last_check_time >= 2:
                last_check_time = current_time
                
                history = self.get_history(prompt_id)
                if history and prompt_id in history:
                    # 检查是否已完成
                    outputs = history[prompt_id].get("outputs", {})
                    if outputs:
                        elapsed = time.time() - start_time
                        print(f"\n任务已完成! 耗时: {elapsed:.2f}秒")
                        return True
                    else:
                        # 任务可能仍在执行中
                        elapsed = time.time() - start_time
                        print(f"\r等待任务完成，已耗时: {elapsed:.2f}秒", end="")
                
            # 短暂暂停，避免过度轮询
            time.sleep(0.5)
        
        print(f"\n等待超时 ({MAX_WAIT_TIME}秒)")
        return False
    def run_test(self, test_image=None):
        """Run a test with an optional test image"""
        print("\n--- Running ComfyUI Test ---")
        
        # Load workflow
        workflow = self.load_workflow()
        if not workflow:
            print("Failed to load workflow")
            return False
            
        if test_image:
            if not os.path.exists(test_image):
                print(f"Test image not found: {test_image}")
                return False
                
            # # Create a test session directory
            # session_id = f"test_session_{int(time.time())}"
            # test_output_dir = os.path.join(self.output_folder, session_id)
            # os.makedirs(test_output_dir, exist_ok=True)
            
            print(f"Processing test image: {test_image}")
            results = self.process_image(test_image)
            
            if results:
                print("\nTest completed successfully!")
                print("Generated images:")
                for res_path in results:
                    print(f"- {res_path}")
                return True
            else:
                print("Test failed - no output generated")
                return False
        else:
            print("No test image provided")
            return False
# 主函数部分保持不变
if __name__ == "__main__":
    # 创建测试器实例
    tester = ComfyUITester(
        server_address=SERVER_ADDRESS,
        workflow_file=WORKFLOW_FILE,
        output_folder=OUTPUT_FOLDER # 使用在脚本顶部定义的输出文件夹
    )
    
    # --- 修改开始: 处理目录中的所有图片 ---
    input_image_dir = "data/initial_frames" # 定义输入图片目录
    # 查找目录中所有的 jpg 和 png 文件 (可以根据需要添加其他格式)
    image_files = glob.glob(os.path.join(input_image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(input_image_dir, "*.png"))
    
    if not image_files:
        print(f"在目录 {input_image_dir} 中未找到任何图片文件 (.jpg, .png)")
    else:
        print(f"找到 {len(image_files)} 个图片文件，将逐一处理:")
        # 对找到的每个图片文件运行测试
        for image_path in image_files:
            print(f"\n--- 开始处理图片: {os.path.basename(image_path)} ---")
            tester.run_test(image_path) # 调用 run_test 处理单个图片
            print(f"--- 完成处理图片: {os.path.basename(image_path)} ---")
            # 可选：在处理每个图像之间添加短暂的暂停
            # time.sleep(2)
    
    # --- 修改结束 ---
    
    # 确保监控器在退出前关闭 (如果启动了)
    if hasattr(tester, 'monitor') and tester.monitor.running:
        tester.monitor.stop_monitoring()

    print("\n--- 所有处理完成 ---")