# --- START OF MERGED FILE comfyui_redrawer_combined_chinese.py ---

import json
import requests
import time
import threading
import os
import random
import sys
from datetime import datetime
import uuid
import websocket
import re
import glob
import base64 # 如果 prompt_reasoning 使用它，则需要导入，最好有
from prompt_reasoning import generate_anime_prompt # 假设此文件与脚本在同一目录或PYTHONPATH中

# --------------- 配置参数 ---------------
SERVER_ADDRESS = "http://36.143.229.115:8188/"  # ComfyUI 服务器地址
# 默认工作流，会被角色特定设置覆盖
DEFAULT_WORKFLOW_FILE = "workflow/FLUX-0506-nanpei.json"
OUTPUT_FOLDER = "data/tmp/opt"  # 所有结果的中央输出文件夹
MAX_WAIT_TIME = 300  # 作业最大等待时间（秒）

# 节点ID (确保这些在您的工作流中是一致的，或者如果需要处理变体)
IMAGE_INPUT_NODE_ID = "74" # 输入图像节点的ID
PROMPT_NODE_ID = "227"    # 目标提示词节点的ID
# --------------- 配置参数结束 ---------------

class ComfyUITester:
    """用于与ComfyUI交互并处理图像的工具。"""

    def __init__(self, server_address, workflow_file, output_folder, char=None):
        self.server_address = server_address
        self.api_url = server_address.rstrip('/')
        self.workflow_file = workflow_file # 初始工作流
        if char:
            self.set_workflow_for_char(char) # 如果提供了char，则使用特定于角色的工作流覆盖
        self.output_folder = output_folder
        self.client_id = str(uuid.uuid4())
        print(f"ComfyUI 重绘器已初始化 (客户端 ID: {self.client_id[:8]}...)")
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"输出目录已设置为: {os.path.abspath(self.output_folder)}")
        print(f"当前使用的工作流: {self.workflow_file}")


    def set_workflow_for_char(self, char):
        """根据角色类型设置工作流文件。"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 正确地将每个角色映射到其特定的工作流
        char_workflows = {
            "nanzhu": "workflow/FLUX-0506-nanzhu.json",
            "nvzhu":  "workflow/FLUX-0506-nvzhu.json",
            "nanpei": "workflow/FLUX-0506-nanpei.json"
        }
        new_workflow_relative_path = char_workflows.get(char)
        if new_workflow_relative_path:
            candidate_path = os.path.join(script_dir, new_workflow_relative_path)
            if os.path.exists(candidate_path):
                self.workflow_file = candidate_path
            elif os.path.exists(new_workflow_relative_path): # 检查它是否是绝对路径或相对于当前工作目录的路径
                 self.workflow_file = new_workflow_relative_path
            else:
                print(f"警告: 角色工作流未找到: {new_workflow_relative_path} 或 {candidate_path}。将使用当前设置: {self.workflow_file}")
                return self.workflow_file # 保持当前设置
            print(f"工作流已设置为: {self.workflow_file} (角色: {char})")
        else:
            print(f"未知的角色类型 '{char}'，将使用当前工作流: {self.workflow_file}")
        return self.workflow_file

    def load_workflow(self):
        """加载工作流文件。"""
        try:
            # 首先尝试相对于脚本加载，然后是绝对路径/当前工作目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # self.workflow_file 可以是绝对路径或相对于脚本的路径
            if os.path.isabs(self.workflow_file):
                 workflow_path = self.workflow_file
            else:
                 workflow_path = os.path.join(script_dir, self.workflow_file)


            if not os.path.exists(workflow_path):
                # 回退：尝试 self.workflow_file 是否相对于当前工作目录而不是脚本目录
                if os.path.exists(self.workflow_file):
                    workflow_path = self.workflow_file
                else:
                    print(f"错误: 工作流文件未找到: {self.workflow_file} (尝试了绝对路径/相对于脚本/相对于当前工作目录)")
                    return None

            with open(workflow_path, 'r', encoding='utf-8') as f:
                print(f"成功加载工作流: {workflow_path}")
                return json.load(f)
        except Exception as e:
            print(f"加载工作流 '{self.workflow_file}' 失败: {e}")
            return None

    def update_workflow(self, workflow, image_filename, generated_prompt: str | None):
        """
        使用输入图像和生成的提示词更新工作流。
        """
        if not workflow:
            return None
        modified_workflow = json.loads(json.dumps(workflow)) # 深拷贝

        # 更新图像节点
        if IMAGE_INPUT_NODE_ID not in modified_workflow:
            print(f"错误: 图像输入节点 ID '{IMAGE_INPUT_NODE_ID}' 在工作流中未找到。")
            return None
        if "inputs" not in modified_workflow[IMAGE_INPUT_NODE_ID]:
            print(f"错误: 节点 '{IMAGE_INPUT_NODE_ID}' 缺少 'inputs' 字段。")
            return None
        modified_workflow[IMAGE_INPUT_NODE_ID]["inputs"]["image"] = image_filename
        print(f"已更新图像输入节点 {IMAGE_INPUT_NODE_ID} 的图像为: {image_filename}")

        # 更新提示词节点
        if generated_prompt:
            if PROMPT_NODE_ID not in modified_workflow:
                print(f"警告: 提示词节点 ID '{PROMPT_NODE_ID}' 未找到。无法更新提示词。")
            elif "inputs" not in modified_workflow[PROMPT_NODE_ID]:
                print(f"警告: 提示词节点 '{PROMPT_NODE_ID}' 缺少 'inputs' 字段。无法更新提示词。")
            elif "text" not in modified_workflow[PROMPT_NODE_ID]["inputs"]:
                 print(f"警告: 提示词节点 '{PROMPT_NODE_ID}' 的 'inputs' 中缺少 'text' 键。无法更新提示词。")
            else:
                modified_workflow[PROMPT_NODE_ID]["inputs"]["text"] = generated_prompt
                print(f"已更新提示词节点 {PROMPT_NODE_ID} 的提示词为生成的动漫风格提示词。")
        else:
             print(f"未提供生成的提示词，节点 {PROMPT_NODE_ID} 将使用工作流中的默认提示词。")

        # 为 KSampler 随机化种子
        random_seed = random.randint(0, 2**32 - 1)
        for node_id, node_data in modified_workflow.items():
            if "class_type" in node_data and "KSampler" in node_data["class_type"]:
                if "inputs" in node_data and "seed" in node_data["inputs"]:
                    node_data["inputs"]["seed"] = random_seed
                    print(f"已更新 KSampler 节点 {node_id} 的种子为: {random_seed}")
                    # 假设只有一个主要的KSampler，在找到第一个后中断。
                    # 如果多个KSampler需要不同的种子或特定的更新，此逻辑需要更复杂。
                    break
        return modified_workflow

    def upload_image(self, image_path, subfolder=""):
        """将图像文件上传到 ComfyUI 服务器的输入目录。"""
        if not os.path.exists(image_path):
            print(f"错误: 无法上传，图像文件不存在: {image_path}")
            return None
        filename = os.path.basename(image_path)
        upload_url = f"{self.api_url}/upload/image"
        print(f"  准备上传图像: {image_path}，文件名为 {filename} 到 {upload_url}")

        try:
            with open(image_path, 'rb') as f:
                files = {'image': (filename, f, 'image/png')} # 明确指定 image/png, 如果其他类型常见则调整
                data = {'overwrite': 'true'}
                if subfolder:
                    data['subfolder'] = subfolder
                
                response = requests.post(upload_url, files=files, data=data, timeout=60)
                response.raise_for_status() # 对 4xx/5xx 错误抛出 HTTPError
                upload_data = response.json()
                server_filename = upload_data.get('name', filename)
                print(f"  图像上传成功: {server_filename} (服务器已确认)")
                return server_filename
        except requests.exceptions.HTTPError as http_err:
            print(f"  图像上传 HTTP 错误: {http_err}")
            if hasattr(http_err, 'response') and http_err.response is not None:
                try: print(f"  服务器响应内容: {http_err.response.text}")
                except Exception: pass
        except requests.exceptions.ConnectionError as conn_err:
            print(f"  图像上传连接错误: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            print(f"  图像上传超时: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"  图像上传失败 (RequestException): {req_err}")
        except Exception as e:
            print(f"  处理上传时发生意外错误: {e}")
            import traceback
            traceback.print_exc() # 打印完整的堆栈跟踪
        return None

    def send_prompt(self, workflow):
        """将工作流提交到服务器。"""
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        data = json.dumps({'prompt': workflow, 'client_id': self.client_id})
        try:
            response = requests.post(f"{self.api_url}/prompt", headers=headers, data=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"提交工作流错误: {e}")
            return None

    def get_history(self, prompt_id):
        """获取给定提示 ID 的执行历史。"""
        headers = {'Accept': 'application/json'}
        try:
            response = requests.get(f"{self.api_url}/history/{prompt_id}", headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"获取提示 {prompt_id} 的历史记录错误: {e}")
            return None

    def download_output_images(self, history, prompt_id, output_dir_for_this_run):
        """从历史记录中下载生成的图像到指定目录。"""
        if not history or prompt_id not in history:
            print("未找到用于下载的执行历史。")
            return []
        
        # 确保此运行的特定输出目录存在 (它应该是 self.output_folder)
        os.makedirs(output_dir_for_this_run, exist_ok=True)
        
        outputs = history[prompt_id].get('outputs', {})
        downloaded_files = []
        
        image_nodes_outputs = {
            node_id: node_output['images']
            for node_id, node_output in outputs.items()
            if 'images' in node_output and isinstance(node_output['images'], list)
        }

        if not image_nodes_outputs:
            print("在工作流输出中未找到图像。")
            return []

        print(f"准备从服务器下载图像到: {output_dir_for_this_run}")
        for node_id, images in image_nodes_outputs.items():
            for image_data in images:
                filename = image_data.get('filename')
                subfolder = image_data.get('subfolder', '') # 处理子文件夹为None或缺失的情况
                img_type = image_data.get('type')
                
                if not filename:
                    print(f"  跳过没有文件名的图像数据: {image_data}")
                    continue

                # 构建URL，确保在子文件夹为空时正确处理
                image_url_parts = [f"{self.api_url}/view?filename={requests.utils.quote(filename)}"]
                if subfolder:
                    image_url_parts.append(f"subfolder={requests.utils.quote(subfolder)}")
                image_url_parts.append(f"type={img_type}")
                image_url = "&".join(image_url_parts)
                
                # 如果多个节点输出同名图像，确保文件名唯一 (对于最终输出不太常见)
                # 为简单起见，我们只使用给定的文件名。如果发生冲突，请添加前缀/后缀。
                local_path = os.path.join(output_dir_for_this_run, filename)
                
                try:
                    print(f"  正在下载: {filename} (来自节点 {node_id}, URL: {image_url})")
                    response = requests.get(image_url, stream=True, timeout=60)
                    response.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"  已保存到: {local_path}")
                    downloaded_files.append(os.path.abspath(local_path))
                except requests.exceptions.RequestException as e:
                    print(f"  下载图像失败 {filename}: {e}")
                except Exception as e:
                    print(f"  保存图像时出错 {filename}: {e}")
        return downloaded_files

    def process_image(self, image_path):
        """处理单个图像：生成提示、加载工作流、更新、提交、等待、下载结果。"""
        image_name = os.path.basename(image_path)
        print(f"------------------------------")
        print(f"开始处理图像: {image_name}")

        print(f"  正在为图像生成动漫风格提示词...")
        anime_prompt = generate_anime_prompt(image_path) # 假设这个函数存在且能工作
        if anime_prompt:
            print(f"  成功生成提示词 (前300字符): {anime_prompt[:300]}...")
        else:
            print("  未能生成动漫风格提示词，将使用工作流中的默认提示词。")

        workflow = self.load_workflow()
        if not workflow:
            print("无法加载工作流，跳过处理此图像。")
            return []

        uploaded_filename = self.upload_image(image_path)
        if not uploaded_filename:
             print("图像上传失败，无法继续处理此图像。")
             return []

        modified_workflow = self.update_workflow(workflow, uploaded_filename, anime_prompt)
        if not modified_workflow:
            print("更新工作流失败，跳过处理此图像。")
            return []

        # 此特定运行的输出将进入 ComfyUITester 的 output_folder
        print(f"本次运行的输出将保存到: {self.output_folder}")

        prompt_response = self.send_prompt(modified_workflow)
        if not prompt_response or 'prompt_id' not in prompt_response:
            print("提交工作流失败。")
            return []

        prompt_id = prompt_response['prompt_id']
        print(f"工作流已提交, Prompt ID: {prompt_id}")

        # 使用 API 轮询，因为 WebSocket 可能不稳定或需要更多设置
        # completed = self.wait_for_completion_with_progress(prompt_id)
        completed = self.wait_for_completion(prompt_id) # 切换到API轮询


        if completed:
            final_history = self.get_history(prompt_id)
            if final_history:
                 status = final_history.get(prompt_id, {}).get('status', {}).get('status_str', '未知')
                 if status == 'success':
                     print("工作流执行成功!")
                     # 传递 self.output_folder 用于下载
                     output_images = self.download_output_images(final_history, prompt_id, self.output_folder)
                     print(f"成功下载 {len(output_images)} 张图片到 {self.output_folder}")
                     return output_images
                 else:
                     print(f"工作流执行完成但状态为: {status}")
                     outputs = final_history.get(prompt_id, {}).get('outputs', {})
                     for node_id, node_output in outputs.items():
                         if 'errors' in node_output: print(f"  节点 {node_id} 错误: {node_output['errors']}")
                     return []
            else:
                print("任务标记为完成，但无法获取最终历史记录。")
                return []
        else:
            print(f"工作流未能成功完成 (可能超时或出错)。")
            history = self.get_history(prompt_id) # 无论如何尝试获取历史记录以获取错误详细信息
            if history and prompt_id in history:
                outputs = history[prompt_id].get('outputs', {})
                for node_id, node_output in outputs.items():
                    if 'errors' in node_output: print(f"  节点 {node_id} 错误: {node_output['errors']}")
            return []

    def wait_for_completion_with_progress(self, prompt_id):
        """使用 WebSocket 等待工作流执行完成并获取进度更新。"""
        ws_url = self.api_url.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws'
        ws_url = f"{ws_url}?clientId={self.client_id}"
        start_time = time.time(); completed = False; last_progress_val = None
        ws = None # 初始化 ws 为 None
        try:
            print(f"正在连接 WebSocket: {ws_url}");
            ws = websocket.create_connection(ws_url, timeout=10)
            print("WebSocket 已连接。正在监听进度更新...")
            while time.time() - start_time < MAX_WAIT_TIME:
                try:
                    ws.settimeout(1.0) # 为 recv 设置超时
                    message_str = ws.recv()
                    if not message_str: continue # 跳过空消息

                    data = json.loads(message_str)
                    msg_type = data.get("type")
                    msg_data = data.get("data", {})
                    
                    # 如果存在，请确保消息是针对当前 prompt_id 的
                    message_prompt_id = msg_data.get("prompt_id")
                    if message_prompt_id and message_prompt_id != prompt_id:
                        continue

                    if msg_type == "progress" and msg_data:
                        value = msg_data.get("value", 0)
                        max_val = msg_data.get("max", 1)
                        if max_val > 0 and (last_progress_val is None or last_progress_val != value):
                            progress_percent = (value / max_val) * 100
                            progress_bar_len = 50
                            filled_len = int(progress_bar_len * value // max_val)
                            bar = "█" * filled_len + "-" * (progress_bar_len - filled_len) # 使用更像进度条的字符
                            progress_text = f"进度: {value}/{max_val} [{bar}] {progress_percent:.1f}%"
                            print(f"\r{progress_text}", end="", flush=True)
                            last_progress_val = value
                    elif msg_type == "status" and msg_data:
                        status_data = msg_data.get("status", {})
                        exec_info = status_data.get("exec_info", {})
                        queue_remaining = exec_info.get("queue_remaining", 0)
                        if queue_remaining > 0:
                            print(f"\n队列状态: 剩余 {queue_remaining} 个任务。")
                    elif msg_type == "executed" and msg_data.get("prompt_id") == prompt_id:
                        print("\n任务执行完成 (通过 WebSocket)!")
                        completed = True
                        break
                    elif msg_type == "execution_error" and msg_data.get("prompt_id") == prompt_id:
                        error_msg = msg_data.get('exception_message', '未知的执行错误')
                        print(f"\n执行错误 (通过 WebSocket): {error_msg}")
                        # 如果可用，您可能还想查看 node_errors
                        node_errors = msg_data.get('node_errors', {})
                        for node_id_err, error_details in node_errors.items(): # 避免与外部 node_id 冲突
                            print(f"  节点 {node_id_err} 错误: 类型 {error_details.get('class_type')}, 消息: {error_details.get('exception_message')}")
                        completed = False # 明确标记为未成功完成
                        break
                except websocket.WebSocketTimeoutException:
                    # 1秒内未收到消息，将API检查作为回退或继续等待
                    elapsed = time.time() - start_time
                    if int(elapsed) % 5 == 0: # 每5秒超时检查一次API
                        history = self.get_history(prompt_id)
                        if history and prompt_id in history and history[prompt_id].get("outputs"):
                             print("\n在 WebSocket 等待期间通过 API 轮询检测到任务完成。")
                             completed = True; break
                except json.JSONDecodeError:
                    print(f"\nWebSocket: 收到非 JSON 消息: {message_str}")
                except Exception as e:
                    print(f"\nWebSocket 错误: {e}。中止 WebSocket 等待。")
                    break # 在其他 WebSocket 错误时跳出 while 循环
            
            if not completed and (time.time() - start_time >= MAX_WAIT_TIME):
                print(f"\nWebSocket 等待超时 ({MAX_WAIT_TIME}秒)。")

        except ConnectionRefusedError:
            print(f"WebSocket 连接被拒绝于 {ws_url}。ComfyUI 服务器是否正在运行并启用了 WebSocket？")
        except websocket.WebSocketException as e: # 捕获通用的 WebSocket 错误，如握手失败
            print(f"WebSocket 通用异常: {e}。正在尝试 API 轮询。")
        except Exception as e: # 捕获 WebSocket 设置期间的其他潜在错误
            print(f"连接或维持 WebSocket 连接失败: {e}。将使用 API 轮询。")
        finally:
            if ws:
                try:
                    ws.close()
                    print("\nWebSocket 连接已关闭。")
                except Exception as e_close:
                    print(f"关闭 WebSocket 时出错: {e_close}")
        
        # 如果 WebSocket 失败或超时但未完成，请尝试 API 轮询作为最终检查
        if not completed:
            print("回退到 API 轮询以获取完成状态...")
            completed = self.wait_for_completion(prompt_id)
            
        return completed

    def wait_for_completion(self, prompt_id):
        """使用 API 轮询等待工作流完成。"""
        start_time = time.time()
        last_check_time = 0
        print(f"正在通过 API 轮询等待提示 {prompt_id} 的完成...")
        while time.time() - start_time < MAX_WAIT_TIME:
            current_time = time.time()
            # 每2秒轮询一次
            if current_time - last_check_time >= 2:
                last_check_time = current_time
                history = self.get_history(prompt_id)
                if history and prompt_id in history:
                    status_info = history[prompt_id].get("status", {})
                    status_str = status_info.get("status_str")
                    outputs_exist = bool(history[prompt_id].get("outputs"))

                    if outputs_exist or status_str in ['success', 'failed', 'error', 'cancelled']:
                         elapsed = time.time() - start_time
                         success_status = status_str == 'success' and outputs_exist
                         print(f"\n任务已完成或出错。状态: {status_str}, 是否找到输出: {outputs_exist}, 耗时: {elapsed:.2f}秒")
                         return success_status
                    else:
                         elapsed = time.time() - start_time
                         q_rem = status_info.get("exec_info", {}).get("queue_remaining", "N/A")
                         print(f"\r等待任务完成... 队列剩余: {q_rem}, 已耗时: {elapsed:.2f}秒", end="", flush=True)
            time.sleep(0.5) # 短暂休眠以避免忙等待

        print(f"\nAPI 轮询在 {MAX_WAIT_TIME} 秒后超时 (提示 ID: {prompt_id})。")
        return False

    def run_test(self, test_image_path=None):
        """为单个测试图像运行处理。"""
        print(f"\n--- 运行 ComfyUI 重绘: {os.path.basename(test_image_path)} ---")
        if not test_image_path or not os.path.exists(test_image_path):
            print(f"测试图片未找到或未提供: {test_image_path}")
            return False
        
        print(f"正在处理测试图片: {test_image_path}")
        results = self.process_image(test_image_path)
        
        if results:
            print("\n处理完成!")
            print("生成的图片:")
            for res_path in results:
                print(f"  - {res_path}")
            return True
        else:
            print("\n处理失败 - 未生成输出或发生错误。")
            return False

# --------------- 主要执行逻辑 ---------------
if __name__ == "__main__":
    # 为每种角色类型定义配置
    character_processing_configs = [
        {
            "char_type": "nanpei",
            "input_dir": "data/tmp/nanpei",
            "workflow": "workflow/FLUX-0506-nanpei.json" # 相对于脚本或当前工作目录
        },
        {
            "char_type": "nanzhu",
            "input_dir": "data/tmp/nanzhu",
            "workflow": "workflow/FLUX-0506-nanzhu.json" # 相对于脚本或当前工作目录
        },
        {
            "char_type": "nvzhu",
            "input_dir": "data/tmp/nvzhu",
            "workflow": "workflow/FLUX-0506-nvzhu.json"   # 相对于脚本或当前工作目录
        },
    ]

    total_successful_runs = 0
    total_failed_runs = 0
    
    # 所有运行的通用输出文件夹
    # OUTPUT_FOLDER 是全局定义的

    for config in character_processing_configs:
        char_type = config["char_type"]
        input_image_dir = config["input_dir"]
        workflow_file = config["workflow"] # 这将被 ComfyUITester 使用

        print(f"\n\n======================================================================")
        print(f"===== 开始处理角色类型: {char_type.upper()} =====")
        print(f"===== 输入目录: {input_image_dir} =====")
        print(f"===== 工作流文件: {workflow_file} =====")
        print(f"===== 输出目录: {os.path.abspath(OUTPUT_FOLDER)} =====")
        print(f"======================================================================")

        # 为此角色类型及其特定工作流实例化测试器
        # 'char' 参数将使 __init__ 调用 set_workflow_for_char
        tester = ComfyUITester(
            server_address=SERVER_ADDRESS,
            workflow_file=workflow_file, # 此工作流现在是特定的
            output_folder=OUTPUT_FOLDER,   # 所有输出到同一个文件夹
            char=char_type                 # 这确保如果逻辑复杂，则设置正确的工作流
        )

        image_extensions = ("*.jpg", "*.png", "*.jpeg")
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_image_dir, ext)))

        if not image_files:
            print(f"在目录 {input_image_dir} 中未找到任何图片文件。")
            continue

        print(f"在 {input_image_dir} 中找到 {len(image_files)} 张图片。将逐一处理...")
        
        char_successful_runs = 0
        char_failed_runs = 0

        for image_path in image_files:
            print(f"\n----- 开始处理图片: {os.path.basename(image_path)} (类型: {char_type}) -----")
            success = tester.run_test(image_path)
            if success:
                char_successful_runs += 1
                total_successful_runs += 1
                print(f"----- 成功处理图片: {os.path.basename(image_path)} (类型: {char_type}) -----")
            else:
                char_failed_runs += 1
                total_failed_runs += 1
                print(f"----- 处理图片失败: {os.path.basename(image_path)} (类型: {char_type}) -----")
            # time.sleep(2) # 图片之间的可选延迟

        print(f"\n--- 角色类型总结: {char_type.upper()} ---")
        print(f"成功处理: {char_successful_runs} 张图片")
        print(f"处理失败: {char_failed_runs} 张图片")

    print(f"\n\n======================================================================")
    print(f"===== 所有处理完成 =====")
    print(f"总共成功处理的图片 (所有类型): {total_successful_runs}")
    print(f"总共处理失败的图片 (所有类型): {total_failed_runs}")
    print(f"所有结果已保存到: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"======================================================================")

# --- END OF MERGED FILE comfyui_redrawer_combined_chinese.py ---