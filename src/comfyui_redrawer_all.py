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

SERVER_ADDRESS = "http://36.143.229.169:8188/"  # ComfyUI 服务器地址

# 默认工作流，会被角色特定设置覆盖
DEFAULT_WORKFLOW_FILE = "workflow/FLUX-0506-nanpei.json"
OUTPUT_FOLDER = "data/tmp/opt"  # 所有结果的中央输出文件夹
MAX_WAIT_TIME = 300  # 作业最大等待时间（秒）

# 节点ID
IMAGE_INPUT_NODE_ID = "74" # 主输入图像节点的ID
PROMPT_NODE_ID = "227"    # 目标提示词节点的ID
MASK_IMAGE_NODE_ID = "190" # 蒙版图像节点的ID (根据你的日志是 "190")
FIXED_MASK_LOCAL_PATH = "data/tmp/clipspace-mask-1120941.png" # 固定蒙版的本地路径

# --------------- 配置参数结束 ---------------

class ComfyUITester:
    """用于与ComfyUI交互并处理图像的工具。"""

    def __init__(self, server_address, workflow_file, output_folder, char=None):
        self.server_address = server_address
        self.api_url = server_address.rstrip('/')
        self.workflow_file = workflow_file
        self.output_folder = output_folder
        self.client_id = str(uuid.uuid4())
        print(f"ComfyUI 重绘器已初始化 (客户端 ID: {self.client_id[:8]}...)")
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"输出目录已设置为: {os.path.abspath(self.output_folder)}")
        # char 参数现在在 __init__ 中不直接用来设置工作流，因为工作流由外部传入
        print(f"当前使用的工作流 (由配置指定): {self.workflow_file}")
        if char: # 只是为了记录角色信息
            print(f"当前处理角色: {char}")


    def load_workflow(self):
        """加载工作流文件。"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if os.path.isabs(self.workflow_file):
                 workflow_path = self.workflow_file
            else:
                 workflow_path = os.path.join(script_dir, self.workflow_file)

            if not os.path.exists(workflow_path):
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

    def update_workflow(self, workflow, main_image_server_filename: str, generated_prompt: str | None, mask_image_server_filename: str | None = None):
        """
        使用输入图像、生成的提示词以及可选的蒙版图像更新工作流。
        """
        if not workflow:
            return None
        modified_workflow = json.loads(json.dumps(workflow)) # 深拷贝

        # 更新主图像节点
        if IMAGE_INPUT_NODE_ID not in modified_workflow:
            print(f"错误: 主图像输入节点 ID '{IMAGE_INPUT_NODE_ID}' 在工作流中未找到。")
            return None
        if "inputs" not in modified_workflow[IMAGE_INPUT_NODE_ID]:
            print(f"错误: 节点 '{IMAGE_INPUT_NODE_ID}' 缺少 'inputs' 字段。")
            return None
        modified_workflow[IMAGE_INPUT_NODE_ID]["inputs"]["image"] = main_image_server_filename
        print(f"已更新主图像输入节点 {IMAGE_INPUT_NODE_ID} 的图像为: {main_image_server_filename}")

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
                print(f"已更新提示词节点 {PROMPT_NODE_ID} 的提示词。")
        else:
             print(f"未提供生成的提示词，节点 {PROMPT_NODE_ID} 将使用工作流中的默认提示词。")

        # 更新蒙版图像节点 (如果提供了蒙版文件名且节点存在)
        if mask_image_server_filename and MASK_IMAGE_NODE_ID:
            if MASK_IMAGE_NODE_ID not in modified_workflow:
                print(f"警告: 蒙版图像节点 ID '{MASK_IMAGE_NODE_ID}' 在工作流中未找到。无法更新蒙版。")
            elif "inputs" not in modified_workflow[MASK_IMAGE_NODE_ID]:
                print(f"警告: 蒙版节点 '{MASK_IMAGE_NODE_ID}' 缺少 'inputs' 字段。无法更新蒙版。")
            elif "image" not in modified_workflow[MASK_IMAGE_NODE_ID]["inputs"]: # 假设蒙版节点也是用 "image" 键
                print(f"警告: 蒙版节点 '{MASK_IMAGE_NODE_ID}' 的 'inputs' 中缺少 'image' 键。无法更新蒙版。")
            else:
                modified_workflow[MASK_IMAGE_NODE_ID]["inputs"]["image"] = mask_image_server_filename
                print(f"已更新蒙版图像节点 {MASK_IMAGE_NODE_ID} 的图像为: {mask_image_server_filename}")
        elif MASK_IMAGE_NODE_ID and not mask_image_server_filename:
            print(f"警告: 定义了蒙版节点ID '{MASK_IMAGE_NODE_ID}' 但未提供蒙版文件名，将使用工作流中的默认蒙版（如果存在）。")


        # 为 KSampler 随机化种子
        random_seed = random.randint(0, 2**32 - 1)
        for node_id, node_data in modified_workflow.items():
            if "class_type" in node_data and "KSampler" in node_data["class_type"]:
                if "inputs" in node_data and "seed" in node_data["inputs"]:
                    node_data["inputs"]["seed"] = random_seed
                    print(f"已更新 KSampler 节点 {node_id} 的种子为: {random_seed}")
                    break
        return modified_workflow

    def _upload_single_image(self, image_path: str, subfolder: str = "", image_type_for_log: str = "图像"):
        """内部通用图像上传函数。"""
        if not os.path.exists(image_path):
            print(f"错误: 无法上传，{image_type_for_log}文件不存在: {image_path}")
            return None, None # 返回 (文件名, 子文件夹)
        
        filename = os.path.basename(image_path)
        upload_url = f"{self.api_url}/upload/image"
        log_prefix_thread = f"[{threading.current_thread().name}] " if not threading.current_thread().name.startswith("MainThread") else ""
        print(f"{log_prefix_thread}  准备上传{image_type_for_log}: {image_path}，文件名为 {filename} 到 {upload_url} (子文件夹: '{subfolder if subfolder else '根目录'}')")

        try:
            with open(image_path, 'rb') as f:
                # ComfyUI 的 'image' 字段名是固定的
                files = {'image': (filename, f, 'image/png')} # 假设蒙版也是png，如果不是需要调整
                data = {'overwrite': 'true'}
                if subfolder: # 如果指定了服务器端的子文件夹
                    data['subfolder'] = subfolder
                
                response = requests.post(upload_url, files=files, data=data, timeout=60)
                response.raise_for_status()
                upload_data = response.json()
                
                print(f"{log_prefix_thread}  {image_type_for_log}上传响应数据: {upload_data}")
                
                server_filename = upload_data.get('name')
                server_subfolder = upload_data.get('subfolder', '') # 服务器可能返回它实际保存的子文件夹

                if not server_filename:
                    print(f"错误: {image_type_for_log}上传成功，但服务器响应中未包含文件名。")
                    return None, None

                # 构造 LoadImage 节点期望的文件名格式 (subfolder/filename or filename)
                if server_subfolder:
                    final_image_reference = f"{server_subfolder}/{server_filename}"
                else:
                    final_image_reference = server_filename
                
                print(f"{log_prefix_thread}  {image_type_for_log}上传成功: {final_image_reference} (服务器已确认)")
                return final_image_reference #直接返回节点可用的引用
                
        except requests.exceptions.HTTPError as http_err:
            print(f"{log_prefix_thread}  {image_type_for_log}上传 HTTP 错误: {http_err}")
            if hasattr(http_err, 'response') and http_err.response is not None:
                try: print(f"{log_prefix_thread}  服务器响应内容: {http_err.response.text}")
                except Exception: pass
        except requests.exceptions.ConnectionError as conn_err:
            print(f"{log_prefix_thread}  {image_type_for_log}上传连接错误: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            print(f"{log_prefix_thread}  {image_type_for_log}上传超时: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"{log_prefix_thread}  {image_type_for_log}上传失败 (RequestException): {req_err}")
        except Exception as e:
            print(f"{log_prefix_thread}  处理{image_type_for_log}上传时发生意外错误: {e}")
            import traceback
            traceback.print_exc()
        return None # 发生错误时返回 None

    def upload_main_image(self, image_path: str):
        """上传主图像。返回服务器上可引用的文件名（可能包含子文件夹）。"""
        # 主图片通常直接上传到 input 根目录
        return self._upload_single_image(image_path, subfolder="", image_type_for_log="主图像")

    def upload_fixed_mask(self, mask_local_path: str):
        """
        上传固定的蒙版图像。
        蒙版通常需要上传到服务器 input 目录下的特定子文件夹 (例如 "clipspace")，
        以匹配工作流中 LoadImage 节点的期望。
        返回服务器上可引用的文件名（包含子文件夹，例如 "clipspace/mask_name.png"）。
        """
        expected_server_subfolder = "clipspace" 
        uploaded_mask_ref = self._upload_single_image(
            mask_local_path, 
            subfolder=expected_server_subfolder, 
            image_type_for_log="固定蒙版"
        )
        return uploaded_mask_ref


    def send_prompt(self, workflow):
        """将工作流提交到服务器。"""
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        payload_dict = {'prompt': workflow, 'client_id': self.client_id}
        data = json.dumps(payload_dict)
        
        log_prefix_thread = f"[{threading.current_thread().name}] " if not threading.current_thread().name.startswith("MainThread") else ""

        try:
            response = requests.post(f"{self.api_url}/prompt", headers=headers, data=data, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"{log_prefix_thread}提交工作流 HTTP 错误: {http_err}")
            if hasattr(http_err, 'response') and http_err.response is not None:
                print(f"{log_prefix_thread}  服务器响应状态码: {http_err.response.status_code}")
                # print(f"{log_prefix_thread}  服务器响应头部: {http_err.response.headers}")
                try:
                    error_details = http_err.response.json()
                    print(f"{log_prefix_thread}  服务器响应内容 (JSON): {json.dumps(error_details, indent=2, ensure_ascii=False)}")
                except json.JSONDecodeError:
                    print(f"{log_prefix_thread}  服务器响应内容 (Text): {http_err.response.text}")
            return None
        except Exception as e:
            print(f"{log_prefix_thread}提交工作流错误: {e}")
            return None

    def get_history(self, prompt_id):
        """获取给定提示 ID 的执行历史。"""
        headers = {'Accept': 'application/json'}
        log_prefix_thread = f"[{threading.current_thread().name}] " if not threading.current_thread().name.startswith("MainThread") else ""
        try:
            response = requests.get(f"{self.api_url}/history/{prompt_id}", headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"{log_prefix_thread}获取提示 {prompt_id} 的历史记录错误: {e}")
            return None

    def download_output_images(self, history, prompt_id, output_dir_for_this_run):
        """从历史记录中下载生成的图像到指定目录，并添加时间戳到文件名。"""
        if not history or prompt_id not in history:
            print("未找到用于下载的执行历史。")
            return []
        
        os.makedirs(output_dir_for_this_run, exist_ok=True)
        
        outputs = history[prompt_id].get('outputs', {})
        downloaded_files = []
        
        image_nodes_outputs = {
            node_id: node_output['images']
            for node_id, node_output in outputs.items()
            if 'images' in node_output and isinstance(node_output['images'], list)
        }

        if not image_nodes_outputs:
            print(f"在工作流输出中 (Prompt ID: {prompt_id}) 未找到图像。")
            return []
        
        log_prefix_thread = f"[{threading.current_thread().name}] " if not threading.current_thread().name.startswith("MainThread") else ""
        print(f"{log_prefix_thread}准备从服务器下载图像到: {output_dir_for_this_run} (Prompt ID: {prompt_id})")

        # 在 ComfyUITester.download_output_images 方法中
    # ... (方法开始部分的代码) ...
        for node_id, images in image_nodes_outputs.items():
            for image_data in images:
                print(f"{log_prefix_thread}  收到的图像数据 (来自节点 {node_id}): {image_data}") # 保持这个打印

                filename = image_data.get('filename')
                subfolder = image_data.get('subfolder', '')  # 默认为空字符串
                img_type = image_data.get('type')

                if not filename: # 文件名是必需的
                    print(f"{log_prefix_thread}  跳过没有文件名的图像数据: {image_data}")
                    continue

                # 构建参数字典
                url_params = {'filename': filename}
                
                # subfolder 参数：ComfyUI /view API 通常需要它，即使为空
                url_params['subfolder'] = subfolder if subfolder is not None else "" 

                url_params['type'] = 'output'

                name_part, ext_part = os.path.splitext(filename)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                new_filename_with_timestamp = f"{name_part}_{timestamp}{ext_part}"
                local_path = os.path.join(output_dir_for_this_run, new_filename_with_timestamp)
                
                try:
                    # 在发送请求前打印最终的参数
                    print(f"{log_prefix_thread}  准备下载。请求URL: {self.api_url}/view, 请求参数: {url_params}")
                    
                    response = requests.get(f"{self.api_url}/view", params=url_params, stream=True, timeout=60)
                    
                    response.raise_for_status() # 如果是404或其他错误会在这里抛出异常
                    
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"{log_prefix_thread}  已保存到: {local_path}")
                    downloaded_files.append(os.path.abspath(local_path))
                except requests.exceptions.HTTPError as http_err: # HTTPError 现在会被 raise_for_status() 捕获
                    print(f"{log_prefix_thread}  下载图像 HTTP 错误 ({http_err.response.status_code}) for {filename}: {http_err}")
                    print(f"{log_prefix_thread}    请求的URL: {http_err.request.url}") # 打印导致错误的请求URL
                    # print(f"{log_prefix_thread}    服务器响应: {http_err.response.text}") # 可选，如果404响应体有内容
                except requests.exceptions.RequestException as e:
                    print(f"{log_prefix_thread}  下载图像失败 {filename}: {e}")
                except Exception as e:
                    print(f"{log_prefix_thread}  保存图像时出错 {filename}: {e}")
        
        if not downloaded_files and image_nodes_outputs: # 如果有输出但一个都没下载成功
            print(f"{log_prefix_thread}警告：在历史输出中找到了图像数据，但未能成功下载任何图像。请检查工作流输出节点和下载逻辑。")

        return downloaded_files

    def process_image(self, image_path, fixed_mask_local_path): # 添加蒙版路径参数
        """处理单个图像：上传固定蒙版、生成提示、加载工作流、更新、提交、等待、下载结果。"""
        image_name = os.path.basename(image_path)
        
        log_prefix_thread = f"[{threading.current_thread().name}] " if not threading.current_thread().name.startswith("MainThread") else ""

        print(f"{log_prefix_thread}------------------------------")
        print(f"{log_prefix_thread}开始处理主图像: {image_name} (使用工作流: {os.path.basename(self.workflow_file)})")

        # 1. 上传固定的蒙版图像
        uploaded_mask_ref = None
        if fixed_mask_local_path and MASK_IMAGE_NODE_ID: # 只有在需要时才上传
            if not os.path.exists(fixed_mask_local_path):
                print(f"{log_prefix_thread}错误: 指定的固定蒙版路径不存在: {fixed_mask_local_path}。节点 {MASK_IMAGE_NODE_ID} 将不会被更新。")
            else:
                uploaded_mask_ref = self.upload_fixed_mask(fixed_mask_local_path)
                if not uploaded_mask_ref:
                    print(f"{log_prefix_thread}固定蒙版 '{fixed_mask_local_path}' 上传失败。节点 {MASK_IMAGE_NODE_ID} 可能使用默认值或导致错误。")
        elif MASK_IMAGE_NODE_ID:
            print(f"{log_prefix_thread}未提供固定蒙版本地路径，但定义了蒙版节点ID '{MASK_IMAGE_NODE_ID}'。它将使用工作流中的默认值。")


        # 2. 为主图像生成提示
        print(f"{log_prefix_thread}  正在为图像 '{image_name}' 生成动漫风格提示词...")
        anime_prompt = None
        try:
            anime_prompt = generate_anime_prompt(image_path)
            if anime_prompt:
                print(f"{log_prefix_thread}  成功为 '{image_name}' 生成提示词 (前50字符): {anime_prompt[:50]}...")
            else:
                print(f"{log_prefix_thread}  未能为 '{image_name}' 生成动漫风格提示词或提示词为空，将使用工作流中的默认提示词。")
        except Exception as e:
            print(f"{log_prefix_thread}  生成动漫风格提示词时发生错误 ({image_name}): {e}。将使用默认提示词。")


        # 3. 加载工作流
        workflow = self.load_workflow()
        if not workflow:
            print(f"{log_prefix_thread}无法加载工作流 '{self.workflow_file}'，跳过处理图像 '{image_name}'。")
            return []

        # 4. 上传主图像
        uploaded_main_image_ref = self.upload_main_image(image_path)
        if not uploaded_main_image_ref:
             print(f"{log_prefix_thread}主图像 '{image_name}' 上传失败，无法继续处理。")
             return []
        print(f"{log_prefix_thread}  主图像 '{image_name}' 已作为 '{uploaded_main_image_ref}' 上传/引用")

        # 5. 更新工作流 (传入主图引用和蒙版图引用)
        modified_workflow = self.update_workflow(workflow, uploaded_main_image_ref, anime_prompt, uploaded_mask_ref)
        if not modified_workflow:
            print(f"{log_prefix_thread}更新工作流失败 (图像: '{image_name}'), 跳过处理。")
            return []
        
        # 此特定运行的输出将进入 ComfyUITester 的 output_folder
        print(f"{log_prefix_thread}本次运行的输出将保存到: {self.output_folder}")

        prompt_response = self.send_prompt(modified_workflow)
        if not prompt_response or 'prompt_id' not in prompt_response:
            print(f"{log_prefix_thread}提交工作流失败 (图像: '{image_name}')。")
            return []

        prompt_id = prompt_response['prompt_id']
        print(f"{log_prefix_thread}工作流已为 '{image_name}' 提交, Prompt ID: {prompt_id}")

        # completed = self.wait_for_completion(prompt_id) # 旧的调用方式
        completed, final_history_from_wait = self.wait_for_completion(prompt_id) # <--- 新的调用方式

        if completed and final_history_from_wait: # 确保 completed 为 True 且确实得到了历史记录
            # final_history = self.get_history(prompt_id) # <--- 不再需要重新获取
            status_obj = final_history_from_wait.get(prompt_id, {}) # 直接使用返回的历史
            status_str = status_obj.get('status', {}).get('status_str', '未知')
            
            # 这里的判断条件可以简化，因为 wait_for_completion 返回 True 时已确认 success 和 outputs
            # 但为了保险，再检查一次
            if status_str == 'success' and status_obj.get('outputs'):
                print(f"{log_prefix_thread}工作流执行成功 (Prompt ID: {prompt_id}, 图像: {image_name})!")
                # 使用 final_history_from_wait 进行下载
                output_images = self.download_output_images(final_history_from_wait, prompt_id, self.output_folder)
                if output_images: # 检查是否真的下载到了图片
                    print(f"{log_prefix_thread}成功为 '{image_name}' (Prompt ID: {prompt_id}) 下载 {len(output_images)} 张图片到 {self.output_folder}")
                    return output_images # 返回下载的图片路径列表
                else:
                    print(f"{log_prefix_thread}工作流成功但未能下载任何图片 (Prompt ID: {prompt_id}, 图像: {image_name})。检查下载逻辑或工作流输出节点。")
                    return [] # 下载失败也算整体失败
            else: # 理论上如果 completed=True，不应该走到这里，但作为防御
                print(f"{log_prefix_thread}工作流执行完成但最终状态检查为 '{status_str}' 或无输出 (Prompt ID: {prompt_id}, 图像: {image_name})")
                outputs = status_obj.get('outputs', {})
                for node_id_err, node_output in outputs.items():
                    if 'errors' in node_output: print(f"{log_prefix_thread}  节点 {node_id_err} 错误: {node_output['errors']}")
                return []
        elif completed and not final_history_from_wait: # completed 为 True 但没有历史返回 (不应发生)
            print(f"{log_prefix_thread}任务标记为完成，但 wait_for_completion 未能返回历史记录 (Prompt ID: {prompt_id}, 图像: {image_name})。")
            return []
        else: # completed 为 False
            print(f"{log_prefix_thread}工作流未能成功完成 (Prompt ID: {prompt_id}, 图像: {image_name}, 可能超时或出错)。")
            # 如果 final_history_from_wait 包含失败的历史记录，可以尝试从中提取错误
            if final_history_from_wait and prompt_id in final_history_from_wait:
                status_obj = final_history_from_wait.get(prompt_id, {})
                print(f"{log_prefix_thread}  获取到的历史状态: {status_obj.get('status', {}).get('status_str', '未知')}")
                outputs = status_obj.get('outputs', {})
                for node_id_err, node_output in outputs.items():
                    if 'errors' in node_output: print(f"{log_prefix_thread}  节点 {node_id_err} 错误: {node_output['errors']}")
            else: # 如果连失败的历史记录都没有
                history_fallback = self.get_history(prompt_id) # 最后尝试获取一次
                if history_fallback and prompt_id in history_fallback:
                    status_obj = history_fallback.get(prompt_id, {})
                    print(f"{log_prefix_thread}  (回退)获取到的历史状态: {status_obj.get('status', {}).get('status_str', '未知')}")

            return []

    # 在 ComfyUITester 类中
    def wait_for_completion(self, prompt_id, is_fallback=False):
        """
        使用 API 轮询等待工作流完成。
        返回: (bool: 是否成功, dict: 包含输出的成功历史记录或None)
        """
        current_thread_name = threading.current_thread().name
        log_prefix = f"[{current_thread_name}, P_ID:{prompt_id[:6]}] " if not current_thread_name.startswith("MainThread") else f"[P_ID:{prompt_id[:6]}] "
        
        start_time = time.time()
        last_log_time = 0 

        if not is_fallback:
            print(f"{log_prefix}正在通过 API 轮询等待提示 {prompt_id} 的完成...")

        while time.time() - start_time < MAX_WAIT_TIME:
            time.sleep(0.5) 
            current_time = time.time()
            
            if current_time - last_log_time >= 2.0: # 控制API调用频率
                history_response = self.get_history(prompt_id) # 注意: get_history 内部有自己的打印

                if history_response and prompt_id in history_response:
                    status_obj = history_response[prompt_id]
                    status_info = status_obj.get("status", {})
                    status_str = status_info.get("status_str") if status_info else "未知(status字段缺失)"
                    outputs_exist = bool(status_obj.get("outputs"))

                    if outputs_exist and status_str == 'success': # 严格判断成功且有输出
                        elapsed = time.time() - start_time
                        print(f"{log_prefix}任务已通过 API 轮询确认成功并找到输出。状态: {status_str}, 耗时: {elapsed:.2f}秒")
                        return True, history_response # <--- 返回 True 和捕获到的历史记录
                    elif status_str in ['failed', 'error', 'cancelled'] or (outputs_exist and status_str != 'success'): # 明确失败或有输出但状态不对
                        elapsed = time.time() - start_time
                        print(f"{log_prefix}任务已通过 API 轮询确认结束但状态非完全成功。状态: {status_str}, 是否找到输出: {outputs_exist}, 耗时: {elapsed:.2f}秒")
                        return False, history_response # <--- 返回 False 和捕获到的历史记录 (可能包含错误信息)
                    else: # 仍在运行或排队，或状态未知但无输出
                        elapsed = time.time() - start_time
                        q_rem = status_info.get("exec_info", {}).get("queue_remaining", "N/A") if status_info else "N/A"
                        # 只在必要时打印等待日志
                        if current_time - last_log_time >= 10.0 or last_log_time == 0 :
                            print(f"{log_prefix}API 轮询等待中... 状态: {status_str}, 队列剩余: {q_rem}, 已耗时: {elapsed:.1f}s / {MAX_WAIT_TIME}s")
                            last_log_time = current_time # 更新日志时间戳，避免频繁打印
                else: 
                    # get_history 返回 None 或 prompt_id 不在 history_response 中
                    elapsed = time.time() - start_time
                    if current_time - last_log_time >= 10.0 or last_log_time == 0:
                        print(f"{log_prefix}API 轮询: 无法获取到 Prompt ID {prompt_id} 的有效历史记录。已耗时: {elapsed:.1f}s")
                        last_log_time = current_time # 更新日志时间戳

            if not (current_time - last_log_time >= 10.0 or last_log_time == 0): # 如果没打印日志
                if current_time - last_log_time >= 2.0 : # 但API已调用
                    last_log_time = current_time # 更新，使得下次等待10s打印

        print(f"{log_prefix}API 轮询在 {MAX_WAIT_TIME} 秒后超时 (提示 ID: {prompt_id})。")
        return False, None # <--- 超时返回 False 和 None

    def run_test(self, test_image_path=None, fixed_mask_path=None): # 添加蒙版路径参数
        """为单个测试图像运行处理。"""
        log_prefix_thread = f"[{threading.current_thread().name}] " if not threading.current_thread().name.startswith("MainThread") else ""
        print(f"{log_prefix_thread}\n--- 运行 ComfyUI 重绘: {os.path.basename(test_image_path)} ---")

        if not test_image_path or not os.path.exists(test_image_path):
            print(f"{log_prefix_thread}测试图片未找到或未提供: {test_image_path}")
            return False
        
        print(f"{log_prefix_thread}正在处理测试图片: {test_image_path}")
        if fixed_mask_path and os.path.exists(fixed_mask_path):
            print(f"{log_prefix_thread}将使用固定蒙版: {fixed_mask_path}")
        elif fixed_mask_path:
            print(f"{log_prefix_thread}警告: 指定的固定蒙版路径不存在: {fixed_mask_path}")
        
        results = self.process_image(test_image_path, fixed_mask_path) # 传递蒙版路径
        
        if results:
            print(f"{log_prefix_thread}\n处理完成!")
            print(f"{log_prefix_thread}生成的图片:")
            for res_path in results:
                print(f"{log_prefix_thread}  - {res_path}")
            return True
        else:
            print(f"{log_prefix_thread}\n处理失败 - 未生成输出或发生错误。")
            return False

# --------------- 辅助函数：用于线程 (如果恢复并发) ---------------
def run_test_threaded(tester_instance, image_path_for_thread, run_num_for_thread, total_runs_for_thread, results_list_for_thread, char_type_for_log, fixed_mask_local_path_for_thread):
    thread_name = threading.current_thread().name 
    base_image_name = os.path.basename(image_path_for_thread)
    print(f"线程 {thread_name}: 开始处理原图 '{base_image_name}' (类型: {char_type_for_log}) - 第 {run_num_for_thread}/{total_runs_for_thread} 个并发任务")
    
    success = False # 默认为失败
    try:
        # 确保 run_test 返回的是布尔值
        success = tester_instance.run_test(image_path_for_thread, fixed_mask_local_path_for_thread)
        if not isinstance(success, bool): # 防御性编程
            print(f"线程 {thread_name}: 警告 - tester_instance.run_test 返回了非布尔值: {success} (类型: {type(success)})。将视为失败。")
            success = False
    except Exception as e:
        print(f"线程 {thread_name}: 在执行 tester_instance.run_test 时发生严重错误: {e}")
        import traceback
        traceback.print_exc() # 打印完整的堆栈跟踪
        success = False # 发生异常也算失败
    finally:
        # 无论成功与否（除非线程被外部杀死），都尝试记录结果
        results_list_for_thread.append(success) 
    
    if success:
        print(f"线程 {thread_name}: 原图 '{base_image_name}' (类型: {char_type_for_log}) - 第 {run_num_for_thread}/{total_runs_for_thread} 个并发任务处理成功")
    else:
        print(f"线程 {thread_name}: 原图 '{base_image_name}' (类型: {char_type_for_log}) - 第 {run_num_for_thread}/{total_runs_for_thread} 个并发任务处理失败")


# --------------- 主要执行逻辑 ---------------
if __name__ == "__main__":
    # 为每种角色类型定义配置
    character_processing_configs = [
        {
            "char_type": "nanzhu",
            "input_dir": "data/tmp/nanzhu",
            "workflow": "workflow/FLUX-0506-nanzhu.json"
        },
        {
            "char_type": "nvzhu",
            "input_dir": "data/tmp/nvzhu",
            "workflow": "workflow/FLUX-0506-nvzhu.json"
        },
        {
            "char_type": "nanpei",
            "input_dir": "data/tmp/nanpei",
            "workflow": "workflow/FLUX-0506-nanpei.json"
        },
    ]

    total_successful_task_runs = 0 # 总的成功并发任务数
    total_failed_task_runs = 0     # 总的失败并发任务数

    # --- 每个原图要并发运行的任务数 ---
    CONCURRENT_TASKS_PER_IMAGE = 1 # 设置为5以进行并发处理

    DELAY_BETWEEN_IMAGES = 2 # 例如暂停5秒

    # 确保 FIXED_MASK_LOCAL_PATH 定义的蒙版文件存在
    if not os.path.exists(FIXED_MASK_LOCAL_PATH):
        print(f"严重错误: 全局固定蒙版文件 '{FIXED_MASK_LOCAL_PATH}' 未找到。程序将退出。")
        sys.exit(1) # 需要 import sys
    else:
        print(f"将为每个任务使用固定的本地蒙版文件: {FIXED_MASK_LOCAL_PATH}")

    # 开始处理不同角色类型
    for config in character_processing_configs:
        char_type = config["char_type"]
        input_image_dir = config["input_dir"]
        workflow_file_for_char = config["workflow"]

        print(f"\n\n======================================================================")
        print(f"===== 开始处理角色类型: {char_type.upper()} =====")
        print(f"===== 输入目录: {input_image_dir} =====")
        print(f"===== 工作流文件: {workflow_file_for_char} =====")
        print(f"===== 输出目录: {os.path.abspath(OUTPUT_FOLDER)} =====")
        print(f"===== 每张原图将启动: {CONCURRENT_TASKS_PER_IMAGE} 个{'并发' if CONCURRENT_TASKS_PER_IMAGE > 1 else ''}生成任务 =====") # 根据并发数调整日志
        print(f"======================================================================")

        # 查找当前角色类型的图片文件
        image_extensions = ("*.jpg", "*.png", "*.jpeg", "*.webp")
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_image_dir, ext)))
        image_files.sort() # 确保处理顺序一致

        if not image_files:
            print(f"在目录 {input_image_dir} 中未找到任何图片文件 ({', '.join(image_extensions)})。跳过此角色类型。")
            continue

        print(f"在 {input_image_dir} 中找到 {len(image_files)} 张图片...")

        char_successful_tasks = 0 # 此角色类型成功的任务数
        char_failed_tasks = 0     # 此角色类型失败的任务数

        # 遍历当前角色类型的每张图片
        for image_path in image_files:
            image_base_name = os.path.basename(image_path)
            print(f"\n===== 开始{'并发' if CONCURRENT_TASKS_PER_IMAGE > 1 else ''}处理原图: {image_base_name} (类型: {char_type}) ... =====")

            # !!! 为每张图片创建一个新的 ComfyUITester 实例 !!!
            # (这是你之前的策略，有助于隔离状态，但可能开销稍大)
            print(f"为图片 '{image_base_name}' 创建新的 ComfyUITester 实例...")
            current_tester_instance = ComfyUITester(
                server_address=SERVER_ADDRESS,
                workflow_file=workflow_file_for_char,
                output_folder=OUTPUT_FOLDER,
                char=char_type
            )
            
            threads = [] # 存储当前图片对应的线程对象
            thread_results = [] # 存储当前图片所有并发任务的结果 (True/False)

            # 启动并发任务
            for run_number in range(1, CONCURRENT_TASKS_PER_IMAGE + 1):
                if CONCURRENT_TASKS_PER_IMAGE > 1:
                    thread_name = f"任务-{char_type}-{os.path.splitext(image_base_name)[0]}-并发-{run_number}"
                    thread = threading.Thread(
                        target=run_test_threaded,
                        # 确保这里的参数与 run_test_threaded 函数定义完全一致
                        args=(current_tester_instance, image_path, run_number, CONCURRENT_TASKS_PER_IMAGE, thread_results, char_type, FIXED_MASK_LOCAL_PATH),
                        name=thread_name
                    )
                    # --- 关键修正：添加并启动线程 ---
                    threads.append(thread)
                    thread.start()
                    # --- 修正结束 ---
                    # time.sleep(0.1) # 可选的轻微启动延迟
                else: # 单任务执行逻辑 (CONCURRENT_TASKS_PER_IMAGE = 1)
                    print(f"\n----- 原图 '{image_base_name}' - 第 {run_number}/{CONCURRENT_TASKS_PER_IMAGE} 轮处理开始 -----")
                    success = current_tester_instance.run_test(image_path, FIXED_MASK_LOCAL_PATH)
                    thread_results.append(success) # 记录结果
                    if success:
                        print(f"----- 原图 '{image_base_name}' - 第 {run_number}/{CONCURRENT_TASKS_PER_IMAGE} 轮处理成功 -----")
                    else:
                        print(f"----- 原图 '{image_base_name}' - 第 {run_number}/{CONCURRENT_TASKS_PER_IMAGE} 轮处理失败 -----")

            # 等待当前图片的所有并发线程完成
            if CONCURRENT_TASKS_PER_IMAGE > 1:
                print(f"主线程：已为图片 '{image_base_name}' 启动 {len(threads)} 个工作线程，现在开始等待它们完成...")
                for i, thread in enumerate(threads):
                    print(f"主线程：正在等待线程 {thread.name}...")
                    # 使用 join 等待线程结束，设置超时
                    thread.join(timeout=MAX_WAIT_TIME + 60) # 超时比任务最大等待时间稍长
                    if thread.is_alive():
                        # 如果线程超时后仍在运行，打印警告
                        print(f"警告: 线程 {thread.name} 在 {MAX_WAIT_TIME + 60} 秒超时后仍未结束。")
                    else:
                        print(f"主线程：线程 {thread.name} 已结束。")
                print(f"主线程：图片 '{image_base_name}' 的所有线程已处理完毕（或超时）。")

            # 统计当前图片的处理结果
            # 添加日志以查看 thread_results 的内容
            print(f"主线程：开始统计图片 '{image_base_name}' 的结果，thread_results 列表内容: {thread_results}")
            successful_tasks_for_this_image = sum(1 for res in thread_results if res is True)

            failed_tasks_for_this_image = len(thread_results) - successful_tasks_for_this_image

            # 更新总计数器
            char_successful_tasks += successful_tasks_for_this_image
            char_failed_tasks += failed_tasks_for_this_image
            total_successful_task_runs += successful_tasks_for_this_image
            total_failed_task_runs += failed_tasks_for_this_image

            print(f"===== 原图 '{image_base_name}' 的 {CONCURRENT_TASKS_PER_IMAGE} 个任务尝试处理完成: 成功 {successful_tasks_for_this_image} 个, 失败 {failed_tasks_for_this_image} 个 (基于记录的结果数: {len(thread_results)}) =====")

            # --- 在处理下一张主图片前加入延时 ---
            if DELAY_BETWEEN_IMAGES > 0:
                 print(f"\n### 处理完图片 '{image_base_name}'。暂停 {DELAY_BETWEEN_IMAGES} 秒后处理下一张... ###\n")
                 time.sleep(DELAY_BETWEEN_IMAGES)
            # --- 延时结束 ---

        # 打印当前角色类型的总结
        print(f"\n--- 角色类型总结: {char_type.upper()} ---")
        print(f"总共成功的并发任务数: {char_successful_tasks}")
        print(f"总共失败的并发任务数: {char_failed_tasks}")
        total_tasks_attempted_char = char_successful_tasks + char_failed_tasks
        if total_tasks_attempted_char > 0:
             success_rate_char = (char_successful_tasks / total_tasks_attempted_char) * 100
             print(f"成功率: {success_rate_char:.2f}%")
        else:
             print("未执行任何任务。")


    # 所有角色类型处理完毕，打印最终总结
    print(f"\n\n======================================================================")
    print(f"===== 所有处理完成 =====")
    print(f"总共成功的并发任务数 (所有类型): {total_successful_task_runs}")
    print(f"总共失败的并发任务数 (所有类型): {total_failed_task_runs}")
    total_tasks_attempted_all = total_successful_task_runs + total_failed_task_runs
    if total_tasks_attempted_all > 0:
        success_rate_all = (total_successful_task_runs / total_tasks_attempted_all) * 100
        print(f"总体成功率: {success_rate_all:.2f}%")
    else:
        print("未执行任何任务。")
    print(f"所有结果已保存到: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"======================================================================")