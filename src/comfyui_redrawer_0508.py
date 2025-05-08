# --- START OF FILE comfyui_redrawer_0508_optimized_final_cn_logfix.py ---
import json
import requests
import time
import os
import random
import sys
from datetime import datetime # 用于时间戳
import uuid
import glob
import re # 用于从场景文件夹名称中提取数字
import base64 # 如果 prompt_reasoning 使用它，则需要导入
from tqdm import tqdm, trange # 导入tqdm用于进度条和 trange (如果需要迭代数字)
from concurrent.futures import ThreadPoolExecutor, as_completed # 用于并行处理

# 确保此文件可用且线程安全 (如果它有全局状态)
# 目前，我们假设 generate_anime_prompt 可以从多个线程安全调用
from prompt_reasoning import generate_anime_prompt as original_generate_anime_prompt

# --------------- 配置参数 ---------------
SERVER_IPS = [
    "http://36.143.229.172:8188",
    "http://36.143.229.173:8188",
    "http://36.143.229.116:8188",
    "http://36.143.229.117:8188"
]
NUM_WORKERS = len(SERVER_IPS) # 并发任务数，与IP数量相同

# --- 迭代控制 ---
NUM_ITERATIONS = 8 # 所有图片迭代次数

# --- 路径定义 ---
BASE_INPUT_DIR = "data/250508"
MASKS_BASE_DIR = "data/mask"
SUBTITLE_MASK_FILENAME = "subtitle-mask.png"
WORKFLOW_BASE_DIR = "workflow"
OUTPUT_FOLDER = "data/250508/opt"

# --- 工作流特定配置 ---
WORKFLOW_SKIP_SCENE_MASK = "FLUX-0508-02.json" # 此工作流不需要为节点190上传场景蒙版

# --- 服务器和工作流节点 ID ---
MAX_WAIT_TIME = 360 # 等待ComfyUI任务完成的最大秒数
MASKS_SERVER_SUBFOLDER = "clipspace" #蒙版上传到服务器的子文件夹

IMAGE_INPUT_NODE_ID = "74"    # 主图像输入节点ID
PROMPT_NODE_ID = "227"   # 提示词输入节点ID
SCENE_MASK_NODE_ID = "190" # 场景蒙版节点ID
SUBTITLE_MASK_NODE_ID = "229" # 字幕蒙版节点ID

DELAY_BETWEEN_SUBMISSIONS = 0.05 # 提交任务到线程池之间的微小延迟（秒）

# --- 日志详细程度 ---
VERBOSE_LOGGING = False # 设置为 True 以启用更详细的内部日志（例如API轮询）

# --------------- 配置参数结束 ---------------

# 包装原始的 generate_anime_prompt 以便控制其日志输出
def generate_anime_prompt_with_custom_log(image_path: str, log_func_info, log_func_error, log_func_verbose) -> str | None:
    """
    包装器函数，调用 prompt_reasoning.py 中的 generate_anime_prompt，
    并通过传入的日志函数来记录过程和结果。

    Args:
        image_path: 输入图像的文件路径。
        log_func_info: 用于记录普通信息的日志函数 (例如 self._log_info)。
        log_func_error: 用于记录错误的日志函数 (例如 self._log_error)。
        log_func_verbose: 用于记录详细信息的日志函数 (例如 self._log_verbose)。

    Returns:
        生成的动漫风格提示词字符串，如果出错则返回 None。
    """
    base_image_name = os.path.basename(image_path)
    log_func_info(f"    [提示生成] 开始为图像 '{base_image_name}' 生成提示词...")

    # 调用原始的、位于 prompt_reasoning.py 中的函数
    # 原始函数内部有自己的 print 语句用于错误报告和状态。
    # 理想情况下，原始函数应该返回错误信息或抛出异常，而不是只打印。
    # 但基于现有代码，我们先直接调用它。
    try:
        # 确保 AIHUBMIX_API_KEY 在 prompt_reasoning.py 能够访问的环境中被设置
        # 如果原始函数因为 API KEY 问题或其他配置问题返回 None，这里也会是 None
        prompt = original_generate_anime_prompt(image_path)

        if prompt:
            log_func_info(f"    [提示生成] 成功为 '{base_image_name}' 获取到提示词。")
            # 具体的提示词内容打印由调用方 (process_image) 处理，以控制长度和格式
            return prompt.strip() # 确保去除多余空白
        else:
            # 如果 prompt 为 None，说明 original_generate_anime_prompt 内部可能遇到了问题
            # 并且它应该已经在其自己的 print 语句中输出了错误原因。
            log_func_error(f"    [提示生成] 未能为 '{base_image_name}' 生成有效提示词 (原始函数返回 None)。请检查控制台是否有来自 'prompt_reasoning.py' 的更详细错误信息。")
            return None
            
    except Exception as e:
        # 捕获调用 original_generate_anime_prompt 时可能发生的任何未被其内部处理的异常
        log_func_error(f"    [提示生成] 调用 'original_generate_anime_prompt' 时发生意外错误 (图像: {base_image_name}): {e}")
        return None


class ComfyUITester:
    """用于与ComfyUI交互并处理图像的工具类。"""

    def __init__(self, server_address, workflow_file_path, output_folder, context_info="", verbose=VERBOSE_LOGGING):
        self.server_address = server_address.rstrip('/') 
        self.api_url = self.server_address 
        self.workflow_file_path = workflow_file_path
        self.output_folder = output_folder
        self.client_id = str(uuid.uuid4())
        self.context_info = context_info # 用于日志，包含服务器、任务等信息
        self.verbose = verbose
        os.makedirs(self.output_folder, exist_ok=True)

    def _log_verbose(self, message):
        """记录详细日志（仅当VERBOSE_LOGGING为True时）。使用tqdm.write安全打印。"""
        if self.verbose:
            tqdm.write(f"VERBOSE [{self.context_info} - 客户端 {self.client_id[:6]}] {message}")

    def _log_info(self, message): 
        """记录普通信息日志。使用tqdm.write安全打印。"""
        tqdm.write(f"INFO [{self.context_info} - 客户端 {self.client_id[:6]}] {message}")

    def _log_error(self, message):
        """记录错误日志。使用tqdm.write安全打印。"""
        tqdm.write(f"错误 [{self.context_info} - 客户端 {self.client_id[:6]}] {message}")


    def load_workflow(self):
        """加载工作流JSON文件。"""
        try:
            if not os.path.exists(self.workflow_file_path):
                self._log_error(f"工作流文件未找到: {self.workflow_file_path}")
                return None
            with open(self.workflow_file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
                self._log_verbose(f"  成功加载工作流: {self.workflow_file_path}")
                return workflow_data
        except Exception as e:
            self._log_error(f"加载工作流 '{self.workflow_file_path}' 失败: {e}")
            return None

    def _upload_single_image(self, image_path: str, subfolder: str = "", image_type_for_log: str = "图像"):
        """内部通用图像上传函数。"""
        if not os.path.exists(image_path):
            self._log_error(f"无法上传，{image_type_for_log}文件不存在: {image_path}")
            return None
        
        filename = os.path.basename(image_path)
        upload_url = f"{self.api_url}/upload/image"
        self._log_verbose(f"    准备上传 {image_type_for_log}: {filename} 到服务器子文件夹 '{subfolder if subfolder else '根目录'}'")

        try:
            with open(image_path, 'rb') as f:
                files = {'image': (filename, f, 'image/png')} # 假设都是PNG，可根据需要调整
                data = {'overwrite': 'true'}
                if subfolder:
                    data['subfolder'] = subfolder
                
                response = requests.post(upload_url, files=files, data=data, timeout=120) 
                response.raise_for_status() # 如果状态码是4xx或5xx，则引发HTTPError
                upload_data = response.json()
                
                server_filename = upload_data.get('name')
                server_subfolder = upload_data.get('subfolder', '')

                if not server_filename:
                    self._log_error(f"{image_type_for_log}上传成功，但服务器响应中未包含文件名。")
                    return None

                final_image_reference = f"{server_subfolder}/{server_filename}" if server_subfolder else server_filename
                self._log_verbose(f"    {image_type_for_log}上传成功: {final_image_reference}")
                return final_image_reference
                
        except requests.exceptions.HTTPError as http_err:
            self._log_error(f"{image_type_for_log}上传时发生 HTTP 错误: {http_err}")
            if hasattr(http_err, 'response') and http_err.response is not None:
                try: self._log_verbose(f"      服务器响应内容: {http_err.response.text}")
                except Exception: pass
        except requests.exceptions.RequestException as req_err: # 更通用的网络错误，如DNS解析失败、连接超时等
            self._log_error(f"处理{image_type_for_log}上传时发生网络请求错误: {req_err}")
        except Exception as e:
            self._log_error(f"处理{image_type_for_log}上传时发生意外错误: {e}")
        return None

    def upload_main_image(self, image_path: str):
        """上传主图像。"""
        return self._upload_single_image(image_path, subfolder="", image_type_for_log="主图像")

    def upload_specific_mask(self, mask_local_path: str, server_target_subfolder: str, mask_type_log: str):
        """上传特定类型的蒙版图像。"""
        if not mask_local_path or not os.path.exists(mask_local_path):
            self._log_verbose(f"警告: {mask_type_log} 路径未提供或文件不存在: {mask_local_path}。跳过上传。")
            return None
        return self._upload_single_image(mask_local_path, subfolder=server_target_subfolder, image_type_for_log=mask_type_log)

    def update_workflow(self, workflow, main_image_ref: str, generated_prompt: str | None, 
                        scene_mask_ref: str | None, subtitle_mask_ref: str | None):
        """根据提供的图像引用和提示词更新工作流JSON。"""
        if not workflow: return None
        modified_workflow = json.loads(json.dumps(workflow)) # 深拷贝以避免修改原始加载的工作流

        if IMAGE_INPUT_NODE_ID in modified_workflow:
            modified_workflow[IMAGE_INPUT_NODE_ID]["inputs"]["image"] = main_image_ref
            self._log_verbose(f"    已更新主图像节点 {IMAGE_INPUT_NODE_ID} 为: {main_image_ref}")
        else:
            self._log_error(f"主图像节点 ID '{IMAGE_INPUT_NODE_ID}' 在工作流中未找到。")
            return None # 关键节点缺失，无法继续

        if generated_prompt:
            if PROMPT_NODE_ID in modified_workflow:
                modified_workflow[PROMPT_NODE_ID]["inputs"]["text"] = generated_prompt
                self._log_verbose(f"    已使用生成的提示词更新提示词节点 {PROMPT_NODE_ID}。")
            else:
                self._log_verbose(f"警告: 提示词节点 ID '{PROMPT_NODE_ID}' 未找到。无法更新提示词。")
        else:
            self._log_verbose(f"    未提供生成的提示词。提示词节点 {PROMPT_NODE_ID} 将使用工作流中的默认提示词。")

        if scene_mask_ref and SCENE_MASK_NODE_ID: 
            if SCENE_MASK_NODE_ID in modified_workflow:
                modified_workflow[SCENE_MASK_NODE_ID]["inputs"]["image"] = scene_mask_ref
                self._log_verbose(f"    已更新场景蒙版节点 {SCENE_MASK_NODE_ID} 为: {scene_mask_ref}")
            else:
                self._log_verbose(f"警告: 场景蒙版节点 ID '{SCENE_MASK_NODE_ID}' 在工作流中未找到，但提供了场景蒙版引用 {scene_mask_ref}。")
        elif SCENE_MASK_NODE_ID and not scene_mask_ref : 
             self._log_verbose(f"    未向节点 {SCENE_MASK_NODE_ID} 提供场景蒙版引用 (或被有意跳过)。节点将使用其默认设置（如有）。")


        if subtitle_mask_ref and SUBTITLE_MASK_NODE_ID:
            if SUBTITLE_MASK_NODE_ID in modified_workflow:
                modified_workflow[SUBTITLE_MASK_NODE_ID]["inputs"]["image"] = subtitle_mask_ref
                self._log_verbose(f"    已更新字幕蒙版节点 {SUBTITLE_MASK_NODE_ID} 为: {subtitle_mask_ref}")
            else:
                self._log_verbose(f"警告: 字幕蒙版节点 ID '{SUBTITLE_MASK_NODE_ID}' 未找到。无法更新字幕蒙版。")
        elif SUBTITLE_MASK_NODE_ID and subtitle_mask_ref is None: 
            self._log_verbose(f"    未提供字幕蒙版引用给节点 {SUBTITLE_MASK_NODE_ID}。")
            
        random_seed = random.randint(0, 2**32 - 1)
        sampler_updated = False
        for node_id, node_data in modified_workflow.items():
            if "class_type" in node_data and "KSampler" in node_data["class_type"]: 
                if "inputs" in node_data and "seed" in node_data["inputs"]:
                    node_data["inputs"]["seed"] = random_seed
                    self._log_verbose(f"    已更新 KSampler 节点 {node_id} 的种子为: {random_seed}")
                    sampler_updated = True
        if not sampler_updated:
            self._log_verbose(f"警告: 工作流中未找到任何 KSampler 节点来更新种子。")
        return modified_workflow

    def send_prompt(self, workflow):
        """向ComfyUI服务器发送已准备好的工作流（prompt）。"""
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        payload_dict = {'prompt': workflow, 'client_id': self.client_id}
        data = json.dumps(payload_dict)
        try:
            response = requests.post(f"{self.api_url}/prompt", headers=headers, data=data, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            self._log_error(f"提交工作流时发生 HTTP 错误: {http_err}")
            if hasattr(http_err, 'response') and http_err.response is not None:
                self._log_verbose(f"      服务器响应 (状态码 {http_err.response.status_code}): {http_err.response.text}")
            return None
        except requests.exceptions.RequestException as req_err:
            self._log_error(f"提交工作流时发生网络请求错误: {req_err}")
            return None
        except Exception as e:
            self._log_error(f"提交工作流时发生错误: {e}")
            return None

    def get_history(self, prompt_id):
        """获取指定prompt_id的执行历史。"""
        try:
            response = requests.get(f"{self.api_url}/history/{prompt_id}", headers={'Accept': 'application/json'}, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e: 
            self._log_verbose(f"获取提示 {prompt_id} 的历史记录时出错 (网络/HTTP): {e}")
            return None
        except Exception as e:
            self._log_verbose(f"获取提示 {prompt_id} 的历史记录时发生意外错误: {e}")
            return None


    def download_output_images(self, history, prompt_id, output_dir_for_run, original_image_basename, current_iteration_num):
        """从服务器下载执行结果中的图像。"""
        if not history or prompt_id not in history:
            self._log_verbose("  未找到用于下载的执行历史。")
            return []
        
        os.makedirs(output_dir_for_run, exist_ok=True)
        
        outputs = history[prompt_id].get('outputs', {})
        downloaded_files_paths = []
        images_to_download = []
        for node_id, node_output in outputs.items():
            if 'images' in node_output and isinstance(node_output['images'], list):
                for img_data in node_output['images']:
                    if img_data.get('type') == 'output':
                        images_to_download.append(img_data)

        if not images_to_download:
            self._log_verbose(f"  在工作流输出中 (提示ID: {prompt_id}) 未找到 'output' 类型的图像。")
            return []
        
        self._log_verbose(f"    准备从服务器下载 {len(images_to_download)} 张 'output' 类型的图像 (提示ID: {prompt_id})")

        original_base, original_ext = os.path.splitext(original_image_basename)
        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3] 

        for idx, image_data in enumerate(images_to_download):
            server_filename = image_data.get('filename')
            subfolder = image_data.get('subfolder', '')

            if not server_filename:
                self._log_verbose(f"    跳过没有文件名的图像数据: {image_data}")
                continue

            name_part_to_use = original_base
            if len(images_to_download) > 1: 
                name_part_to_use = f"{original_base}_output_{idx}" 
            
            if current_iteration_num > 1: 
                final_local_filename = f"{name_part_to_use}-iter{current_iteration_num}_{timestamp_str}{original_ext}"
            else: 
                final_local_filename = f"{name_part_to_use}_{timestamp_str}{original_ext}"
            
            local_path = os.path.join(output_dir_for_run, final_local_filename)
            url_params = {'filename': server_filename, 'subfolder': subfolder} 
            
            try:
                self._log_verbose(f"      正在下载: 服务器文件 '{subfolder}/{server_filename}' 另存为 '{final_local_filename}'")
                response = requests.get(f"{self.api_url}/view", params=url_params, stream=True, timeout=180) 
                response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                self._log_verbose(f"      已保存到: {local_path}")
                downloaded_files_paths.append(os.path.abspath(local_path))
            except requests.exceptions.HTTPError as http_err:
                self._log_error(f"下载图像 HTTP 错误 ({http_err.response.status_code}) for {server_filename}: {http_err}")
                self._log_verbose(f"        请求的URL: {http_err.request.url}")
            except requests.exceptions.RequestException as req_err:
                self._log_error(f"下载图像时发生网络请求错误 {server_filename}: {req_err}")
            except Exception as e:
                self._log_error(f"下载或保存图像时出错 {server_filename}: {e}")
        
        if not downloaded_files_paths and images_to_download:
            self._log_verbose(f"  警告：找到图像数据但未能下载任何图像 ({original_image_basename})。")
        return downloaded_files_paths

    def wait_for_completion(self, prompt_id, progress_bar_instance=None):
        """等待ComfyUI任务完成，并定期更新进度条（如果提供）。"""
        log_prefix_short = f"提示ID:{prompt_id[:6]}" 
        start_time_wait = time.time() 
        last_log_time = 0 
        self._log_verbose(f"{log_prefix_short} 正在等待任务在服务器 {self.server_address} 上完成...")

        while time.time() - start_time_wait < MAX_WAIT_TIME:
            time.sleep(1.0) 
            current_time_loop = time.time()
            history_response = self.get_history(prompt_id)

            if history_response and prompt_id in history_response:
                status_obj = history_response[prompt_id]
                status_info = status_obj.get("status", {})
                status_str = status_info.get("status_str", "未知状态")
                outputs_exist = bool(status_obj.get("outputs"))

                if progress_bar_instance: 
                    current_elapsed_time_wait = time.time() - start_time_wait
                    q_rem = status_info.get("exec_info", {}).get("queue_remaining", "N/A")
                    
                    # 提取服务器IP的最后一部分作为简称
                    server_short_name = self.server_address.split('.')[-1].split(':')[0]
                    # 构造一个简洁的后缀
                    postfix_ctx_parts = self.context_info.split(" ")
                    # e.g. "Srv172 It1 Sc1 Sh01 Img23.00_00_0" -> "Srv172 It1 Sc1 Sh01"
                    if len(postfix_ctx_parts) > 4:
                         short_context = " ".join(postfix_ctx_parts[:4]) # 服务器 迭代 场景 镜头
                    else:
                         short_context = self.context_info # Fallback

                    postfix_message = f"{short_context} {log_prefix_short} {status_str}, 队列:{q_rem}, 耗时:{current_elapsed_time_wait:.0f}s"
                    progress_bar_instance.set_postfix_str(postfix_message, refresh=False)

                if outputs_exist and status_str == 'success':
                    elapsed_wait = time.time() - start_time_wait
                    self._log_verbose(f"{log_prefix_short} 任务成功。状态: {status_str}, 耗时: {elapsed_wait:.2f}秒")
                    if progress_bar_instance: progress_bar_instance.set_postfix_str(f"{short_context} {log_prefix_short} 成功! ({elapsed_wait:.1f}s)", refresh=True)
                    return True, history_response, elapsed_wait 
                elif status_str in ['failed', 'error', 'cancelled'] or \
                     (outputs_exist and status_str not in ['success', 'running', 'pending']): 
                    elapsed_wait = time.time() - start_time_wait
                    self._log_error(f"{log_prefix_short} 任务失败或出错。状态: {status_str}, 耗时: {elapsed_wait:.2f}秒")
                    if progress_bar_instance: progress_bar_instance.set_postfix_str(f"{short_context} {log_prefix_short} 失败/错误! ({status_str})", refresh=True)
                    if status_obj.get("outputs"): 
                        for node_id_err, node_output in status_obj["outputs"].items():
                            if 'errors' in node_output:
                                self._log_error(f"  {log_prefix_short} 节点 {node_id_err} 错误: {node_output['errors']}")
                    return False, history_response, elapsed_wait 
                else: 
                    if current_time_loop - last_log_time >= 10.0 and self.verbose: 
                        elapsed_wait = time.time() - start_time_wait
                        q_rem = status_info.get("exec_info", {}).get("queue_remaining", "N/A")
                        self._log_verbose(f"{log_prefix_short} API 轮询等待中... 状态: {status_str}, 队列剩余: {q_rem}, 已耗时: {elapsed_wait:.1f}s / {MAX_WAIT_TIME}s")
                        last_log_time = current_time_loop
                        if progress_bar_instance: progress_bar_instance.refresh() 
            else: 
                if current_time_loop - last_log_time >= 10.0 and self.verbose: 
                    elapsed_wait = time.time() - start_time_wait
                    self._log_verbose(f"{log_prefix_short} API 轮询: 无法获取到提示 ID {prompt_id} 的有效历史记录。已耗时: {elapsed_wait:.1f}s")
                    last_log_time = current_time_loop
                    if progress_bar_instance: progress_bar_instance.refresh()
        
        elapsed_wait_timeout = time.time() - start_time_wait
        self._log_error(f"{log_prefix_short} 超时 ({MAX_WAIT_TIME}秒) 等待提示 ID: {prompt_id}。")
        if progress_bar_instance: progress_bar_instance.set_postfix_str(f"{self.context_info.split(' ')[0]} {log_prefix_short} 超时!", refresh=True)
        return False, None, elapsed_wait_timeout 

    def process_image(self, main_image_path, scene_mask_local_path, subtitle_mask_local_path,
                      original_image_basename, current_iteration_num, progress_bar_instance=None):
        """处理单个图像的完整流程：上传、更新工作流、发送、等待、下载。"""
        task_start_time = time.time() 
        self._log_info(f"开始处理: {original_image_basename} (迭代: {current_iteration_num}) 使用工作流 {os.path.basename(self.workflow_file_path)}")
        self._log_verbose(f"    主图像: {main_image_path}")
        
        workflow = self.load_workflow()
        if not workflow:
            self._log_error(f"无法加载工作流 {self.workflow_file_path} ({original_image_basename})。跳过。")
            return [], 0 
        
        uploaded_main_image_ref = self.upload_main_image(main_image_path)
        if not uploaded_main_image_ref:
             self._log_error(f"主图像 '{original_image_basename}' 上传失败。跳过。")
             return [], 0 
        
        uploaded_scene_mask_ref = None
        current_workflow_basename = os.path.basename(self.workflow_file_path)

        if current_workflow_basename == WORKFLOW_SKIP_SCENE_MASK:
            self._log_verbose(f"    工作流 {current_workflow_basename} 指定跳过场景蒙版 (节点 {SCENE_MASK_NODE_ID})。")
        elif SCENE_MASK_NODE_ID: 
            if scene_mask_local_path: 
                self._log_verbose(f"    工作流 {current_workflow_basename} 将使用场景蒙版。本地路径: {scene_mask_local_path}")
                uploaded_scene_mask_ref = self.upload_specific_mask(scene_mask_local_path, MASKS_SERVER_SUBFOLDER, "场景蒙版")
                if not uploaded_scene_mask_ref: 
                    self._log_verbose(f"警告: 场景蒙版 '{scene_mask_local_path}' (节点 {SCENE_MASK_NODE_ID}) 上传失败。")
            else: 
                self._log_verbose(f"    工作流 {current_workflow_basename} 可使用场景蒙版 (节点 {SCENE_MASK_NODE_ID})，但此次未提供本地场景蒙版路径。")
        else: 
            self._log_verbose(f"    场景蒙版节点 ID 未配置，不尝试上传场景蒙版。")
        
        uploaded_subtitle_mask_ref = None
        if subtitle_mask_local_path and SUBTITLE_MASK_NODE_ID:
            self._log_verbose(f"    字幕蒙版: {subtitle_mask_local_path if subtitle_mask_local_path else '无'}")
            uploaded_subtitle_mask_ref = self.upload_specific_mask(subtitle_mask_local_path, MASKS_SERVER_SUBFOLDER, "字幕蒙版")
            if not uploaded_subtitle_mask_ref:
                 self._log_verbose(f"警告: 字幕蒙版 '{subtitle_mask_local_path}' 上传失败。")
        elif SUBTITLE_MASK_NODE_ID: 
            self._log_verbose(f"    字幕蒙版节点ID ({SUBTITLE_MASK_NODE_ID}) 已配置但未提供本地字幕蒙版路径。")

        anime_prompt = None
        # 使用包装后的函数，并将 self._log_verbose (或 self._log_info) 作为日志回调
        # generate_anime_prompt_with_custom_log 内部已经有日志输出了
        try:
            anime_prompt = generate_anime_prompt_with_custom_log(main_image_path, self._log_verbose, self._log_error, self._log_verbose) # 或者 self._log_info 如果希望提示词生成过程总是可见

            # 在这里添加对 anime_prompt 结果的日志输出
            if anime_prompt:
                # 为了美观，可以控制打印长度
                max_prompt_log_length = 150 # 定义希望在日志中显示的最大提示词长度
                logged_prompt = anime_prompt
                if len(anime_prompt) > max_prompt_log_length:
                    logged_prompt = anime_prompt[:max_prompt_log_length] + "..."
                
                # 使用 _log_info 或 _log_verbose 来打印生成的提示词
                # 如果提示词内容非常重要，总是希望看到，用 _log_info
                # 如果只是调试时或详细模式下想看，用 _log_verbose
                self._log_info(f"    生成的提示词 (图像: {original_image_basename}): \"{logged_prompt}\"")
            else:
                # generate_anime_prompt_with_custom_log 内部应该已经记录了生成失败或为空的情况
                # 但这里可以再加一个简短的确认（如果需要）
                self._log_info(f"    未获取到有效提示词 (图像: {original_image_basename})，将使用工作流默认值。")

        except Exception as e:
            self._log_error(f"    调用提示生成函数时发生严重错误 ({original_image_basename}): {e}。将使用默认提示词。")
            anime_prompt = None # 确保出错时 anime_prompt 为 None

        # 下一行就是紧接着的 update_workflow 调用
        modified_workflow = self.update_workflow(workflow, uploaded_main_image_ref, anime_prompt,
                                                uploaded_scene_mask_ref, 
                                                uploaded_subtitle_mask_ref)
        if not modified_workflow:
            self._log_error(f"更新工作流失败 ({original_image_basename})。跳过。")
            return [], 0
        
        prompt_response = self.send_prompt(modified_workflow)
        if not prompt_response or 'prompt_id' not in prompt_response:
            self._log_error(f"提交工作流失败 ({original_image_basename})。跳过。")
            return [], 0
        prompt_id = prompt_response['prompt_id']
        self._log_info(f"任务已提交: {original_image_basename}, 提示ID: {prompt_id[:8]}")

        completed, final_history, time_spent_waiting = self.wait_for_completion(prompt_id, progress_bar_instance)

        if completed and final_history:
            self._log_verbose(f"    工作流执行成功 (提示ID: {prompt_id})。等待耗时: {time_spent_waiting:.2f}秒。")
            output_images = self.download_output_images(final_history, prompt_id, self.output_folder,
                                                        original_image_basename, current_iteration_num)
            task_duration = time.time() - task_start_time 
            if output_images:
                self._log_info(f"成功处理并下载 {len(output_images)} 张图片 for {original_image_basename}。任务耗时: {task_duration:.2f}秒。")
                return output_images, task_duration 
            else:
                self._log_error(f"工作流成功但未能下载任何图片 (提示ID: {prompt_id}, {original_image_basename})。任务耗时: {task_duration:.2f}秒。")
                return [], task_duration 
        else:
            task_duration = time.time() - task_start_time
            self._log_error(f"工作流失败或超时 (提示ID: {prompt_id}, {original_image_basename})。任务耗时: {task_duration:.2f}秒 (其中等待 {time_spent_waiting:.2f}秒)。")
            if final_history and prompt_id in final_history: 
                 status_obj = final_history[prompt_id]
                 outputs = status_obj.get('outputs', {})
                 for node_id_err, node_output in outputs.items():
                    if 'errors' in node_output:
                        self._log_error(f"      节点 {node_id_err} 报告错误: {node_output['errors']}")
            return [], task_duration

# --------------- 主要执行逻辑 ---------------
if __name__ == "__main__":
    overall_start_time = time.time() 
    total_images_to_process_overall = 0
    tasks_to_run = [] 

    # 预扫描时的日志直接用 print，因为 tqdm 实例还没创建
    print("程序开始：扫描文件并计算总任务数...")
    for iter_num_calc in range(1, NUM_ITERATIONS + 1):
        scene_folders_calc = sorted([
            d for d in os.listdir(BASE_INPUT_DIR) 
            if os.path.isdir(os.path.join(BASE_INPUT_DIR, d)) and d.startswith("场景")
        ])
        for scene_folder_name_calc in scene_folders_calc:
            scene_full_path_calc = os.path.join(BASE_INPUT_DIR, scene_folder_name_calc)
            scene_num_match_calc = re.search(r'\d+', scene_folder_name_calc)
            scene_num_str_calc = scene_num_match_calc.group(0) if scene_num_match_calc else "未知场景ID"

            shot_folders_calc = sorted([
                d for d in os.listdir(scene_full_path_calc)
                if os.path.isdir(os.path.join(scene_full_path_calc, d)) and d.isdigit() 
            ])
            for shot_folder_name_calc in shot_folders_calc:
                shot_images_dir_calc = os.path.join(scene_full_path_calc, shot_folder_name_calc)
                current_workflow_filename_calc = f"FLUX-0508-{shot_folder_name_calc}.json"
                current_workflow_path_calc = os.path.join(WORKFLOW_BASE_DIR, current_workflow_filename_calc)
                if not os.path.exists(current_workflow_path_calc):
                    print(f"警告 [预扫描]: 工作流文件 {current_workflow_path_calc} 未找到，将跳过场景 {scene_folder_name_calc} 中的镜头 {shot_folder_name_calc} 的所有图像。")
                    continue 

                image_files_calc = sorted([
                    f for f in os.listdir(shot_images_dir_calc) 
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                ])
                if not image_files_calc:
                    if VERBOSE_LOGGING: print(f"信息 [预扫描]: 在 {shot_images_dir_calc} 中未找到图像文件。")
                    continue

                for image_filename_calc in image_files_calc:
                    total_images_to_process_overall += 1
                    tasks_to_run.append((iter_num_calc, scene_folder_name_calc, shot_folder_name_calc, image_filename_calc, scene_num_str_calc))
    
    if total_images_to_process_overall == 0:
        print("未找到任何需要处理的图像任务。请检查输入目录、文件结构和工作流文件。程序将退出。")
        sys.exit(0)

    print(f"总共需要处理 {total_images_to_process_overall} 个图像任务，将使用 {NUM_WORKERS} 个并发工作线程。")

    total_images_processed_successfully = 0
    total_images_failed = 0
    individual_task_durations = [] 

    global_subtitle_mask_path = os.path.join(MASKS_BASE_DIR, SUBTITLE_MASK_FILENAME)
    if not os.path.exists(global_subtitle_mask_path):
        print(f"严重错误: 全局字幕蒙版 '{global_subtitle_mask_path}' 未找到。程序将退出。")
        sys.exit(1)
    else:
        if VERBOSE_LOGGING: print(f"信息: 将使用全局字幕蒙版: {global_subtitle_mask_path}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    if VERBOSE_LOGGING: print(f"信息: 全局输出文件夹: {os.path.abspath(OUTPUT_FOLDER)}")

    futures_map = {} 

    with tqdm(total=len(tasks_to_run), unit="张图", ncols=120, 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
              disable=False, file=sys.stdout) as pbar_main: # tqdm输出到stdout
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for i, (iter_num, scene_folder_name, shot_folder_name, image_filename, scene_num_str) in enumerate(tasks_to_run):
                current_server_ip = SERVER_IPS[i % NUM_WORKERS] 
                
                scene_full_path = os.path.join(BASE_INPUT_DIR, scene_folder_name)
                
                current_scene_mask_path = os.path.join(MASKS_BASE_DIR, f"scene-{scene_num_str}-mask.png")
                if not os.path.exists(current_scene_mask_path):
                    if VERBOSE_LOGGING: 
                        # 使用tqdm.write打印，即使在提交任务阶段，如果pbar_main已初始化
                        tqdm.write(f"VERBOSE [任务准备]: 场景 {scene_folder_name} 的场景蒙版 '{current_scene_mask_path}' 未找到。如果工作流需要，将不使用此蒙版。")
                    current_scene_mask_path = None 

                shot_images_dir = os.path.join(scene_full_path, shot_folder_name)
                current_workflow_filename = f"FLUX-0508-{shot_folder_name}.json"
                current_workflow_path = os.path.join(WORKFLOW_BASE_DIR, current_workflow_filename)
                
                full_image_path = os.path.join(shot_images_dir, image_filename)
                
                server_id_for_log = current_server_ip.split('.')[-1].split(':')[0] 
                context_log = f"服务器{server_id_for_log} 迭代{iter_num} 场景{scene_num_str} 镜头{shot_folder_name} 图像{image_filename[:10]}"

                tester = ComfyUITester(
                    server_address=current_server_ip,
                    workflow_file_path=current_workflow_path, 
                    output_folder=OUTPUT_FOLDER, 
                    context_info=context_log,
                    verbose=VERBOSE_LOGGING
                )

                future = executor.submit(
                    tester.process_image, 
                    main_image_path=full_image_path,
                    scene_mask_local_path=current_scene_mask_path, 
                    subtitle_mask_local_path=global_subtitle_mask_path,
                    original_image_basename=image_filename,
                    current_iteration_num=iter_num,
                    progress_bar_instance=pbar_main 
                )
                futures_map[future] = (image_filename, current_server_ip, context_log) 
                if DELAY_BETWEEN_SUBMISSIONS > 0:
                    time.sleep(DELAY_BETWEEN_SUBMISSIONS) 

            tqdm.write(f"所有 {len(tasks_to_run)} 个任务已提交到线程池。正在等待各个任务完成...")

            for future in as_completed(futures_map):
                img_fname, srv_ip, ctx_log = futures_map[future]
                
                # 描述更新得不那么频繁，或者只在任务完成时更新
                # pbar_main.set_description_str(f"最近完成: {img_fname[:15]} (服务器 {srv_ip.split('.')[-1].split(':')[0]})", refresh=False)
                
                try:
                    processed_results, task_time = future.result() 

                    if processed_results and isinstance(processed_results, list) and len(processed_results) > 0:
                        total_images_processed_successfully += 1
                        individual_task_durations.append(task_time) 
                    else:
                        total_images_failed += 1
                except Exception as e:
                    total_images_failed += 1
                    tqdm.write(f"主线程捕获到处理图像 {img_fname} (来自 {ctx_log}) 时发生严重错误: {e}") 
                finally:
                    pbar_main.update(1) 
                    if pbar_main.n % (NUM_WORKERS if NUM_WORKERS > 0 else 1) == 0 or pbar_main.n == pbar_main.total : # 每N个任务或最后一个任务时刷新
                        pbar_main.refresh()
            if len(tasks_to_run) > 0 : pbar_main.refresh() 

    overall_end_time = time.time()
    total_script_duration_seconds = overall_end_time - overall_start_time
    
    # 最终总结使用普通 print，因为 tqdm 循环已结束
    print(f"\n\n{'='*20} 所有处理已完成 {'='*20}")
    print(f"总迭代轮数: {NUM_ITERATIONS}")
    print(f"计划处理的任务总数: {total_images_to_process_overall}")
    print(f"总共成功处理的图像任务数: {total_images_processed_successfully}")
    print(f"总共失败的图像任务数: {total_images_failed}")
    
    if total_images_to_process_overall > 0: 
         success_rate = (total_images_processed_successfully / total_images_to_process_overall) * 100
         print(f"总体成功率 (基于计划任务): {success_rate:.2f}%")
    else:
        print("未计划任何任务。")

    if individual_task_durations:
        avg_task_duration = sum(individual_task_durations) / len(individual_task_durations)
        print(f"单个成功任务的平均处理时间: {avg_task_duration:.2f} 秒")
    
    if total_script_duration_seconds > 0 and total_images_processed_successfully > 0:
        overall_throughput = total_images_processed_successfully / total_script_duration_seconds
        print(f"整体系统吞吐量: {overall_throughput:.2f} 张图像/秒")

    print(f"脚本总执行时间: {time.strftime('%H:%M:%S', time.gmtime(total_script_duration_seconds))}")
    print(f"所有结果已保存到: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"{'='*50}")

# --- END OF FILE comfyui_redrawer_0508_optimized_final_cn_logfix.py ---