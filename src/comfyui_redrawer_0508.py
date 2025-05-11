# --- START OF FILE comfyui_redrawer_0508_optimized_final_cn_log_beautify_v4_tqdm.py ---
import json
import requests
import time
import os
import random
import sys
from datetime import datetime
import uuid
# import glob
import re
# import base64
from tqdm import tqdm # <--- 重新引入 tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
# import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from prompt_reasoning import generate_anime_prompt as original_generate_anime_prompt, PromptGenerationError


# --------------- 配置参数 ---------------
SERVER_IPS = [
    "http://36.143.229.172:8188",
    "http://36.143.229.173:8188",
    "http://36.143.229.116:8188",
    "http://36.143.229.117:8188",
    "http://36.143.229.119:8188",
    "http://36.143.229.120:8188",
    "http://36.143.229.121:8188",
]
NUM_WORKERS = len(SERVER_IPS)

NUM_ITERATIONS = 8

BASE_INPUT_DIR = "data/250508"
MASKS_BASE_DIR = "data/mask"
SUBTITLE_MASK_FILENAME = "subtitle-mask.png"
WORKFLOW_BASE_DIR = "workflow"
OUTPUT_FOLDER = os.path.join(BASE_INPUT_DIR, "opt_auto")

WORKFLOW_SKIP_SCENE_MASK = "FLUX-0508-02.json" # 特定工作流文件名，用于跳过场景蒙版

MAX_WAIT_TIME = 360 # ComfyUI任务最大等待时间（秒）
MASKS_SERVER_SUBFOLDER = "clipspace" # 蒙版上传到服务器的子文件夹名称

IMAGE_INPUT_NODE_ID = "74" # ComfyUI 工作流中主图像输入节点的ID
PROMPT_NODE_ID = "227" # ComfyUI 工作流中提示词输入节点的ID
SCENE_MASK_NODE_ID = "190" # ComfyUI 工作流中场景蒙版输入节点的ID
SUBTITLE_MASK_NODE_ID = "229" # ComfyUI 工作流中字幕蒙版输入节点的ID

DELAY_BETWEEN_SUBMISSIONS = 0.05 # 提交任务到线程池的延迟（秒），0表示无延迟

VERBOSE_LOGGING = False # 是否启用详细输出
# --------------- 配置参数结束 ---------------

def generate_anime_prompt_wrapper(image_path: str, log_func_info, log_func_error, log_func_verbose) -> str | None:
    """
    包装原始的提示词生成函数，增加错误处理和状态报告。
    """
    base_image_name = os.path.basename(image_path)
    def status_reporter(message: str):
        # 此处的 log_func_verbose 来自 ComfyUITester 实例，将使用 tqdm.write
        log_func_verbose(f"    [提示API状态] {message}")
    try:
        prompt = original_generate_anime_prompt(image_path, status_callback=status_reporter)
        return prompt
    except PromptGenerationError as pge:
        log_func_error(f"    [提示生成失败] 图像 '{base_image_name}': {pge}")
        return None
    except Exception as e:
        log_func_error(f"    [提示生成意外错误] 图像 '{base_image_name}': 调用原始提示函数时发生 {type(e).__name__}: {e}")
        return None

class ComfyUITester:
    def __init__(self, server_address, workflow_file_path, output_folder, context_info="", verbose=VERBOSE_LOGGING):
        self.server_address = server_address.rstrip('/')
        self.api_url = self.server_address # ComfyUI API 的基础URL
        self.workflow_file_path = workflow_file_path
        self.output_folder = output_folder
        self.client_id = str(uuid.uuid4()) # 为每个 ComfyUI 会话生成唯一的客户端ID
        self.context_info = context_info # 用于输出的上下文信息，如服务器、迭代、场景等
        self.verbose = verbose

    def _print_message(self, level_prefix, message):
        """
        统一的消息输出方法，使用 tqdm.write 以避免干扰进度条。
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        # 使用 tqdm.write 确保输出打印在进度条上方或不破坏进度条
        tqdm.write(f"{timestamp} {level_prefix} [{self.context_info} - 客户端 {self.client_id[:6]}] {message}")

    def _log_verbose(self, message):
        if self.verbose:
            self._print_message("详细", message)

    def _log_info(self, message):
        self._print_message("信息", message)

    def _log_error(self, message):
        self._print_message("错误", message)

    def load_workflow(self):
        """加载 JSON 工作流文件。"""
        try:
            if not os.path.exists(self.workflow_file_path):
                self._log_error(f"工作流文件未找到: {self.workflow_file_path}")
                return None
            with open(self.workflow_file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            self._log_verbose(f"  成功加载工作流: {os.path.basename(self.workflow_file_path)}")
            return workflow_data
        except Exception as e:
            self._log_error(f"加载工作流 '{os.path.basename(self.workflow_file_path)}' 失败: {e}")
            return None

    def _upload_single_image(self, image_path: str, subfolder: str = "", image_type_for_log: str = "图像"):
        """通用图像上传方法。"""
        if not os.path.exists(image_path):
            self._log_error(f"无法上传，{image_type_for_log}文件不存在: {image_path}")
            return None

        filename = os.path.basename(image_path)
        upload_url = f"{self.api_url}/upload/image"
        self._log_verbose(f"    准备上传 {image_type_for_log}: {filename} (到服务器子文件夹 '{subfolder if subfolder else '根目录'}')")

        try:
            with open(image_path, 'rb') as f:
                files = {'image': (filename, f, 'image/png')} # 假设都是png，可根据需要调整
                data = {'overwrite': 'true'} # 覆盖同名文件
                if subfolder: data['subfolder'] = subfolder

                response = requests.post(upload_url, files=files, data=data, timeout=120) # 120秒超时
                response.raise_for_status() # 如果HTTP状态码是4xx或5xx，则抛出HTTPError
                upload_data = response.json()
                server_filename = upload_data.get('name')
                server_subfolder = upload_data.get('subfolder', '') # ComfyUI可能返回子文件夹

                if not server_filename:
                    self._log_error(f"{image_type_for_log} '{filename}' 上传成功，但服务器响应中未包含文件名。")
                    return None
                
                # 构建服务器上的完整图像引用路径
                final_image_reference = f"{server_subfolder}/{server_filename}" if server_subfolder else server_filename
                self._log_verbose(f"    {image_type_for_log} '{filename}' 上传成功: {final_image_reference}")
                return final_image_reference
        except requests.exceptions.HTTPError as http_err:
            self._log_error(f"{image_type_for_log} '{filename}' 上传时发生 HTTP 错误: {http_err}")
            if hasattr(http_err, 'response') and http_err.response is not None:
                try: self._log_verbose(f"      服务器响应 ({http_err.response.status_code}): {http_err.response.text[:200]}...")
                except Exception: pass
        except requests.exceptions.RequestException as req_err: # 包括网络连接错误等
            self._log_error(f"处理{image_type_for_log} '{filename}' 上传时发生网络请求错误: {req_err}")
        except Exception as e:
            self._log_error(f"处理{image_type_for_log} '{filename}' 上传时发生意外错误: {e}")
        return None

    def upload_main_image(self, image_path: str):
        """上传主图像到服务器根输入目录。"""
        return self._upload_single_image(image_path, subfolder="", image_type_for_log="主图像")

    def upload_specific_mask(self, mask_local_path: str, server_target_subfolder: str, mask_type_log: str):
        """上传特定类型的蒙版到服务器指定子文件夹。"""
        if not mask_local_path or not os.path.exists(mask_local_path):
            self._log_verbose(f"警告: {mask_type_log} 路径未提供或文件不存在 ('{mask_local_path}')。跳过上传。")
            return None
        return self._upload_single_image(mask_local_path, subfolder=server_target_subfolder, image_type_for_log=mask_type_log)

    def update_workflow(self, workflow, main_image_ref: str, generated_prompt: str | None,
                        scene_mask_ref: str | None, subtitle_mask_ref: str | None):
        """
        更新工作流中的输入节点：主图像、提示词、场景蒙版、字幕蒙版，并设置随机种子。
        """
        if not workflow: return None
        modified_workflow = json.loads(json.dumps(workflow)) # 深拷贝工作流以避免修改原始模板

        # 更新主图像输入
        if IMAGE_INPUT_NODE_ID in modified_workflow:
            modified_workflow[IMAGE_INPUT_NODE_ID]["inputs"]["image"] = main_image_ref
            self._log_verbose(f"    已更新主图像节点 {IMAGE_INPUT_NODE_ID} 为: {main_image_ref}")
        else:
            self._log_error(f"主图像节点 ID '{IMAGE_INPUT_NODE_ID}' 在工作流中未找到。")
            return None # 关键节点缺失，无法继续

        # 更新提示词输入
        if generated_prompt: # 如果成功生成了AI提示词
            if PROMPT_NODE_ID in modified_workflow:
                modified_workflow[PROMPT_NODE_ID]["inputs"]["text"] = generated_prompt
                self._log_verbose(f"    已使用生成提示词更新节点 {PROMPT_NODE_ID}。")
            else:
                self._log_verbose(f"警告: 提示词节点 ID '{PROMPT_NODE_ID}' 未找到。无法更新提示词。")
        else: # 未生成提示词，使用工作流中的默认提示词
            self._log_verbose(f"    未提供生成提示词。节点 {PROMPT_NODE_ID} 将使用工作流默认值。")

        # 更新场景蒙版输入
        if SCENE_MASK_NODE_ID: # 仅当配置了场景蒙版节点ID时才处理
            if scene_mask_ref: # 如果上传了场景蒙版并获得了引用
                if SCENE_MASK_NODE_ID in modified_workflow:
                    modified_workflow[SCENE_MASK_NODE_ID]["inputs"]["image"] = scene_mask_ref
                    self._log_verbose(f"    已更新场景蒙版节点 {SCENE_MASK_NODE_ID} 为: {scene_mask_ref}")
                else: # 节点ID配置了，但在工作流中找不到
                    self._log_verbose(f"警告: 场景蒙版节点 ID '{SCENE_MASK_NODE_ID}' 未在工作流中找到，但提供了蒙版引用。")
            else: # 未提供场景蒙版引用 (可能是有意跳过或上传失败)
                 self._log_verbose(f"    未向节点 {SCENE_MASK_NODE_ID} 提供场景蒙版引用 (或被有意跳过)。")

        # 更新字幕蒙版输入
        if SUBTITLE_MASK_NODE_ID: # 仅当配置了字幕蒙版节点ID时才处理
            if subtitle_mask_ref: # 如果上传了字幕蒙版并获得了引用
                if SUBTITLE_MASK_NODE_ID in modified_workflow:
                    modified_workflow[SUBTITLE_MASK_NODE_ID]["inputs"]["image"] = subtitle_mask_ref
                    self._log_verbose(f"    已更新字幕蒙版节点 {SUBTITLE_MASK_NODE_ID} 为: {subtitle_mask_ref}")
                else: # 节点ID配置了，但在工作流中找不到
                    self._log_verbose(f"警告: 字幕蒙版节点 ID '{SUBTITLE_MASK_NODE_ID}' 未在工作流中找到。")
            else: # 未提供字幕蒙版引用
                self._log_verbose(f"    未提供字幕蒙版引用给节点 {SUBTITLE_MASK_NODE_ID}。")
        
        # 为所有KSampler节点设置随机种子
        random_seed = random.randint(0, 2**32 - 1)
        sampler_updated_count = 0
        for node_id, node_data in modified_workflow.items():
            if "class_type" in node_data and "KSampler" in node_data["class_type"]: # 检查是否为 KSampler 节点
                if "inputs" in node_data and "seed" in node_data["inputs"]:
                    node_data["inputs"]["seed"] = random_seed
                    self._log_verbose(f"    已更新 KSampler 节点 {node_id} 的种子为: {random_seed}")
                    sampler_updated_count +=1
        if sampler_updated_count == 0:
            self._log_verbose(f"警告: 工作流中未找到任何 KSampler 节点来更新种子。")
        elif sampler_updated_count > 1:
             self._log_verbose(f"注意: 工作流中更新了 {sampler_updated_count} 个 KSampler 节点的种子。")
        return modified_workflow

    def send_prompt(self, workflow):
        """向ComfyUI服务器发送已准备好的工作流。"""
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        payload_dict = {'prompt': workflow, 'client_id': self.client_id}
        data = json.dumps(payload_dict)
        try:
            response = requests.post(f"{self.api_url}/prompt", headers=headers, data=data, timeout=60) # 60秒超时
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            self._log_error(f"提交工作流时发生 HTTP 错误: {http_err}")
            if hasattr(http_err, 'response') and http_err.response is not None:
                self._log_verbose(f"      服务器响应 ({http_err.response.status_code}): {http_err.response.text[:200]}...")
            return None
        except requests.exceptions.RequestException as req_err:
            self._log_error(f"提交工作流时发生网络请求错误: {req_err}")
            return None
        except Exception as e:
            self._log_error(f"提交工作流时发生错误: {e}")
            return None

    def get_history(self, prompt_id):
        """获取指定prompt_id的执行历史/状态。"""
        try:
            # API 端点 /history/{prompt_id}
            response = requests.get(f"{self.api_url}/history/{prompt_id}", headers={'Accept': 'application/json'}, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # 在详细模式下，记录这些轮询错误，否则可能会产生大量不必要的输出
            if self.verbose: self._log_verbose(f"获取提示 {prompt_id} 的历史记录时出错 (网络/HTTP): {e}")
            return None # 返回 None 表示获取历史失败
        except Exception as e:
            if self.verbose: self._log_verbose(f"获取提示 {prompt_id} 的历史记录时发生意外错误: {e}")
            return None

    def download_output_images(self, history, prompt_id, output_dir_for_run, original_image_basename, current_iteration_num):
        """从ComfyUI历史记录中下载标记为'output'的图像。"""
        if not history or prompt_id not in history:
            self._log_verbose("  未找到用于下载的执行历史。")
            return []

        outputs = history[prompt_id].get('outputs', {})
        downloaded_files_paths = []
        images_to_download = []
        # 遍历历史记录中的所有输出节点
        for node_id, node_output in outputs.items():
            if 'images' in node_output and isinstance(node_output['images'], list):
                for img_data in node_output['images']:
                    # 仅下载标记为 'output' 类型的图像
                    if img_data.get('type') == 'output':
                        images_to_download.append(img_data)

        if not images_to_download:
            self._log_verbose(f"  在工作流输出中 (提示ID: {prompt_id}) 未找到 'output' 类型的图像。")
            return []

        self._log_verbose(f"    准备从服务器下载 {len(images_to_download)} 张 'output' 图像 (提示ID: {prompt_id})")

        original_base, original_ext = os.path.splitext(original_image_basename)
        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3] # 精确到毫秒的时间戳

        for idx, image_data in enumerate(images_to_download):
            server_filename = image_data.get('filename')
            subfolder = image_data.get('subfolder', '') # 服务器上的子文件夹

            if not server_filename:
                self._log_verbose(f"    跳过没有文件名的图像数据: {image_data}")
                continue

            # 构建本地保存文件名
            name_part_to_use = original_base
            if len(images_to_download) > 1: # 如果有多个输出图像，添加索引
                name_part_to_use = f"{original_base}_output_{idx}"
            
            if current_iteration_num > 1: # 如果是多次迭代中的一次
                final_local_filename = f"{name_part_to_use}-iter{current_iteration_num}_{timestamp_str}{original_ext}"
            else: # 首次迭代或单次运行
                final_local_filename = f"{name_part_to_use}_{timestamp_str}{original_ext}"

            local_path = os.path.join(output_dir_for_run, final_local_filename)
            # API 端点 /view 用于下载图像
            url_params = {'filename': server_filename, 'subfolder': subfolder}
            
            try:
                self._log_verbose(f"      正在下载: 服务器文件 '{subfolder}/{server_filename}' 到 '{final_local_filename}'")
                response = requests.get(f"{self.api_url}/view", params=url_params, stream=True, timeout=180) # 180秒下载超时
                response.raise_for_status()

                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
                self._log_verbose(f"      已保存到: {local_path}")
                downloaded_files_paths.append(os.path.abspath(local_path))
            except requests.exceptions.HTTPError as http_err:
                self._log_error(f"下载图像 '{server_filename}' HTTP 错误 ({http_err.response.status_code}): {http_err}")
            except requests.exceptions.RequestException as req_err:
                self._log_error(f"下载图像 '{server_filename}' 时发生网络请求错误: {req_err}")
            except Exception as e:
                self._log_error(f"下载或保存图像 '{server_filename}' 时出错: {e}")
        
        if not downloaded_files_paths and images_to_download: # 如果有图像数据但一个也没下载成功
             self._log_verbose(f"  警告：找到图像数据但未能下载任何图像 ({original_image_basename})。")
        return downloaded_files_paths

    def wait_for_completion(self, prompt_id):
        """等待ComfyUI任务完成，轮询状态。"""
        log_prefix_short = f"提示ID:{prompt_id[:6]}" # 用于日志的缩短版提示ID
        start_time_wait = time.time()
        last_log_time = 0 # 用于控制详细日志的输出频率
        server_log_id = self.server_address.split('//')[-1].split(':')[0] # 从服务器地址提取IP用于日志

        self._log_verbose(f"{log_prefix_short} 监视服务器 {server_log_id} 上的任务...")

        while time.time() - start_time_wait < MAX_WAIT_TIME:
            time.sleep(1.0) # 轮询间隔
            current_time_loop = time.time()
            history_response = self.get_history(prompt_id) # 获取最新状态

            if history_response and prompt_id in history_response:
                status_obj = history_response[prompt_id]
                status_info = status_obj.get("status", {})
                status_str = status_info.get("status_str", "未知状态")
                outputs_exist = bool(status_obj.get("outputs")) # 检查 'outputs' 字段是否存在
                q_rem = status_info.get("exec_info", {}).get("queue_remaining", "N/A") # 队列中剩余任务
                current_elapsed_time_wait = time.time() - start_time_wait

                # 详细日志：如果任务仍在运行/等待，或距离上次日志已有一段时间
                if self.verbose and (status_str in ['running', 'pending'] or (current_time_loop - last_log_time >= 15.0)):
                    self._log_verbose(f"    [{log_prefix_short} on Srv {server_log_id}] 状态: {status_str}, 队列: {q_rem}, 等待: {current_elapsed_time_wait:.0f}s")
                    last_log_time = current_time_loop

                # 检查任务是否成功完成 (有输出且状态为 'success')
                if outputs_exist and status_str == 'success':
                    elapsed_wait = time.time() - start_time_wait
                    self._log_info(f"  任务 {log_prefix_short} (Srv {server_log_id}) 成功。状态: {status_str}, API等待: {elapsed_wait:.2f}秒")
                    return True, history_response, elapsed_wait
                # 检查任务是否失败或出现错误
                elif status_str in ['failed', 'error', 'cancelled'] or \
                     (outputs_exist and status_str not in ['success', 'running', 'pending']): # 异常状态
                    elapsed_wait = time.time() - start_time_wait
                    self._log_error(f"  任务 {log_prefix_short} (Srv {server_log_id}) 失败/错误。状态: {status_str}, API等待: {elapsed_wait:.2f}秒")
                    if status_obj.get("outputs"): # 尝试打印错误节点的具体信息
                        for node_id_err, node_output in status_obj["outputs"].items():
                            if 'errors' in node_output:
                                self._log_error(f"    {log_prefix_short} 节点 {node_id_err} 错误: {node_output['errors']}")
                    return False, history_response, elapsed_wait
            
            else: # 获取历史记录失败 (网络问题或服务器忙)
                if self.verbose and (current_time_loop - last_log_time >= 10.0): # 避免过于频繁的轮询失败日志
                    elapsed_wait = time.time() - start_time_wait
                    self._log_verbose(f"    [{log_prefix_short} on Srv {server_log_id}] API轮询: 无法获取历史。已等待:{elapsed_wait:.1f}s")
                    last_log_time = current_time_loop
        
        # 超时处理
        elapsed_wait_timeout = time.time() - start_time_wait
        self._log_error(f"任务 {log_prefix_short} (Srv {server_log_id}) 超时 ({MAX_WAIT_TIME}秒)。")
        return False, None, elapsed_wait_timeout

    def process_image(self, main_image_path, scene_mask_local_path, subtitle_mask_local_path,
                      original_image_basename, current_iteration_num):
        """
        处理单个图像的完整流程：加载工作流、上传图像和蒙版、生成提示、更新工作流、
        提交任务、等待完成、下载结果。
        """
        task_start_time = time.time()
        self._log_info(f"开始处理: '{original_image_basename}' (迭代 {current_iteration_num}, 工作流: {os.path.basename(self.workflow_file_path)})")

        workflow = self.load_workflow()
        if not workflow:
            return [], 0 # 返回空列表和0时长表示失败

        # 1. 上传主图像
        uploaded_main_image_ref = self.upload_main_image(main_image_path)
        if not uploaded_main_image_ref:
             return [], 0

        # 2. 上传场景蒙版 (如果需要且存在)
        uploaded_scene_mask_ref = None
        current_workflow_basename = os.path.basename(self.workflow_file_path)
        if current_workflow_basename == WORKFLOW_SKIP_SCENE_MASK: # 特定工作流跳过场景蒙版
            self._log_verbose(f"    工作流 {current_workflow_basename} 跳过场景蒙版 (节点 {SCENE_MASK_NODE_ID})。")
        elif SCENE_MASK_NODE_ID: # 检查是否配置了场景蒙版节点ID
            if scene_mask_local_path: # 检查是否提供了场景蒙版路径
                self._log_verbose(f"    工作流 {current_workflow_basename} 将使用场景蒙版: '{os.path.basename(scene_mask_local_path)}'")
                uploaded_scene_mask_ref = self.upload_specific_mask(scene_mask_local_path, MASKS_SERVER_SUBFOLDER, "场景蒙版")
                if not uploaded_scene_mask_ref:
                    self._log_verbose(f"警告: 场景蒙版 '{os.path.basename(scene_mask_local_path)}' 上传失败。")
            else: # 配置了节点ID但未提供蒙版路径
                self._log_verbose(f"    工作流 {current_workflow_basename} 可用场景蒙版，但未提供蒙版路径。")

        # 3. 上传字幕蒙版 (如果需要且存在)
        uploaded_subtitle_mask_ref = None
        if subtitle_mask_local_path and SUBTITLE_MASK_NODE_ID: # 检查路径和节点ID是否都存在
            self._log_verbose(f"    将使用字幕蒙版: '{os.path.basename(subtitle_mask_local_path)}'")
            uploaded_subtitle_mask_ref = self.upload_specific_mask(subtitle_mask_local_path, MASKS_SERVER_SUBFOLDER, "字幕蒙版")
            if not uploaded_subtitle_mask_ref:
                 self._log_verbose(f"警告: 字幕蒙版 '{os.path.basename(subtitle_mask_local_path)}' 上传失败。")
        elif SUBTITLE_MASK_NODE_ID: # 配置了节点ID但未提供蒙版路径
            self._log_verbose(f"    字幕蒙版节点ID ({SUBTITLE_MASK_NODE_ID}) 已配置但未提供路径。")

        # 4. AI生成提示词
        anime_prompt = None
        self._log_info(f"  -> 正在为 '{original_image_basename}' 请求AI生成提示词...")
        anime_prompt = generate_anime_prompt_wrapper(
            main_image_path,
            self._log_info, # 传递修改后的输出方法
            self._log_error,
            self._log_verbose
        )

        if anime_prompt:
            max_prompt_log_length = 120 # 输出时截断过长的提示词
            logged_prompt = anime_prompt
            if len(anime_prompt) > max_prompt_log_length:
                logged_prompt = anime_prompt[:max_prompt_log_length].replace('\n', ' ') + "..."
            self._log_info(f"  <- AI生成提示词 (图像: {original_image_basename}): \"{logged_prompt}\"")
        else:
            self._log_info(f"  <- 未能为 '{original_image_basename}' 获取AI提示词，将使用工作流默认值。")

        # 5. 更新工作流
        modified_workflow = self.update_workflow(workflow, uploaded_main_image_ref, anime_prompt,
                                                 uploaded_scene_mask_ref, uploaded_subtitle_mask_ref)
        if not modified_workflow:
            return [], 0

        # 6. 提交工作流到ComfyUI
        prompt_response = self.send_prompt(modified_workflow)
        if not prompt_response or 'prompt_id' not in prompt_response:
            return [], 0 # 提交失败
        prompt_id = prompt_response['prompt_id']
        self._log_info(f"  ComfyUI任务已提交 (图像: {original_image_basename}, 提示ID: {prompt_id[:8]})")

        # 7. 等待任务完成
        completed, final_history, time_spent_waiting = self.wait_for_completion(prompt_id)
        task_duration = time.time() - task_start_time # 总任务耗时

        # 8. 处理结果
        if completed and final_history:
            output_images = self.download_output_images(final_history, prompt_id, OUTPUT_FOLDER,
                                                        original_image_basename, current_iteration_num)
            if output_images:
                self._log_info(f"成功完成并下载 {len(output_images)} 张图片 for '{original_image_basename}'。总耗时: {task_duration:.2f}s。")
                return output_images, task_duration
            else:
                self._log_error(f"工作流成功但下载图片失败 (提示ID: {prompt_id}, {original_image_basename})。总耗时: {task_duration:.2f}s。")
                return [], task_duration # 成功但下载失败也算部分失败
        else: # 任务失败或超时
            self._log_error(f"处理 '{original_image_basename}' 失败/超时 (提示ID: {prompt_id or 'N/A'})。总耗时: {task_duration:.2f}s (其中API等待 {time_spent_waiting:.2f}s)。")
            if final_history and prompt_id in final_history and not completed: # 如果有历史但未完成 (例如，错误)
                 status_obj = final_history[prompt_id]
                 outputs = status_obj.get('outputs', {})
                 for node_id_err, node_output in outputs.items(): # 打印详细的节点错误
                    if 'errors' in node_output and self.verbose: # 仅在详细模式下打印
                        self._log_error(f"      补充错误: 节点 {node_id_err} 报告错误: {node_output['errors']}")
            return [], task_duration

# --------------- 主要执行逻辑 ---------------
if __name__ == "__main__":
    # 确保在程序开始时，输出文件夹存在
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    # 使用 print 进行初始消息输出，这些消息会在 tqdm 进度条初始化之前显示
    print(f"所有输出将保存到: {os.path.abspath(OUTPUT_FOLDER)}")

    overall_start_time = time.time()
    tasks_to_run = [] # 存储所有待处理任务的元组列表

    print("程序开始：扫描文件并计算总任务数...")
    # 任务扫描逻辑: 预先收集所有任务，以便准确显示总进度
    for iter_num_calc in range(1, NUM_ITERATIONS + 1):
        scene_folders_calc = sorted([
            d for d in os.listdir(BASE_INPUT_DIR)
            if os.path.isdir(os.path.join(BASE_INPUT_DIR, d)) and d.startswith("场景")
        ])
        for scene_folder_name_calc in scene_folders_calc:
            scene_full_path_calc = os.path.join(BASE_INPUT_DIR, scene_folder_name_calc)
            scene_num_match_calc = re.search(r'\d+', scene_folder_name_calc) # 从场景文件夹名提取场景编号
            scene_num_str_calc = scene_num_match_calc.group(0) if scene_num_match_calc else "未知场景ID"

            shot_folders_calc = sorted([
                d for d in os.listdir(scene_full_path_calc)
                if os.path.isdir(os.path.join(scene_full_path_calc, d)) and d.isdigit() # 镜头文件夹是数字
            ])
            for shot_folder_name_calc in shot_folders_calc:
                shot_images_dir_calc = os.path.join(scene_full_path_calc, shot_folder_name_calc)
                # 根据镜头号确定工作流文件，例如 FLUX-0508-01.json, FLUX-0508-02.json
                current_workflow_filename_calc = f"FLUX-0508-{shot_folder_name_calc}.json"
                current_workflow_path_calc = os.path.join(WORKFLOW_BASE_DIR, current_workflow_filename_calc)

                if not os.path.exists(current_workflow_path_calc):
                    print(f"警告 [预扫描]: 工作流文件 '{current_workflow_path_calc}' 未找到，跳过场景 '{scene_folder_name_calc}' /镜头 '{shot_folder_name_calc}' 的任务添加。")
                    continue

                image_files_calc = sorted([
                    f for f in os.listdir(shot_images_dir_calc)
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                ])
                if not image_files_calc:
                    if VERBOSE_LOGGING: print(f"信息 [预扫描]: 在 '{shot_images_dir_calc}' 中未找到图像文件。")
                    continue

                for image_filename_calc in image_files_calc:
                    # 添加任务元组: (迭代号, 场景文件夹名, 镜头文件夹名, 图像文件名, 场景编号字符串)
                    tasks_to_run.append((iter_num_calc, scene_folder_name_calc, shot_folder_name_calc, image_filename_calc, scene_num_str_calc))
    
    total_tasks_to_process = len(tasks_to_run)

    if total_tasks_to_process == 0:
        print("未找到任何需要处理的图像任务。请检查输入目录、文件结构和工作流文件。程序将退出。")
        sys.exit(0)

    print(f"总共需要处理 {total_tasks_to_process} 个图像任务，使用 {NUM_WORKERS} 个并发工作线程。")

    tasks_completed_so_far = 0 # 新增：已完成任务计数器 (包括成功和失败)
    total_images_processed_successfully = 0
    total_images_failed = 0
    individual_task_durations = [] # 存储每个成功任务的耗时

    # 确保全局字幕蒙版路径在 ComfyUITester 实例化之前定义和检查
    global_subtitle_mask_path = os.path.join(MASKS_BASE_DIR, SUBTITLE_MASK_FILENAME)
    if not os.path.exists(global_subtitle_mask_path):
        print(f"严重错误: 全局字幕蒙版 '{global_subtitle_mask_path}' 未找到。程序将退出。")
        sys.exit(1)
    else:
        # 此处使用 print 是因为 tqdm 循环还未开始
        if VERBOSE_LOGGING: print(f"信息: 将使用全局字幕蒙版: {global_subtitle_mask_path}")

    futures_map = {} # 用于将 future 对象映射回其原始任务信息
    # 此 print 在 tqdm 循环之外，是安全的
    print(f"\n--- 准备提交 {total_tasks_to_process} 个任务到线程池 ---")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 任务提交循环
        for i, (iter_num, scene_folder_name, shot_folder_name, image_filename, scene_num_str) in enumerate(tasks_to_run):
            current_server_ip = SERVER_IPS[i % NUM_WORKERS] # 轮询服务器IP
            scene_full_path = os.path.join(BASE_INPUT_DIR, scene_folder_name)
            
            # 构建当前场景的蒙版路径，例如 scene-1-mask.png
            current_scene_mask_path = os.path.join(MASKS_BASE_DIR, f"scene-{scene_num_str}-mask.png")
            if not os.path.exists(current_scene_mask_path):
                if VERBOSE_LOGGING:
                    # 使用 tqdm.write 因为这可能与进度条初始化或早期更新同时发生
                    tqdm.write(f"详细 [任务准备]: 场景 '{scene_folder_name}' 的特定蒙版 '{current_scene_mask_path}' 未找到。将不使用场景蒙版。")
                current_scene_mask_path = None # 如果蒙版不存在，则不使用

            shot_images_dir = os.path.join(scene_full_path, shot_folder_name)
            current_workflow_filename = f"FLUX-0508-{shot_folder_name}.json"
            current_workflow_path = os.path.join(WORKFLOW_BASE_DIR, current_workflow_filename)
            full_image_path = os.path.join(shot_images_dir, image_filename)

            # 为每个任务创建上下文日志信息
            server_id_for_log = current_server_ip.split('.')[-1].split(':')[0] # 提取IP的最后一部分用于日志
            # 修改 context_log 以包含任务提交时的总任务数，i+1 是当前提交的任务序号
            context_log = f"任务 {i+1}/{total_tasks_to_process} Srv{server_id_for_log} It{iter_num} Sc{scene_num_str} Sh{shot_folder_name} Img:{image_filename[:10]}"


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
                subtitle_mask_local_path=global_subtitle_mask_path, # 全局字幕蒙版
                original_image_basename=image_filename,
                current_iteration_num=iter_num
            )
            futures_map[future] = (image_filename, current_server_ip, context_log)
            if DELAY_BETWEEN_SUBMISSIONS > 0:
                time.sleep(DELAY_BETWEEN_SUBMISSIONS) # 避免瞬间提交过多任务，给服务器缓冲
        
        # 所有任务已提交到线程池
        # 使用 tqdm.write 来确保此消息打印在进度条上方（如果已有活动进度条）或正常打印
        tqdm.write(f"\n所有 {total_tasks_to_process} 个任务已提交。开始处理并等待完成...\n")

        # 使用 tqdm 包装 as_completed 来显示进度条
        for future in tqdm(as_completed(futures_map),
                            total=total_tasks_to_process,
                            desc="处理图像",  # 进度条描述
                            unit="张",       # 每个项目的单位
                            ncols=120 if sys.stdout.isatty() else None,  # 尝试设置宽度，如果非tty则不设置
                            dynamic_ncols=True, # 允许动态调整宽度
                            file=sys.stdout,    # 确保输出到 stdout
                            # 详细的进度条格式，包含速率和剩余时间
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
                            ):
            img_fname, srv_ip, ctx_log = futures_map[future] # 从映射中取回任务信息
            tasks_completed_so_far += 1 # 无论成功失败，都算一个任务已尝试完成

            task_succeeded_in_future = False
            current_task_actual_duration = None

            try:
                processed_results, task_duration_for_this_future = future.result() # 获取任务执行结果
                current_task_actual_duration = task_duration_for_this_future # 保存本次任务的耗时
                if processed_results and isinstance(processed_results, list) and len(processed_results) > 0:
                    total_images_processed_successfully += 1
                    individual_task_durations.append(task_duration_for_this_future)
                    task_succeeded_in_future = True # 标记此任务成功
                else: # 如果 process_image 返回空列表或处理失败
                    total_images_failed += 1
            except CancelledError: # 捕获任务被取消的错误
                total_images_failed += 1
                # 使用 tqdm.write 打印错误信息，以避免破坏进度条
                tqdm.write(f"错误 [主循环]: 任务 {ctx_log} 被取消。")
            except Exception as e: # 捕获其他未预料的异常
                total_images_failed += 1
                # 使用 tqdm.write 打印错误信息
                tqdm.write(f"错误 [主循环]: 处理 '{img_fname}' (来自 {ctx_log}) 时发生严重错误: {type(e).__name__}: {e}")
            
            # --- 在每个任务完成后打印当前总体进度 ---
            current_elapsed_script_time = time.time() - overall_start_time
            # 平均成功速度: 基于成功处理的图像数和脚本总运行时间
            avg_speed_for_successful_tasks = total_images_processed_successfully / current_elapsed_script_time if current_elapsed_script_time > 0 and total_images_processed_successfully > 0 else 0.0
            
            last_task_duration_info_str = ""
            if task_succeeded_in_future and current_task_actual_duration is not None:
                last_task_duration_info_str = f" 最近成功任务耗时: {current_task_actual_duration:.2f}s."
            elif current_task_actual_duration is not None: # 任务完成了（有耗时记录）但可能未成功（例如下载失败）
                 last_task_duration_info_str = f" 最近任务(可能未完全成功)耗时: {current_task_actual_duration:.2f}s."

            # 构造进度信息字符串, 速度保留三位小数
            avg_speed_str = f"{avg_speed_for_successful_tasks:.3f}" 
            progress_summary_message = (
                f"--- [全局进度 {tasks_completed_so_far}/{total_tasks_to_process}] --- "
                f"成功: {total_images_processed_successfully}, 失败: {total_images_failed}. "
                f"脚本已运行: {time.strftime('%H:%M:%S', time.gmtime(current_elapsed_script_time))}. "
                f"平均成功速度: {avg_speed_str} 张/秒.{last_task_duration_info_str}"
            )
            tqdm.write(progress_summary_message)
            # --- 打印结束 ---

    overall_end_time = time.time()
    total_script_duration_seconds = overall_end_time - overall_start_time

    # 最终总结使用 print，因为此时 tqdm 进度条已结束
    print("\n") 
    print(f"{'='*20} 所有处理已完成 {'='*20}")
    print(f"总迭代轮数: {NUM_ITERATIONS}")
    print(f"计划处理的任务总数: {total_tasks_to_process}")
    print(f"总共成功处理的图像任务数: {total_images_processed_successfully}")
    print(f"总共失败的图像任务数: {total_images_failed}")

    if total_tasks_to_process > 0:
         success_rate = (total_images_processed_successfully / total_tasks_to_process) * 100
         print(f"总体成功率 (基于计划任务): {success_rate:.2f}%")
    else:
        print("未计划任何任务。")

    if individual_task_durations:
        avg_task_duration = sum(individual_task_durations) / len(individual_task_durations)
        print(f"单个成功任务的平均处理时间: {avg_task_duration:.2f} 秒")
    else:
        if total_images_processed_successfully > 0: # 如果有成功任务但列表为空（理论上不应发生）
             print("成功处理了图像，但未记录单个任务时长。")
        elif total_tasks_to_process > 0 : # 如果没有成功任务
            print("没有成功处理的任务，无法计算平均处理时间。")


    if total_script_duration_seconds > 0 and total_images_processed_successfully > 0:
        # 这里的吞吐量是基于整个脚本运行时间和成功处理的图像数
        overall_throughput = total_images_processed_successfully / total_script_duration_seconds
        print(f"整体系统吞吐量: {overall_throughput:.3f} 张图像/秒") # 也调整为3位小数

    print(f"脚本总执行时间: {time.strftime('%H:%M:%S', time.gmtime(total_script_duration_seconds))}")
    print(f"所有结果已保存到: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"{'='*50}")