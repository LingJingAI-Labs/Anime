# --- START OF FILE comfyui_redrawer_0508_final_unified_workflow.py ---
import json
import requests
import time
import os
import random
import sys
from datetime import datetime
import uuid
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError

# --- Set up Python Path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# --- Try importing prompt_reasoning, handle if not found ---
try:
    from prompt_reasoning import generate_anime_prompt as original_generate_anime_prompt, PromptGenerationError
    PROMPT_REASONING_AVAILABLE = True
except ImportError:
    print("错误: 无法导入 'prompt_reasoning' 模块。AI提示生成将不可用。")
    print("请确保 'prompt_reasoning.py' 文件与此脚本位于同一目录或在 Python 路径中。")
    PROMPT_REASONING_AVAILABLE = False
    # Define dummy class/function if needed elsewhere, though not strictly required by current usage
    class PromptGenerationError(Exception): pass
    def original_generate_anime_prompt(*args, **kwargs):
        raise NotImplementedError("prompt_reasoning module not available")

# --------------- Configuration Parameters ---------------
SERVER_IPS = [
    "comfyui-demo.lingjingai.cn",
    "comfyui-demo2.lingjingai.cn",
    "comfyui-demo3.lingjingai.cn",      
    "comfyui-demo4.lingjingai.cn",
    "comfyui-demo5.lingjingai.cn",
    "comfyui-demo6.lingjingai.cn",
    "comfyui-demo7.lingjingai.cn",
    "comfyui-demo8.lingjingai.cn",
    "comfyui-demo9.lingjingai.cn",
    "comfyui-demo10.lingjingai.cn",
    "comfyui-demo11.lingjingai.cn",
    "comfyui-demo12.lingjingai.cn",
]
NUM_WORKERS = len(SERVER_IPS) # Number of concurrent workers = number of servers

NUM_ITERATIONS = 8 # Number of times to iterate over all images

BASE_INPUT_DIR = "data/250511" # Root directory containing scene folders
MASKS_BASE_DIR = "data/mask" # Directory containing mask files (scene & subtitle)
SUBTITLE_MASK_FILENAME = "subtitle-mask.png" # Filename for the global subtitle mask
WORKFLOW_BASE_DIR = "workflow" # Directory containing the base workflow JSON
OUTPUT_FOLDER = os.path.join(BASE_INPUT_DIR, "opt_auto_unified") # Output directory for generated images

# --- Unified Workflow and Dynamic LoRA Configuration ---
BASE_WORKFLOW_FILENAME = "FLUX-0508-base.json" # Single workflow file to load for all tasks
LORA_NODE_ID = "12" # The ID of the "Lora Loader Stack" node in the workflow
DEFAULT_LORA_STRENGTH_02 = 1.0 # Default strength for lora_02 when a specific LoRA is used

# --- !! 人工编辑区: 定义镜头文件夹名称到 LoRA 文件名的映射 !! ---
# Format: "shot_folder_name_string": "LoRA_filename_in_ComfyUI"
# Special Case: "00" maps to "None", which disables lora_02 and sets strength to 0.
LORA_MAPPING = {
    "00": "None", # Special case: Disable lora_02
    "01": "wuji/char/陈天极/char01陈天极.safetensors",
    "03": "wuji/char/欧阳南/char03欧阳南.safetensors",
    "04": "wuji/char/柳如月/char04柳如月.safetensors",
    "10": "wuji/char/方天行/char10方天行.safetensors",
    "11": "wuji/char/金蕴/char11金蕴.safetensors",
    "13": "wuji/char/外门长老1/char13外门长老1.safetensors",
    "14": "wuji/char/外门长老2/char14外门长老2.safetensors",
    "15": "wuji/char/魏不凡/char15魏不凡.safetensors",
    "16": "wuji/char/外门长老3/char16外门长老3.safetensors",
    "18": "wuji/char/器灵/char18器灵.safetensors",
    "19": "wuji/char/黑供奉/char19黑供奉.safetensors",
    "20": "wuji/char/火长老/char20火长老.safetensors",
    "21": "wuji/char/叶天穹/char21叶天穹.safetensors",
    "25": "wuji/char/金长老/char25金长老.safetensors",
    "26": "wuji/char/木长老/char26木长老.safetensors",
    "27": "wuji/char/土长老/char27土长老.safetensors",
    "28": "wuji/char/水长老/char28水长老.safetensors",
    "30": "wuji/char/小师妹/char30小师妹.safetensors",
}
# --- !! 人工编辑区结束 !! ---

# --- Define which shots skip the scene mask ---
# List of shot folder names (strings) that should ignore scene masks
SHOTS_TO_SKIP_SCENE_MASK = ["02"]

# --- Node IDs in the Workflow ---
# These must match the node IDs in your BASE_WORKFLOW_FILENAME JSON
IMAGE_INPUT_NODE_ID = "74"     # Node ID for loading the main input image
PROMPT_NODE_ID = "227"         # Node ID for the positive prompt input
SCENE_MASK_NODE_ID = "190"     # Node ID for loading the scene mask (can be None or "" to disable)
SUBTITLE_MASK_NODE_ID = "229"  # Node ID for loading the subtitle mask (can be None or "" to disable)

# --- Execution Control ---
MAX_WAIT_TIME = 360             # Max seconds to wait for a ComfyUI task to complete
MASKS_SERVER_SUBFOLDER = "clipspace" # Subfolder on ComfyUI server to upload masks into
DELAY_BETWEEN_SUBMISSIONS = 0.05 # Seconds delay between submitting tasks to the thread pool (0 for none)
VERBOSE_LOGGING = False         # Enable detailed logs (True/False)
# --------------- End of Configuration ---------------

def generate_anime_prompt_wrapper(image_path: str, log_func_info, log_func_error, log_func_verbose) -> str | None:
    """
    Wrapper for the AI prompt generation function with error handling and status reporting.
    Uses the globally available 'original_generate_anime_prompt'.
    Returns the generated prompt string or None on failure.
    """
    if not PROMPT_REASONING_AVAILABLE:
        log_func_info(f"  跳过AI提示词生成 ('prompt_reasoning' 未加载) for '{os.path.basename(image_path)}'.")
        return None

    base_image_name = os.path.basename(image_path)
    def status_reporter(message: str):
        log_func_verbose(f"    [提示API状态:{base_image_name}] {message}")

    log_func_info(f"  -> 正在为 '{base_image_name}' 请求AI生成提示词...")
    try:
        prompt = original_generate_anime_prompt(image_path, status_callback=status_reporter)
        if prompt:
            max_prompt_log_length = 120
            logged_prompt = prompt.replace('\n', ' ') # Log on single line
            if len(logged_prompt) > max_prompt_log_length:
                logged_prompt = logged_prompt[:max_prompt_log_length] + "..."
            log_func_info(f"  <- AI提示词 ({base_image_name}): \"{logged_prompt}\"")
        else:
            # Handle case where the function returns None without error (e.g., API couldn't process)
            log_func_info(f"  <- AI未能为 '{base_image_name}' 生成提示词 (返回 None)。将使用默认值。")
        return prompt # Return the prompt or None
    except PromptGenerationError as pge:
        log_func_error(f"    [提示生成失败] 图像 '{base_image_name}': {pge}")
        return None
    except Exception as e:
        log_func_error(f"    [提示生成意外错误] 图像 '{base_image_name}': 调用提示函数时发生 {type(e).__name__}: {e}")
        return None


class ComfyUITester:
    """Handles communication and task processing with a ComfyUI server instance."""
    def __init__(self, server_address, workflow_file_path, output_folder, context_info="", verbose=VERBOSE_LOGGING):
        self.server_address = server_address.rstrip('/')
        self.api_url = self.server_address # Base URL for ComfyUI API
        self.workflow_file_path = workflow_file_path # Path to the base workflow JSON
        self.output_folder = output_folder
        self.client_id = str(uuid.uuid4()) # Unique ID for this session
        self.context_info = context_info # Info for log messages (server, task ID, etc.)
        self.verbose = verbose

    def _print_message(self, level_prefix, message):
        """Unified message output using tqdm.write to avoid progress bar conflicts."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        tqdm.write(f"{timestamp} {level_prefix} [{self.context_info} - Client {self.client_id[:6]}] {message}")

    def _log_verbose(self, message):
        if self.verbose:
            self._print_message("详细", message)

    def _log_info(self, message):
        self._print_message("信息", message)

    def _log_error(self, message):
        self._print_message("错误", message)

    def load_workflow(self) -> dict | None:
        """Loads the base workflow JSON file."""
        try:
            if not os.path.exists(self.workflow_file_path):
                self._log_error(f"基础工作流文件未找到: {self.workflow_file_path}")
                return None
            with open(self.workflow_file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            self._log_verbose(f"  成功加载基础工作流: {os.path.basename(self.workflow_file_path)}")
            return workflow_data
        except json.JSONDecodeError as json_err:
            self._log_error(f"加载基础工作流 '{os.path.basename(self.workflow_file_path)}' 失败: JSON 格式错误 - {json_err}")
            return None
        except Exception as e:
            self._log_error(f"加载基础工作流 '{os.path.basename(self.workflow_file_path)}' 失败: {type(e).__name__}: {e}")
            return None

    def _upload_single_image(self, image_path: str, subfolder: str = "", image_type_for_log: str = "图像") -> str | None:
        """Uploads a single image file to the ComfyUI server."""
        if not os.path.exists(image_path):
            # Log as error because the calling function expected the file
            self._log_error(f"无法上传，{image_type_for_log}文件不存在: {image_path}")
            return None

        filename = os.path.basename(image_path)
        upload_url = f"{self.api_url}/upload/image"
        log_subfolder_text = f"到服务器子文件夹 '{subfolder}'" if subfolder else "到服务器根目录"
        self._log_verbose(f"    准备上传 {image_type_for_log}: '{filename}' ({log_subfolder_text})")

        try:
            with open(image_path, 'rb') as f:
                # Determine mime type based on extension for robustness
                _, ext = os.path.splitext(filename.lower())
                mime_type = 'image/png' # Default
                if ext == '.jpg' or ext == '.jpeg':
                    mime_type = 'image/jpeg'
                elif ext == '.webp':
                    mime_type = 'image/webp'

                files = {'image': (filename, f, mime_type)}
                data = {'overwrite': 'true'} # Always overwrite existing files on server
                if subfolder:
                    data['subfolder'] = subfolder

                response = requests.post(upload_url, files=files, data=data, timeout=120) # 120s timeout for upload
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                upload_data = response.json()

                server_filename = upload_data.get('name')
                server_subfolder = upload_data.get('subfolder', '') # Get subfolder returned by server

                if not server_filename:
                    self._log_error(f"{image_type_for_log} '{filename}' 上传成功，但服务器响应中缺少'name'字段。")
                    return None

                # Construct the reference path as ComfyUI expects it (subfolder/filename)
                final_image_reference = f"{server_subfolder}/{server_filename}" if server_subfolder else server_filename
                self._log_verbose(f"    {image_type_for_log} '{filename}' 上传成功: '{final_image_reference}'")
                return final_image_reference

        except requests.exceptions.HTTPError as http_err:
            self._log_error(f"{image_type_for_log} '{filename}' 上传时发生 HTTP 错误: {http_err}")
            if hasattr(http_err, 'response') and http_err.response is not None:
                try:
                    # Attempt to log server response body for debugging
                    self._log_verbose(f"      服务器响应 ({http_err.response.status_code}): {http_err.response.text[:500]}...")
                except Exception: pass # Ignore errors during response logging
        except requests.exceptions.RequestException as req_err:
            self._log_error(f"处理{image_type_for_log} '{filename}' 上传时发生网络请求错误: {req_err}")
        except IOError as io_err:
             self._log_error(f"读取文件 '{image_path}' 时发生 IO 错误: {io_err}")
        except Exception as e:
            self._log_error(f"处理{image_type_for_log} '{filename}' 上传时发生意外错误: {type(e).__name__}: {e}")

        return None # Return None if any error occurred

    def upload_main_image(self, image_path: str) -> str | None:
        """Uploads the main input image to the server's root input directory."""
        return self._upload_single_image(image_path, subfolder="", image_type_for_log="主图像")

    def upload_specific_mask(self, mask_local_path: str, server_target_subfolder: str, mask_type_log: str) -> str | None:
        """Uploads a mask file to a specific subfolder on the server."""
        # Check existence before calling the generic upload function
        if not mask_local_path or not os.path.exists(mask_local_path):
            # This might be expected (e.g., scene mask not existing), log verbosely
            self._log_verbose(f"提示: {mask_type_log} 路径未提供或文件不存在 ('{mask_local_path}'). 跳过上传。")
            return None
        return self._upload_single_image(mask_local_path, subfolder=server_target_subfolder, image_type_for_log=mask_type_log)

    def update_workflow(self, workflow: dict, main_image_ref: str, generated_prompt: str | None,
                        scene_mask_ref: str | None, subtitle_mask_ref: str | None,
                        shot_folder_name: str) -> dict | None:
        """
        Updates the loaded workflow dictionary with dynamic inputs based on the current task.
        Modifies: Main image, prompt, scene mask, subtitle mask, lora_02/strength_02, and sampler seeds.
        Returns the modified workflow dictionary or None if a critical error occurs.
        """
        if not workflow:
            self._log_error("update_workflow 收到无效的工作流 (None)。")
            return None
        # Create a deep copy to avoid modifying the original template unintentionally
        modified_workflow = json.loads(json.dumps(workflow))

        # --- Update Main Image Input ---
        if IMAGE_INPUT_NODE_ID in modified_workflow:
            # Check if node structure is as expected
            if "inputs" in modified_workflow[IMAGE_INPUT_NODE_ID] and "image" in modified_workflow[IMAGE_INPUT_NODE_ID]["inputs"]:
                 modified_workflow[IMAGE_INPUT_NODE_ID]["inputs"]["image"] = main_image_ref
                 self._log_verbose(f"    更新主图像节点 '{IMAGE_INPUT_NODE_ID}' 为: '{main_image_ref}'")
            else:
                 self._log_error(f"主图像节点 '{IMAGE_INPUT_NODE_ID}' 结构不正确 (缺少 'inputs' 或 'image' 键)。")
                 return None # Critical failure
        else:
            self._log_error(f"主图像节点 ID '{IMAGE_INPUT_NODE_ID}' 在工作流中未找到。")
            return None # Critical failure

        # --- Update Prompt Input ---
        if PROMPT_NODE_ID in modified_workflow:
             # Check if node structure is as expected
             if "inputs" in modified_workflow[PROMPT_NODE_ID] and "text" in modified_workflow[PROMPT_NODE_ID]["inputs"]:
                if generated_prompt:
                    modified_workflow[PROMPT_NODE_ID]["inputs"]["text"] = generated_prompt
                    self._log_verbose(f"    更新提示词节点 '{PROMPT_NODE_ID}' 为AI生成内容。")
                else:
                    # Use default prompt from the loaded workflow JSON
                    default_prompt = modified_workflow[PROMPT_NODE_ID]["inputs"].get("text", "未定义默认提示")
                    self._log_verbose(f"    未提供AI提示词。节点 '{PROMPT_NODE_ID}' 使用工作流默认值: '{default_prompt[:50]}...'")
             else:
                 self._log_verbose(f"警告: 提示词节点 '{PROMPT_NODE_ID}' 结构不正确 (缺少 'inputs' 或 'text' 键)。")
        else:
             # If prompt node is optional, only log verbosely
             self._log_verbose(f"提示: 提示词节点 ID '{PROMPT_NODE_ID}' 在工作流中未找到。")

        # --- Update Scene Mask Input ---
        if SCENE_MASK_NODE_ID: # Only process if the node ID is configured
            if SCENE_MASK_NODE_ID in modified_workflow:
                 # Check node structure
                 if "inputs" in modified_workflow[SCENE_MASK_NODE_ID] and "image" in modified_workflow[SCENE_MASK_NODE_ID]["inputs"]:
                    if scene_mask_ref: # If mask was uploaded successfully
                        modified_workflow[SCENE_MASK_NODE_ID]["inputs"]["image"] = scene_mask_ref
                        self._log_verbose(f"    更新场景蒙版节点 '{SCENE_MASK_NODE_ID}' 为: '{scene_mask_ref}'")
                    else: # Mask not provided or upload failed
                        # Keep the default value from the workflow (might be 'None' or a placeholder)
                        default_mask = modified_workflow[SCENE_MASK_NODE_ID]["inputs"].get("image", "未定义默认蒙版")
                        self._log_verbose(f"    未提供场景蒙版引用。节点 '{SCENE_MASK_NODE_ID}' 使用工作流默认值: '{default_mask}'")
                 else:
                     self._log_verbose(f"警告: 场景蒙版节点 '{SCENE_MASK_NODE_ID}' 结构不正确 (缺少 'inputs' 或 'image' 键)。")
            else:
                 self._log_verbose(f"警告: 场景蒙版节点 ID '{SCENE_MASK_NODE_ID}' 已配置，但在工作流中未找到。")

        # --- Update Subtitle Mask Input ---
        if SUBTITLE_MASK_NODE_ID: # Only process if the node ID is configured
            if SUBTITLE_MASK_NODE_ID in modified_workflow:
                 # Check node structure
                 if "inputs" in modified_workflow[SUBTITLE_MASK_NODE_ID] and "image" in modified_workflow[SUBTITLE_MASK_NODE_ID]["inputs"]:
                    if subtitle_mask_ref: # If mask was uploaded successfully
                        modified_workflow[SUBTITLE_MASK_NODE_ID]["inputs"]["image"] = subtitle_mask_ref
                        self._log_verbose(f"    更新字幕蒙版节点 '{SUBTITLE_MASK_NODE_ID}' 为: '{subtitle_mask_ref}'")
                    else: # Mask not provided or upload failed
                        default_mask = modified_workflow[SUBTITLE_MASK_NODE_ID]["inputs"].get("image", "未定义默认蒙版")
                        self._log_verbose(f"    未提供字幕蒙版引用。节点 '{SUBTITLE_MASK_NODE_ID}' 使用工作流默认值: '{default_mask}'")
                 else:
                    self._log_verbose(f"警告: 字幕蒙版节点 '{SUBTITLE_MASK_NODE_ID}' 结构不正确 (缺少 'inputs' 或 'image' 键)。")
            else:
                 self._log_verbose(f"警告: 字幕蒙版节点 ID '{SUBTITLE_MASK_NODE_ID}' 已配置，但在工作流中未找到。")

        # --- Update LoRA Node (lora_02 and strength_02) ---
        if LORA_NODE_ID in modified_workflow:
            lora_node = modified_workflow[LORA_NODE_ID]
            # Check structure before accessing inputs
            if "inputs" in lora_node and "lora_02" in lora_node["inputs"] and "strength_02" in lora_node["inputs"]:
                if shot_folder_name in LORA_MAPPING:
                    lora_file_to_use = LORA_MAPPING[shot_folder_name]
                    # Handle the special "00" or explicit "None" mapping
                    if shot_folder_name == "00" or lora_file_to_use == "None":
                        lora_node["inputs"]["lora_02"] = "None"
                        lora_node["inputs"]["strength_02"] = 0.0
                        self._log_verbose(f"    更新 LoRA 节点 '{LORA_NODE_ID}': lora_02='None', strength_02=0.0 (镜头 '{shot_folder_name}')")
                    else:
                        # Use the mapped LoRA file and default strength
                        lora_node["inputs"]["lora_02"] = lora_file_to_use
                        lora_node["inputs"]["strength_02"] = DEFAULT_LORA_STRENGTH_02
                        self._log_verbose(f"    更新 LoRA 节点 '{LORA_NODE_ID}': lora_02='{lora_file_to_use}', strength_02={DEFAULT_LORA_STRENGTH_02} (镜头 '{shot_folder_name}')")
                else:
                    # Shot folder not found in mapping, use defaults from workflow JSON
                    default_lora = lora_node["inputs"].get("lora_02", "未定义")
                    default_strength = lora_node["inputs"].get("strength_02", "未定义")
                    self._log_verbose(f"警告: 镜头 '{shot_folder_name}' 在 LORA_MAPPING 中未找到。节点 '{LORA_NODE_ID}' 使用工作流默认值 (LoRA: '{default_lora}', Strength: {default_strength})。")
            else:
                # Log error if structure is wrong, but don't necessarily stop the whole process unless critical
                self._log_error(f"LoRA 节点 '{LORA_NODE_ID}' 结构不正确 (缺少 'inputs', 'lora_02', 或 'strength_02' 键)。无法动态调整 LoRA。")
        else:
            # If LoRA node is optional or not always present, just log verbosely
            self._log_verbose(f"提示: LoRA 节点 ID '{LORA_NODE_ID}' 在工作流中未找到。跳过 LoRA 调整。")

        # --- Update Random Seeds for Samplers ---
        random_seed = random.randint(0, 2**32 - 1) # Generate one seed per workflow execution
        sampler_updated_count = 0
        for node_id, node_data in modified_workflow.items():
            # Check for common sampler node types
            if "class_type" in node_data and ("KSampler" in node_data["class_type"] or "SamplerCustom" in node_data["class_type"]):
                if "inputs" in node_data and "seed" in node_data["inputs"]:
                    node_data["inputs"]["seed"] = random_seed
                    self._log_verbose(f"    更新 Sampler 节点 '{node_id}' 种子为: {random_seed}")
                    sampler_updated_count += 1
                elif "inputs" in node_data and "noise_seed" in node_data["inputs"]: # Some samplers use 'noise_seed'
                     node_data["inputs"]["noise_seed"] = random_seed
                     self._log_verbose(f"    更新 Sampler 节点 '{node_id}' 噪声种子为: {random_seed}")
                     sampler_updated_count += 1

        if sampler_updated_count == 0:
            self._log_verbose(f"警告: 工作流中未找到任何已知 Sampler 节点来更新种子。")
        elif sampler_updated_count > 1 and self.verbose:
             self._log_verbose(f"注意: 工作流中更新了 {sampler_updated_count} 个 Sampler 节点的种子。")

        return modified_workflow # Return the fully updated workflow

    def send_prompt(self, workflow: dict) -> dict | None:
        """Sends the prepared workflow (prompt) to the ComfyUI server's /prompt endpoint."""
        if not workflow:
            self._log_error("send_prompt 收到无效的工作流 (None)。")
            return None

        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        payload_dict = {'prompt': workflow, 'client_id': self.client_id}
        try:
            data = json.dumps(payload_dict)
        except TypeError as json_err:
             self._log_error(f"无法序列化工作流为 JSON: {json_err}")
             return None

        prompt_url = f"{self.api_url}/prompt"
        try:
            response = requests.post(prompt_url, headers=headers, data=data, timeout=60) # 60s timeout for submission
            response.raise_for_status()
            response_data = response.json()
            if 'prompt_id' not in response_data:
                 self._log_error(f"提交工作流成功，但响应缺少 'prompt_id': {response_data}")
                 return None
            return response_data # Should contain 'prompt_id', 'number', 'node_errors'
        except requests.exceptions.HTTPError as http_err:
            self._log_error(f"提交工作流到 {prompt_url} 时发生 HTTP 错误: {http_err}")
            if hasattr(http_err, 'response') and http_err.response is not None:
                try: self._log_verbose(f"      服务器响应 ({http_err.response.status_code}): {http_err.response.text[:500]}...")
                except Exception: pass
            return None
        except requests.exceptions.RequestException as req_err:
            self._log_error(f"提交工作流到 {prompt_url} 时发生网络请求错误: {req_err}")
            return None
        except Exception as e:
            self._log_error(f"提交工作流到 {prompt_url} 时发生意外错误: {type(e).__name__}: {e}")
            return None

    def get_history(self, prompt_id: str) -> dict | None:
        """Retrieves the execution history/status for a given prompt ID."""
        if not prompt_id: return None
        history_url = f"{self.api_url}/history/{prompt_id}"
        try:
            response = requests.get(history_url, headers={'Accept': 'application/json'}, timeout=30) # 30s timeout for history check
            response.raise_for_status()
            return response.json() # Returns the history object for the prompt_id
        except requests.exceptions.RequestException as e:
            # Network/HTTP errors during polling are common, only log verbosely
            if self.verbose: self._log_verbose(f"获取提示 {prompt_id} 的历史时出错 (网络/HTTP): {e}")
            return None # Indicate failure to get history
        except Exception as e:
            if self.verbose: self._log_verbose(f"获取提示 {prompt_id} 的历史时发生意外错误: {type(e).__name__}: {e}")
            return None

    def download_output_images(self, history: dict, prompt_id: str, output_dir_for_run: str,
                               original_image_basename: str, current_iteration_num: int) -> list[str]:
        """
        Downloads images marked as 'output' type from the completed task's history.
        Returns a list of absolute paths to the downloaded files.
        """
        downloaded_files_paths = []
        if not history or prompt_id not in history:
            self._log_verbose("  下载失败：未找到有效的执行历史。")
            return downloaded_files_paths

        # The history for a prompt_id contains node outputs
        history_entry = history.get(prompt_id, {})
        outputs = history_entry.get('outputs', {})
        if not outputs:
            self._log_verbose(f"  提示 {prompt_id} 的历史中未找到 'outputs' 字段。")
            return downloaded_files_paths

        images_to_download = []
        # Iterate through each node's output in the history
        for node_id, node_output in outputs.items():
            if isinstance(node_output, dict) and 'images' in node_output and isinstance(node_output['images'], list):
                for img_data in node_output['images']:
                    # Check if it's a dictionary and has type 'output'
                    if isinstance(img_data, dict) and img_data.get('type') == 'output':
                        # Check if essential keys exist
                        if 'filename' in img_data:
                             images_to_download.append(img_data)
                        else:
                            self._log_verbose(f"    在节点 {node_id} 的输出中找到 'output' 图像，但缺少 'filename' 字段: {img_data}")

        if not images_to_download:
            self._log_verbose(f"  在提示 {prompt_id} 的工作流输出中未找到标记为 'output' 类型的图像。")
            return downloaded_files_paths

        self._log_verbose(f"    准备从服务器下载 {len(images_to_download)} 张 'output' 图像 (提示ID: {prompt_id})")

        original_base, original_ext = os.path.splitext(original_image_basename)
        # Ensure original_ext starts with a dot, default to .png if missing
        if not original_ext: original_ext = '.png'
        elif not original_ext.startswith('.'): original_ext = '.' + original_ext

        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3] # Millisecond timestamp

        for idx, image_data in enumerate(images_to_download):
            server_filename = image_data.get('filename')
            subfolder = image_data.get('subfolder', '') # Server subfolder (usually empty for outputs)
            # image_type = image_data.get('type') # Should be 'output'

            if not server_filename: # Should have been filtered already, but double-check
                self._log_verbose(f"    跳过索引 {idx} 处无文件名的图像数据。")
                continue

            # Construct local filename
            name_part_to_use = original_base
            # Add index if multiple output images from the same original input
            if len(images_to_download) > 1:
                name_part_to_use = f"{original_base}_output_{idx}"

            # Add iteration number if applicable
            if current_iteration_num > 1:
                final_local_filename = f"{name_part_to_use}-iter{current_iteration_num}_{timestamp_str}{original_ext}"
            else: # First iteration or single run
                final_local_filename = f"{name_part_to_use}_{timestamp_str}{original_ext}"

            local_path = os.path.join(output_dir_for_run, final_local_filename)

            # Prepare parameters for the /view API endpoint
            # 'type' might be needed depending on server config (temp vs output)
            url_params = {'filename': server_filename, 'type': 'output'}
            if subfolder: url_params['subfolder'] = subfolder

            view_url = f"{self.api_url}/view"
            try:
                self._log_verbose(f"      下载: 服务器文件 '{server_filename}' (类型: output, 子目录: '{subfolder or '无'}') 到 '{final_local_filename}'")
                # Use stream=True for potentially large images
                response = requests.get(view_url, params=url_params, stream=True, timeout=180) # 180s timeout for download
                response.raise_for_status()

                # Write the image content to the local file
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192): # Read in chunks
                        f.write(chunk)
                self._log_verbose(f"      已保存到: {local_path}")
                downloaded_files_paths.append(os.path.abspath(local_path)) # Store absolute path

            except requests.exceptions.HTTPError as http_err:
                self._log_error(f"下载图像 '{server_filename}' 时发生 HTTP 错误 ({http_err.response.status_code}): {http_err}")
            except requests.exceptions.RequestException as req_err:
                self._log_error(f"下载图像 '{server_filename}' 时发生网络请求错误: {req_err}")
            except IOError as io_err:
                 self._log_error(f"写入文件 '{local_path}' 时发生 IO 错误: {io_err}")
            except Exception as e:
                self._log_error(f"下载或保存图像 '{server_filename}' 时出错: {type(e).__name__}: {e}")

        if not downloaded_files_paths and images_to_download:
             # Logged as warning because workflow might have succeeded, just download failed
             self._log_verbose(f"  警告：找到 {len(images_to_download)} 张图像数据但未能下载任何图像 ({original_image_basename})。")

        return downloaded_files_paths

    def wait_for_completion(self, prompt_id: str) -> tuple[bool, dict | None, float]:
        """
        Waits for the ComfyUI task associated with prompt_id to complete by polling the /history endpoint.
        Returns: (success_status, final_history_object | None, time_spent_waiting)
        """
        if not prompt_id:
            return False, None, 0.0

        log_prefix_short = f"提示ID:{prompt_id[:6]}" # Short ID for logs
        start_time_wait = time.time()
        last_log_time = 0
        server_ip_short = self.server_address.split('//')[-1].split(':')[0] # Extract IP for logging

        self._log_verbose(f"{log_prefix_short} 监视服务器 {server_ip_short} 上的任务...")

        while True:
            current_loop_time = time.time()
            elapsed_wait = current_loop_time - start_time_wait

            # Check for timeout
            if elapsed_wait >= MAX_WAIT_TIME:
                self._log_error(f"任务 {log_prefix_short} (Srv {server_ip_short}) 超时 ({MAX_WAIT_TIME}秒)。")
                return False, None, elapsed_wait

            # Get the latest history for the prompt
            history_response = self.get_history(prompt_id)

            if history_response and prompt_id in history_response:
                status_obj = history_response[prompt_id]
                status_info = status_obj.get("status", {})
                status_str = status_info.get("status_str", "未知状态")
                # Check the 'completed' flag provided by newer ComfyUI versions
                exec_completed = status_info.get("completed", False)
                outputs_exist = bool(status_obj.get("outputs")) # Check if the 'outputs' key exists
                q_rem = status_info.get("exec_info", {}).get("queue_remaining", "N/A")

                # Log progress periodically or if status changes
                if self.verbose and (current_loop_time - last_log_time >= 15.0 or status_str not in ['running', 'pending']):
                    self._log_verbose(f"    [{log_prefix_short} on Srv {server_ip_short}] 状态: {status_str}, 完成标志: {exec_completed}, 队列: {q_rem}, 等待: {elapsed_wait:.0f}s")
                    last_log_time = current_loop_time

                # --- Check for Completion Status ---
                # Use 'exec_completed' flag as primary indicator if available
                if exec_completed:
                    if status_str == 'success':
                        # Double check if outputs exist, log warning if not
                        if not outputs_exist:
                             self._log_error(f"  任务 {log_prefix_short} (Srv {server_ip_short}) 状态为 'success' 且完成标志为 True，但历史记录缺少 'outputs'。 API 等待: {elapsed_wait:.2f}秒")
                             # Treat as failure if outputs are missing despite success status
                             return False, history_response, elapsed_wait
                        else:
                            self._log_info(f"  任务 {log_prefix_short} (Srv {server_ip_short}) 成功完成。状态: {status_str}, API 等待: {elapsed_wait:.2f}秒")
                            return True, history_response, elapsed_wait
                    else: # Completed, but status is not 'success' (e.g., error, failed)
                        self._log_error(f"  任务 {log_prefix_short} (Srv {server_ip_short}) 完成但状态为失败/错误。状态: {status_str}, API 等待: {elapsed_wait:.2f}秒")
                        # Log node-specific errors if available
                        if outputs_exist:
                            for node_id_err, node_output in status_obj["outputs"].items():
                                if isinstance(node_output, dict) and 'errors' in node_output:
                                    self._log_error(f"    {log_prefix_short} 节点 {node_id_err} 报告错误: {node_output['errors']}")
                        return False, history_response, elapsed_wait

            else: # Failed to get history (network error, server busy, prompt ID not found yet)
                if self.verbose and (current_loop_time - last_log_time >= 10.0): # Log polling failures less frequently
                    self._log_verbose(f"    [{log_prefix_short} on Srv {server_ip_short}] API 轮询: 无法获取历史。已等待:{elapsed_wait:.1f}s")
                    last_log_time = current_loop_time

            # Wait before the next poll
            time.sleep(1.0) # Polling interval

        # This part should ideally not be reached due to the timeout check inside the loop
        # return False, None, time.time() - start_time_wait


    def process_image(self, main_image_path: str, scene_mask_local_path: str | None, subtitle_mask_local_path: str | None,
                      original_image_basename: str, current_iteration_num: int, shot_folder_name: str) -> tuple[list[str], float]:
        """
        Processes a single image through the entire pipeline:
        Load WF -> Upload -> Gen Prompt -> Update WF -> Submit -> Wait -> Download.
        Returns a tuple: (list_of_downloaded_image_paths, total_task_duration).
        """
        task_start_time = time.time()
        self._log_info(f"开始处理: '{original_image_basename}' (迭代 {current_iteration_num}, 镜头 {shot_folder_name})")

        # --- 1. Load Base Workflow ---
        workflow_template = self.load_workflow()
        if not workflow_template:
            return [], time.time() - task_start_time # Return empty list and duration on failure

        # --- 2. Upload Main Image ---
        uploaded_main_image_ref = self.upload_main_image(main_image_path)
        if not uploaded_main_image_ref:
             self._log_error(f"主图像上传失败 for '{original_image_basename}'. 中止处理。")
             return [], time.time() - task_start_time

        # --- 3. Upload Scene Mask (Conditional) ---
        uploaded_scene_mask_ref = None
        # Check if scene masks are enabled globally (node ID configured)
        if SCENE_MASK_NODE_ID:
            # Check if this specific shot should skip the mask
            if shot_folder_name in SHOTS_TO_SKIP_SCENE_MASK:
                self._log_verbose(f"    镜头 '{shot_folder_name}' 在跳过列表 ({SHOTS_TO_SKIP_SCENE_MASK}) 中，跳过场景蒙版处理。")
            else:
                # Attempt to upload if path exists (upload_specific_mask handles non-existence)
                uploaded_scene_mask_ref = self.upload_specific_mask(scene_mask_local_path, MASKS_SERVER_SUBFOLDER, "场景蒙版")
                if not uploaded_scene_mask_ref and scene_mask_local_path and os.path.exists(scene_mask_local_path):
                    # Log warning only if the file existed but upload failed
                    self._log_verbose(f"警告: 场景蒙版 '{os.path.basename(scene_mask_local_path)}' 上传失败。")
        # else: Scene masks disabled globally, do nothing

        # --- 4. Upload Subtitle Mask (Conditional) ---
        uploaded_subtitle_mask_ref = None
        # Check if subtitle masks are enabled globally (node ID configured)
        if SUBTITLE_MASK_NODE_ID:
            # Attempt upload (upload_specific_mask handles non-existence of the global file)
            uploaded_subtitle_mask_ref = self.upload_specific_mask(subtitle_mask_local_path, MASKS_SERVER_SUBFOLDER, "字幕蒙版")
            if not uploaded_subtitle_mask_ref and subtitle_mask_local_path and os.path.exists(subtitle_mask_local_path):
                 # Log warning only if the file existed but upload failed
                 self._log_verbose(f"警告: 字幕蒙版 '{os.path.basename(subtitle_mask_local_path)}' 上传失败。")
        # else: Subtitle masks disabled globally, do nothing

        # --- 5. Generate AI Prompt (Optional) ---
        anime_prompt = generate_anime_prompt_wrapper(
            main_image_path,
            self._log_info,
            self._log_error,
            self._log_verbose
        )
        # Note: If generate_anime_prompt_wrapper returns None, update_workflow will handle it

        # --- 6. Update Workflow with Dynamic Values ---
        modified_workflow = self.update_workflow(
            workflow=workflow_template,
            main_image_ref=uploaded_main_image_ref,
            generated_prompt=anime_prompt,
            scene_mask_ref=uploaded_scene_mask_ref,
            subtitle_mask_ref=uploaded_subtitle_mask_ref,
            shot_folder_name=shot_folder_name
        )
        if not modified_workflow:
            self._log_error(f"更新工作流失败 for '{original_image_basename}'. 中止处理。")
            return [], time.time() - task_start_time

        # --- 7. Submit Workflow to ComfyUI ---
        prompt_response = self.send_prompt(modified_workflow)
        if not prompt_response or 'prompt_id' not in prompt_response:
            self._log_error(f"提交工作流失败 for '{original_image_basename}'. 中止处理。")
            # Attempt to log node errors if available in response
            if prompt_response and 'node_errors' in prompt_response:
                 self._log_error(f"  服务器报告节点错误: {prompt_response['node_errors']}")
            return [], time.time() - task_start_time
        prompt_id = prompt_response['prompt_id']
        self._log_info(f"  ComfyUI 任务已提交 (图像: {original_image_basename}, 提示ID: {prompt_id[:8]})")

        # --- 8. Wait for Task Completion ---
        completed, final_history, time_spent_waiting = self.wait_for_completion(prompt_id)
        task_duration = time.time() - task_start_time # Total time for this image

        # --- 9. Process Results (Download or Handle Failure) ---
        if completed and final_history:
            # Workflow completed successfully on the server, now download images
            downloaded_images = self.download_output_images(
                final_history, prompt_id, OUTPUT_FOLDER,
                original_image_basename, current_iteration_num
            )
            if downloaded_images:
                self._log_info(f"成功完成并下载 {len(downloaded_images)} 张图片 for '{original_image_basename}'。总耗时: {task_duration:.2f}s。")
                return downloaded_images, task_duration
            else:
                # Workflow succeeded, but download failed
                self._log_error(f"工作流成功但下载图片失败 (提示ID: {prompt_id}, {original_image_basename})。总耗时: {task_duration:.2f}s。")
                # Return empty list indicating failure for this task
                return [], task_duration
        else:
            # Task failed on server, timed out, or completed with error status
            # Errors should have been logged during wait_for_completion
            self._log_error(f"处理 '{original_image_basename}' 失败/超时 (提示ID: {prompt_id or 'N/A'})。总耗时: {task_duration:.2f}s (API 等待 {time_spent_waiting:.2f}s)。")
            return [], task_duration # Return empty list for failure


# --------------- Main Execution Logic ---------------
if __name__ == "__main__":
    # --- Initial Setup ---
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        print(f"输出将保存到: {os.path.abspath(OUTPUT_FOLDER)}")
    except OSError as e:
        print(f"严重错误: 无法创建输出目录 '{OUTPUT_FOLDER}': {e}")
        sys.exit(1)

    # --- Check Base Workflow ---
    base_workflow_full_path = os.path.join(WORKFLOW_BASE_DIR, BASE_WORKFLOW_FILENAME)
    if not os.path.exists(base_workflow_full_path):
        print(f"严重错误: 基础工作流文件 '{base_workflow_full_path}' 未找到。请确保它存在于 '{WORKFLOW_BASE_DIR}' 目录中。程序将退出。")
        sys.exit(1)
    else:
        print(f"将使用基础工作流: {base_workflow_full_path}")

    # --- Check Global Subtitle Mask (only if node is configured) ---
    global_subtitle_mask_path = None # Default to None
    if SUBTITLE_MASK_NODE_ID:
        potential_subtitle_mask_path = os.path.join(MASKS_BASE_DIR, SUBTITLE_MASK_FILENAME)
        if os.path.exists(potential_subtitle_mask_path):
            global_subtitle_mask_path = potential_subtitle_mask_path
            if VERBOSE_LOGGING: print(f"信息: 将使用全局字幕蒙版: {global_subtitle_mask_path}")
        else:
            # Subtitle node is configured, but file is missing
            print(f"警告: 字幕蒙版节点 '{SUBTITLE_MASK_NODE_ID}' 已配置，但全局字幕蒙版 '{potential_subtitle_mask_path}' 未找到。将不使用字幕蒙版。")
            # global_subtitle_mask_path remains None
    else:
        # Subtitle node not configured
        if VERBOSE_LOGGING: print(f"信息: 未配置字幕蒙版节点ID ({SUBTITLE_MASK_NODE_ID})，不加载全局字幕蒙版。")
        # global_subtitle_mask_path remains None

    # --- Prepare Task List ---
    overall_start_time = time.time()
    tasks_to_run = [] # List of tuples: (iter_num, scene_name, shot_name, img_name, scene_num_str)

    print("\n程序开始：扫描输入目录并准备任务列表...")
    total_files_scanned = 0
    total_images_found = 0
    scenes_found = set()
    shots_found = set()

    # Use try-except for directory scanning
    try:
        for iter_num_calc in range(1, NUM_ITERATIONS + 1):
            scene_folders_calc = sorted([
                d for d in os.listdir(BASE_INPUT_DIR)
                if os.path.isdir(os.path.join(BASE_INPUT_DIR, d)) and d.startswith("场景") # Assuming scene folders start with "场景"
            ])
            if not scene_folders_calc and iter_num_calc == 1:
                 print(f"警告 [扫描]: 在 '{BASE_INPUT_DIR}' 中未找到 '场景' 开头的文件夹。")

            for scene_folder_name_calc in scene_folders_calc:
                scenes_found.add(scene_folder_name_calc)
                scene_full_path_calc = os.path.join(BASE_INPUT_DIR, scene_folder_name_calc)
                # Extract scene number (digits) from folder name
                scene_num_match_calc = re.search(r'\d+', scene_folder_name_calc)
                scene_num_str_calc = scene_num_match_calc.group(0) if scene_num_match_calc else scene_folder_name_calc # Use full name if no digits

                shot_folders_calc = sorted([
                    d for d in os.listdir(scene_full_path_calc)
                    if os.path.isdir(os.path.join(scene_full_path_calc, d)) and d.isdigit() # Shot folders are purely digits
                ])
                if not shot_folders_calc and iter_num_calc == 1:
                    if VERBOSE_LOGGING: print(f"信息 [扫描]: 在 '{scene_full_path_calc}' 中未找到数字命名的镜头文件夹。")

                for shot_folder_name_calc in shot_folders_calc:
                    shots_found.add(f"{scene_folder_name_calc}/{shot_folder_name_calc}")
                    shot_images_dir_calc = os.path.join(scene_full_path_calc, shot_folder_name_calc)

                    image_files_calc = sorted([
                        f for f in os.listdir(shot_images_dir_calc)
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                    ])
                    total_files_scanned += len(os.listdir(shot_images_dir_calc)) # Count all files for info

                    if not image_files_calc:
                        if VERBOSE_LOGGING and iter_num_calc == 1: print(f"信息 [扫描]: 在 '{shot_images_dir_calc}' 中未找到图像文件。")
                        continue

                    # Check LoRA mapping during scan (only warn once per missing shot)
                    if iter_num_calc == 1 and shot_folder_name_calc not in LORA_MAPPING:
                         print(f"警告 [扫描]: 在 LORA_MAPPING 配置中未找到镜头 '{shot_folder_name_calc}' 的映射。将使用工作流默认 LoRA/强度。")

                    for image_filename_calc in image_files_calc:
                        total_images_found += 1
                        tasks_to_run.append((iter_num_calc, scene_folder_name_calc, shot_folder_name_calc, image_filename_calc, scene_num_str_calc))

    except FileNotFoundError:
        print(f"严重错误: 输入目录 '{BASE_INPUT_DIR}' 未找到。程序将退出。")
        sys.exit(1)
    except Exception as scan_err:
        print(f"严重错误: 扫描输入目录时发生错误: {scan_err}")
        sys.exit(1)


    total_tasks_to_process = len(tasks_to_run)
    print(f"扫描完成。找到 {len(scenes_found)} 个场景, {len(shots_found)} 个镜头。")
    print(f"总共扫描文件数 (所有类型): {total_files_scanned}") # Informative
    print(f"总共找到图像文件数: {total_images_found}") # Base images found
    print(f"总迭代次数: {NUM_ITERATIONS}")


    if total_tasks_to_process == 0:
        print("未找到任何需要处理的图像任务（图像文件数 x 迭代次数 = 0）。请检查输入目录结构和内容。程序将退出。")
        sys.exit(0)

    print(f"总共需要处理的任务数 (图像 x 迭代): {total_tasks_to_process}")
    print(f"将使用 {NUM_WORKERS} 个并发工作线程 (每个服务器一个)。")

    # Print LoRA logic confirmation
    print(f"\nLoRA 节点 '{LORA_NODE_ID}' 的 lora_02/strength_02 将根据镜头动态设置:")
    print(f"  - 镜头 '00' 或映射为 'None': lora_02='None', strength_02=0.0")
    print(f"  - 其他映射镜头: lora_02=映射文件, strength_02={DEFAULT_LORA_STRENGTH_02}")
    print(f"  - 未映射镜头: 使用工作流默认值")
    # Print Scene Mask skip confirmation
    if SHOTS_TO_SKIP_SCENE_MASK and SCENE_MASK_NODE_ID:
        print(f"将跳过以下镜头的场景蒙版处理 (如果蒙版存在): {SHOTS_TO_SKIP_SCENE_MASK}")
    elif not SCENE_MASK_NODE_ID:
         print(f"场景蒙版处理已禁用 (SCENE_MASK_NODE_ID 未配置)。")


    # --- Initialize Counters and Thread Pool ---
    tasks_completed_count = 0
    tasks_succeeded_count = 0 # Specifically counts tasks returning downloaded images
    tasks_failed_count = 0    # Counts tasks that error, timeout, or fail download
    individual_task_durations = [] # Store durations of successful tasks

    futures_map = {} # Dictionary to map Future objects back to task details
    print(f"\n--- 准备提交 {total_tasks_to_process} 个任务到线程池 ---")

    # --- ThreadPoolExecutor for Concurrency ---
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # --- Task Submission Loop ---
        for i, (iter_num, scene_folder_name, shot_folder_name, image_filename, scene_num_str) in enumerate(tasks_to_run):
            current_server_ip = SERVER_IPS[i % NUM_WORKERS] # Simple round-robin server assignment
            scene_full_path = os.path.join(BASE_INPUT_DIR, scene_folder_name)

            # --- Determine Scene Mask Path for this Task ---
            current_scene_mask_path = None
            if SCENE_MASK_NODE_ID: # Only look for mask if the node is configured
                potential_mask_path = os.path.join(MASKS_BASE_DIR, f"scene-{scene_num_str}-mask.png")
                # The mask path is passed even if it doesn't exist; upload function handles it
                current_scene_mask_path = potential_mask_path

            shot_images_dir = os.path.join(scene_full_path, shot_folder_name)
            full_image_path = os.path.join(shot_images_dir, image_filename)

            # --- Create Context String for Logging ---
            server_id_for_log = current_server_ip.split('.')[-1].split(':')[0] # Get last IP octet
            # Use a slightly more compact log context
            context_log = f"任务 {i+1}/{total_tasks_to_process} Srv{server_id_for_log} It{iter_num} Sc{scene_num_str}-Sh{shot_folder_name}"

            # --- Instantiate ComfyUITester for this task ---
            tester = ComfyUITester(
                server_address=current_server_ip,
                workflow_file_path=base_workflow_full_path, # Always use the base workflow
                output_folder=OUTPUT_FOLDER,
                context_info=context_log, # Pass the context for logging within the instance
                verbose=VERBOSE_LOGGING
            )

            # --- Submit the process_image method to the executor ---
            future = executor.submit(
                tester.process_image, # Method to run
                # Arguments for process_image:
                main_image_path=full_image_path,
                scene_mask_local_path=current_scene_mask_path, # Pass potential path
                subtitle_mask_local_path=global_subtitle_mask_path, # Pass checked global path
                original_image_basename=image_filename,
                current_iteration_num=iter_num,
                shot_folder_name=shot_folder_name # Crucial for LoRA logic
            )
            # Store relevant info to retrieve when the future completes
            futures_map[future] = (image_filename, context_log, shot_folder_name)

            # Optional delay to prevent flooding servers
            if DELAY_BETWEEN_SUBMISSIONS > 0:
                time.sleep(DELAY_BETWEEN_SUBMISSIONS)

        # --- Progress Reporting and Result Collection ---
        tqdm.write(f"\n所有 {total_tasks_to_process} 个任务已提交。开始处理并等待完成...\n")

        # Use tqdm with as_completed for a progress bar
        for future in tqdm(as_completed(futures_map),
                            total=total_tasks_to_process, # Total number of futures submitted
                            desc="处理图像",              # Progress bar description
                            unit="个任务",               # Unit displayed
                            ncols=120 if sys.stdout.isatty() else None, # Adjust width if possible
                            dynamic_ncols=True,          # Allow dynamic width adjustment
                            file=sys.stdout,             # Ensure output to standard out
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
                           ):

            img_fname, ctx_log, shot_name = futures_map[future] # Retrieve task info
            tasks_completed_count += 1 # Increment completion counter (success or fail)

            task_succeeded = False
            current_task_duration = 0.0

            try:
                # Get the result from the completed future
                # process_image returns (list_of_paths, duration)
                processed_results, task_duration = future.result()
                current_task_duration = task_duration # Store duration

                # Check if the result indicates success (non-empty list of downloaded files)
                if isinstance(processed_results, list) and len(processed_results) > 0:
                    tasks_succeeded_count += 1
                    individual_task_durations.append(task_duration)
                    task_succeeded = True
                else:
                    # Task completed but resulted in failure (error, timeout, download fail)
                    tasks_failed_count += 1
                    # Error messages should have been logged inside process_image/wait_for_completion

            except CancelledError:
                tasks_failed_count += 1
                tqdm.write(f"错误 [主循环]: 任务 {ctx_log} (图像: {img_fname}) 被工作线程取消。")
            except Exception as e: # Catch unexpected errors during future.result()
                tasks_failed_count += 1
                tqdm.write(f"严重错误 [主循环]: 获取任务 {ctx_log} (图像: {img_fname}) 结果时出错: {type(e).__name__}: {e}")
                # Duration might not be available if error occurred before return

            # --- Update Overall Progress Summary (using tqdm.write) ---
            current_elapsed_script_time = time.time() - overall_start_time
            # Calculate average speed based on successful tasks only
            avg_speed_successful = tasks_succeeded_count / current_elapsed_script_time if current_elapsed_script_time > 0 and tasks_succeeded_count > 0 else 0.0

            # Format duration string for the last completed task
            last_task_duration_info = ""
            if current_task_duration > 0:
                 status_indicator = "(成功)" if task_succeeded else "(失败)"
                 last_task_duration_info = f" 最近任务{status_indicator}耗时: {current_task_duration:.2f}s."

            avg_speed_str = f"{avg_speed_successful:.3f}" # Format speed
            # Construct the summary message for tqdm.write
            progress_summary = (
                f"进度: {tasks_completed_count}/{total_tasks_to_process}. "
                f"成功: {tasks_succeeded_count}, 失败: {tasks_failed_count}. "
                f"已运行: {time.strftime('%H:%M:%S', time.gmtime(current_elapsed_script_time))}. "
                f"平均成功速度: {avg_speed_str} 任务/秒.{last_task_duration_info}"
            )
            tqdm.write(progress_summary) # Print summary above the progress bar

    # --- Final Summary ---
    overall_end_time = time.time()
    total_script_duration_seconds = overall_end_time - overall_start_time

    print("\n") # Add spacing before final summary
    print(f"{'='*25} 所有处理已完成 {'='*25}")
    print(f"总迭代轮数: {NUM_ITERATIONS}")
    print(f"使用基础工作流: {BASE_WORKFLOW_FILENAME}")
    print(f"LoRA 节点 '{LORA_NODE_ID}' 已根据镜头动态调整。")
    if SHOTS_TO_SKIP_SCENE_MASK and SCENE_MASK_NODE_ID: print(f"已尝试跳过镜头 {SHOTS_TO_SKIP_SCENE_MASK} 的场景蒙版。")
    print("-" * 60)
    print(f"计划处理的任务总数: {total_tasks_to_process}")
    print(f"完成的任务总数 (成功+失败): {tasks_completed_count}")
    print(f"成功完成的任务数 (生成并下载): {tasks_succeeded_count}")
    print(f"失败/未下载的任务数: {tasks_failed_count}")

    if total_tasks_to_process > 0:
         # Calculate success rate based on successful tasks vs. total planned
         success_rate = (tasks_succeeded_count / total_tasks_to_process) * 100
         print(f"总体成功率 (基于计划任务): {success_rate:.2f}%")
    else:
        print("未计划任何任务。")

    # Calculate average duration only if there were successful tasks
    if individual_task_durations:
        avg_task_duration = sum(individual_task_durations) / len(individual_task_durations)
        print(f"单个成功任务的平均处理时间: {avg_task_duration:.2f} 秒")
        # Optional: Min/Max duration
        # print(f"  - 最快成功任务: {min(individual_task_durations):.2f} 秒")
        # print(f"  - 最慢成功任务: {max(individual_task_durations):.2f} 秒")
    elif tasks_succeeded_count == 0 and total_tasks_to_process > 0:
        print("没有成功处理的任务，无法计算平均处理时间。")

    # Calculate overall throughput based on successful tasks and total script time
    if total_script_duration_seconds > 0 and tasks_succeeded_count > 0:
        overall_throughput = tasks_succeeded_count / total_script_duration_seconds
        print(f"整体系统吞吐量 (基于成功任务): {overall_throughput:.3f} 任务/秒")

    print("-" * 60)
    print(f"脚本总执行时间: {time.strftime('%H:%M:%S', time.gmtime(total_script_duration_seconds))} ({total_script_duration_seconds:.2f} 秒)")
    print(f"所有成功生成的图像已保存到: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"{'='*60}")

# --- END OF FILE comfyui_redrawer_0508_final_unified_workflow.py ---