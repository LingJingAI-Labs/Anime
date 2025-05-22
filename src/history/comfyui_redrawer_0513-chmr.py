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

# --- 设置 Python Path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# --- 尝试导入 prompt_reasoning，处理未找到的情况 ---
try:
    from prompt_reasoning import generate_anime_prompt as original_generate_anime_prompt, PromptGenerationError
    PROMPT_REASONING_AVAILABLE = True
except ImportError:
    print("错误: 无法导入 'prompt_reasoning' 模块。AI提示生成将不可用。")
    print("请确保 'prompt_reasoning.py' 文件与此脚本位于同一目录或在 Python 路径中。")
    PROMPT_REASONING_AVAILABLE = False
    class PromptGenerationError(Exception): pass
    def original_generate_anime_prompt(*args, **kwargs):
        raise NotImplementedError("prompt_reasoning module not available")

# --------------- 配置参数 ---------------
SERVER_IPS = [
    "http://comfyui-demo.lingjingai.cn",
    "http://comfyui-demo2.lingjingai.cn",
    "http://comfyui-demo3.lingjingai.cn",
    "http://comfyui-demo4.lingjingai.cn",
    "http://comfyui-demo5.lingjingai.cn",
    "http://comfyui-demo6.lingjingai.cn",
    "http://comfyui-demo7.lingjingai.cn",
    "http://comfyui-demo8.lingjingai.cn",
    "http://comfyui-demo9.lingjingai.cn",
    "http://comfyui-demo10.lingjingai.cn",
    "http://comfyui-demo11.lingjingai.cn",
    "http://comfyui-demo12.lingjingai.cn",
    # 根据需要添加其他服务器IP
]
NUM_WORKERS = len(SERVER_IPS) # 并发工作线程数 = 服务器数量

NUM_ITERATIONS = 10 # 对所有图像的迭代次数

BASE_INPUT_DIR = "data/250513" # 包含场景文件夹的根目录 (主图的本地路径)

# --- MODIFIED: 本地 Mask 文件路径配置 ---
# 这些是 Mask 文件在您本地计算机上的路径。脚本会将它们上传到 ComfyUI 服务器。
BASE_MASK_DIR_LOCAL = "data/mask" # 本地 Mask 文件的基础目录
SCENE_MASK_LOCAL_FILENAME_TEMPLATE = "scene-{scene_num}-mask.png" # 本地场景 Mask 文件名模板，{scene_num} 会被替换 (如果 SCENE_MASK_NODE_ID 被使用)
SUBTITLE_MASK_LOCAL_FILENAME = "subtitle-mask.png"          # 本地字幕 Mask 的固定文件名。如果不用，设为 None 或 ""
MASK_UPLOAD_SUBFOLDER_ON_SERVER = "clipspace"          # Mask 在 ComfyUI 服务器 input 目录中上传到的子文件夹 (例如 "masks_from_script")。为空则上传到 input 根目录。
# --- End of Mask Filenames ---

WORKFLOW_BASE_DIR = "workflow" # 包含基础工作流 JSON 的目录 (本地路径)
OUTPUT_FOLDER = os.path.join(BASE_INPUT_DIR, "opt_auto") # 生成图像的输出目录 (本地路径)

# --- 统一工作流和动态 LoRA 配置 ---
BASE_WORKFLOW_FILENAME = "chmr-0513-base.json" # 用于所有任务的单个工作流文件
LORA_NODE_ID = "12" # 工作流中 "Lora Loader Stack" 节点的 ID
DEFAULT_LORA_STRENGTH_02 = 1.0 # 使用特定 LoRA 时 lora_02 的默认强度

# --- !! 人工编辑区: 定义镜头文件夹名称到 LoRA 文件名的映射 !! ---
LORA_MAPPING = {
    "00": "None",
    "01": "mori/char01秦云#/char01_V3.safetensors",
    "02": "mori/char02萧灵#/char02.safetensors",
    "03": "mori/char03蔡成安#/char03.safetensors",
    "04": "mori/char04蔡晓霞#/char04.safetensors",
    "07": "mori/char07张思思#/char07.safetensors",
    "09": "mori/char09虎哥#/char09.safetensors",
    "10": "mori/char10绿衣男#/char10.safetensors",
    "11": "mori/char11周学兵#/char11.safetensors",
    "12": "mori/char12售货员#/char12.safetensors",
    "13": "mori/char13王大妈#/char13.safetensors",
    "15": "mori/char15周雪#/char15.safetensors",
    "16": "mori/char16刘大爷#/char16.safetensors",
}
# --- !! 人工编辑区结束 !! ---

LORA_PROMPTS = { # 用于更新节点 "216" (或其他指定节点) 的文本输入
    "00": "",
    "01": "a man char01 with short black hair, wearing a brown jacket over a white t-shirt, Wearing a black watch with a white dial,",
    "02": "a woman char02, long black dress, bare_shoulders, ",
    "03": "a man char03, black mid-length suit, unbuttoned, white shirt, white tissue",
    "04": "A woman character 04, brown hair, gold necklace and earrings, red dress, bare shoulders.",
    "07": "This is an image of the woman cha07 wearing White skirt",
    "09": "This is a picture of old man char09, bald head, floral shirt, red suit, black pants",
    "10": "This is a picture of man char10, tender green long-sleeved double pocket shirt, black pants, yellow skin, round glasses, messy and fluffy curly hair, mature",
    "11": "This is a picture of man char11, bald head, (brown skin: 1.5), neck chain, camouflage vest inside, black sleeveless jacket, black work pants, brown coarse wrist short wrist guards exposed to the back of the hand",
    "12": "a woman char12, brown eyes, brown footwear, brown hair, braided hair, bangs, short sleeves, brown skirt, long skirt, white shirt",
    "13": "Here is a picture of an old woman named char13. She wears a brown-red cheongsam with a floral pattern. Her hair was tied in a high gray bun and her expression was arrogant",
    "15": "a woman char15, light brown short hair to shoulders, ugly, fat, small head, light and dark purple mid-length sleeve dress, pink and brown skin, bow on the chest",
    "16": "This is a picture of old man char16,gray shirt，Khaki pants",
}

# --- 定义哪些镜头跳过场景 Mask ---
SHOTS_TO_SKIP_SCENE_MASK = ["02"] # 如果 SCENE_MASK_NODE_ID 被使用，这些镜头将跳过场景Mask

# --- 工作流中的节点 ID ---
IMAGE_INPUT_NODE_ID = "74"     # 加载主输入图像 (上传的) 的节点 ID
PROMPT_NODE_ID = "146"         # Positive Prompt 输入的节点 ID (通常由AI生成提示或LoRA特定提示覆盖)
SCENE_MASK_NODE_ID = None      # 场景 Mask 节点 ID。如果工作流中无此节点或不想使用，请设为 None。原值 "190"
SUBTITLE_MASK_NODE_ID = "214"  # 其 'inputs.image' 将被设置为上传的字幕 Mask 的节点 ID
TEXT_MULTILINE_NODE_ID = "216" # 需要根据 LORA_PROMPTS 更新 'inputs.text' 的多行文本节点 ID

# --- 执行控制 ---
MAX_WAIT_TIME = 360             # 等待 ComfyUI 任务完成的最长时间 (秒)
DELAY_BETWEEN_SUBMISSIONS = 0.05 # 向线程池提交任务之间的延迟 (秒)，0 表示无延迟
VERBOSE_LOGGING = False         # 启用详细日志 (True/False)
# --------------- End of Configuration ---------------

def generate_anime_prompt_wrapper(image_path: str, log_func_info, log_func_error, log_func_verbose) -> str | None:
    """AI提示生成函数的包装器"""
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
            logged_prompt = prompt.replace('\n', ' ')
            if len(logged_prompt) > max_prompt_log_length:
                logged_prompt = logged_prompt[:max_prompt_log_length] + "..."
            log_func_info(f"  <- AI提示词 ({base_image_name}): \"{logged_prompt}\"")
        else:
            log_func_info(f"  <- AI未能为 '{base_image_name}' 生成提示词 (返回 None)。将使用默认值。")
        return prompt
    except PromptGenerationError as pge:
        log_func_error(f"    [提示生成失败] 图像 '{base_image_name}': {pge}")
        return None
    except Exception as e:
        log_func_error(f"    [提示生成意外错误] 图像 '{base_image_name}': 调用提示函数时发生 {type(e).__name__}: {e}")
        return None

class ComfyUITester:
    """处理与 ComfyUI 服务器实例的通信和任务处理。"""
    def __init__(self, server_address, workflow_file_path, output_folder, context_info="", verbose=VERBOSE_LOGGING):
        self.server_address = server_address.rstrip('/')
        self.api_url = self.server_address
        self.workflow_file_path = workflow_file_path
        self.output_folder = output_folder
        self.client_id = str(uuid.uuid4())
        self.context_info = context_info
        self.verbose = verbose

    def _print_message(self, level_prefix, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        tqdm.write(f"{timestamp} {level_prefix} [{self.context_info} - Client {self.client_id[:6]}] {message}")

    def _log_verbose(self, message):
        if self.verbose: self._print_message("详细", message)
    def _log_info(self, message): self._print_message("信息", message)
    def _log_error(self, message): self._print_message("错误", message)

    def load_workflow(self) -> dict | None:
        """加载基础工作流 JSON 文件。"""
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
        """上传单个图像文件到 ComfyUI 服务器。"""
        if not os.path.exists(image_path):
            self._log_error(f"无法上传，本地 {image_type_for_log} 文件不存在: {image_path}")
            return None
        filename = os.path.basename(image_path)
        upload_url = f"{self.api_url}/upload/image"
        log_subfolder_text = f"到服务器 input/{subfolder if subfolder else ''}" if subfolder else "到服务器 input 根目录"
        self._log_verbose(f"    准备上传 {image_type_for_log}: '{filename}' (从本地: '{image_path}') {log_subfolder_text}")
        try:
            with open(image_path, 'rb') as f:
                _, ext = os.path.splitext(filename.lower())
                mime_type = 'image/png'
                if ext == '.jpg' or ext == '.jpeg': mime_type = 'image/jpeg'
                elif ext == '.webp': mime_type = 'image/webp'
                files = {'image': (filename, f, mime_type)}
                data = {'overwrite': 'true'} # 通常建议为 true，以防重复上传同名文件
                if subfolder: data['subfolder'] = subfolder.strip('/') # 确保子文件夹路径格式正确
                
                response = requests.post(upload_url, files=files, data=data, timeout=120)
                response.raise_for_status()
                upload_data = response.json()
                server_filename = upload_data.get('name')
                server_subfolder = upload_data.get('subfolder', '')

                if not server_filename:
                    self._log_error(f"{image_type_for_log} '{filename}' 上传成功，但服务器响应中缺少'name'字段。")
                    return None
                
                final_image_reference = f"{server_subfolder.strip('/')}/{server_filename}" if server_subfolder else server_filename
                self._log_verbose(f"    {image_type_for_log} '{filename}' 上传成功，服务器引用: '{final_image_reference}'")
                return final_image_reference
        except requests.exceptions.HTTPError as http_err:
            self._log_error(f"{image_type_for_log} '{filename}' 上传时发生 HTTP 错误: {http_err}")
            if hasattr(http_err, 'response') and http_err.response is not None:
                try: self._log_verbose(f"      服务器响应 ({http_err.response.status_code}): {http_err.response.text[:500]}...")
                except Exception: pass
        except requests.exceptions.RequestException as req_err:
            self._log_error(f"处理{image_type_for_log} '{filename}' 上传时发生网络请求错误: {req_err}")
        except IOError as io_err:
             self._log_error(f"读取本地文件 '{image_path}' 时发生 IO 错误: {io_err}")
        except Exception as e:
            self._log_error(f"处理{image_type_for_log} '{filename}' 上传时发生意外错误: {type(e).__name__}: {e}")
        return None

    def upload_main_image(self, image_path: str) -> str | None:
        """上传主输入图像到服务器的 input 目录 (根目录或指定子目录)。"""
        return self._upload_single_image(image_path, subfolder="", image_type_for_log="主图像")

    def update_workflow(self, workflow: dict, main_image_ref: str, generated_prompt: str | None,
                        shot_folder_name: str, scene_num_str: str) -> dict | None:
        """
        更新加载的工作流字典。
        主图像, 提示词 (PROMPT_NODE_ID), LoRA, 种子会被更新。
        场景 Mask (如果 SCENE_MASK_NODE_ID 被配置且节点存在) 和字幕 Mask 将从本地上传 (如果配置且文件存在)，然后更新对应节点。
        新增：更新 TEXT_MULTILINE_NODE_ID 的文本内容。
        """
        if not workflow:
            self._log_error("update_workflow 收到无效的工作流 (None)。")
            return None
        modified_workflow = json.loads(json.dumps(workflow)) # 深拷贝

        # --- 更新主图像输入 (已上传的) ---
        if IMAGE_INPUT_NODE_ID in modified_workflow:
            if "inputs" in modified_workflow[IMAGE_INPUT_NODE_ID] and "image" in modified_workflow[IMAGE_INPUT_NODE_ID]["inputs"]:
                 modified_workflow[IMAGE_INPUT_NODE_ID]["inputs"]["image"] = main_image_ref
                 self._log_verbose(f"    更新主图像节点 '{IMAGE_INPUT_NODE_ID}' 为上传的: '{main_image_ref}'")
            else:
                 self._log_error(f"主图像节点 '{IMAGE_INPUT_NODE_ID}' 结构不正确。")
                 return None 
        else:
            self._log_error(f"主图像节点 ID '{IMAGE_INPUT_NODE_ID}' 在工作流中未找到。")
            return None

        # --- 更新AI生成的提示词输入 (PROMPT_NODE_ID) ---
        if PROMPT_NODE_ID in modified_workflow:
             if "inputs" in modified_workflow[PROMPT_NODE_ID] and "text" in modified_workflow[PROMPT_NODE_ID]["inputs"]:
                if generated_prompt: # AI生成的提示优先
                    modified_workflow[PROMPT_NODE_ID]["inputs"]["text"] = generated_prompt
                    self._log_verbose(f"    更新提示词节点 '{PROMPT_NODE_ID}' 为AI生成内容。")
                else: # AI未生成提示，保留工作流默认
                    default_prompt = modified_workflow[PROMPT_NODE_ID]["inputs"].get("text", "未定义默认提示")
                    self._log_verbose(f"    未提供AI提示词。节点 '{PROMPT_NODE_ID}' 使用工作流默认值: '{default_prompt[:50]}...'")
             else:
                 self._log_verbose(f"警告: 提示词节点 '{PROMPT_NODE_ID}' 结构不正确。")
        else:
             self._log_verbose(f"提示: 提示词节点 ID '{PROMPT_NODE_ID}' 在工作流中未找到。")

        # --- 更新场景 Mask  ---
        if SCENE_MASK_NODE_ID: 
            if SCENE_MASK_NODE_ID in modified_workflow:
                if "inputs" in modified_workflow[SCENE_MASK_NODE_ID] and "image" in modified_workflow[SCENE_MASK_NODE_ID]["inputs"]:
                    if shot_folder_name in SHOTS_TO_SKIP_SCENE_MASK:
                        default_mask_ref = modified_workflow[SCENE_MASK_NODE_ID]["inputs"].get("image", "未定义默认蒙版")
                        self._log_verbose(f"    镜头 '{shot_folder_name}' 在跳过列表。场景蒙版节点 '{SCENE_MASK_NODE_ID}' 保留工作流默认值: '{default_mask_ref}'")
                    else:
                        scene_mask_fname = SCENE_MASK_LOCAL_FILENAME_TEMPLATE.format(scene_num=scene_num_str)
                        uploaded_scene_mask_ref =  os.path.join('/data/comfyui/input/wuji/mask', scene_mask_fname)
                        modified_workflow[SCENE_MASK_NODE_ID]["inputs"]["image"] = uploaded_scene_mask_ref
                        self._log_verbose(f"    场景Mask节点 '{SCENE_MASK_NODE_ID}' 更新为硬编码路径: '{uploaded_scene_mask_ref}' (基于文件名 '{scene_mask_fname}')")
                else:
                     self._log_verbose(f"警告: 场景蒙版节点 '{SCENE_MASK_NODE_ID}' 结构不正确 (缺少 inputs 或 image 键)。")
            else:
                 self._log_verbose(f"警告: 场景蒙版节点 ID '{SCENE_MASK_NODE_ID}' 已配置，但在工作流中未找到。")
        else:
            self._log_verbose("提示: SCENE_MASK_NODE_ID 未配置或设为 None，跳过场景 Mask 处理。")

        # --- 更新字幕 Mask (从本地上传) ---
        if SUBTITLE_MASK_NODE_ID:
            if SUBTITLE_MASK_NODE_ID in modified_workflow:
                if "inputs" in modified_workflow[SUBTITLE_MASK_NODE_ID] and "image" in modified_workflow[SUBTITLE_MASK_NODE_ID]["inputs"]:
                    modified_workflow[SUBTITLE_MASK_NODE_ID]["inputs"]["image"] = "/data/comfyui/input/wuji/mask/subtitle-mask.png"
                    self._log_verbose(f"    字幕Mask节点 '{SUBTITLE_MASK_NODE_ID}' 更新为硬编码路径: '/data/comfyui/input/wuji/mask/subtitle-mask.png'")
                else:
                    self._log_verbose(f"警告: 字幕蒙版节点 '{SUBTITLE_MASK_NODE_ID}' 结构不正确 (缺少 inputs 或 image 键)。")
            else:
                 self._log_verbose(f"警告: 字幕蒙版节点 ID '{SUBTITLE_MASK_NODE_ID}' 已配置，但在工作流中未找到。")

        # --- 更新 LoRA 节点 ---
        if LORA_NODE_ID in modified_workflow:
            lora_node = modified_workflow[LORA_NODE_ID]
            if "inputs" in lora_node and "lora_02" in lora_node["inputs"] and "strength_02" in lora_node["inputs"]:
                if shot_folder_name in LORA_MAPPING:
                    lora_file_to_use = LORA_MAPPING[shot_folder_name]
                    if shot_folder_name == "00" or lora_file_to_use == "None":
                        lora_node["inputs"]["lora_02"] = "None"
                        lora_node["inputs"]["strength_02"] = 0.0
                        self._log_verbose(f"    更新 LoRA 节点 '{LORA_NODE_ID}': lora_02='None', strength_02=0.0 (镜头 '{shot_folder_name}')")
                    else:
                        lora_node["inputs"]["lora_02"] = lora_file_to_use
                        lora_node["inputs"]["strength_02"] = DEFAULT_LORA_STRENGTH_02
                        self._log_verbose(f"    更新 LoRA 节点 '{LORA_NODE_ID}': lora_02='{lora_file_to_use}', strength_02={DEFAULT_LORA_STRENGTH_02} (镜头 '{shot_folder_name}')")
                else:
                    default_lora = lora_node["inputs"].get("lora_02", "未定义")
                    default_strength = lora_node["inputs"].get("strength_02", "未定义")
                    self._log_verbose(f"警告: 镜头 '{shot_folder_name}' 在 LORA_MAPPING 中未找到。节点 '{LORA_NODE_ID}' 使用工作流默认值 (LoRA: '{default_lora}', Strength: {default_strength})。")
            else:
                self._log_error(f"LoRA 节点 '{LORA_NODE_ID}' 结构不正确。无法动态调整 LoRA。")
        else:
            self._log_verbose(f"提示: LoRA 节点 ID '{LORA_NODE_ID}' 在工作流中未找到。跳过 LoRA 调整。")

        # --- 新增: 更新多行文本节点 (TEXT_MULTILINE_NODE_ID) 的 'text' 输入 ---
        if TEXT_MULTILINE_NODE_ID in modified_workflow:
            text_node = modified_workflow[TEXT_MULTILINE_NODE_ID]
            if "inputs" in text_node and "text" in text_node["inputs"]:
                # 优先使用 LORA_PROMPTS 中与 shot_folder_name 匹配的条目
                # 如果 shot_folder_name 不在 LORA_PROMPTS 中, 则后备到 LORA_PROMPTS["00"] 的内容
                # 如果 LORA_PROMPTS["00"] 也不存在, 则后备到空字符串 ""
                prompt_text_to_use = LORA_PROMPTS.get(shot_folder_name, LORA_PROMPTS.get("00", ""))
                
                text_node["inputs"]["text"] = prompt_text_to_use
                
                log_prompt_text = prompt_text_to_use.replace('\n', ' ') # 确保日志中不换行
                if len(log_prompt_text) > 70: log_prompt_text = log_prompt_text[:70] + "..."

                if shot_folder_name in LORA_PROMPTS:
                    self._log_verbose(f"    更新多行文本节点 '{TEXT_MULTILINE_NODE_ID}' 的 'text' 为 (来自LORA_PROMPTS for '{shot_folder_name}'): \"{log_prompt_text}\"")
                else:
                    self._log_info(f"    镜头 '{shot_folder_name}' 在 LORA_PROMPTS 中未找到明确映射。多行文本节点 '{TEXT_MULTILINE_NODE_ID}' 的 'text' 已设置为 LORA_PROMPTS['00'] 的内容 (或空字符串): \"{log_prompt_text}\"")
            else:
                self._log_error(f"多行文本节点 '{TEXT_MULTILINE_NODE_ID}' 结构不正确 (缺少 'inputs' 或 'inputs.text')。无法更新文本。")
        else:
            self._log_verbose(f"提示: 多行文本节点 ID '{TEXT_MULTILINE_NODE_ID}' 在工作流中未找到。跳过其文本更新。")


        # --- 更新 Sampler 种子 ---
        random_seed = random.randint(0, 2**32 - 1)
        sampler_updated_count = 0
        for node_id, node_data in modified_workflow.items():
            if "class_type" in node_data and ("KSampler" in node_data["class_type"] or "SamplerCustom" in node_data["class_type"]):
                if "inputs" in node_data and "seed" in node_data["inputs"]:
                    node_data["inputs"]["seed"] = random_seed
                    self._log_verbose(f"    更新 Sampler 节点 '{node_id}' 种子为: {random_seed}")
                    sampler_updated_count += 1
                elif "inputs" in node_data and "noise_seed" in node_data["inputs"]: # 有些自定义 Sampler 可能用 noise_seed
                     node_data["inputs"]["noise_seed"] = random_seed
                     self._log_verbose(f"    更新 Sampler 节点 '{node_id}' 噪声种子为: {random_seed}")
                     sampler_updated_count += 1
        if sampler_updated_count == 0:
            self._log_verbose(f"警告: 工作流中未找到任何已知 Sampler 节点来更新种子。")
        elif sampler_updated_count > 1 and self.verbose:
             self._log_verbose(f"注意: 工作流中更新了 {sampler_updated_count} 个 Sampler 节点的种子。")

        return modified_workflow

    def send_prompt(self, workflow: dict) -> dict | None:
        """发送准备好的工作流 (prompt) 到 ComfyUI 服务器。"""
        if not workflow:
            self._log_error("send_prompt 收到无效的工作流 (None)。")
            return None
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        payload_dict = {'prompt': workflow, 'client_id': self.client_id}
        try: data = json.dumps(payload_dict)
        except TypeError as json_err:
             self._log_error(f"无法序列化工作流为 JSON: {json_err}")
             return None
        prompt_url = f"{self.api_url}/api/prompt"
        try:
            response = requests.post(prompt_url, headers=headers, data=data, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            if 'prompt_id' not in response_data:
                 self._log_error(f"提交工作流成功，但响应缺少 'prompt_id': {response_data}")
                 return None
            return response_data
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
        """获取指定 prompt ID 的执行历史/状态。"""
        if not prompt_id: return None
        history_url = f"{self.api_url}/history/{prompt_id}"
        try:
            response = requests.get(history_url, headers={'Accept': 'application/json'}, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if self.verbose: self._log_verbose(f"获取提示 {prompt_id} 的历史时出错 (网络/HTTP): {e}")
            return None
        except Exception as e:
            if self.verbose: self._log_verbose(f"获取提示 {prompt_id} 的历史时发生意外错误: {type(e).__name__}: {e}")
            return None

    def download_output_images(self, history: dict, prompt_id: str, output_dir_for_run: str,
                               original_image_basename: str, current_iteration_num: int) -> list[str]:
        """从完成任务的历史记录中下载标记为 'output' 类型的图像。"""
        downloaded_files_paths = []
        if not history or prompt_id not in history:
            self._log_verbose("  下载失败：未找到有效的执行历史。")
            return downloaded_files_paths
        history_entry = history.get(prompt_id, {})
        outputs = history_entry.get('outputs', {})
        if not outputs:
            self._log_verbose(f"  提示 {prompt_id} 的历史中未找到 'outputs' 字段。")
            return downloaded_files_paths
        
        images_to_download = []
        for node_id, node_output in outputs.items():
            if isinstance(node_output, dict) and 'images' in node_output and isinstance(node_output['images'], list):
                for img_data in node_output['images']:
                    if isinstance(img_data, dict) and img_data.get('type') == 'output':
                        if 'filename' in img_data:
                            images_to_download.append(img_data)
                        else:
                            self._log_verbose(f"    在节点 {node_id} 的输出中找到 'output' 图像，但缺少 'filename' 字段: {img_data}")
        
        if not images_to_download:
            self._log_verbose(f"  在提示 {prompt_id} 的工作流输出中未找到标记为 'output' 类型的图像。")
            return downloaded_files_paths

        self._log_verbose(f"    准备从服务器下载 {len(images_to_download)} 张 'output' 图像 (提示ID: {prompt_id})")
        
        original_base, original_ext = os.path.splitext(original_image_basename)
        if not original_ext: original_ext = '.png' 
        elif not original_ext.startswith('.'): original_ext = '.' + original_ext

        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]

        for idx, image_data in enumerate(images_to_download):
            server_filename = image_data.get('filename')
            subfolder = image_data.get('subfolder', '') 
            if not server_filename:
                continue

            name_part_to_use = original_base
            if len(images_to_download) > 1: 
                name_part_to_use = f"{original_base}_output_{idx}"
            
            if current_iteration_num > 1:
                final_local_filename = f"{name_part_to_use}-iter{current_iteration_num}_{timestamp_str}{original_ext}"
            else:
                final_local_filename = f"{name_part_to_use}_{timestamp_str}{original_ext}"
            
            local_path = os.path.join(output_dir_for_run, final_local_filename)

            url_params = {'filename': server_filename, 'type': 'output'}
            if subfolder:
                url_params['subfolder'] = subfolder
            
            view_url = f"{self.api_url}/view"
            
            try:
                self._log_verbose(f"      下载: 服务器文件 '{server_filename}' (类型: output, 子目录: '{subfolder or '无'}') 到 '{final_local_filename}'")
                response = requests.get(view_url, params=url_params, stream=True, timeout=180)
                response.raise_for_status()
                
                os.makedirs(os.path.dirname(local_path), exist_ok=True) 
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                self._log_verbose(f"      已保存到: {local_path}")
                downloaded_files_paths.append(os.path.abspath(local_path))
            except requests.exceptions.HTTPError as http_err:
                self._log_error(f"下载图像 '{server_filename}' 时发生 HTTP 错误 ({http_err.response.status_code}): {http_err}")
            except requests.exceptions.RequestException as req_err:
                self._log_error(f"下载图像 '{server_filename}' 时发生网络请求错误: {req_err}")
            except IOError as io_err:
                 self._log_error(f"写入文件 '{local_path}' 时发生 IO 错误: {io_err}")
            except Exception as e:
                self._log_error(f"下载或保存图像 '{server_filename}' 时出错: {type(e).__name__}: {e}")
        
        if not downloaded_files_paths and images_to_download: 
             self._log_verbose(f"  警告：找到 {len(images_to_download)} 张图像数据但未能下载任何图像 ({original_image_basename})。")
        time.sleep(2)
        return downloaded_files_paths

    def wait_for_completion(self, prompt_id: str) -> tuple[bool, dict | None, float]:
        """通过轮询 /history 端点等待 ComfyUI 任务完成。"""
        if not prompt_id: return False, None, 0.0
        
        log_prefix_short = f"提示ID:{prompt_id[:6]}" 
        start_time_wait = time.time()
        last_log_time = 0
        server_ip_short = self.server_address.split('//')[-1].split(':')[0] 

        self._log_verbose(f"{log_prefix_short} 监视服务器 {server_ip_short} 上的任务...")

        while True:
            current_loop_time = time.time()
            elapsed_wait = current_loop_time - start_time_wait

            if elapsed_wait >= MAX_WAIT_TIME:
                self._log_error(f"任务 {log_prefix_short} (Srv {server_ip_short}) 超时 ({MAX_WAIT_TIME}秒)。")
                return False, None, elapsed_wait

            history_response = self.get_history(prompt_id)

            if history_response and prompt_id in history_response:
                status_obj = history_response[prompt_id]
                status_info = status_obj.get("status", {})
                status_str = status_info.get("status_str", "未知状态")
                exec_completed = status_info.get("completed", False) 
                outputs_exist = bool(status_obj.get("outputs"))
                q_rem = status_info.get("exec_info", {}).get("queue_remaining", "N/A")

                if self.verbose and (current_loop_time - last_log_time >= 15.0 or status_str not in ['running', 'pending']): 
                    self._log_verbose(f"    [{log_prefix_short} on Srv {server_ip_short}] 状态: {status_str}, 完成标志: {exec_completed}, 队列: {q_rem}, 等待: {elapsed_wait:.0f}s")
                    last_log_time = current_loop_time
                
                if exec_completed:
                    if status_str == 'success': 
                        if not outputs_exist: 
                             self._log_error(f"  任务 {log_prefix_short} (Srv {server_ip_short}) 状态为 'success' 且完成标志为 True，但历史记录缺少 'outputs'。 API 等待: {elapsed_wait:.2f}秒")
                             return False, history_response, elapsed_wait 
                        else:
                            self._log_info(f"  任务 {log_prefix_short} (Srv {server_ip_short}) 成功完成。状态: {status_str}, API 等待: {elapsed_wait:.2f}秒")
                            return True, history_response, elapsed_wait
                    else: 
                        self._log_error(f"  任务 {log_prefix_short} (Srv {server_ip_short}) 完成但状态为失败/错误。状态: {status_str}, API 等待: {elapsed_wait:.2f}秒")
                        if outputs_exist: 
                            for node_id_err, node_output in status_obj["outputs"].items():
                                if isinstance(node_output, dict) and 'errors' in node_output:
                                    self._log_error(f"    {log_prefix_short} 节点 {node_id_err} 报告错误: {node_output['errors']}")
                        return False, history_response, elapsed_wait
            else: 
                if self.verbose and (current_loop_time - last_log_time >= 10.0): 
                    self._log_verbose(f"    [{log_prefix_short} on Srv {server_ip_short}] API 轮询: 无法获取历史。已等待:{elapsed_wait:.1f}s")
                    last_log_time = current_loop_time
            
            time.sleep(1.0) 

    def process_image(self, main_image_path: str, original_image_basename: str,
                      current_iteration_num: int, shot_folder_name: str, scene_num_str: str) -> tuple[list[str], float]:
        """
        处理单个图像的完整流程:
        加载 WF -> (主图像路径已硬编码修改) -> 生成 Prompt -> 更新 WF (Masks路径已硬编码修改, 新增TEXT_MULTILINE_NODE_ID文本更新) -> 提交 -> 等待 -> 下载。
        返回元组: (下载的图像路径列表, 总任务时长)。
        """
        task_start_time = time.time()
        self._log_info(f"开始处理: '{original_image_basename}' (迭代 {current_iteration_num}, 镜头 {shot_folder_name}, 场景号 {scene_num_str})")
        time.sleep(2.0)
        # --- 1. 加载基础工作流 ---
        workflow_template = self.load_workflow()
        if not workflow_template:
            return [], time.time() - task_start_time
        
        # --- 2. 主图像路径处理 (使用硬编码转换规则) ---
        # 原始问题行: uploaded_main_image_ref = main_image_path.replace("data/250511", "/data/comfyui/input/chmr/250513")
        # 修改后的行: 将本地的 BASE_INPUT_DIR (例如 "data/250511") 替换为服务器上图像存放的基础路径 (例如 "/data/comfyui/input/chmr")
        uploaded_main_image_ref = main_image_path.replace(BASE_INPUT_DIR, "/data/comfyui/input/chmr")
        self._log_verbose(f"    主图像服务器路径设置为: '{uploaded_main_image_ref}' (基于本地: '{main_image_path}')")


        # --- 3. 生成 AI 提示词 (可选, 用于 PROMPT_NODE_ID) ---
        anime_prompt_for_main_slot = generate_anime_prompt_wrapper(
            main_image_path, self._log_info, self._log_error, self._log_verbose
        )

        # --- 4. 更新工作流 (Masks路径已硬编码在update_workflow中, 新增TEXT_MULTILINE_NODE_ID文本更新) ---
        modified_workflow = self.update_workflow(
            workflow=workflow_template,
            main_image_ref=uploaded_main_image_ref,
            generated_prompt=anime_prompt_for_main_slot, # 这个是给 PROMPT_NODE_ID 的
            shot_folder_name=shot_folder_name, # 这个用于 LoRA 和 TEXT_MULTILINE_NODE_ID
            scene_num_str=scene_num_str 
        )
        if not modified_workflow:
            self._log_error(f"更新工作流失败 for '{original_image_basename}'. 中止处理。")
            return [], time.time() - task_start_time

        # --- 5. 提交工作流到 ComfyUI ---
        prompt_response = self.send_prompt(modified_workflow)
        if not prompt_response or 'prompt_id' not in prompt_response:
            self._log_error(f"提交工作流失败 for '{original_image_basename}'. 中止处理。")
            if prompt_response and 'node_errors' in prompt_response: 
                 self._log_error(f"  服务器报告节点错误: {prompt_response['node_errors']}")
            return [], time.time() - task_start_time
        prompt_id = prompt_response['prompt_id']
        self._log_info(f"  ComfyUI 任务已提交 (图像: {original_image_basename}, 提示ID: {prompt_id[:8]})")

        # --- 6. 等待任务完成 ---
        completed, final_history, time_spent_waiting = self.wait_for_completion(prompt_id)
        task_duration = time.time() - task_start_time 
        time.sleep(0.5)
        # --- 7. 处理结果 ---
        if completed and final_history:
            downloaded_images = self.download_output_images(
                final_history, prompt_id, OUTPUT_FOLDER,
                original_image_basename, current_iteration_num
            )
            if downloaded_images:
                self._log_info(f"成功完成并下载 {len(downloaded_images)} 张图片 for '{original_image_basename}'。总耗时: {task_duration:.2f}s。")
                return downloaded_images, task_duration
            else:
                self._log_error(f"工作流成功但下载图片失败 (提示ID: {prompt_id}, {original_image_basename})。总耗时: {task_duration:.2f}s。")
                return [], task_duration
        else:
            self._log_error(f"处理 '{original_image_basename}' 失败/超时 (提示ID: {prompt_id or 'N/A'})。总耗时: {task_duration:.2f}s (API 等待 {time_spent_waiting:.2f}s)。")
            return [], task_duration

# --------------- Main Execution Logic ---------------
if __name__ == "__main__":
    # --- 初始设置 ---
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        print(f"输出将保存到: {os.path.abspath(OUTPUT_FOLDER)}")
    except OSError as e:
        print(f"严重错误: 无法创建输出目录 '{OUTPUT_FOLDER}': {e}")
        sys.exit(1)

    # --- 检查基础工作流 ---
    base_workflow_full_path = os.path.join(WORKFLOW_BASE_DIR, BASE_WORKFLOW_FILENAME)
    if not os.path.exists(base_workflow_full_path):
        print(f"严重错误: 基础工作流文件 '{base_workflow_full_path}' 未找到。请确保它存在于 '{WORKFLOW_BASE_DIR}' 目录中。程序将退出。")
        sys.exit(1)
    else:
        print(f"将使用基础工作流: {base_workflow_full_path}")

    # --- 打印 Mask 配置信息 ---
    print("\nMask 和其他动态节点配置:")
    if SCENE_MASK_NODE_ID: # SCENE_MASK_NODE_ID is None
        print(f"  - 场景 Mask (节点 {SCENE_MASK_NODE_ID}):")
        # ... (原逻辑，当前由于SCENE_MASK_NODE_ID=None不会执行)
    else:
        print("  - 场景 Mask: SCENE_MASK_NODE_ID 未配置 (设为 None)，不处理场景 Mask。")

    if SUBTITLE_MASK_NODE_ID:
        print(f"  - 字幕 Mask (节点 {SUBTITLE_MASK_NODE_ID}):")
        print(f"    - 将在工作流中硬编码字幕 Mask 路径为: '/data/comfyui/input/wuji/mask/subtitle-mask.png'")
    else:
         print("  - 字幕 Mask: SUBTITLE_MASK_NODE_ID 未配置，不处理。")
    
    print("  (注意: 主图像和Mask路径当前在脚本中是基于特定规则进行硬编码转换或直接设置的)")

    # --- 打印动态文本节点配置信息 ---
    if TEXT_MULTILINE_NODE_ID:
        print(f"  - 多行文本节点 (节点 {TEXT_MULTILINE_NODE_ID}):")
        print(f"    - 其 'inputs.text' 将根据当前镜头的 'shot_folder_name' 从 LORA_PROMPTS 字典中动态填充。")
        print(f"    - 如果镜头文件夹名在 LORA_PROMPTS 中没有直接匹配，将使用 LORA_PROMPTS['00'] 的内容作为后备。")
    else:
        print("  - 多行文本节点: TEXT_MULTILINE_NODE_ID 未配置，不进行动态文本更新。")


    # --- 准备任务列表 ---
    overall_start_time = time.time()
    tasks_to_run = []

    print("\n程序开始：扫描输入目录并准备任务列表...")
    total_files_scanned = 0
    total_images_found_in_scan = 0 
    scenes_found = set()
    shots_found = set()

    try:
        scene_folders_calc = sorted([
            d for d in os.listdir(BASE_INPUT_DIR)
            if os.path.isdir(os.path.join(BASE_INPUT_DIR, d)) and d.startswith("场景")
        ])
        if not scene_folders_calc:
             print(f"警告 [扫描]: 在 '{BASE_INPUT_DIR}' 中未找到 '场景' 开头的文件夹。")

        for scene_folder_name_calc in scene_folders_calc:
            scenes_found.add(scene_folder_name_calc)
            scene_full_path_calc = os.path.join(BASE_INPUT_DIR, scene_folder_name_calc)
            scene_num_match_calc = re.search(r'\d+', scene_folder_name_calc) 
            scene_num_str_calc = scene_num_match_calc.group(0) if scene_num_match_calc else scene_folder_name_calc 

            shot_folders_calc = sorted([
                d for d in os.listdir(scene_full_path_calc)
                if os.path.isdir(os.path.join(scene_full_path_calc, d)) and d.isdigit() 
            ])
            if not shot_folders_calc and VERBOSE_LOGGING:
                print(f"信息 [扫描]: 在 '{scene_full_path_calc}' 中未找到数字命名的镜头文件夹。")

            for shot_folder_name_calc in shot_folders_calc:
                shots_found.add(f"{scene_folder_name_calc}/{shot_folder_name_calc}")
                shot_images_dir_calc = os.path.join(scene_full_path_calc, shot_folder_name_calc)
                
                current_shot_files = os.listdir(shot_images_dir_calc)
                total_files_scanned += len(current_shot_files)

                image_files_calc = sorted([
                    f for f in current_shot_files
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                ])
                
                if not image_files_calc:
                    if VERBOSE_LOGGING: print(f"信息 [扫描]: 在 '{shot_images_dir_calc}' 中未找到图像文件。")
                    continue

                if shot_folder_name_calc not in LORA_MAPPING:
                     print(f"警告 [扫描]: 在 LORA_MAPPING 配置中未找到镜头 '{shot_folder_name_calc}' 的映射。将使用工作流默认 LoRA/强度。")
                
                # 为 TEXT_MULTILINE_NODE_ID 检查 LORA_PROMPTS
                if shot_folder_name_calc not in LORA_PROMPTS and "00" not in LORA_PROMPTS: # 仅当特定镜头和 "00" 都不在时警告
                     print(f"警告 [扫描]: 镜头 '{shot_folder_name_calc}' 在 LORA_PROMPTS 中未找到，且 LORA_PROMPTS 中无 '00' 后备项。节点 '{TEXT_MULTILINE_NODE_ID}' 的文本可能为空。")
                elif shot_folder_name_calc not in LORA_PROMPTS : # 特定镜头不在，但有 "00"
                     if VERBOSE_LOGGING or shot_folder_name_calc != "00": # 避免 "00" 自身的后备情况也打印
                         print(f"提示 [扫描]: 镜头 '{shot_folder_name_calc}' 在 LORA_PROMPTS 中无直接映射，将使用 LORA_PROMPTS['00'] 的内容更新节点 '{TEXT_MULTILINE_NODE_ID}'。")


                for image_filename_calc in image_files_calc:
                    total_images_found_in_scan += 1
                    for iter_num_calc in range(1, NUM_ITERATIONS + 1):
                        tasks_to_run.append((
                            iter_num_calc,
                            scene_folder_name_calc,
                            shot_folder_name_calc,
                            image_filename_calc,
                            scene_num_str_calc, 
                        ))
    except FileNotFoundError:
        print(f"严重错误: 输入目录 '{BASE_INPUT_DIR}' 未找到。程序将退出。")
        sys.exit(1)
    except Exception as scan_err:
        print(f"严重错误: 扫描输入目录时发生错误: {scan_err}")
        sys.exit(1)

    total_tasks_to_process = len(tasks_to_run)
    print(f"扫描完成。找到 {len(scenes_found)} 个场景, {len(shots_found)} 个镜头。")
    print(f"总共扫描文件数 (所有类型): {total_files_scanned}")
    print(f"总共找到不同图像文件数 (未计迭代): {total_images_found_in_scan}") 
    print(f"总迭代次数: {NUM_ITERATIONS}")

    if total_tasks_to_process == 0:
        print("未找到任何需要处理的图像任务。请检查输入目录结构和内容。程序将退出。")
        sys.exit(0)

    print(f"总共需要处理的任务数 (图像 x 迭代): {total_tasks_to_process}")
    print(f"将使用 {NUM_WORKERS} 个并发工作线程 (每个服务器一个)。")

    print(f"\nLoRA 节点 '{LORA_NODE_ID}' 的 lora_02/strength_02 将根据镜头动态设置:")
    print(f"  - 镜头 '00' 或映射为 'None': lora_02='None', strength_02=0.0")
    print(f"  - 其他映射镜头: lora_02=映射文件, strength_02={DEFAULT_LORA_STRENGTH_02}")
    print(f"  - 未映射镜头: 使用工作流默认值")


    # --- 初始化计数器和线程池 ---
    tasks_completed_count = 0
    tasks_succeeded_count = 0
    tasks_failed_count = 0
    individual_task_durations = []
    futures_map = {} 
    print(f"\n--- 准备提交 {total_tasks_to_process} 个任务到线程池 ---")

    # --- 线程池并发执行 ---
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # --- 任务提交循环 ---
        for i, (iter_num, scene_folder_name, shot_folder_name, image_filename, scene_num_str) in enumerate(tasks_to_run):
            
            current_server_ip = SERVER_IPS[i % NUM_WORKERS] 
            scene_full_path = os.path.join(BASE_INPUT_DIR, scene_folder_name)
            shot_images_dir = os.path.join(scene_full_path, shot_folder_name)
            full_image_path = os.path.join(shot_images_dir, image_filename) 

            server_id_for_log = current_server_ip.split('//')[-1].split('.')[0] 
            context_log = f"任务 {i+1}/{total_tasks_to_process} Srv{server_id_for_log} It{iter_num} Sc{scene_num_str}-Sh{shot_folder_name}"

            tester = ComfyUITester(
                server_address=current_server_ip,
                workflow_file_path=base_workflow_full_path,
                output_folder=OUTPUT_FOLDER,
                context_info=context_log,
                verbose=VERBOSE_LOGGING
            )

            future = executor.submit(
                tester.process_image,
                main_image_path=full_image_path,
                original_image_basename=image_filename,
                current_iteration_num=iter_num,
                shot_folder_name=shot_folder_name,
                scene_num_str=scene_num_str 
            )
            futures_map[future] = (image_filename, context_log, shot_folder_name) 

            if DELAY_BETWEEN_SUBMISSIONS > 0:
                time.sleep(DELAY_BETWEEN_SUBMISSIONS)

        # --- 进度报告和结果收集 ---
        tqdm.write(f"\n所有 {total_tasks_to_process} 个任务已提交。开始处理并等待完成...\n")

        for future in tqdm(as_completed(futures_map), 
                            total=total_tasks_to_process, 
                            desc="处理图像", unit="个任务", 
                            ncols=120 if sys.stdout.isatty() else None, 
                            dynamic_ncols=True, file=sys.stdout, 
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]' 
                           ):
            
            img_fname, ctx_log, _ = futures_map[future] 
            tasks_completed_count += 1

            task_succeeded = False
            current_task_duration = 0.0

            try:
                processed_results, task_duration = future.result() 
                current_task_duration = task_duration
                if isinstance(processed_results, list) and len(processed_results) > 0: 
                    tasks_succeeded_count += 1
                    individual_task_durations.append(task_duration)
                    task_succeeded = True
                else:
                    tasks_failed_count += 1
            except CancelledError: 
                tasks_failed_count += 1
                tqdm.write(f"错误 [主循环]: 任务 {ctx_log} (图像: {img_fname}) 被工作线程取消。")
            except Exception as e: 
                tasks_failed_count += 1
                tqdm.write(f"严重错误 [主循环]: 获取任务 {ctx_log} (图像: {img_fname}) 结果时出错: {type(e).__name__}: {e}")

            current_elapsed_script_time = time.time() - overall_start_time
            avg_speed_successful = tasks_succeeded_count / current_elapsed_script_time if current_elapsed_script_time > 0 and tasks_succeeded_count > 0 else 0.0
            
            last_task_duration_info = ""
            if current_task_duration > 0:
                 status_indicator = "(成功)" if task_succeeded else "(失败)"
                 last_task_duration_info = f" 最近任务{status_indicator}耗时: {current_task_duration:.2f}s."
            
            avg_speed_str = f"{avg_speed_successful:.3f}"
            progress_summary = (
                f"进度: {tasks_completed_count}/{total_tasks_to_process}. "
                f"成功: {tasks_succeeded_count}, 失败: {tasks_failed_count}. "
                f"已运行: {time.strftime('%H:%M:%S', time.gmtime(current_elapsed_script_time))}. "
                f"平均成功速度: {avg_speed_str} 任务/秒.{last_task_duration_info}"
            )
            tqdm.write(progress_summary) 

    # --- 最终总结 ---
    overall_end_time = time.time()
    total_script_duration_seconds = overall_end_time - overall_start_time

    print("\n") 
    print(f"{'='*25} 所有处理已完成 {'='*25}")
    print(f"总迭代轮数: {NUM_ITERATIONS}")
    print(f"使用基础工作流: {BASE_WORKFLOW_FILENAME}")
    
    print("-" * 60)
    print("动态节点调整总结:")
    print(f"  - LoRA 节点 '{LORA_NODE_ID}' 已根据镜头动态调整。")
    
    if SCENE_MASK_NODE_ID:
        print(f"  - 场景 Mask (节点 {SCENE_MASK_NODE_ID}): 已尝试设置 (从本地 '{BASE_MASK_DIR_LOCAL}/{SCENE_MASK_LOCAL_FILENAME_TEMPLATE}' 上传, 跳过: {SHOTS_TO_SKIP_SCENE_MASK if SHOTS_TO_SKIP_SCENE_MASK else '无'})")
    else:
        print(f"  - 场景 Mask: SCENE_MASK_NODE_ID 未配置 (设为 None)，未处理。")
        
    if SUBTITLE_MASK_NODE_ID: 
        print(f"  - 字幕 Mask (节点 {SUBTITLE_MASK_NODE_ID}): 已尝试设置 (使用硬编码路径 '/data/comfyui/input/wuji/mask/subtitle-mask.png')")
    else:
        print(f"  - 字幕 Mask: SUBTITLE_MASK_NODE_ID 未配置，未处理。")

    if TEXT_MULTILINE_NODE_ID:
        print(f"  - 多行文本 (节点 {TEXT_MULTILINE_NODE_ID}): 其 'inputs.text' 已根据 LORA_PROMPTS 和镜头文件夹名动态填充。")
    else:
        print(f"  - 多行文本: TEXT_MULTILINE_NODE_ID 未配置，未处理。")
    
    # MASK_UPLOAD_SUBFOLDER_ON_SERVER 仅在实际发生上传时相关，当前硬编码路径逻辑不直接使用它。
    if MASK_UPLOAD_SUBFOLDER_ON_SERVER and (False): # (False) placeholder
         print(f"  - Mask 文件 (若通过脚本上传) 会尝试上传到服务器子目录: '{MASK_UPLOAD_SUBFOLDER_ON_SERVER}'")

    print("-" * 60)
    print(f"计划处理的任务总数: {total_tasks_to_process}")
    print(f"完成的任务总数 (成功+失败): {tasks_completed_count}")
    print(f"成功完成的任务数 (生成并下载): {tasks_succeeded_count}")
    print(f"失败/未下载的任务数: {tasks_failed_count}")
    if total_tasks_to_process > 0:
         success_rate = (tasks_succeeded_count / total_tasks_to_process) * 100
         print(f"总体成功率 (基于计划任务): {success_rate:.2f}%")
    else: print("未计划任何任务。")

    if individual_task_durations: 
        avg_task_duration = sum(individual_task_durations) / len(individual_task_durations)
        print(f"单个成功任务的平均处理时间: {avg_task_duration:.2f} 秒")
    elif tasks_succeeded_count == 0 and total_tasks_to_process > 0:
        print("没有成功处理的任务，无法计算平均处理时间。")
    
    if total_script_duration_seconds > 0 and tasks_succeeded_count > 0:
        overall_throughput = tasks_succeeded_count / total_script_duration_seconds
        print(f"整体系统吞吐量 (基于成功任务): {overall_throughput:.3f} 任务/秒")
    print("-" * 60)
    print(f"脚本总执行时间: {time.strftime('%H:%M:%S', time.gmtime(total_script_duration_seconds))} ({total_script_duration_seconds:.2f} 秒)")
    print(f"所有成功生成的图像已保存到: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"{'='*60}")