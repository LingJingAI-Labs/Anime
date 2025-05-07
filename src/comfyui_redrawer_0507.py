# --- START OF FILE comfyui_redrawer_all_new_cn_progress.py ---
import json
import requests
import time
import os
import random
import sys
from datetime import datetime
import uuid
import glob
import re # 用于从场景文件夹名称中提取数字
import base64 # 如果 prompt_reasoning 使用它，则需要导入
from tqdm import tqdm # 导入tqdm用于进度条
from prompt_reasoning import generate_anime_prompt # 确保此文件可用

# --------------- 配置参数 ---------------
SERVER_ADDRESS = "http://36.143.229.169:8188/"  # ComfyUI 服务器地址

# --- 迭代控制 ---
NUM_ITERATIONS = 4 # 所有图片迭代次数

# --- 路径定义 ---
BASE_INPUT_DIR = "data/250507"
MASKS_BASE_DIR = "data/mask"
SUBTITLE_MASK_FILENAME = "subtitle-mask.png"
WORKFLOW_BASE_DIR = "workflow"
OUTPUT_FOLDER = "data/250507/opt"

# --- 服务器和工作流节点 ID ---
MAX_WAIT_TIME = 360
MASKS_SERVER_SUBFOLDER = "clipspace"

IMAGE_INPUT_NODE_ID = "74"
PROMPT_NODE_ID = "227"
SCENE_MASK_NODE_ID = "190"
SUBTITLE_MASK_NODE_ID = "229"

DELAY_BETWEEN_IMAGES = 1

# --- 日志详细程度 ---
VERBOSE_LOGGING = False # 设置为 True 以启用更详细的内部日志

# --------------- 配置参数结束 ---------------

class ComfyUITester:
    """用于与ComfyUI交互并处理图像的工具。"""

    def __init__(self, server_address, workflow_file_path, output_folder, context_info="", verbose=VERBOSE_LOGGING):
        self.server_address = server_address
        self.api_url = server_address.rstrip('/')
        self.workflow_file_path = workflow_file_path
        self.output_folder = output_folder
        self.client_id = str(uuid.uuid4())
        self.context_info = context_info
        self.verbose = verbose # 控制日志详细程度
        # 初始化时只打印关键信息
        # print(f"ComfyUI 处理器已为 {self.context_info} 初始化 (客户端 ID: {self.client_id[:8]})")
        os.makedirs(self.output_folder, exist_ok=True)
        # if self.verbose:
        #     print(f"  输出目录: {os.path.abspath(self.output_folder)}")
        #     print(f"  使用工作流: {self.workflow_file_path}")


    def _log_verbose(self, message):
        if self.verbose:
            print(message)

    def load_workflow(self):
        """加载工作流JSON文件。"""
        try:
            if not os.path.exists(self.workflow_file_path):
                print(f"错误: 工作流文件未找到: {self.workflow_file_path}")
                return None
            with open(self.workflow_file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
                self._log_verbose(f"  成功加载工作流: {self.workflow_file_path}")
                return workflow_data
        except Exception as e:
            print(f"错误: 加载工作流 '{self.workflow_file_path}' 失败: {e}")
            return None

    def _upload_single_image(self, image_path: str, subfolder: str = "", image_type_for_log: str = "图像"):
        """内部通用图像上传函数。"""
        if not os.path.exists(image_path):
            print(f"错误: 无法上传，{image_type_for_log}文件不存在: {image_path}")
            return None
        
        filename = os.path.basename(image_path)
        upload_url = f"{self.api_url}/upload/image"
        self._log_verbose(f"    准备上传 {image_type_for_log}: {filename} 到服务器子文件夹 '{subfolder if subfolder else '根目录'}'")

        try:
            with open(image_path, 'rb') as f:
                files = {'image': (filename, f, 'image/png')}
                data = {'overwrite': 'true'}
                if subfolder:
                    data['subfolder'] = subfolder
                
                response = requests.post(upload_url, files=files, data=data, timeout=60)
                response.raise_for_status()
                upload_data = response.json()
                
                server_filename = upload_data.get('name')
                server_subfolder = upload_data.get('subfolder', '')

                if not server_filename:
                    print(f"错误: {image_type_for_log}上传成功，但服务器响应中未包含文件名。")
                    return None

                final_image_reference = f"{server_subfolder}/{server_filename}" if server_subfolder else server_filename
                self._log_verbose(f"    {image_type_for_log}上传成功: {final_image_reference}")
                return final_image_reference
                
        except requests.exceptions.HTTPError as http_err:
            print(f"错误: {image_type_for_log}上传时发生 HTTP 错误: {http_err}")
            if hasattr(http_err, 'response') and http_err.response is not None:
                try: self._log_verbose(f"      服务器响应内容: {http_err.response.text}")
                except Exception: pass
        except Exception as e:
            print(f"错误: 处理{image_type_for_log}上传时发生意外错误: {e}")
        return None

    def upload_main_image(self, image_path: str):
        return self._upload_single_image(image_path, subfolder="", image_type_for_log="主图像")

    def upload_specific_mask(self, mask_local_path: str, server_target_subfolder: str, mask_type_log: str):
        if not mask_local_path or not os.path.exists(mask_local_path):
            self._log_verbose(f"警告: {mask_type_log} 路径未提供或文件不存在: {mask_local_path}。跳过上传。")
            return None
        return self._upload_single_image(mask_local_path, subfolder=server_target_subfolder, image_type_for_log=mask_type_log)

    def update_workflow(self, workflow, main_image_ref: str, generated_prompt: str | None, 
                        scene_mask_ref: str | None, subtitle_mask_ref: str | None):
        if not workflow: return None
        modified_workflow = json.loads(json.dumps(workflow))

        if IMAGE_INPUT_NODE_ID in modified_workflow:
            modified_workflow[IMAGE_INPUT_NODE_ID]["inputs"]["image"] = main_image_ref
            self._log_verbose(f"    已更新主图像节点 {IMAGE_INPUT_NODE_ID} 为: {main_image_ref}")
        else:
            print(f"错误: 主图像节点 ID '{IMAGE_INPUT_NODE_ID}' 在工作流中未找到。")
            return None

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
                self._log_verbose(f"警告: 场景蒙版节点 ID '{SCENE_MASK_NODE_ID}' 未找到。无法更新场景蒙版。")
        elif SCENE_MASK_NODE_ID and scene_mask_ref is None: # 明确表示需要但未提供
             self._log_verbose(f"    未提供场景蒙版引用给节点 {SCENE_MASK_NODE_ID}。")

        if subtitle_mask_ref and SUBTITLE_MASK_NODE_ID:
            if SUBTITLE_MASK_NODE_ID in modified_workflow:
                modified_workflow[SUBTITLE_MASK_NODE_ID]["inputs"]["image"] = subtitle_mask_ref
                self._log_verbose(f"    已更新字幕蒙版节点 {SUBTITLE_MASK_NODE_ID} 为: {subtitle_mask_ref}")
            else:
                self._log_verbose(f"警告: 字幕蒙版节点 ID '{SUBTITLE_MASK_NODE_ID}' 未找到。无法更新字幕蒙版。")
        elif SUBTITLE_MASK_NODE_ID and subtitle_mask_ref is None: # 明确表示需要但未提供
            self._log_verbose(f"    未提供字幕蒙版引用给节点 {SUBTITLE_MASK_NODE_ID}。")
            
        random_seed = random.randint(0, 2**32 - 1)
        for node_id, node_data in modified_workflow.items():
            if "class_type" in node_data and "KSampler" in node_data["class_type"]:
                if "inputs" in node_data and "seed" in node_data["inputs"]:
                    node_data["inputs"]["seed"] = random_seed
                    self._log_verbose(f"    已更新 KSampler 节点 {node_id} 的种子为: {random_seed}")
                    break
        return modified_workflow

    def send_prompt(self, workflow):
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        payload_dict = {'prompt': workflow, 'client_id': self.client_id}
        data = json.dumps(payload_dict)
        try:
            response = requests.post(f"{self.api_url}/prompt", headers=headers, data=data, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"错误：提交工作流时发生 HTTP 错误: {http_err}")
            if hasattr(http_err, 'response') and http_err.response is not None:
                self._log_verbose(f"      服务器响应 (状态码 {http_err.response.status_code}): {http_err.response.text}")
            return None
        except Exception as e:
            print(f"错误：提交工作流时发生错误: {e}")
            return None

    def get_history(self, prompt_id):
        try:
            response = requests.get(f"{self.api_url}/history/{prompt_id}", headers={'Accept': 'application/json'}, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self._log_verbose(f"获取提示 {prompt_id} 的历史记录时出错: {e}") # 仅在verbose时打印错误
            return None

    def download_output_images(self, history, prompt_id, output_dir_for_run, original_image_basename, current_iteration_num):
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
                    images_to_download.append(img_data)

        if not images_to_download:
            self._log_verbose(f"  在工作流输出中 (Prompt ID: {prompt_id}) 未找到图像。")
            return []
        
        self._log_verbose(f"    准备从服务器下载 {len(images_to_download)} 张图像 (Prompt ID: {prompt_id})")

        original_base, original_ext = os.path.splitext(original_image_basename)

        for idx, image_data in enumerate(images_to_download):
            server_filename = image_data.get('filename')
            subfolder = image_data.get('subfolder', '')
            img_type = image_data.get('type', 'output')

            if not server_filename:
                self._log_verbose(f"    跳过没有文件名的图像数据: {image_data}")
                continue

            name_part_to_use = original_base
            if len(images_to_download) > 1: 
                name_part_to_use = f"{original_base}_{idx}" 
            
            if current_iteration_num > 1:
                final_local_filename = f"{name_part_to_use}-{current_iteration_num}{original_ext}"
            else:
                final_local_filename = f"{name_part_to_use}{original_ext}"
            
            local_path = os.path.join(output_dir_for_run, final_local_filename)
            url_params = {'filename': server_filename, 'subfolder': subfolder, 'type': img_type}
            
            try:
                self._log_verbose(f"      正在下载: 服务器文件 '{server_filename}' 另存为 '{final_local_filename}'")
                response = requests.get(f"{self.api_url}/view", params=url_params, stream=True, timeout=120)
                response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                self._log_verbose(f"      已保存到: {local_path}")
                downloaded_files_paths.append(os.path.abspath(local_path))
            except requests.exceptions.HTTPError as http_err:
                print(f"错误: 下载图像 HTTP 错误 ({http_err.response.status_code}) for {server_filename}: {http_err}")
                self._log_verbose(f"        请求的URL: {http_err.request.url}")
            except Exception as e:
                print(f"错误: 下载或保存图像时出错 {server_filename}: {e}")
        
        if not downloaded_files_paths and images_to_download:
            self._log_verbose(f"  警告：找到图像数据但未能下载任何图像 ({original_image_basename})。")
        return downloaded_files_paths

    def wait_for_completion(self, prompt_id, progress_bar_instance=None): # 接收tqdm实例
        log_prefix = f"[P_ID:{prompt_id[:6]}]"
        start_time = time.time()
        last_log_time = 0
        self._log_verbose(f"{log_prefix} 正在等待任务完成...")

        while time.time() - start_time < MAX_WAIT_TIME:
            time.sleep(1.0)
            current_time = time.time()
            history_response = self.get_history(prompt_id)

            if history_response and prompt_id in history_response:
                status_obj = history_response[prompt_id]
                status_info = status_obj.get("status", {})
                status_str = status_info.get("status_str", "未知")
                outputs_exist = bool(status_obj.get("outputs"))

                if progress_bar_instance: # 更新进度条描述
                    current_elapsed_time = time.time() - start_time
                    q_rem = status_info.get("exec_info", {}).get("queue_remaining", "N/A")
                    progress_bar_instance.set_postfix_str(f"{log_prefix} 状态: {status_str}, 队列: {q_rem}, 已耗时: {current_elapsed_time:.0f}s", refresh=True)


                if outputs_exist and status_str == 'success':
                    elapsed = time.time() - start_time
                    self._log_verbose(f"{log_prefix} 任务成功。状态: {status_str}, 耗时: {elapsed:.2f}秒")
                    if progress_bar_instance: progress_bar_instance.set_postfix_str(f"{log_prefix} 成功! ({elapsed:.1f}s)", refresh=True)
                    return True, history_response
                elif status_str in ['failed', 'error', 'cancelled'] or \
                     (outputs_exist and status_str not in ['success', 'running', 'pending']):
                    elapsed = time.time() - start_time
                    print(f"{log_prefix} 任务失败或出错。状态: {status_str}, 耗时: {elapsed:.2f}秒")
                    if progress_bar_instance: progress_bar_instance.set_postfix_str(f"{log_prefix} 失败/错误! ({status_str})", refresh=True)
                    if status_obj.get("outputs"):
                        for node_id_err, node_output in status_obj["outputs"].items():
                            if 'errors' in node_output:
                                print(f"  {log_prefix} 节点 {node_id_err} 错误: {node_output['errors']}")
                    return False, history_response
                else: 
                    if current_time - last_log_time >= 10.0 and self.verbose: # 详细日志时才打印轮询
                        elapsed = time.time() - start_time
                        q_rem = status_info.get("exec_info", {}).get("queue_remaining", "N/A")
                        print(f"{log_prefix} API 轮询等待中... 状态: {status_str}, 队列剩余: {q_rem}, 已耗时: {elapsed:.1f}s / {MAX_WAIT_TIME}s")
                        last_log_time = current_time
            else: 
                if current_time - last_log_time >= 10.0 and self.verbose: # 详细日志时才打印轮询
                    elapsed = time.time() - start_time
                    print(f"{log_prefix} API 轮询: 无法获取到 Prompt ID {prompt_id} 的有效历史记录。已耗时: {elapsed:.1f}s")
                    last_log_time = current_time
        
        print(f"{log_prefix} 超时 ({MAX_WAIT_TIME}秒) 等待提示 ID: {prompt_id}。")
        if progress_bar_instance: progress_bar_instance.set_postfix_str(f"{log_prefix} 超时!", refresh=True)
        return False, None

    def process_image(self, main_image_path, scene_mask_local_path, subtitle_mask_local_path,
                      original_image_basename, current_iteration_num, progress_bar_instance=None):
        # print(f"--- 处理: {original_image_basename} (迭代: {current_iteration_num}) ---") # 由进度条描述代替
        self._log_verbose(f"    主图像: {main_image_path}")
        self._log_verbose(f"    场景蒙版: {scene_mask_local_path if scene_mask_local_path else '无'}")
        self._log_verbose(f"    字幕蒙版: {subtitle_mask_local_path if subtitle_mask_local_path else '无'}")

        workflow = self.load_workflow()
        if not workflow:
            print(f"错误: 无法加载工作流 {self.workflow_file_path} ({original_image_basename})。跳过。")
            return []

        uploaded_main_image_ref = self.upload_main_image(main_image_path)
        if not uploaded_main_image_ref:
             print(f"错误: 主图像 '{original_image_basename}' 上传失败。跳过。")
             return []
        
        uploaded_scene_mask_ref = None
        if scene_mask_local_path and SCENE_MASK_NODE_ID:
            uploaded_scene_mask_ref = self.upload_specific_mask(scene_mask_local_path, MASKS_SERVER_SUBFOLDER, "场景蒙版")
            if not uploaded_scene_mask_ref and self.verbose: # 仅在verbose时打印上传失败警告
                print(f"警告: 场景蒙版 '{scene_mask_local_path}' 上传失败。")
        
        uploaded_subtitle_mask_ref = None
        if subtitle_mask_local_path and SUBTITLE_MASK_NODE_ID:
            uploaded_subtitle_mask_ref = self.upload_specific_mask(subtitle_mask_local_path, MASKS_SERVER_SUBFOLDER, "字幕蒙版")
            if not uploaded_subtitle_mask_ref and self.verbose: # 仅在verbose时打印上传失败警告
                 print(f"警告: 字幕蒙版 '{subtitle_mask_local_path}' 上传失败。")

        anime_prompt = None
        self._log_verbose(f"    正在为图像 '{original_image_basename}' 生成提示词...")
        try:
            anime_prompt = generate_anime_prompt(main_image_path)
            if anime_prompt:
                self._log_verbose(f"    成功生成提示词 (前30字符): {anime_prompt[:30]}...")
            else:
                self._log_verbose(f"    提示词为空，使用默认。")
        except Exception as e:
            self._log_verbose(f"    生成提示词时出错 ({original_image_basename}): {e}。使用默认。")

        modified_workflow = self.update_workflow(workflow, uploaded_main_image_ref, anime_prompt,
                                                 uploaded_scene_mask_ref, uploaded_subtitle_mask_ref)
        if not modified_workflow:
            print(f"错误: 更新工作流失败 ({original_image_basename})。跳过。")
            return []
        
        prompt_response = self.send_prompt(modified_workflow)
        if not prompt_response or 'prompt_id' not in prompt_response:
            print(f"错误: 提交工作流失败 ({original_image_basename})。跳过。")
            return []
        prompt_id = prompt_response['prompt_id']
        print(f"  任务已提交: {original_image_basename}, Prompt ID: {prompt_id[:8]}...") # 关键日志

        completed, final_history = self.wait_for_completion(prompt_id, progress_bar_instance)

        if completed and final_history:
            self._log_verbose(f"    工作流执行成功 (Prompt ID: {prompt_id})。")
            output_images = self.download_output_images(final_history, prompt_id, self.output_folder,
                                                        original_image_basename, current_iteration_num)
            if output_images:
                self._log_verbose(f"    成功下载 {len(output_images)} 张图片。")
                return output_images
            else:
                print(f"警告: 工作流成功但未能下载任何图片 (Prompt ID: {prompt_id}, {original_image_basename})。")
                return []
        else:
            print(f"错误: 工作流失败或超时 (Prompt ID: {prompt_id}, {original_image_basename})。")
            if final_history and prompt_id in final_history: # 尝试记录错误
                 status_obj = final_history[prompt_id]
                 outputs = status_obj.get('outputs', {})
                 for node_id_err, node_output in outputs.items():
                    if 'errors' in node_output:
                        print(f"      节点 {node_id_err} 报告错误: {node_output['errors']}")
            return []

# --------------- 主要执行逻辑 ---------------
if __name__ == "__main__":
    start_time_total = time.time()
    total_images_to_process_overall = 0 # 用于tqdm的总计数
    tasks_to_run = [] # 存储所有待处理任务的元组 (iter_num, scene_folder_name, shot_folder_name, image_filename)

    # --- 预计算总任务数 ---
    print("正在扫描文件并计算总任务数...")
    for iter_num_calc in range(1, NUM_ITERATIONS + 1):
        scene_folders_calc = sorted([
            d for d in os.listdir(BASE_INPUT_DIR) 
            if os.path.isdir(os.path.join(BASE_INPUT_DIR, d)) and d.startswith("场景")
        ])
        for scene_folder_name_calc in scene_folders_calc:
            scene_full_path_calc = os.path.join(BASE_INPUT_DIR, scene_folder_name_calc)
            shot_folders_calc = sorted([
                d for d in os.listdir(scene_full_path_calc)
                if os.path.isdir(os.path.join(scene_full_path_calc, d)) and d.isdigit()
            ])
            for shot_folder_name_calc in shot_folders_calc:
                shot_images_dir_calc = os.path.join(scene_full_path_calc, shot_folder_name_calc)
                current_workflow_filename_calc = f"FLUX-0507-{shot_folder_name_calc}.json"
                current_workflow_path_calc = os.path.join(WORKFLOW_BASE_DIR, current_workflow_filename_calc)
                if not os.path.exists(current_workflow_path_calc):
                    continue # 跳过没有工作流的镜头

                image_files_calc = sorted([
                    f for f in os.listdir(shot_images_dir_calc) 
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                ])
                for image_filename_calc in image_files_calc:
                    total_images_to_process_overall += 1
                    tasks_to_run.append((iter_num_calc, scene_folder_name_calc, shot_folder_name_calc, image_filename_calc))
    
    if total_images_to_process_overall == 0:
        print("未找到任何需要处理的图像任务。请检查输入目录和文件结构。")
        sys.exit(0)

    print(f"总共需要处理 {total_images_to_process_overall} 个图像任务。")

    # --- 初始化计数器 ---
    total_images_processed_successfully = 0
    total_images_failed = 0

    global_subtitle_mask_path = os.path.join(MASKS_BASE_DIR, SUBTITLE_MASK_FILENAME)
    if not os.path.exists(global_subtitle_mask_path):
        print(f"严重错误: 全局字幕蒙版 '{global_subtitle_mask_path}' 未找到。程序将退出。")
        sys.exit(1)
    else:
        if VERBOSE_LOGGING: print(f"将使用全局字幕蒙版: {global_subtitle_mask_path}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    if VERBOSE_LOGGING: print(f"全局输出文件夹: {os.path.abspath(OUTPUT_FOLDER)}")

    # --- 使用tqdm创建进度条 ---
    with tqdm(total=total_images_to_process_overall, unit="张图", ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:
        for iter_num, scene_folder_name, shot_folder_name, image_filename in tasks_to_run:
            pbar.set_description(f"迭代 {iter_num}/{NUM_ITERATIONS}, {scene_folder_name}/{shot_folder_name}")
            
            scene_full_path = os.path.join(BASE_INPUT_DIR, scene_folder_name)
            scene_num_match = re.search(r'\d+', scene_folder_name)
            scene_num = scene_num_match.group(0) if scene_num_match else "未知"
            
            current_scene_mask_path = os.path.join(MASKS_BASE_DIR, f"scene-{scene_num}-mask.png")
            if not os.path.exists(current_scene_mask_path):
                if VERBOSE_LOGGING: print(f"警告: 场景 {scene_folder_name} 的场景蒙版 '{current_scene_mask_path}' 未找到。")
                current_scene_mask_path = None

            shot_images_dir = os.path.join(scene_full_path, shot_folder_name)
            current_workflow_filename = f"FLUX-0507-{shot_folder_name}.json"
            current_workflow_path = os.path.join(WORKFLOW_BASE_DIR, current_workflow_filename)
            # 在预计算时已检查工作流是否存在，此处不再重复检查
            
            full_image_path = os.path.join(shot_images_dir, image_filename)
            short_img_name = image_filename[:15] + "..." if len(image_filename) > 18 else image_filename
            pbar.set_postfix_str(f"处理中: {short_img_name}", refresh=True)
            
            context_log = f"迭代 {iter_num}, 场景 {scene_num}, 镜头 {shot_folder_name}, 图像 {image_filename}"
            tester = ComfyUITester(
                server_address=SERVER_ADDRESS,
                workflow_file_path=current_workflow_path,
                output_folder=OUTPUT_FOLDER,
                context_info=context_log,
                verbose=VERBOSE_LOGGING
            )

            processed_results = tester.process_image(
                main_image_path=full_image_path,
                scene_mask_local_path=current_scene_mask_path,
                subtitle_mask_local_path=global_subtitle_mask_path,
                original_image_basename=image_filename,
                current_iteration_num=iter_num,
                progress_bar_instance=pbar # 传递tqdm实例
            )

            if processed_results:
                total_images_processed_successfully += 1
            else:
                total_images_failed += 1
            
            pbar.update(1) # 更新进度条

            # DELAY_BETWEEN_IMAGES 逻辑可以保留，如果需要的话
            # last_task_check = (iter_num == tasks_to_run[-1][0] and \
            #                    scene_folder_name == tasks_to_run[-1][1] and \
            #                    shot_folder_name == tasks_to_run[-1][2] and \
            #                    image_filename == tasks_to_run[-1][3])
            # if DELAY_BETWEEN_IMAGES > 0 and not last_task_check:
            #      if VERBOSE_LOGGING: print(f"    暂停 {DELAY_BETWEEN_IMAGES} 秒...")
            #      time.sleep(DELAY_BETWEEN_IMAGES)
            # 由于tqdm会持续刷新，短时间延迟可能不明显，如果需要显著延迟，可以取消注释并调整
            if DELAY_BETWEEN_IMAGES > 0:
                 time.sleep(DELAY_BETWEEN_IMAGES)


    # --- 最终总结 ---
    end_time_total = time.time()
    total_duration_seconds = end_time_total - start_time_total
    print(f"\n\n{'='*20} 所有处理完成 {'='*20}")
    print(f"总迭代次数: {NUM_ITERATIONS}")
    print(f"总共成功处理的图像任务数: {total_images_processed_successfully}")
    print(f"总共失败的图像任务数: {total_images_failed}")
    total_attempted = total_images_processed_successfully + total_images_failed
    if total_attempted > 0:
        success_rate = (total_images_processed_successfully / total_attempted) * 100
        print(f"总体成功率: {success_rate:.2f}%")
    else:
        print("未尝试任何任务。")
    print(f"总执行时间: {time.strftime('%H:%M:%S', time.gmtime(total_duration_seconds))}")
    print(f"所有结果已保存到: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"{'='*50}")

# --- END OF FILE comfyui_redrawer_all_new_cn_progress.py ---