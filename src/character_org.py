import os
import base64
import argparse
import requests
import json
import shutil
from pathlib import Path
import time # For potential rate limiting

# 从配置文件导入设置
try:
    import config as cfg
except ImportError:
    print("错误：无法导入配置文件 'config.py'。请确保该文件存在且路径正确。")
    exit(1)

CHARACTER_MAPPING = {
    "00": "None",
    "01": "秦云",
    "02": "萧灵",
    "03": "蔡成安",
    "04": "蔡晓霞",
    "07": "张思思",
    "09": "虎哥",
    "10": "绿衣男",
    "11": "周学兵",
    "12": "售货员",
    "13": "王大妈",
    "15": "周雪",
    "16": "刘大爷", 
}

# --- Helper Function: Encode Image to Base64 ---
def encode_image_to_base64(image_path: str) -> str | None:
    """将图像文件编码为 Base64 字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"错误：无法找到图像文件 '{image_path}'")
        return None
    except Exception as e:
        print(f"错误：读取或编码图像 '{image_path}' 时出错: {e}")
        return None

# --- Core Function: Get Character Info via Aihubmix ---
def get_character_info_from_image(
    scene_image_path: str,
    reference_image_path: str,
    current_model_id: str,
    max_tokens: int,
    timeout: int
) -> dict | None:
    """
    使用 Aihubmix API 分析场景图像，并根据参考图像识别角色，返回JSON格式的角色信息。
    """
    if not cfg.AIHUBMIX_API_KEY:
        print("错误：AIHUBMIX_API_KEY 未在 config.py 中配置或环境变量未设置。")
        return None

    base64_scene_image = encode_image_to_base64(scene_image_path)
    if not base64_scene_image:
        return None

    base64_reference_image = encode_image_to_base64(reference_image_path)
    if not base64_reference_image:
        return None

    scene_image_url = f"data:image/jpeg;base64,{base64_scene_image}"
    reference_image_url = f"data:image/jpeg;base64,{base64_reference_image}"

    headers = {
        "Authorization": f"Bearer {cfg.AIHUBMIX_API_KEY}",
        "Content-Type": "application/json",
    }

    user_prompt = (
        "你将收到两张图片：第一张是“角色参考图”，其中包含多个标记了名字的角色肖像；第二张是“场景图”。\n"
        "你的任务是：\n"
        "1. 仔细查看“场景图”。\n"
        "2. 根据“角色参考图”中提供的角色信息，识别出“场景图”中出现了哪些角色。\n"
        "3. 注意：角色在“场景图”中可能穿着不同的服装，请主要依据面部特征进行识别。\n"
        "4. 对于每个在“场景图”中识别出的角色，请指出其在图中的大致位置（只有如下3种位置：'left', 'center', 'right'）。\n"
        "5. 最终，你必须严格按照以下 JSON 格式输出结果，不要包含任何额外的解释、Markdown标记或说明文字。只输出纯JSON字符串：\n"
        "{\n"
        '  "count": <识别到的角色数量 (整数)>,\n'
        '  "people": [\n'
        "    {\n"
        '      "name": "<角色在参考图中的名字 (字符串)>",\n'
        '      "position": "<角色在场景图中的位置 (字符串)>"\n'
        "    }\n"
        "    // ... 如果有更多角色，继续添加对象\n"
        "  ]\n"
        "}\n"
        "如果“场景图”中没有出现“角色参考图”中的任何角色，或者场景图本身为空镜，请输出：\n" # Clarified for empty shot
        "{\n"
        '  "count": 0,\n'
        '  "people": []\n'
        "}\n"
        "请确保角色名字与“角色参考图”中的标签完全一致。"
    )

    payload_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": reference_image_url, "detail": "high"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": scene_image_url, "detail": "high"},
                },
            ],
        }
    ]

    data = {
        "model": current_model_id,
        "messages": payload_messages,
        "max_tokens": max_tokens,
    }

    print(f"正在通过 Aihubmix 使用模型 '{current_model_id}' 分析图像 '{Path(scene_image_path).name}'...")
    api_response_data = None
    try:
        response = requests.post(
            cfg.AIHUBMIX_API_URL,
            headers=headers,
            data=json.dumps(data),
            timeout=timeout
        )
        response.raise_for_status()
        api_response_data = response.json()
        raw_content = api_response_data.get('choices', [{}])[0].get('message', {}).get('content')

        if raw_content:
            if raw_content.strip().startswith("```json"):
                raw_content = raw_content.strip()[7:]
                if raw_content.strip().endswith("```"):
                    raw_content = raw_content.strip()[:-3]
            
            try:
                response_json_content = json.loads(raw_content.strip())
                return response_json_content
            except json.JSONDecodeError as json_err:
                print(f"错误：API 返回的内容不是有效的 JSON 格式。错误: {json_err}")
                print(f"API 原始返回内容: {raw_content}")
                return None
        else:
            print(f"错误：API 返回了有效响应，但未找到生成的文本。API 响应: {api_response_data}")
            return None

    except requests.exceptions.Timeout:
        print(f"错误：请求 Aihubmix API 超时（超过 {timeout} 秒）。")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"错误：Aihubmix API 返回 HTTP 错误: {e.response.status_code} {e.response.reason}")
        try:
            error_details = e.response.json()
            print(f"错误详情: {error_details}")
        except json.JSONDecodeError:
            print(f"无法解析的错误响应体: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"错误：请求 Aihubmix API 时出错: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"错误：解析 Aihubmix API 响应结构时出错: {e}。")
        print(f"请检查 API 响应是否符合预期结构。API 响应: {api_response_data if api_response_data else 'N/A'}")
        return None
    except Exception as e:
        print(f"错误：发生未知错误: {e}")
        return None

def load_character_map(map_file_path: str | None) -> dict:
    default_map = getattr(cfg, 'DEFAULT_CHARACTER_MAP', {})
    if map_file_path:
        try:
            with open(map_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告：角色映射文件 '{map_file_path}' 未找到，将使用默认映射。")
        except json.JSONDecodeError:
            print(f"警告：角色映射文件 '{map_file_path}' 格式错误，将使用默认映射。")
        except Exception as e:
            print(f"警告：加载角色映射文件时出错 '{e}'，将使用默认映射。")
    return default_map

# --- Main Execution Block ---
# --- START OF if __name__ == "__main__": BLOCK ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 Aihubmix API 分析图像中的角色，生成JSON元数据，并根据角色整理文件。"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/initial_frames/",
        help="包含需要分析的场景图像的文件夹路径。"
    )
    parser.add_argument(
        "--ref_image",
        type=str,
        default="data/char-ref.jpeg", # Default reference image path
        help="包含角色名称标签的角色参考图文件路径。如果未提供，将默认使用 'data/char-ref.jpeg'。"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/tmp", # This is the base for 00, 01, etc. character folders
        help="存放整理后角色文件夹和文件的根目录 (默认为: data/tmp)。"
    )
    parser.add_argument(
        "--map_file",
        type=str,
        default=None,
        help="可选的 JSON 文件路径，用于定义角色名到两位数文件夹名的映射。如果未提供，则使用配置文件中的默认映射。"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=getattr(cfg, 'MODEL_ID', 'gpt-4o-mini'), # Get MODEL_ID from config.py
        help=f"指定要使用的模型 ID (默认为配置文件中的值)。"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0, # Default delay of 1 second between API calls
        help="每次API调用之间的延迟时间（秒），以避免速率限制 (默认为: 1.0)。"
    )

    args = parser.parse_args()

    # Ensure config has necessary attributes, provide defaults if missing from config.py
    current_model_id = args.model
    max_tokens = getattr(cfg, 'MAX_TOKENS_OUTPUT', 1024)
    request_timeout = getattr(cfg, 'REQUEST_TIMEOUT', 120)

    # Load character map
    character_map = load_character_map(args.map_file)
    if not character_map: # Should use default from config if map_file is None or fails
        print("警告：角色映射为空或加载失败。如果配置文件中没有定义默认映射，脚本可能无法正确分类。")
    print(f"使用的角色映射: {character_map}")

    input_path = Path(args.input_dir)
    # output_base_dir is where 00, 01, etc. folders (final classified output) will be created
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Define and create the temporary directory for initial JSON saves
    # This directory will be inside the main output_dir
    temp_json_dir = output_base_dir / "temp_json_files" # Renamed for clarity
    temp_json_dir.mkdir(parents=True, exist_ok=True)
    print(f"临时JSON文件将首先保存在: {temp_json_dir}")


    if not input_path.is_dir():
        print(f"错误：输入路径 '{args.input_dir}' 不是一个有效的目录。")
        exit(1)

    # Use parser.get_default('ref_image') to show the actual default in error message
    if not Path(args.ref_image).is_file():
        print(f"错误：参考图像 '{args.ref_image}' 未找到或不是一个文件。")
        print(f"请确保默认路径 '{parser.get_default('ref_image')}' 有效，或者通过 --ref_image 参数提供一个正确的路径。")
        exit(1)

    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    
    scene_image_files = sorted([ # Sorted for consistent processing order
        f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions
    ])

    if not scene_image_files:
        print(f"在目录 '{args.input_dir}' 中没有找到支持的图像文件。")
        exit(1)
    
    total_images = len(scene_image_files)
    print(f"找到 {total_images} 张图像进行处理...")

    for i, scene_image_file_path in enumerate(scene_image_files):
        print(f"\n--- 处理图像: {scene_image_file_path.name} ({i+1}/{total_images}) ---")
        
        # 1. Get character info from API
        character_data = get_character_info_from_image(
            str(scene_image_file_path), 
            args.ref_image,
            current_model_id,
            max_tokens,
            request_timeout
        )

        if not character_data:
            print(f"未能从API获取图像 '{scene_image_file_path.name}' 的角色信息或解析响应。跳过此图像。")
            if args.delay > 0 and i < total_images - 1: # Delay only if not the last image
                time.sleep(args.delay)
            continue

        # 2. Save the JSON data to the temporary directory
        json_filename = scene_image_file_path.stem + ".json"
        # JSON files are initially saved in the temp_json_dir
        temp_json_output_path = temp_json_dir / json_filename 

        try:
            with open(temp_json_output_path, 'w', encoding='utf-8') as f:
                json.dump(character_data, f, ensure_ascii=False, indent=2)
            print(f"角色信息已临时保存到: {temp_json_output_path}")
        except IOError as e:
            print(f"错误：无法写入临时JSON文件 '{temp_json_output_path}': {e}。跳过此图像的文件整理。")
            if args.delay > 0 and i < total_images - 1:
                time.sleep(args.delay)
            continue

        # 3. Organize files based on detected characters
        # Validate structure needed for organization
        if not ("people" in character_data and \
                isinstance(character_data.get("people"), list) and \
                "count" in character_data):
            print(f"警告：图像 '{scene_image_file_path.name}' 的JSON数据结构不符合预期 (缺少 'people' 或 'count')。无法进行文件整理。")
            print(f"获取到的数据: {character_data}")
            # Optionally, decide if you want to keep or delete the temp JSON in this case
            # For now, it will remain in the temp_json_dir
            if args.delay > 0 and i < total_images - 1:
                time.sleep(args.delay)
            continue 

        json_copied_to_final_destination = False # Flag to track if JSON was copied for this image

        # Case 1: Empty shot (no characters recognized by API matching reference)
        if character_data.get("count") == 0:
            print(f"对于图像 '{scene_image_file_path.name}': API 指示为空镜或未识别出与参考图匹配的角色。")
            empty_shot_folder_code = "00"
            # Character folders are directly under output_base_dir (e.g., data/tmp/00)
            character_folder = output_base_dir / empty_shot_folder_code
            character_folder.mkdir(parents=True, exist_ok=True)

            # Copy original image to "00"
            try:
                dest_image_path = character_folder / scene_image_file_path.name
                shutil.copy2(scene_image_file_path, dest_image_path)
                print(f"图像 '{scene_image_file_path.name}' (空镜/无匹配) 已复制到 '{dest_image_path}'")
            except Exception as e:
                print(f"错误：复制图像 '{scene_image_file_path.name}' 到 '{character_folder}' 失败: {e}")

            # Copy JSON file from temporary location to "00"
            try:
                dest_json_path = character_folder / json_filename
                if temp_json_output_path.exists(): # Source is now the temp JSON path
                    shutil.copy2(temp_json_output_path, dest_json_path)
                    print(f"JSON '{json_filename}' (空镜/无匹配) 已从临时位置复制到 '{dest_json_path}'")
                    json_copied_to_final_destination = True
                else:
                    print(f"错误: 源临时JSON文件 '{temp_json_output_path}' 未找到，无法复制。")
            except Exception as e:
                print(f"错误：复制JSON '{json_filename}' 到 '{character_folder}' 失败: {e}")

        # Case 2: Characters were potentially found (count > 0)
        else:
            found_characters_in_map_for_this_image = False # For specific message later
            people_list = character_data.get("people", []) # Should have items if count > 0
            
            if not people_list and character_data.get("count", 0) > 0:
                 print(f"注意: API 报告了 {character_data['count']} 个角色，但 'people' 列表为空。图像 '{scene_image_file_path.name}' 将不会被分类到角色文件夹中。")
            
            for person in people_list:
                name = person.get("name")
                if name and character_map and name in character_map:
                    found_characters_in_map_for_this_image = True
                    character_code = character_map[name]
                    character_folder = output_base_dir / character_code
                    character_folder.mkdir(parents=True, exist_ok=True)

                    # Copy original image
                    try:
                        dest_image_path = character_folder / scene_image_file_path.name
                        shutil.copy2(scene_image_file_path, dest_image_path)
                        print(f"图像 '{scene_image_file_path.name}' 已复制到 '{dest_image_path}' (角色: {name})")
                    except Exception as e:
                        print(f"错误：复制图像 '{scene_image_file_path.name}' 到 '{character_folder}' (角色: {name}) 失败: {e}")

                    # Copy JSON file from temporary location
                    try:
                        dest_json_path = character_folder / json_filename
                        if temp_json_output_path.exists(): # Source is temp JSON path
                             shutil.copy2(temp_json_output_path, dest_json_path)
                             print(f"JSON '{json_filename}' 已从临时位置复制到 '{dest_json_path}' (角色: {name})")
                             json_copied_to_final_destination = True # Mark as copied
                        else:
                            print(f"错误: 源临时JSON文件 '{temp_json_output_path}' 未找到，无法复制。")
                    except Exception as e:
                        print(f"错误：复制JSON '{json_filename}' 到 '{character_folder}' (角色: {name}) 失败: {e}")
                elif name: # Character detected by API but not in our user-defined map
                    print(f"注意：在图像 '{scene_image_file_path.name}' 中检测到角色 '{name}'，但该角色不在角色映射中或映射为空，将忽略此角色的分类。")
            
            # This message is for cases where API found people, but NONE of them matched our character_map
            if not found_characters_in_map_for_this_image and people_list: 
                print(f"对于图像 '{scene_image_file_path.name}': API 识别出角色，但这些角色均未在提供的角色映射中找到。图像不会被分类到数字角色文件夹。")

        # Delete the JSON from the temporary directory if it was successfully copied to a final destination
        if json_copied_to_final_destination and temp_json_output_path.exists():
            try:
                os.remove(temp_json_output_path)
                print(f"已从临时目录删除: {temp_json_output_path.name}")
            except OSError as e:
                print(f"错误：无法从临时目录删除JSON文件 '{temp_json_output_path}': {e}")
        elif temp_json_output_path.exists(): # If it exists but wasn't copied to a final destination
             print(f"注意：图像 '{scene_image_file_path.name}' 的临时JSON文件保留在 '{temp_json_dir}' 中，因为它未被分类到任何最终的角色文件夹。")
        
        # Delay between API calls, except for the last image
        if args.delay > 0 and i < total_images - 1:
            print(f"等待 {args.delay} 秒后处理下一张图片...")
            time.sleep(args.delay)
    
    # Optional: Clean up the temp_json_dir if it's empty after processing all files
    # This is useful if you intend to delete all temp JSONs successfully after they are moved
    try:
        if temp_json_dir.exists() and not any(temp_json_dir.iterdir()): # Check if directory is empty
            print(f"\n临时JSON目录 '{temp_json_dir}' 为空，正在尝试删除...")
            # os.rmdir(temp_json_dir) # Use os.rmdir for empty directory, or shutil.rmtree if it might have subdirs (not expected here)
            shutil.rmtree(temp_json_dir) # shutil.rmtree is safer if by any chance something else got put there
            print(f"临时JSON目录 '{temp_json_dir}' 已删除。")
        elif temp_json_dir.exists():
            print(f"\n临时JSON目录 '{temp_json_dir}' 中仍有文件，未删除该目录。检查该目录以获取未分类或处理失败的JSON。")
    except Exception as e:
        print(f"尝试清理临时JSON目录 '{temp_json_dir}' 时出错: {e}")

    print("\n--- 所有图像处理完毕 ---")