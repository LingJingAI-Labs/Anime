# --- START OF FILE character_org_modified_concurrent.py ---

import os
import base64
import argparse
import requests
import json
import shutil # 用于文件复制
from pathlib import Path
import time # 仍然保留，以防某些特定情况下需要
import concurrent.futures # 用于并发处理

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

# --- 辅助函数：将图像编码为 Base64 ---
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

# --- 核心函数：通过 Aihubmix 获取角色代码 ---
def get_character_codes_from_image(
    scene_image_path: str,
    reference_image_path: str,
    current_model_id: str,
    max_tokens: int,
    timeout: int,
    character_mapping_for_prompt: dict,
    image_display_name: str # 用于日志记录
) -> list[str] | None:
    """
    使用 Aihubmix API 分析场景图像，并根据参考图像识别角色，
    返回这些角色在 CHARACTER_MAPPING 中对应的代码列表。
    """
    if not cfg.AIHUBMIX_API_KEY:
        print(f"错误 ({image_display_name})：AIHUBMIX_API_KEY 未在 config.py 中配置或环境变量未设置。")
        return None

    base64_scene_image = encode_image_to_base64(scene_image_path)
    if not base64_scene_image:
        print(f"错误 ({image_display_name})：编码场景图失败。")
        return None

    base64_reference_image = encode_image_to_base64(reference_image_path)
    if not base64_reference_image:
        print(f"错误 ({image_display_name})：编码参考图失败。")
        return None

    scene_image_url = f"data:image/jpeg;base64,{base64_scene_image}"
    reference_image_url = f"data:image/jpeg;base64,{base64_reference_image}"

    headers = {
        "Authorization": f"Bearer {cfg.AIHUBMIX_API_KEY}",
        "Content-Type": "application/json",
    }

    mapping_prompt_str = "请参考以下角色代码和名称的映射：\n"
    for code, name in character_mapping_for_prompt.items():
        if code != "00":
            mapping_prompt_str += f'- 代码 "{code}": 角色名 "{name}"\n'

    user_prompt = (
        "你将收到两张图片：第一张是“角色参考图”，其中可能包含多个已知角色及其名字；第二张是“场景图”。\n"
        "你的任务是：\n"
        "1. 仔细查看“场景图”。\n"
        "2. 根据“角色参考图”中提供的角色信息，识别出“场景图”中出现了哪些角色。\n"
        "3. 注意：角色在“场景图”中可能穿着不同的服装，请主要依据面部特征进行识别。\n"
        f"{mapping_prompt_str}"
        "4. 根据你在“场景图”中识别出的角色，并对照上面提供的映射关系，返回这些角色对应的代码列表。\n"
        "5. 最终，你必须严格按照 Python列表 (list of strings) 的格式输出结果。例如： `[\"01\", \"07\"]`。\n"
        "6. 如果“场景图”中没有出现上述映射中的任何角色，或者场景图本身为空镜，请务必输出：`[\"00\"]`。\n"
        "7. 不要包含任何额外的解释、Markdown标记或说明文字。只输出纯粹的 Python列表 字符串。\n"
        "请确保角色名字的识别基于“角色参考图”，然后将这些识别出的名字映射到给定的代码。"
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

    print(f"INFO ({image_display_name}): 正在通过 Aihubmix 使用模型 '{current_model_id}' 分析...")
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
            clean_content = raw_content.strip()
            if clean_content.startswith("```json"):
                clean_content = clean_content[7:]
                if clean_content.endswith("```"):
                    clean_content = clean_content[:-3]
            elif clean_content.startswith("```python"):
                clean_content = clean_content[9:]
                if clean_content.endswith("```"):
                    clean_content = clean_content[:-3]
            elif clean_content.startswith("```"):
                 clean_content = clean_content[3:]
                 if clean_content.endswith("```"):
                    clean_content = clean_content[:-3]
            clean_content = clean_content.strip()

            try:
                parsed_codes = json.loads(clean_content)
                if isinstance(parsed_codes, list) and all(isinstance(item, str) for item in parsed_codes):
                    valid_codes = [code for code in parsed_codes if code in CHARACTER_MAPPING]
                    if not valid_codes and "00" in parsed_codes:
                        return ["00"]
                    if not valid_codes and parsed_codes:
                         print(f"警告 ({image_display_name})：API返回了代码 {parsed_codes}，但它们不在CHARACTER_MAPPING中。将视为空处理。")
                         return ["00"]
                    return valid_codes if valid_codes else ["00"]
                else:
                    print(f"错误 ({image_display_name})：API 返回的内容不是预期的代码列表格式。内容: {clean_content}")
                    return None
            except json.JSONDecodeError:
                if clean_content in CHARACTER_MAPPING: # 处理API直接返回单个代码字符串的情况
                    return [clean_content]
                print(f"错误 ({image_display_name})：API 返回的内容不是有效的 JSON 列表。内容: {clean_content}")
                return None
        else:
            print(f"错误 ({image_display_name})：API 返回了有效响应，但未找到生成的文本。API 响应: {api_response_data}")
            return None

    except requests.exceptions.Timeout:
        print(f"错误 ({image_display_name})：请求 Aihubmix API 超时（超过 {timeout} 秒）。")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"错误 ({image_display_name})：Aihubmix API 返回 HTTP 错误: {e.response.status_code} {e.response.reason}")
        try:
            error_details = e.response.json()
            print(f"错误详情 ({image_display_name}): {error_details}")
        except json.JSONDecodeError:
            print(f"无法解析的错误响应体 ({image_display_name}): {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"错误 ({image_display_name})：请求 Aihubmix API 时出错: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"错误 ({image_display_name})：解析 Aihubmix API 响应结构时出错: {e}。")
        print(f"请检查 API 响应是否符合预期结构。API 响应 ({image_display_name}): {api_response_data if api_response_data else 'N/A'}")
        return None
    except Exception as e: # pylint: disable=broad-except
        print(f"错误 ({image_display_name})：发生未知错误: {e}")
        return None


# --- 单个图像处理任务函数 ---
def process_single_image_task(
    scene_image_file_path_obj: Path,
    ref_image_path_str: str,
    current_model_id: str,
    max_tokens: int,
    request_timeout: int,
    output_base_dir: Path,
    input_root_path: Path
):
    """处理单个图像：API调用、结果解析、文件复制。"""
    try:
        relative_image_path = scene_image_file_path_obj.relative_to(input_root_path)
    except ValueError:
        relative_image_path = scene_image_file_path_obj.name

    image_display_name = str(relative_image_path) # 用于日志
    print(f"开始处理图像: {image_display_name}")

    character_codes = get_character_codes_from_image(
        str(scene_image_file_path_obj),
        ref_image_path_str,
        current_model_id,
        max_tokens,
        request_timeout,
        CHARACTER_MAPPING, # 传递完整的映射给API函数
        image_display_name
    )

    if character_codes:
        print(f"结果 ({image_display_name}): 识别角色代码: {character_codes}")
        copied_count = 0
        for code in character_codes:
            if code in CHARACTER_MAPPING:
                target_character_dir = output_base_dir / code
                destination_file = target_character_dir / scene_image_file_path_obj.name
                try:
                    # 确保目标子目录存在 (理论上已在主函数预创建，但双重检查无害)
                    target_character_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(scene_image_file_path_obj), str(destination_file))
                    print(f"  ({image_display_name}): 已将图像复制到 '{destination_file}'")
                    copied_count += 1
                except Exception as e: # pylint: disable=broad-except
                    print(f"  错误 ({image_display_name}): 复制图像到 '{destination_file}' 失败: {e}")
            else:
                print(f"  警告 ({image_display_name}): API返回了未在CHARACTER_MAPPING定义的代码 '{code}'，跳过复制。")
        if copied_count == 0 and character_codes != ["00"]: # 如果有有效代码但都复制失败
             print(f"  警告 ({image_display_name}): 未能成功复制图片到任何目标文件夹。")

    else:
        print(f"结果 ({image_display_name}): 未能从API获取角色代码或解析响应。该图片将不会被复制。")

    return image_display_name, True if character_codes else False # 返回图片名和处理状态


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 Aihubmix API 并发分析图像中的角色，并将图片复制到按角色代码分类的子文件夹中。"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/250514-chmr",
        help="包含各剧集源文件夹（如“9集”、“10集”）的根目录路径。脚本会排除名为“场景1”的子文件夹作为输入源。"
    )
    parser.add_argument(
        "--ref_image",
        type=str,
        default="data/char-ref.jpeg",
        help="包含角色名称标签的角色参考图文件路径。 (默认为: data/char-ref.jpeg)。"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=getattr(cfg, 'MODEL_ID', 'gemini-2.5-flash-preview-05-20-nothink'),
        help="指定要使用的模型 ID (默认为配置文件中的值或 'gemini-2.5-flash-preview-05-20-nothink')。"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(10, os.cpu_count() + 4 if os.cpu_count() else 8), # 默认为CPU核心数+4或10中较小者，至少为8
        help="并发处理的工作线程数量 (默认为 CPU核心数+4 和 10 中的较小值, 如果无法获取核心数则为8)。"
    )

    args = parser.parse_args()

    current_model_id = args.model
    max_tokens = getattr(cfg, 'MAX_TOKENS_OUTPUT', 200)
    request_timeout = getattr(cfg, 'REQUEST_TIMEOUT', 120)

    input_root_path = Path(args.input_dir)
    OUTPUT_BASE_DIR = Path("data/250514-chmr/场景1") # 固定输出目录

    if not input_root_path.is_dir():
        print(f"错误：输入根目录 '{args.input_dir}' 不是一个有效的目录。")
        exit(1)
    if not Path(args.ref_image).is_file():
        print(f"错误：参考图像 '{args.ref_image}' 未找到或不是一个文件。")
        exit(1)

    try:
        OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"输出目录 '{OUTPUT_BASE_DIR}' 已确保存在。")
        for code_key in CHARACTER_MAPPING.keys():
            (OUTPUT_BASE_DIR / code_key).mkdir(exist_ok=True)
        print(f"所有角色代码子目录已在 '{OUTPUT_BASE_DIR}' 下确保存在。")
    except OSError as e:
        print(f"错误：创建输出目录 '{OUTPUT_BASE_DIR}' 或其子目录失败: {e}")
        exit(1)

    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    collected_source_files = []
    print(f"\n正在扫描输入目录 '{input_root_path}' 下的子文件夹以查找图片...")
    for item_in_root in input_root_path.iterdir():
        if item_in_root.is_dir():
            if item_in_root.name == OUTPUT_BASE_DIR.name: # 排除 "场景1"
                print(f"  跳过扫描源目录 '{item_in_root.name}' (与输出目标目录同名)。")
                continue
            print(f"  正在扫描子文件夹 '{item_in_root.name}' 中的图片...")
            for image_file_path in item_in_root.glob('**/*'):
                if image_file_path.is_file() and image_file_path.suffix.lower() in image_extensions:
                    collected_source_files.append(image_file_path)
    scene_image_files = sorted(list(set(collected_source_files)))

    if not scene_image_files:
        print(f"在输入目录 '{args.input_dir}' 的合格子文件夹中没有找到支持的图像文件。")
        exit(1)

    total_images = len(scene_image_files)
    print(f"\n总共找到 {total_images} 张图像准备进行并发处理 (使用 {args.workers} 个工作线程)...")
    print(f"将使用以下 CHARACTER_MAPPING (代码: 角色名) 指导LLM:")
    for code, name in CHARACTER_MAPPING.items():
        if code != "00":
            print(f'  "{code}": "{name}"')
    print(f"如果未找到以上角色，或为空镜，LLM应输出: [\"00\"]\n")

    futures = []
    processed_count = 0
    successful_api_calls = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for scene_image_file_path_obj in scene_image_files:
            future = executor.submit(
                process_single_image_task,
                scene_image_file_path_obj,
                args.ref_image,
                current_model_id,
                max_tokens,
                request_timeout,
                OUTPUT_BASE_DIR,
                input_root_path
            )
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            processed_count += 1
            try:
                image_name, api_success = future.result()
                if api_success:
                    successful_api_calls +=1
                print(f"--- 完成 ({processed_count}/{total_images}): {image_name} ---")
            except Exception as e: # pylint: disable=broad-except
                # 异常应该在 task 内部被捕获并打印，这里是备用
                print(f"--- 错误 ({processed_count}/{total_images}): 处理一个图像时发生顶层异常: {e} ---")


    print(f"\n--- 所有 {total_images} 个图像任务已提交并等待完成 ---")
    print(f"总共处理完成: {processed_count} 张图片。")
    print(f"其中 API 成功获取角色代码（或判定为'00'）: {successful_api_calls} 次。")
    print(f"图片已根据识别结果复制到 '{OUTPUT_BASE_DIR}' 下的对应子文件夹。")

# --- END OF FILE character_org_modified_concurrent.py ---