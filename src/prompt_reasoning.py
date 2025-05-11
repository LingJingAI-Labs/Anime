# --- START OF MODIFIED prompt_reasoning.py ---
import os
import base64
import argparse # argparse 仅用于 if __name__ == "__main__" 测试块
import requests 
import json

# --- Configuration ---
AIHUBMIX_API_KEY = os.getenv("AIHUBMIX_API_KEY")
AIHUBMIX_API_URL = "https://aihubmix.com/v1/chat/completions"
MODEL_ID = "gemini-2.5-flash-preview-04-17-nothink" 
MAX_TOKENS_OUTPUT = 300 
REQUEST_TIMEOUT = 60 

class PromptGenerationError(Exception):
    """自定义异常，用于提示生成过程中的错误。"""
    pass

# --- Helper Function: Encode Image to Base64 ---
def encode_image_to_base64(image_path: str) -> str: # 返回 str, 出错则抛异常
    """将图像文件编码为 Base64 字符串。出错时抛出 PromptGenerationError。"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise PromptGenerationError(f"无法找到图像文件 '{image_path}'")
    except Exception as e:
        raise PromptGenerationError(f"读取或编码图像 '{image_path}' 时出错: {e}")

# --- Core Function: Generate Anime Prompt via Aihubmix ---
def generate_anime_prompt(image_path: str, status_callback=None) -> str: # 返回 str, 出错则抛异常
    """
    使用 Aihubmix API 分析图像，并生成动漫风格的提示词。
    出错时抛出 PromptGenerationError。
    status_callback: 一个可选函数(str) -> None，用于报告进度/状态信息。
    """
    if not AIHUBMIX_API_KEY:
        raise PromptGenerationError("AIHUBMIX_API_KEY 环境变量未设置。请设置环境变量 'AIHUBMIX_API_KEY' 后重试。例如：export AIHUBMIX_API_KEY='你的密钥'")

    image_basename = os.path.basename(image_path)

    if status_callback:
        status_callback(f"为 '{image_basename}' 编码图像...")
    
    # 1. 编码图像 (encode_image_to_base64 内部会抛出 PromptGenerationError)
    base64_image = encode_image_to_base64(image_path)

    image_url = f"data:image/jpeg;base64,{base64_image}" 

    headers = {
        "Authorization": f"Bearer {AIHUBMIX_API_KEY}",
        "Content-Type": "application/json",
    }

    user_prompt_text = ( # Renamed for clarity
        "请仔细分析提供的这张中国古装短剧的图像。\n"
        "你的任务是生成一个详细的英文文本提示词 (prompt)，这个提示词可以被 AI 图像生成模型（Stable Diffusion）使用，来创作出这张图像的【2D动漫风格】版本。\n"
        "提示词应该包含关键元素、主体、构图、色彩、光线、氛围，并明确指示生成【动漫或动画艺术风格】(anime or animation art style)。\n"
        "请专注于捕捉原始图像的精髓，并将其转化为动漫美学。\n"
        "如果出现人物，注意描述人物的服装、动作、姿态、表情等。\n"
        "如果图像中没有出现人物/人脸，则在提示词开头添加'Empty shot'\n"
        "输出只需要包含生成的英文提示词本身，不要添加任何额外的解释或说明文字。"
    )

    payload_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "high" 
                    },
                },
            ],
        }
    ]

    data = {
        "model": MODEL_ID, # 使用全局 MODEL_ID
        "messages": payload_messages,
        "max_tokens": MAX_TOKENS_OUTPUT
    }

    if status_callback:
        status_callback(f"正在通过 Aihubmix (模型: {MODEL_ID}) 为 '{image_basename}' 生成提示词...")
    
    response_obj = None # 用于在异常处理中引用 response 对象
    response_json_content = None # 用于在异常处理中引用已解析的json
    try:
        response_obj = requests.post(
            AIHUBMIX_API_URL,
            headers=headers,
            data=json.dumps(data), 
            timeout=REQUEST_TIMEOUT 
        )
        response_obj.raise_for_status() # 检查 HTTP 错误

        response_json_content = response_obj.json() # 解析 JSON

        # 提取提示词 (与你原代码逻辑相同)
        generated_prompt = response_json_content.get('choices', [{}])[0].get('message', {}).get('content')

        if generated_prompt:
            if status_callback:
                status_callback(f"成功为 '{image_basename}' 获取到 API 生成的提示词。")
            return generated_prompt.strip()
        else:
            # API 成功返回但内容不符合预期
            raise PromptGenerationError(f"API 返回了有效响应，但未能从中提取生成的文本。响应内容: {response_json_content}")

    except requests.exceptions.Timeout:
        raise PromptGenerationError(f"请求 Aihubmix API 超时（超过 {REQUEST_TIMEOUT} 秒）。")
    except requests.exceptions.HTTPError as e:
        error_msg = f"Aihubmix API 返回 HTTP 错误: {e.response.status_code} {e.response.reason}"
        try:
            error_details = e.response.json()
            error_msg += f" - 错误详情: {error_details}"
        except json.JSONDecodeError:
            error_msg += f" - 无法解析的错误响应体: {e.response.text}"
        raise PromptGenerationError(error_msg)
    except requests.exceptions.RequestException as e:
        raise PromptGenerationError(f"请求 Aihubmix API 时出错: {e}")
    except json.JSONDecodeError:
        err_text = response_obj.text if response_obj else "无响应对象"
        status_code = response_obj.status_code if response_obj else "N/A"
        raise PromptGenerationError(f"无法解析 Aihubmix API 返回的 JSON 响应。状态码: {status_code}, 响应体: {err_text[:500]}...") # 截断过长的响应体
    except (KeyError, IndexError) as e:
        raise PromptGenerationError(f"解析 Aihubmix API 响应结构时出错: {e}。响应内容: {response_json_content}")
    except Exception as e: # 捕获其他所有未预料的错误
        raise PromptGenerationError(f"生成提示词过程中发生未知错误: {type(e).__name__} - {e}")

# --- Main Execution Block (用于独立测试此文件) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 Aihubmix API 为输入图像生成动漫风格的 AI 绘画提示词。"
    )
    parser.add_argument(
        "--image_path", 
        default="data/250508/场景1/01/32C04 特写主角，向前伸手.png", # 确保这个默认路径存在或修改它
        type=str,      
        help="需要生成提示词的输入图像文件路径。" 
    )
    parser.add_argument(
        "--model",     
        type=str,
        default=MODEL_ID, 
        help=f"指定要使用的模型 ID (默认为: {MODEL_ID})。确保模型支持图像输入。"
    )
    args = parser.parse_args()

    # 更新全局 MODEL_ID 如果命令行指定了 (在 generate_anime_prompt 中已使用全局 MODEL_ID)
    # 注意：如果希望命令行参数覆盖全局，generate_anime_prompt 也需要接收 model_id 参数
    # 为简单起见，这里假设全局 MODEL_ID 是我们想要的，或在调用前已设置好
    
    print(f"测试 `prompt_reasoning.py` 独立运行...")
    print(f"使用的 API 密钥 (前几位): {AIHUBMIX_API_KEY[:5]}..." if AIHUBMIX_API_KEY else "未设置 API 密钥!")
    print(f"使用的模型: {args.model}") # 注意: 当前 generate_anime_prompt 使用全局 MODEL_ID

    # 定义一个简单的回调给独立测试用
    def cli_status_callback(message):
        print(f"[状态回调] {message}")

    try:
        # 确保测试时 MODEL_ID 被正确设置 (如果希望命令行参数生效)
        # 例如: prompt_reasoning.MODEL_ID = args.model (如果 MODEL_ID 是模块级变量)
        # 或者，修改 generate_anime_prompt 以接受 model_id 参数
        
        # 为了让命令行 --model 生效，我们临时修改全局 MODEL_ID (仅在 __main__ 中)
        # 更好的方式是 generate_anime_prompt 接受 model_id 参数
        _original_model_id = MODEL_ID # 保存原始值
        MODEL_ID = args.model # 临时设置为命令行参数值

        anime_prompt = generate_anime_prompt(args.image_path, status_callback=cli_status_callback)
        
        MODEL_ID = _original_model_id # 恢复原始值

        if anime_prompt:
            print("\n--- 生成的动漫风格提示词 ---")
            print(anime_prompt)
        # else: # 如果返回 None (在修改后的版本中，应该是抛出异常)
            # print("\n未能生成提示词。")
    except PromptGenerationError as e:
        print(f"\n错误：生成提示词失败: {e}")
    except Exception as e:
        print(f"\n发生意外的顶级错误: {e}")

# --- END OF MODIFIED prompt_reasoning.py ---