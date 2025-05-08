import os
import base64
import argparse
import requests # 使用 requests 库进行 HTTP 调用
import json

# --- Configuration ---
# 从环境变量读取 Aihubmix API 密钥
# 在终端设置: export AIHUBMIX_API_KEY='你的aihubmix_api_key'
AIHUBMIX_API_KEY = os.getenv("AIHUBMIX_API_KEY")
# Aihubmix API 端点 URL
AIHUBMIX_API_URL = "https://aihubmix.com/v1/chat/completions"
# 指定要使用的模型 ID (确保这个模型支持图像输入，如 gpt-4o, gpt-4-vision-preview 等)
# 你可以根据 Aihubmix 支持的模型列表进行修改
MODEL_ID = "gemini-2.5-flash-preview-04-17-nothink" # 或者 "gpt-4o-mini", "gpt-4-vision-preview" 等
# MODEL_ID = "gpt-4o-mini"
MAX_TOKENS_OUTPUT = 300 # 限制生成提示词的长度
REQUEST_TIMEOUT = 60 # 请求超时时间（秒）

# --- Helper Function: Encode Image to Base64 ---
def encode_image_to_base64(image_path):
    """将图像文件编码为 Base64 字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"错误：无法找到图像文件 '{image_path}'")
        return None
    except Exception as e:
        print(f"错误：读取或编码图像时出错: {e}")
        return None

# --- Core Function: Generate Anime Prompt via Aihubmix ---
def generate_anime_prompt(image_path: str) -> str | None:
    """
    使用 Aihubmix API (背后可能是 OpenAI Vision 模型) 分析图像，并生成动漫风格的提示词。

    Args:
        image_path: 输入图像的文件路径。

    Returns:
        生成的动漫风格提示词字符串，如果出错则返回 None。
    """
    if not AIHUBMIX_API_KEY:
        print("错误：AIHUBMIX_API_KEY 环境变量未设置。")
        print("请设置环境变量 'AIHUBMIX_API_KEY' 后重试。")
        print("例如：export AIHUBMIX_API_KEY='你的密钥'")
        return None

    # 1. 编码图像
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None # 错误信息已在 encode_image_to_base64 中打印

    # 构建图像数据 URL (假设是 JPEG 或 PNG)
    image_url = f"data:image/jpeg;base64,{base64_image}" # 可以根据需要改为 image/png

    # 2. 构建发送给 Aihubmix API 的 Headers 和 Data (Payload)
    headers = {
        "Authorization": f"Bearer {AIHUBMIX_API_KEY}",
        "Content-Type": "application/json",
    }

    # 这里的 user_prompt 指导模型如何工作
    user_prompt = (
        "请仔细分析提供的这张中国古装短剧的图像。\n"
        "你的任务是生成一个详细的英文文本提示词 (prompt)，这个提示词可以被 AI 图像生成模型（Stable Diffusion）使用，来创作出这张图像的【2D动漫风格】版本。\n"
        "提示词应该包含关键元素、主体、构图、色彩、光线、氛围，并明确指示生成【动漫或动画艺术风格】(anime or animation art style)。\n"
        "请专注于捕捉原始图像的精髓，并将其转化为动漫美学。\n"
        "如果出现人物，注意描述人物的服装、动作、姿态、表情等。\n"
        # "如果是男性请务必在提示词开头添加'1boy'\n"
        # "如果是女性请务必在提示词开头添加'1girl, '\n"
        "如果图像中没有出现人物/人脸，则在提示词开头添加'Empty shot'\n"
        "输出只需要包含生成的英文提示词本身，不要添加任何额外的解释或说明文字。"
    )

    # 构建符合 OpenAI Vision API (以及兼容的代理 API) 的消息结构
    payload_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "high" # 'high' 获取更多细节, 'low' 节省 token
                    },
                },
            ],
        }
    ]

    data = {
        "model": MODEL_ID,
        "messages": payload_messages,
        "max_tokens": MAX_TOKENS_OUTPUT # 传递 max_tokens 参数
        # 根据 Aihubmix 文档，可能还需要添加其他参数，例如 temperature, top_p 等
    }

    # 3. 调用 Aihubmix API
    print(f"正在通过 Aihubmix 使用模型 '{MODEL_ID}' 分析图像并生成提示词...")
    response_json = None # 初始化，用于错误处理中访问
    try:
        response = requests.post(
            AIHUBMIX_API_URL,
            headers=headers,
            data=json.dumps(data), # 将 Python 字典转为 JSON 字符串
            timeout=REQUEST_TIMEOUT # 设置超时
        )
        # 检查 HTTP 响应状态码，如果不是 2xx，则引发异常
        response.raise_for_status()

        # 4. 解析响应
        response_json = response.json()

        # 尝试从响应中提取生成的提示词
        # 注意：这里假设 Aihubmix 的响应结构与 OpenAI 兼容
        # 如果 Aihubmix 的响应结构不同，你需要修改下面的代码路径
        generated_prompt = response_json.get('choices', [{}])[0].get('message', {}).get('content')

        if generated_prompt:
             # 去除可能存在的前后多余空格或换行符
            return generated_prompt.strip()
        else:
            print(f"错误：API 返回了有效响应，但未找到生成的文本。响应内容: {response_json}")
            return None

    except requests.exceptions.Timeout:
        print(f"错误：请求 Aihubmix API 超时（超过 {REQUEST_TIMEOUT} 秒）。")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"错误：Aihubmix API 返回 HTTP 错误: {e.response.status_code} {e.response.reason}")
        # 尝试打印返回的错误详情
        try:
            error_details = e.response.json()
            print(f"错误详情: {error_details}")
        except json.JSONDecodeError:
            print(f"无法解析的错误响应体: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        # 处理其他 requests 相关的错误 (如连接错误)
        print(f"错误：请求 Aihubmix API 时出错: {e}")
        return None
    except json.JSONDecodeError:
        # 如果响应不是有效的 JSON
        print(f"错误：无法解析 Aihubmix API 返回的 JSON 响应。状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        return None
    except (KeyError, IndexError) as e:
        # 如果响应 JSON 结构不符合预期 (如缺少 'choices' 或 'message')
        print(f"错误：解析 Aihubmix API 响应结构时出错: {e}。")
        print(f"请检查 API 响应是否符合预期结构。响应内容: {response_json}")
        return None
    except Exception as e:
        # 捕获其他未知错误
        print(f"错误：发生未知错误: {e}")
        return None

# --- Main Execution Block ---
if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description="使用 Aihubmix API (模型如 GPT-4o) 为输入图像生成动漫风格的 AI 绘画提示词。"
    )
    parser.add_argument(
        "--image_path", # 参数名称
        default="data/250508/场景1/01/32C04 特写主角，向前伸手.png",
        type=str,      # 参数类型
        help="需要生成提示词的输入图像文件路径。" # 帮助信息
    )
    parser.add_argument(
        "--model",     # 添加一个可选参数来指定模型
        type=str,
        default=MODEL_ID, # 使用配置中的默认值
        help=f"指定要使用的模型 ID (默认为: {MODEL_ID})。确保模型支持图像输入。"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 更新全局模型 ID（如果通过命令行指定了）
    MODEL_ID = args.model

    # 调用核心函数并打印结果
    anime_prompt = generate_anime_prompt(args.image_path)

    if anime_prompt:
        print("\n--- 生成的动漫风格提示词 ---")
        print(anime_prompt)
    else:
        print("\n未能生成提示词。请检查上面的错误信息。")