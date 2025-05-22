# config_characters.py
import os

# --- Configuration ---
# 从环境变量读取 Aihubmix API 密钥
# 在终端设置: export AIHUBMIX_API_KEY='你的aihubmix_api_key'
AIHUBMIX_API_KEY = os.getenv("AIHUBMIX_API_KEY")
if not AIHUBMIX_API_KEY:
    print("警告: AIHUBMIX_API_KEY 环境变量未设置。脚本可能无法调用 API。")
    print("请设置环境变量 'AIHUBMIX_API_KEY' 后重试。")
    print("例如：export AIHUBMIX_API_KEY='你的密钥'")

# Aihubmix API 端点 URL
AIHUBMIX_API_URL = "https://aihubmix.com/v1/chat/completions"

# 指定要使用的模型 ID
MODEL_ID = "gemini-2.5-flash-preview-05-20-nothink" # 或者 "gpt-4o", "gpt-4o-mini"

# 增加 token 输出以容纳 JSON
MAX_TOKENS_OUTPUT = 1024 # Increased for JSON output
REQUEST_TIMEOUT = 120 # 请求超时时间（秒）, increased for potentially complex analysis

# --- Character Mapping ---
# 用户可以自行修改或者补充
# 键是角色名 (必须与参考图中的标签以及模型可能识别出的名称一致)
# 值是两位数字的文件夹名
DEFAULT_CHARACTER_MAP = {
    "秦云": "01",
    "萧灵": "02",
    "蔡成安": "03",
    "蔡晓霞 ": "03",
    "张思思": "07",
    "虎哥": "09",
    "绿衣男": "10",
    "周学兵": "11",
    "售货员": "12",
    "王大妈": "13",
    "周雪": "15",
    "刘大爷": "16",
}