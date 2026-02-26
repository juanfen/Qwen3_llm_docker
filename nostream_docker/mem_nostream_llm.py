import os
import re
import json
import warnings
import uuid
import time
from typing import List
from bs4 import BeautifulSoup
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from transformers import AutoTokenizer
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

device = "cuda:1"

app = FastAPI(
    title="扬州文化旅游专家聊天机器人",
    description="基于Qwen模型的扬州文化旅游专家聊天机器人"
)

from cachetools import TTLCache
session_request_count = TTLCache(maxsize=1000, ttl=600)

class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    tools: list[dict] = []
    tool_choice: str = "auto"

def extract_context_block(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    tag = soup.find("context")
    if tag:
        return tag.get_text(strip=True)
    return ""

def extract_context_block_i(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    tag = soup.find("identity")
    if tag:
        return tag.get_text(strip=True)
    return ""


def _blocking_chat_completion(messages, model="Qwen3", timeout=120.0):
    # 同步调用，放到线程池执行
    client = OpenAI(
        base_url="http://192.168.99.105:8194/v1",
        api_key="no-key-needed"
    )
    return client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        response_format={"type": "json_object"},
    )

async def _chat_completion_with_timeout(messages, timeout: float = 120.0):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=4) as pool:
        return await loop.run_in_executor(
            pool, _blocking_chat_completion, messages, "Qwen3", timeout
        )
# ===================================================================

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    query = request.messages[-1]["content"]
    content = request.messages[0]["content"]

    session_id = request.model
    count = session_request_count.get(session_id, 0) + 1
    session_request_count[session_id] = count

    messages = [
        {"role": "system", "content": content},
        {"role": "user", "content": query},
    ]
    # logging.info(f"query: {query}")

    response = await _chat_completion_with_timeout(messages, timeout=120.0)

    choice = response.choices[0].message
    content = getattr(choice, "content", None)

    clean_text = content.strip()
    clean_text = re.sub(r"^```json\s*|\s*```$", "", clean_text, flags=re.IGNORECASE).strip()
    if re.search(r'"\s*memory\s*"\s*:', clean_text):
        memory_list = json.loads(clean_text)
        print(f"响应一：{memory_list}")
        payload = memory_list  # 已经包含 memory 键
        payload_str = json.dumps(payload, ensure_ascii=False)
    else:
        memory_list = json.loads(clean_text)
        print(f"响应二：{memory_list}")
        payload = {"memory": memory_list}  # 不包含则包一层
        payload_str = json.dumps(payload, ensure_ascii=False)

    if "fact" in clean_text:
        output_text = clean_text
    else:
        output_text = payload_str

    return JSONResponse(content={
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "Qwen3",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": output_text},
            "finish_reason": "stop"
        }]
    })

class QwenChatbot:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = OpenAI(
            base_url="http://192.168.99.105:8194/v1",
            api_key="no-key-needed"
        )
        self.sys_template = """你是人"""

    def generate_response(self, user_prompt: str, system_prompt: str):
        # 修复：使用 self.model，避免引用未定义变量 chatbot
        response = self.model.chat.completions.create(
            model="Qwen3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            stream=False,
        )
        return response

class YangzhouTourismChatbot:
    def __init__(self, corpus: List[str], model_path: str):
        self.llm = QwenChatbot(model_path)

if __name__ == "__main__":
    corpus = []
    model_path = "/mnt/Data/jungao/Qwen3"
    chatbot = YangzhouTourismChatbot(corpus, model_path)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8178)