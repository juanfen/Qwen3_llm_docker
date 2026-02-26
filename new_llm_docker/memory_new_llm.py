import os
import re
import json
import torch
import jieba
import warnings
import uuid
import json
import time
from collections import defaultdict

warnings.filterwarnings("ignore",
                        message="Relevance scores must be between 0 and 1",
                        category=UserWarning)
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from glob import glob
from transformers import TextStreamer
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from asyncio import sleep
import asyncio
from typing import AsyncGenerator
from threading import Thread
# from yz_chatbot_tts import TTS
from pydantic import BaseModel
import logging
import numpy as np
import scipy.io.wavfile as wavfile
import io
import base64
from openai import OpenAI
from datetime import datetime
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = "cuda:0"
app = FastAPI(title="扬州文化旅游专家聊天机器人",
              description="基于Qwen模型的扬州文化旅游专家聊天机器人")


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    tools: list[dict] = []
    tool_choice: str = "auto"


def save_messages_to_file(messages, save_dir="logs"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(save_dir, f"chat_{timestamp}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"✅ 对话已保存到: {file_path}")


# 读取请求里的部分prompt
def extract_context_block(text: str) -> str:
    """
    提取 <context> 标签中的内容，去掉首尾空白和换行
    """
    soup = BeautifulSoup(text, "html.parser")
    tag = soup.find("context")
    if tag:
        return tag.get_text(strip=True)
    return ""


def extract_context_block_i(text: str) -> str:
    """
    提取 <identity> 标签中的内容，去掉首尾空白和换行
    """
    soup = BeautifulSoup(text, "html.parser")
    tag = soup.find("identity")
    if tag:
        return tag.get_text(strip=True)
    return ""


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    # 调用你的本地模型
    query = request.messages[-1]['content']
    # print(f"messages:{request.messages}")
    content = chatbot.llm.sys_template + request.messages[0]['content']
    # print(f"content:{content}")

    retrieved_docs = chatbot.retrieval.retrieve(query)
    # context = "\n\n".join([doc.metadata.get("content") for doc, _ in retrieved_docs[:2]])
    texts = [
        (doc.page_content or "").strip()
        for doc, _ in retrieved_docs[:3]
        if hasattr(doc, "page_content") and doc.page_content is not None
    ]
    context = "\n\n".join(texts)
    context_summary = context
    # print(context_summary)

    # print(f"request.messages:{request.messages}")

    current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")

    # message拼接 系统提示词+摘要+mem0检索记忆（5条）+短期上下文窗口（两轮对话）+query

    messages = [
        {
            "role": "system",
            "content": f"""{content}+当用户未表达明确需求或相关问题指定对话内容时，可根据以下内容作为聊天话题发散对话{context_summary}
            """
        },

        # *previous_messages,
        {
            "role": "user",
            "content": query
        },

    ]

    # print(f"query: {query}")
    print(f"messages:{messages}")

    # 构建异步生成器
    async def stream_response():
        full_response = ""
        response = chatbot.llm.model.chat.completions.create(
            model="Qwen3",
            messages=messages,
            stream=True,
            tools=request.tools,
            tool_choice=request.tool_choice,
            extra_body={"repetition_penalty": 1.3}

        )
        # print(response)
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                # 构造标准 OpenAI SSE 响应格式
                full_response += delta.content
                data = {
                    "id": str(uuid.uuid4()),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "Qwen3",
                    "choices": [{
                        "delta": {"content": delta.content},
                        "index": 0,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(0.001)  # 释放事件循环，避免阻塞
        yield "data: [DONE]\n\n"
        # log_entry["response"] = full_response
        # with open("chatbot_footbal_history.json", "a", encoding='UTF-8') as f:
        #     f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return StreamingResponse(stream_response(), media_type="text/event-stream")


class HybridRetrieval:
    def __init__(self, corpus: List[Document]):
        self.corpus = corpus
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="/mnt/Data/chenjiahui/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cuda:0'},
            encode_kwargs={'normalize_embeddings': True})

        # 关键修改1：使用完整内容构建检索索引
        self.vectorstore = FAISS.from_documents(corpus, self.embedding_model)
        self.bm25 = BM25Okapi([list(jieba.cut(doc.page_content)) for doc in corpus])

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        # 1. 获取双路检索结果（基于完整内容）
        embedding_results = self.vectorstore.similarity_search_with_relevance_scores(query, k=40)
        bm25_scores = self.bm25.get_scores(list(jieba.cut(query)))
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:40]

        # 2. 构建RRF排名映射（使用原始doc对象）
        doc_ranks = defaultdict(dict)
        for rank, (doc, _) in enumerate(embedding_results):
            doc_id = doc.metadata["source"]
            doc_ranks[doc_id]["embed_rank"] = rank + 1

        for rank, i in enumerate(bm25_top_indices):
            doc_id = self.corpus[i].metadata["source"]
            doc_ranks[doc_id]["bm25_rank"] = rank + 1

        # 3. 计算RRF分数
        k = 60
        rrf_scores = {}
        for doc_id, ranks in doc_ranks.items():
            score = 1 / (k + ranks.get("embed_rank", float("inf"))) + \
                    6 / (k + ranks.get("bm25_rank", float("inf")))
            score_1 = 1 / (k + ranks.get("embed_rank", float("inf")))
            score_2 = 6 / (k + ranks.get("bm25_rank", float("inf")))
            # print (score_1, score_2)
            rrf_scores[doc_id] = score

        # 4. 关联原始文档
        scored_docs = []
        for doc in self.corpus:
            if doc.metadata["source"] in rrf_scores:
                scored_docs.append((doc, rrf_scores[doc.metadata["source"]]))

        # 5. 按分数排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        import random
        candidates = scored_docs[:20]
        sampled = random.sample(candidates, k=min(5, len(candidates)))
        scored_docs = sampled
        return scored_docs[:top_k]


class QwenChatbot:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = OpenAI(
            base_url="http://192.168.99.105:8194/v1",
            api_key="no-key-needed"
        )
        self.sys_template = """你是小德，在问题超出已知信息范围时，无法确定回答时，请不要胡编乱造"""

        self.template = """##  

## 检索语料库内容

{context} 

你将根据用户问题，在系统提供的检索结果中提取最相关内容用于回答。

## 用户问题

{query}
"""
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    def generate_response(self, query: str, context: str, streamer=None) -> str:
        raise NotImplementedError


class YangzhouTourismChatbot:
    def __init__(self, corpus: List[str], model_path: str):
        self.retrieval = HybridRetrieval(corpus)
        self.llm = QwenChatbot(model_path)

    def chat(self, query: str) -> str:
        retrieved_docs = self.retrieval.retrieve(query, top_k=5)
        context = "\n\n".join([doc for doc, _ in retrieved_docs])
        response = self.llm.generate_response(query, context)
        return response


if __name__ == "__main__":
    corpus = []
    base_path = '/mnt/Data/chenjiahui/topic'

    # 加载所有目录下的文件，不再分类
    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            if any(file_path.endswith(ext) for ext in ['.txt', '.md', '.json']):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 按行处理所有文件
                for line_num, line in enumerate(content.splitlines(), 1):
                    if line.strip():
                        corpus.append(Document(
                            page_content=line.strip(),
                            metadata={
                                "source": f"{file_path}#L{line_num}"
                            }
                        ))

    model_path = "/mnt/Data/jungao/Qwen3"

    # Initialize chatbot
    chatbot = YangzhouTourismChatbot(corpus, model_path)

    # Chat with the bot
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8183)  # 8179

