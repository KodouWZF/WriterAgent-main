import asyncio
import json
import re
import os
import dotenv
import uuid
import httpx
import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from outline_client import A2AOutlineClientWrapper
from content_client import A2AContentClientWrapper
import sqlite3
from datetime import datetime
from typing import Optional
dotenv.load_dotenv()

"""
后端:
- 仅生成"综述类论文"的 Markdown；**不再返回 JSON 结构**，也**不再提供静态资源**。
- 路由改为：
  - POST /api/review_outline  -- 生成大纲（Markdown，支持流式）
  - POST /api/review          -- 生成正文章节（Markdown，支持流式）；可按大纲逐章生成
- 典型用法：先调 /api/review_outline 得到大纲，再根据所选章节标题/序号逐个调 /api/review。
"""
OUTLINE_API = os.environ["OUTLINE_API"]
CONTENT_API = os.environ["CONTENT_API"]

# ==================== 数据库相关功能 ====================
# 初始化SQLite数据库连接和表结构
def init_db():
    """
    初始化数据库，创建SQLite数据库文件和对话历史表
    
    功能说明:
    1. 连接到SQLite数据库(如果不存在会自动创建)
    2. 创建conversations表用于存储对话历史
    3. 表字段说明:
       - id: 主键，自动递增
       - uuid: 唯一标识符，用于关联请求
       - topic: 对话主题
       - outline: 大纲内容
       - content: 正文内容
       - created_at: 记录创建时间，默认为当前时间戳，格式为'YYYY-MM-DD HH:MM:SS'
    """
    # 连接到SQLite数据库，如果文件不存在会自动创建
    conn = sqlite3.connect('conversations.db')
    cursor = conn.cursor()
    
    # 创建对话历史表
    # 使用IF NOT EXISTS避免重复创建表的错误
    # 确保created_at字段使用TEXT类型并以'YYYY-MM-DD HH:MM:SS'格式存储时间戳
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,           -- 主键，自动递增
            uuid TEXT UNIQUE,                               -- 唯一标识符
            topic TEXT,                                     -- 对话主题
            outline TEXT,                                   -- 大纲内容
            content TEXT,                                   -- 正文内容
            created_at TEXT DEFAULT (datetime('now', 'localtime'))  -- 创建时间戳，格式为'YYYY-MM-DD HH:MM:SS'
        )
    ''')
    
    # 提交事务并关闭连接
    conn.commit()
    conn.close()

# 在应用启动时初始化数据库
# 确保每次应用启动时都会检查并创建必要的数据库结构
init_db()

# 保存对话历史到数据库
def save_conversation_to_db(uuid: str, topic: str, outline: str, content: str):
    """
    将完整的对话历史保存到数据库中
    
    参数说明:
    - uuid (str): 请求的唯一标识符，用于追踪和关联特定请求
    - topic (str): 用户提供的主题内容
    - outline (str): 生成的大纲文本内容
    - content (str): 生成的正文文本内容
    
    功能说明:
    1. 连接到SQLite数据库
    2. 使用INSERT OR REPLACE语句保存数据
       - 如果相同uuid的记录已存在，则更新该记录
       - 如果不存在，则插入新记录
    3. 处理可能发生的异常并打印错误信息
    """
    try:
        # 建立数据库连接
        conn = sqlite3.connect('conversations.db')
        cursor = conn.cursor()
        
        # 插入或更新对话记录
        # 使用占位符(?)防止SQL注入攻击
        cursor.execute('''
            INSERT OR REPLACE INTO conversations (uuid, topic, outline, content)
            VALUES (?, ?, ?, ?)
        ''', (uuid, topic, outline, content))
        
        # 提交事务并关闭连接
        conn.commit()
        conn.close()
        
        # 打印成功日志
        print(f"对话历史已保存到数据库，UUID: {uuid}")
    except Exception as e:
        # 打印错误信息便于调试
        print(f"保存对话历史到数据库时出错: {e}")

# ==================== FastAPI应用配置 ====================
app = FastAPI(title="Review Paper Generator (Markdown, Mock)", version="1.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReviewOutlineRequest(BaseModel):
    topic: str = Field(..., description="综述主题，例如 '多模态大模型安全性'")
    language: str = Field("zh", description="'zh' 或 'en'")

class ReviewChapterRequest(BaseModel):
    topic: str = Field(..., description="综述主题")
    outline: str|dict = Field(..., description="大纲数组。如果不传，使用默认模板大纲")
    language: str = Field("zh", description="'zh' 或 'en'")

class FullReviewRequest(BaseModel):
    topic: str = Field(..., description="综述主题")
    language: str = Field("zh", description="'zh' 或 'en'")
    uuid: str = Field(None, description="请求的唯一标识符")
    callback_url: str = Field(None, description="结果回调地址")

async def stream_outline_response(prompt: str):
    """A generator that yields parts of the agent response."""
    outline_wrapper = A2AOutlineClientWrapper(session_id=uuid.uuid4().hex, agent_url=OUTLINE_API)
    async for chunk_data in outline_wrapper.generate(prompt):
        if chunk_data.get("text", "") != "" and ("<think>" in chunk_data.get("text") or "</think>" in chunk_data.get("text")):
            continue
        print(f"生成大纲输出的chunk_data: {chunk_data}")
        yield "data: " + json.dumps(chunk_data,ensure_ascii=False) + "\n\n"

@app.post("/api/outline")
async def api_review_outline(req: ReviewOutlineRequest):
    topic = (req.topic or "示例主题").strip()
    return StreamingResponse(stream_outline_response(topic), media_type="text/event-stream; charset=utf-8")

class AipptContentRequest(BaseModel):
    content: str

async def stream_content_response(markdown_content: str):
    """  # PPT的正文内容生成"""
    # 用正则找到第一个一级标题及之后的内容
    match = re.search(r"(# .*)", markdown_content, flags=re.DOTALL)

    if match:
        result = markdown_content[match.start():]
    else:
        result = markdown_content
    print(f"用户输入的markdown大纲是：{result}")
    content_wrapper = A2AContentClientWrapper(session_id=uuid.uuid4().hex, agent_url=CONTENT_API)
    async for chunk_data in content_wrapper.generate(result):
        print(f"生成文章内容的输出的chunk_data: {chunk_data}")
        yield "data: " + json.dumps(chunk_data,ensure_ascii=False) + "\n\n"
@app.post("/api/review")
async def api_review(req: ReviewChapterRequest):
    topic = (req.topic or "示例主题").strip()
    outline = req.outline
    return StreamingResponse(stream_content_response(outline), media_type="text/event-stream; charset=utf-8")

class GenerateParagraphRequest(BaseModel):
    prompt: str = Field(..., description="要进行改写或者生成的提示词，例如某个段落")
    option: str = Field(..., description="要进行操作的命令，根据不同的操作要求，和prompt进行更多生成")

@app.post("/api/pragraph")
async def api_pragraph(req: GenerateParagraphRequest):
    print(f"进行改写或者生成，prompt是：{req.prompt}")

@app.post("/api/full-review")
async def api_full_review(req: FullReviewRequest):
    """生成完整综述内容（大纲+正文）"""
    # 接收请求后立即确认
    if not req.uuid:
        return JSONResponse({
            "message": "请提供请求的唯一标识符uuid"
        })
    
    # 创建后台任务处理完整流程
    asyncio.create_task(process_full_review_async(req))
    
    # 立即返回响应，不等待任务完成
    return JSONResponse({
        "message": "请求已接收，正在后台处理中，稍等5-8分钟返回结果到指定客户端",
        "uuid": req.uuid,
        "callback_url": req.callback_url
    })

async def process_full_review_async(req: FullReviewRequest):
    """
    异步处理完整综述生成流程的后台任务
    包括生成大纲、生成正文、发送回调和保存数据库等步骤
    """
    topic = (req.topic or "示例主题").strip()
    
    try:
        # 1. 生成大纲
        outline_wrapper = A2AOutlineClientWrapper(session_id=req.uuid, agent_url=OUTLINE_API)
        
        # 收集完整的大纲内容
        outline_content = ""
        async for chunk_data in outline_wrapper.generate(topic):
            if chunk_data.get("text", ""):
                outline_content += chunk_data["text"]

        # 清理大纲内容中的思考标记
        for mt in ["<think>", "</think>"]:
            outline_content = outline_content.replace(f"{mt}", "")

        # 2. 基于大纲生成正文内容
        content_wrapper = A2AContentClientWrapper(session_id=req.uuid, agent_url=CONTENT_API)
        
        # 收集完整的正文内容
        full_content = ""
        async for chunk_data in content_wrapper.generate(outline_content):
            if chunk_data.get("text", ""):
                full_content += chunk_data["text"]

        # 3. 发送回调请求到指定URL
        requests_result = requests.post(
            req.callback_url, 
            json = {
            "outline": outline_content,
            "content": full_content,
            "uuid": req.uuid
            }
        )

        # 4. 保存到数据库
        save_conversation_to_db(req.uuid, topic, outline_content, full_content)
        
        # 根据回调结果做进一步处理
        if requests_result.status_code != 200:
            print(f"回调请求失败，状态码: {requests_result.status_code}")
            
    except Exception as e:
        print(f"处理过程中出错: {e}")
        # 记录错误日志
        import traceback
        traceback.print_exc()

@app.get("/ping")
async def Ping():
    return "PONG"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7800)