import logging
import os

import click
import uvicorn

from adk_agent_executor import ADKAgentExecutor
from dotenv import load_dotenv
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from starlette.routing import Route
from google.adk.agents.run_config import RunConfig, StreamingMode
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from starlette.middleware.cors import CORSMiddleware
from starlette.applications import Starlette
from agent import root_agent

# 加载环境变量
load_dotenv()

# 配置日志格式和级别
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def create_app(agent_card=None):
    """
    创建应用实例的工厂函数
    """
    streaming = os.environ.get("STREAMING") == "true"
    
    if agent_card is None:
        agent_card_name = "outline Agent"
        agent_name = "outline_agent"
        # Agent描述必须清晰
        agent_description = "Generate an outline based on the user's requirements"

        # 定义 agent 的技能
        skill = AgentSkill(
            id=agent_name,
            name=agent_card_name,
            description=agent_description,
            tags=["outline"],
            examples=["outline"],
        )
        
        agent_url = os.environ.get("AGENT_URL", "http://localhost:10050/")
        # 构建 agent 卡片信息
        agent_card = AgentCard(
            name=agent_card_name,
            description=agent_description,
            url=agent_url,
            version="1.0.0",
            defaultInputModes=["text"],
            defaultOutputModes=["text"],
            capabilities=AgentCapabilities(streaming=streaming),
            skills=[skill],
        )

    # 初始化 Runner，管理 agent 的执行、会话、记忆和产物
    logger.info("初始化Runner...")
    runner = Runner(
        app_name=agent_card.name,
        agent=root_agent,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
    )

    # 根据环境变量决定是否启用流式输出
    if streaming:
        logger.info("使用 SSE 流式输出模式")
        run_config = RunConfig(
            streaming_mode=StreamingMode.SSE,
            max_llm_calls=500
        )
    else:
        logger.info("使用普通输出模式")
        run_config = RunConfig(
            streaming_mode=StreamingMode.NONE,
            max_llm_calls=500
        )

    # 初始化 agent 执行器
    agent_executor = ADKAgentExecutor(runner, agent_card, run_config)

    # 请求处理器，管理任务存储和请求分发
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )

    # 构建 Starlette 应用
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    app = a2a_app.build()
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app

# 创建全局应用实例
app = create_app()

@click.command()
@click.option("--host", "host", default="localhost", help="服务器绑定的主机名（默认为 localhost,可以指定具体本机ip）")
@click.option("--port", "port", default=10050, help="服务器监听的端口号（默认为 10050）")
@click.option("--workers", "workers", default=4, help="工作进程数（默认为 4）")
@click.option("--agent_url", "agent_url", default="",help="Agent Card中对外展示和访问的地址")
def main(host: str, port: int, workers: int = 4, agent_url: str = None):
    """
    启动 Outline Agent 服务，支持流式和非流式两种模式。
    """
    logger.info("启动 Outline Agent 服务")
    logger.info(f"工作进程数: {workers}")
    streaming = os.environ.get("STREAMING") == "true"
    logger.info(f"流式模式: {streaming}")
    
    # 构建 agent 卡片信息
    agent_card_name = "outline Agent"
    agent_name = "outline_agent"
    agent_description = "Generate an outline based on the user's requirements"
    
    if not agent_url:
        agent_url = f"http://{host}:{port}/"
        
    agent_card = AgentCard(
        name=agent_card_name,
        description=agent_description,
        url=agent_url,
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=streaming),
        skills=[AgentSkill(
            id=agent_name,
            name=agent_card_name,
            description=agent_description,
            tags=["outline"],
            examples=["outline"],
        )],
    )

    logger.info(f"服务启动中，监听地址: http://{host}:{port}")
    # 如果workers大于1，则使用多进程模式
    if workers > 1:
        logger.info(f"使用多进程模式启动，工作进程数: {workers}")
        # 为多进程模式创建一个包装函数
        def app_factory():
            return create_app(agent_card)
        uvicorn.run("main_api:app_factory", host=host, port=port, workers=workers, factory=True)
    else:
        logger.info("使用单进程模式启动")
        # 使用传入的agent_card创建应用实例
        app_instance = create_app(agent_card)
        # 启动 uvicorn 服务器
        uvicorn.run(app_instance, host=host, port=port)

if __name__ == "__main__":
    main()