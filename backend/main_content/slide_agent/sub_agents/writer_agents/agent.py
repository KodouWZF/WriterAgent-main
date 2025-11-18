import json
import os
import time
import logging
from typing import Dict, List, Any, AsyncGenerator, Optional, Union
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents import LoopAgent, BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from .tools import SearchImage, DocumentSearch
from .utils import stringify_references
from ...config import CONTENT_WRITER_AGENT_CONFIG
from ...create_model import create_model
from . import prompt
from . import index_filter
from . import rotator_logger
from .fast_checker_agent import fast_checker_agent

# 配置：日志目录、前缀、保留天数
LOG_DIR = "logs"
PREFIX = "writer_agent"
KEEP_DAYS = 7  # 保留最近 7 天；改成 0 测试会删除今天之前的所有历史文件
logger = rotator_logger.setup_daily_logger(log_dir=LOG_DIR, prefix=PREFIX, keep_days=KEEP_DAYS)

def my_before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmRequest]:
    """
    在调用LLM模型之前执行的回调函数
    
    Args:
        callback_context (CallbackContext): 回调上下文，包含agent状态信息
        llm_request (LlmRequest): 发送给LLM的请求对象
        
    Returns:
        Optional[LlmRequest]: 返回处理后的请求对象，如果返回None则继续使用原始请求
    """
    agent_name = callback_context.agent_name
    history_length = len(llm_request.contents)
    metadata = callback_context.state.get("metadata")
    print(f"调用了{agent_name}模型前的callback, 现在Agent共有{history_length}条历史记录,metadata数据为：{metadata}")
    logger.info(f"=====>>>1. 调用了{agent_name}.my_before_model_callback, 现在Agent共有{history_length}条历史记录,metadata数据为：{metadata}\n历史会话为：{llm_request.contents}")

    return None

def my_after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    """
    在调用LLM模型之后执行的回调函数
    
    Args:
        callback_context (CallbackContext): 回调上下文，包含agent状态信息
        llm_response (LlmResponse): 从LLM返回的响应对象
        
    Returns:
        Optional[LlmResponse]: 返回处理后的响应对象，如果返回None则继续使用原始响应
    """
    # 1. 检查用户输入，注意如果是llm的stream模式，那么response_data的结果是一个token的结果，还有可能是工具的调用
    agent_name = callback_context.agent_name
    response_parts = llm_response.content.parts
    part_texts =[]
    for one_part in response_parts:
        part_text = getattr(one_part, "text", None)
        if part_text is not None:
            part_texts.append(part_text)
    part_text_content = "\n".join(part_texts)
    metadata = callback_context.state.get("metadata")
    callback_context.state["last_draft"] = part_text_content
    print(f"调用了{agent_name}模型后的callback, 这次模型回复{response_parts}条信息,metadata数据为：{metadata},回复内容是: {part_text_content}")
    logger.info(f"=====>>>2. 调用了{agent_name}.my_after_model_callback,metadata数据为：{metadata},回复内容是: {part_text_content}")

    return None

# --- 1. Custom Callback Functions for PPTWriterSubAgent ---
def my_writer_before_agent_callback(callback_context: CallbackContext) -> None:
    """
    在WriterSubAgent调用LLM之前执行的回调函数
    用于记录当前正在生成的内容块索引和总块数等信息
    
    Args:
        callback_context (CallbackContext): 回调上下文，包含agent状态信息
    """
    agent_name = callback_context.agent_name
    current_part_index: int = callback_context.state.get("current_part_index", 0)  # Default to 0
    parts_plan_num = callback_context.state.get("parts_plan_num")
    metadata = callback_context.state.get("metadata")
    logger.info(f"=====>>>3. 调用了{agent_name}.my_writer_before_agent_callback, metadata数据为：{metadata}，当前块索引：{current_part_index}，总分块索引：{parts_plan_num}")
    # 返回 None，继续调用 LLM
    return None


def my_after_agent_callback(callback_context: CallbackContext) -> None:
    """
    在WriterSubAgent调用LLM之后执行的回调函数
    用于记录LLM生成的内容并保存到会话状态中
    
    Args:
        callback_context (CallbackContext): 回调上下文，包含agent状态信息
    """
    # 获取当前回调上下文中的agent名称
    agent_name = callback_context.agent_name
    # 获取模型最后一次输出的内容（通常是生成的文本）
    model_last_output_content = callback_context._invocation_context.session.events[-1]
    # 提取内容的各个部分（parts）
    response_parts = model_last_output_content.content.parts
    # 初始化一个列表来存储所有文本部分
    part_texts = []
    # 遍历所有内容部分
    for one_part in response_parts:
        # 尝试获取每个部分的文本内容
        part_text = getattr(one_part, "text", None)
        # 如果文本内容存在，则添加到列表中
        if part_text is not None:
            part_texts.append(part_text)
    # 将所有文本部分用换行符连接成一个完整的字符串
    part_text_content = "\n".join(part_texts)
    # 从回调上下文状态中获取元数据
    metadata = callback_context.state.get("metadata")
    # 记录日志，显示agent名称、元数据和回复内容
    logger.info(f"=====>>>4. 调用了{agent_name}.my_after_agent_callback,metadata数据为：{metadata},回复内容是: {part_text_content}")
    # 返回None，表示不修改回调上下文
    return None

class WriterSubAgent(LlmAgent):
    """
    内容撰写子Agent类，继承自LlmAgent
    负责根据大纲结构逐块生成综述内容
    """
    def __init__(self, **kwargs):
        """
        初始化WriterSubAgent实例
        
        Args:
            **kwargs: 传递给父类LlmAgent的其他参数
        """
        super().__init__(
            # Agent名称，用于标识和日志记录
            name="WriterSubAgent",
            # 使用的AI模型配置，指定模型类型和提供商
            model=create_model(model=CONTENT_WRITER_AGENT_CONFIG["model"], provider=CONTENT_WRITER_AGENT_CONFIG["provider"]),
            # Agent的描述信息，说明其功能和使用场景
            description="综述撰写助手，根据要求撰写综述的每一块内容。因为你写的是一块内容，不要在末尾添加参考文献列表。",
            # 动态指令获取函数，用于在运行时生成具体的指令
            instruction=self._get_dynamic_instruction,
            # Agent执行前的回调函数，用于执行预处理操作
            before_agent_callback=my_writer_before_agent_callback,
            # Agent执行后的回调函数，用于执行后处理操作
            after_agent_callback=my_after_agent_callback,
            # 模型调用前的回调函数，可在模型请求前进行处理
            before_model_callback=my_before_model_callback,
            # 模型调用后的回调函数，可在模型响应后进行处理
            after_model_callback=my_after_model_callback,
            # Agent可用的工具列表，这里提供了文档搜索功能
            tools=[DocumentSearch],
            # 其他传递给父类的参数
            **kwargs
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        异步运行WriterSubAgent的核心实现方法
        
        Args:
            ctx (InvocationContext): 调用上下文，包含会话状态等信息
            
        Yields:
            Event: 生成的事件对象
        """
        # 获取总计划块数和当前块索引
        parts_plan_num: int = ctx.session.state.get("parts_plan_num")
        current_part_index: int = ctx.session.state.get("current_part_index", 0)
        rewrite_retry_count_map = ctx.session.state.get("rewrite_retry_count_map", {})
        
        # 清空历史记录，防止历史记录进行干扰
        if int(rewrite_retry_count_map.get(current_part_index, 0)) > 0:
            logger.info(f"=====>>>6. 当前正在进行对: 第{current_part_index}个块重新生成")
            del_history = ctx.session.events.pop()
            logger.info(f"=============>>>删除了最后1个内容块：\n{del_history}")
            logger.info(f"=============>>>删除后的历史记录为：\n{ctx.session.events}")
        else:
            logger.info(f"=====>>>6. 当前计划块数{parts_plan_num}, 正在生成第{current_part_index}块内容，清空历史记录")
            ctx.session.events = []
            
        logger.info(f"=====>>>7. 总的计划块数{parts_plan_num}, 正在生成第{current_part_index}块内容...")
        
        # 根据当前块索引确定要生成的内容结构
        current_part_index: int = ctx.session.state.get("current_part_index", 0)
        if current_part_index == 0:
            # 第0块是摘要部分
            last_struct = ctx.session.state["abstract"]
        else:
            # 其他块是正文部分
            sections = ctx.session.state["sections"]
            last_struct = sections[current_part_index - 1]
        ctx.session.state["last_struct"] = last_struct
        
        # 调用父类逻辑（最终结果）
        async for event in super()._run_async_impl(ctx):
            print(f"{self.name} 收到事件：{event}")
            logger.info(f"=====>>>5. {self.name} 收到事件：{event}")
            # 不返回结果给前端，等待审核通过后返回
            yield event

    def _get_dynamic_instruction(self, ctx: InvocationContext) -> str:
        """
        根据当前上下文动态生成指令(prompt)
        
        Args:
            ctx (InvocationContext): 调用上下文
            
        Returns:
            str: 生成的指令字符串
        """
        # 获取当前块索引
        current_part_index: int = ctx.state.get("current_part_index", 0)
        
        # 获取语言参数
        language = ctx.state.get("language")
        
        if current_part_index == 0:
            # 生成摘要部分
            current_type = "abstract"
            abstract_outline = ctx.state["abstract"]
            print(f"准备生成第一部分，摘要内容:{abstract_outline}")
            part_prompt = prompt.prompt_mapper[current_type]
            title = ctx.state["title"]
            prompt_instruction = prompt.PREFIX_PROMPT.format(TITLE=title, language=language) + part_prompt.format(ABSTRACT_STRUCT=abstract_outline, TITLE=title,language=language)
        else:
            # 生成正文部分
            current_type = "body"
            sections = ctx.state["sections"]
            print(f"总的块数是{len(sections)}，要进行生成的是第{current_part_index-1}块")
            section_outline = sections[current_part_index-1]  # 因为abstract占据了1个索引，所以需要去掉1
            print(f"准备生成正文的第{current_part_index-1}块内容: {section_outline}")
            # 当前生成文章的这块的要求汇入prompt
            # 根据不同的类型，形成不同的prompt
            part_prompt = prompt.prompt_mapper[current_type]
            title = ctx.state["title"]
            existing_text = ctx.state["existing_text"]
            prompt_instruction = prompt.PREFIX_PROMPT.format(TITLE=title, language=language) + part_prompt.format(SECTION_STRUCT=section_outline,language=language)
        print(f"第{current_part_index}块的prompt是：{prompt_instruction}")
        return prompt_instruction


class ControllerAgent(BaseAgent):
    """
    控制器Agent类，继承自BaseAgent
    负责根据检查结果决定是否提交当前块内容并进入下一块，或者要求重新生成
    """
    def __init__(self, **kwargs):
        """
        初始化ControllerAgent实例
        
        Args:
            **kwargs: 传递给父类BaseAgent的其他参数
        """
        super().__init__(
            name="ControllerAgent",
            description="根据 Checker 结果决定是否提交并推进到下一块，或要求重写。",
            **kwargs
        )

    async def _run_async_impl(self, ctx: InvocationContext):
        """
        异步运行ControllerAgent的核心实现方法
        
        Args:
            ctx (InvocationContext): 调用上下文，包含会话状态等信息
            
        Yields:
            Event: 生成的事件对象
        """
        # 设置最大重试次数
        max_retries = 3
        
        # 获取当前状态信息
        parts_plan_num: int = ctx.session.state.get("parts_plan_num")
        idx: int = ctx.session.state.get("current_part_index", 0)
        checker_result = ctx.session.state.get("checker_result")
        retry_map = ctx.session.state.get("rewrite_retry_count_map", {})
        idx_mapping = ctx.session.state.get("idx_mapping", {})

        if checker_result:
            # ✅ 检查通过：提交当前块内容并推进到下一块
            sections = ctx.session.state.get("existing_sections", [])
            sections.append(ctx.session.state.get("last_draft", ""))
            ctx.session.state["existing_sections"] = sections
            ctx.session.state["existing_text"] = "\n".join(sections)
            ctx.session.state["current_part_index"] = idx + 1
            test_sections1 = ctx.session.state.get("existing_sections", [])
            logger.info(f"====================================>>>当块通过时，检查此时sections的内容：{test_sections1}")
            
            # 如果已经到达最后一个块，则结束整个生成过程
            if idx + 1 == parts_plan_num:
                references = ctx.session.state.get("references", {})
                refs_text = stringify_references(references=references)
                if refs_text:
                    final_bib, missing = index_filter.finalize_bibliography(bib_text=refs_text, final_mapping=idx_mapping)
                    logger.info(f"=====>>>12. 最后一条消息结束，有参考资料，即将发送给请求端：{final_bib}")
                    yield Event(author=self.name, content=types.Content(parts=[types.Part(text=final_bib)]))
                else:
                    logger.info(f"=====>>>12. 注意：最后一块内容已经撰写完成，但是参考引用为空，请检查搜索引擎是否正常。")
                yield Event(author=self.name, actions=EventActions(escalate=True))
            else:
                logger.info(f"=====>>>13. 第 {idx} 块校验通过，进入第 {idx+1} 块。")

        else:
            # ❌ 检查未通过：判断是否还能重试
            # 获取当前块的重试次数
            count = int(retry_map.get(idx, 0))
            if count >= max_retries:
                # 达到最大重试次数：警告并继续推进到下一块
                warn = f"第 {idx} 块重试超过 {max_retries} 次，将保留最近草稿（可能仍不完全合规）。"
                sections = ctx.session.state.get("existing_sections", [])
                sections.append(ctx.session.state.get("last_draft", ""))
                ctx.session.state["existing_sections"] = sections
                ctx.session.state["existing_text"] = "\n".join(sections)
                ctx.session.state["current_part_index"] = idx + 1
                test_sections = ctx.session.state.get("existing_sections", [])
                logger.info(f"====================================>>>当块达到最大尝试次数时，检查此时sections的内容：{test_sections}")
                
                # 如果已经到达最后一个块，则结束整个生成过程
                if idx + 1 == parts_plan_num:
                    references = ctx.session.state.get("references", {})
                    refs_text = stringify_references(references=references)
                    if refs_text:
                        final_bib, missing = index_filter.finalize_bibliography(bib_text=refs_text, final_mapping=idx_mapping)
                        yield Event(author=self.name, content=types.Content(parts=[types.Part(text=final_bib)]))
                    logger.info(f"=====>>>12. warning: {warn}")
                    yield Event(author=self.name, actions=EventActions(escalate=True))
                else:
                    logger.info(f"=====>>>12. warning: {warn}")
            else:
                # 还能重试：增加重试计数并触发重写
                retry_map[idx] = count + 1
                ctx.session.state["rewrite_retry_count_map"] = retry_map
                logger.info(f"=====>>>13. 第 {idx} 块未通过，将触发重写（第 {count + 1} 次重试）。")


def my_super_before_agent_callback(callback_context: CallbackContext):
    """
    在LoopAgent调用之前执行的回调函数
    用于初始化各种状态变量
    
    Args:
        callback_context (CallbackContext): 回调上下文
    """
    title = callback_context.state.get("title","")
    sections = callback_context.state.get("sections","")
    abstract = callback_context.state.get("abstract","")
    logger.info(f"=================>>>提取到的标题为：{title}")
    logger.info(f"=================>>>提取到的章节为：{sections}")
    logger.info(f"=================>>>提取到的摘要为：{abstract}")

    # 初始化各种状态变量
    callback_context.state["existing_text"] = ""  # 已经生成的内容
    callback_context.state["idx_mapping"] = {}  # 新旧索引映射
    # 初始化重试次数记录
    if "rewrite_retry_count_map" not in callback_context.state:
        callback_context.state["rewrite_retry_count_map"] = {}
    if "existing_text" not in callback_context.state:
        callback_context.state["existing_text"] = ""
    if "existing_sections" not in callback_context.state:
        callback_context.state["existing_sections"] = []
    if "last_struct" not in callback_context.state:
        callback_context.state["last_struct"] = ""
    if "current_part_index" not in callback_context.state:
        callback_context.state["current_part_index"] = 0
    if "idx_mapping" not in callback_context.state:
        callback_context.state["idx_mapping"] = {}
    return None

# --- 4. WriterGeneratorLoopAgent ---

# 创建循环Agent，用于循环生成各个内容块
writer_generator_loop_agent = LoopAgent(
    name="WriterGeneratorLoopAgent",
    max_iterations=100,  # 设置一个足够大的最大迭代次数，以防万一。主要依赖ConditionAgent停止。
    sub_agents=[
        WriterSubAgent(),   # 生成草稿 -> state["last_draft"]
        fast_checker_agent,     # 校验 -> state["checker_result"] (可选择快速或传统检查器)
        ControllerAgent(),  # 决策：提交/递增/终止 或 触发重写
    ],
    before_agent_callback=my_super_before_agent_callback,
)