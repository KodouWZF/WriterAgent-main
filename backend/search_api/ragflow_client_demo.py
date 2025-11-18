#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ragflow搜索接口调用示例

此文件提供了使用Ragflow API进行知识库搜索的完整示例
包括客户端初始化、参数配置和搜索结果处理
"""

import os
import json
import httpx
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class RagflowSearchClient:
    """
    Ragflow知识库搜索客户端
    用于与Ragflow API进行交互，执行知识库检索操作
    """
    
    def __init__(self, 
                 base_url: Optional[str] = None, 
                 api_key: Optional[str] = None, 
                 dataset_ids: Optional[List[str]] = None,
                 default_limit: int = 10):
        """
        初始化Ragflow客户端
        
        Args:
            base_url: Ragflow服务基础URL，若不提供则从环境变量读取
            api_key: Ragflow API密钥，若不提供则从环境变量读取
            dataset_ids: 数据集ID列表，若不提供则从环境变量读取
            default_limit: 默认返回结果数量
        """
        # 优先使用传入参数，若无则从环境变量读取
        self.base_url = (base_url or os.environ.get("RAGFLOW_BASE_URL"))
        self.api_key = (api_key or os.environ.get("RAGFLOW_API_KEY"))
        
        # 解析数据集ID
        if dataset_ids:
            self.dataset_ids = dataset_ids
        else:
            dataset_ids_str = os.environ.get("RAGFLOW_DATASET_IDS", "")
            self.dataset_ids = [id.strip() for id in dataset_ids_str.split(',') if id.strip()]
        
        self.default_limit = default_limit
        
        # 验证必要的配置
        self._validate_config()
        
        # 配置请求头
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # API端点
        self.retrieval_endpoint = f"{self.base_url.rstrip('/')}/api/v1/retrieval"
        
        print(f"Ragflow客户端初始化成功")
        print(f"  Base URL: {self.base_url}")
        print(f"  数据集数量: {len(self.dataset_ids)}")
        print(f"  API端点: {self.retrieval_endpoint}")
    
    def _validate_config(self):
        """
        验证配置是否完整
        """
        missing = []
        if not self.base_url:
            missing.append("RAGFLOW_BASE_URL")
        if not self.api_key:
            missing.append("RAGFLOW_API_KEY")
        if not self.dataset_ids:
            missing.append("RAGFLOW_DATASET_IDS")
        
        if missing:
            raise ValueError(f"缺少必要的配置项: {', '.join(missing)}")
    
    async def search_knowledge_base(self, 
                                   query: str, 
                                   limit: int = None,
                                   dataset_ids: List[str] = None,
                                   similarity_threshold: float = 0.3) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        在Ragflow知识库中搜索相关内容
        
        Args:
            query: 搜索关键词
            limit: 返回结果数量，默认使用初始化时的default_limit
            dataset_ids: 可选的数据集ID列表，若不提供则使用初始化时的数据集
            similarity_threshold: 相似度阈值，默认为0.3
            
        Returns:
            Tuple[bool, List[Dict]]: (是否成功, 搜索结果列表)
        """
        # 参数处理
        if limit is None:
            limit = self.default_limit
        
        # 使用指定的数据集ID或默认数据集
        search_dataset_ids = dataset_ids if dataset_ids else self.dataset_ids
        
        print(f"\n=== 开始Ragflow搜索 ===")
        print(f"搜索关键词: {query}")
        print(f"返回数量: {limit}")
        print(f"数据集ID: {search_dataset_ids}")
        print(f"相似度阈值: {similarity_threshold}")
        
        # 构建请求参数
        payload = {
            "question": query,
            "dataset_ids": search_dataset_ids,
            "page": 1,
            "page_size": limit,
            "similarity_threshold": similarity_threshold,
            "top_k": min(limit * 10, 1024),  # top_k设为limit的10倍但不超过1024
            "keyword": True,                # 启用关键词搜索
            "highlight": False              # 不启用高亮
        }
        
        try:
            print(f"发送请求到: {self.retrieval_endpoint}")
            print(f"请求参数: {json.dumps(payload, ensure_ascii=False)}")
            
            # 发送HTTP请求
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.post(
                    self.retrieval_endpoint,
                    headers=self.headers,
                    json=payload
                )
            
            # 检查响应状态
            print(f"响应状态码: {response.status_code}")
            
            # 解析响应
            result = response.json()
            print(f"API响应状态: {'成功' if result.get('code') == 0 else '失败'}")
            
            # 检查是否返回错误
            if result.get('code') != 0:
                error_msg = result.get('message', '未知错误')
                print(f"错误信息: {error_msg}")
                return False, []
            
            # 处理成功响应
            data = result.get("data", {})
            chunks = data.get("chunks", [])
            doc_aggs = data.get("doc_aggs", [])
            
            print(f"找到{len(chunks)}个相关片段")
            print(f"来自{len(doc_aggs)}个文档")
            
            # 构建文档ID到文档名称的映射
            doc_id_to_name = {}
            for doc_info in doc_aggs:
                doc_id = doc_info.get("doc_id", "")
                doc_name = doc_info.get("doc_name", "未知文档")
                doc_id_to_name[doc_id] = doc_name
            
            # 格式化搜索结果
            formatted_results = []
            for chunk in chunks:
                # 提取片段信息
                content = chunk.get("content", "")
                document_id = chunk.get("document_id", "")
                similarity = chunk.get("similarity", 0.0)
                kb_id = chunk.get("kb_id", "")
                chunk_id = chunk.get("id", "")
                
                # 获取文档名称
                doc_name = doc_id_to_name.get(document_id, "未知文档")
                
                # 构建结果对象
                result_item = {
                    "title": doc_name,
                    "content": content,
                    "score": similarity,
                    "document_id": document_id,
                    "kb_id": kb_id,
                    "chunk_id": chunk_id,
                    "metadata": {
                        "document_id": document_id,
                        "document_name": doc_name,
                        "kb_id": kb_id,
                        "similarity": similarity,
                        "chunk_id": chunk_id
                    }
                }
                formatted_results.append(result_item)
            
            # 按相似度排序
            formatted_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            
            # 限制返回数量
            final_results = formatted_results[:limit]
            
            print(f"返回{len(final_results)}个搜索结果")
            return True, final_results
            
        except httpx.RequestError as e:
            print(f"HTTP请求错误: {str(e)}")
            return False, []
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {str(e)}")
            print(f"原始响应内容: {response.text}")
            return False, []
        except Exception as e:
            print(f"搜索过程中发生错误: {str(e)}")
            import traceback
            print(f"错误堆栈:\n{traceback.format_exc()}")
            return False, []
    
    def search_sync(self, 
                   query: str, 
                   limit: int = None,
                   dataset_ids: List[str] = None,
                   similarity_threshold: float = 0.3) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        同步调用搜索接口
        
        Args:
            与search_knowledge_base方法相同
            
        Returns:
            与search_knowledge_base方法相同
        """
        # 创建事件循环并执行异步操作
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.search_knowledge_base(
                query=query,
                limit=limit,
                dataset_ids=dataset_ids,
                similarity_threshold=similarity_threshold
            )
        )
    
    def get_answer_from_knowledge_base(self, 
                                      question: str, 
                                      limit: int = None,
                                      dataset_ids: List[str] = None,
                                      similarity_threshold: float = 0.3) -> Tuple[bool, Dict[str, Any]]:
        """
        根据用户问题从知识库获取答案
        
        Args:
            question: 用户提出的问题
            limit: 返回结果数量，默认使用初始化时的default_limit
            dataset_ids: 可选的数据集ID列表，若不提供则使用初始化时的数据集
            similarity_threshold: 相似度阈值，默认为0.3
            
        Returns:
            Tuple[bool, Dict]: (是否成功, 包含答案和源信息的字典)
        """
        print(f"\n=== 从知识库获取答案 ===")
        print(f"用户问题: {question}")
        
        # 调用搜索接口获取相关内容
        success, search_results = self.search_sync(
            query=question,
            limit=limit,
            dataset_ids=dataset_ids,
            similarity_threshold=similarity_threshold
        )
        
        # 如果搜索失败，返回错误信息
        if not success:
            return False, {"error": "搜索失败，无法获取相关信息"}
        
        # 如果没有结果，返回无结果信息
        if not search_results:
            return False, {"error": "未找到与问题相关的信息"}
        
        # 构建答案响应
        answer_info = {
            "question": question,
            "top_result": search_results[0],  # 最相关的结果
            "sources": [],
            "answer": ""
        }
        
        # 提取答案内容和源信息
        content_parts = []
        for result in search_results:
            # 添加到内容部分（用于构建答案）
            content_parts.append(result.get("content", ""))
            
            # 构建源信息
            source_info = {
                "document_id": result.get("document_id", ""),
                "document_name": result.get("title", ""),
                "similarity_score": result.get("score", 0.0),
                "chunk_id": result.get("chunk_id", "")
            }
            answer_info["sources"].append(source_info)
        
        # 构建一个基本的答案（在实际应用中可能需要更复杂的处理或使用LLM生成更连贯的答案）
        # 这里简单地组合最相关的几个结果片段
        answer_content = " ".join([part[:300] for part in content_parts[:3]])  # 取前3个结果的前300个字符
        answer_info["answer"] = answer_content.strip()
        
        print(f"答案构建完成")
        print(f"  源文档数量: {len(answer_info['sources'])}")
        print(f"  答案长度: {len(answer_info['answer'])} 字符")
        
        return True, answer_info

def demo_search():
    """
    演示如何使用RagflowSearchClient进行搜索
    """
    try:
        # 初始化客户端
        client = RagflowSearchClient()
        
        # 示例1: 同步搜索
        print("\n=== 示例1: 同步搜索 ===")
        success, results = client.search_sync(
            query="煤化工业",
            limit=5,
            similarity_threshold=0.4
        )
        
        # 展示搜索结果
        if success and results:
            print(f"\n搜索成功，找到{len(results)}个结果：")
            for i, result in enumerate(results, 1):
                print(f"\n结果{i}:")
                print(f"  标题: {result['title']}")
                print(f"  相似度: {result['score']:.4f}")
                print(f"  文档ID: {result['document_id']}")
                print(f"  内容摘要: {result['content'][:100]}...")
        else:
            print("搜索失败或未找到结果")
        
        # 示例2: 指定数据集搜索
        print("\n=== 示例2: 指定数据集搜索 ===")
        # 可以在这里指定特定的数据集ID进行搜索
        # specific_dataset_ids = ["特定数据集ID"]
        # success, results = client.search_sync(
        #     query="特定查询",
        #     dataset_ids=specific_dataset_ids
        # )
        
    except Exception as e:
        print(f"演示过程中发生错误: {str(e)}")

if __name__ == "__main__":
    """
    主程序入口
    直接运行此文件将执行演示搜索
    """
    print("开始Ragflow搜索接口演示...")
    demo_search()
    print("\n演示结束")