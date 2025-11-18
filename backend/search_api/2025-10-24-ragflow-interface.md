# Ragflow搜索接口调用示例文件生成记录

## 生成时间
2025年10月24日

## 生成内容
创建了完整的Ragflow搜索接口调用示例文件，用于演示如何与Ragflow知识库进行交互和搜索。

## 文件路径
- `backend/search_api/ragflow_client_demo.py` - Ragflow搜索接口调用示例代码

## 功能说明
该示例文件实现了以下功能：

1. **完整的Ragflow客户端类**：`RagflowSearchClient`类封装了所有与Ragflow API交互的功能
2. **环境变量配置**：支持从`.env`文件加载RAGFLOW_BASE_URL、RAGFLOW_API_KEY和RAGFLOW_DATASET_IDS配置
3. **同步与异步搜索**：同时提供异步方法`search_knowledge_base`和同步封装`search_sync`
4. **参数化搜索**：支持自定义查询、结果数量限制、数据集选择和相似度阈值
5. **详细的错误处理**：包含HTTP错误、JSON解析错误和其他异常的完整捕获
6. **结果格式化**：自动将API返回的原始数据转换为结构化的搜索结果
7. **演示功能**：提供`demo_search`函数展示如何使用客户端进行实际搜索

## 主要参数说明

### API请求参数
- `question`: 搜索关键词
- `dataset_ids`: 要搜索的数据集ID列表
- `page_size`: 返回结果数量
- `similarity_threshold`: 相似度过滤阈值，默认为0.3
- `top_k`: 内部检索数量，设为返回数量的10倍但不超过1024
- `keyword`: 是否启用关键词搜索，设为True
- `highlight`: 是否启用高亮显示，设为False

### 返回结果结构
每个搜索结果包含以下字段：
- `title`: 文档标题
- `content`: 匹配的内容片段
- `score`: 相似度得分
- `document_id`: 文档唯一标识符
- `kb_id`: 知识库ID
- `chunk_id`: 内容片段ID
- `metadata`: 详细元数据信息

## 使用方法

1. 确保已配置`.env`文件，包含必要的Ragflow配置信息
2. 直接运行示例文件：`python ragflow_client_demo.py`
3. 或在其他Python代码中导入并使用：
   ```python
   from ragflow_client_demo import RagflowSearchClient
   
   # 初始化客户端
   client = RagflowSearchClient()
   
   # 执行搜索
   success, results = client.search_sync("你的搜索关键词", limit=10)
   ```

## 注意事项
- 确保Ragflow服务可正常访问，且API Key有效
- 数据集ID必须正确配置，否则无法返回相关结果
- 可以根据实际需求调整相似度阈值以获取更精确或更广泛的结果
- 建议在生产环境中添加适当的日志记录和缓存机制以提高性能