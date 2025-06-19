import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
import re
from file_loads import load_docs
from fastapi import FastAPI, Request, Query, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import glob
from fastapi.middleware.cors import CORSMiddleware
from document_loader import DocumentLoader
from pathlib import Path
import json
import urllib.parse

from database import db
import asyncio
from dotenv import load_dotenv
import time
from datetime import datetime
from functools import lru_cache
import hashlib
import pickle
from typing import Dict, List, Tuple, Optional
from contextlib import asynccontextmanager
import requests
import aiohttp
from mcp_api import router as mcp_router
import sqlite3

# 加载环境变量
load_dotenv()

def get_env_or_default(key: str, default: str = "") -> str:
    """获取环境变量，如果不存在则返回默认值"""
    return os.getenv(key, default)

def get_env_int(key: str, default: int = 0) -> int:
    """获取环境变量并转换为整数"""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default

# 直接从环境变量构建模型配置
def get_model_config():
    """从环境变量获取模型配置"""
    return {
        "glm-4-plus": {
            "model": get_env_or_default("GLM_4_PLUS_MODEL", "glm-4-plus"),
            "api_key": get_env_or_default("GLM_4_PLUS_API_KEY", ""),
            "api_base": get_env_or_default("GLM_4_PLUS_API_BASE", "https://open.bigmodel.cn/api/paas/v4/")
        },
        "deepseek": {
            "model": get_env_or_default("DEEPSEEK_MODEL", "DeepSeek-R1"),
            "api_key": get_env_or_default("DEEPSEEK_API_KEY", ""),
            "api_base": get_env_or_default("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        },
        "qwen": {
            "model": get_env_or_default("QWEN_MODEL", "qwen3-235b-a22b"),
            "api_key": get_env_or_default("QWEN_API_KEY", ""),
            "api_base": get_env_or_default("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        },
        "claude3.7": {
            "model": get_env_or_default("CLAUDE_MODEL", "claude-3-7-sonnet"),
            "api_key": get_env_or_default("CLAUDE_API_KEY", ""),
            "api_base": get_env_or_default("CLAUDE_API_BASE", "https://api.anthropic.com/v1")
        }
    }

def get_bocha_config():
    """从环境变量获取Bocha配置"""
    return {
        "api_key": get_env_or_default("BOCHA_API_KEY", ""),
        "api_base": get_env_or_default("BOCHA_API_BASE", "https://api.bochaai.com/v1/web-search"),
        "timeout": get_env_int("BOCHA_TIMEOUT", 30)
    }

# 获取配置
MODEL_CONFIG = get_model_config()
BOCHA_CONFIG = get_bocha_config()
DEFAULT_MODEL = get_env_or_default("DEFAULT_MODEL", "glm-4-plus")

def load_model():
    """加载或初始化模型"""
    local_model_path = 'local_m3e_model'
    if os.path.exists(local_model_path):
        print(f"从本地加载模型: {local_model_path}")
        model = SentenceTransformer(local_model_path)
    else:
        print(f"本地模型不存在，从网络加载: moka-ai/m3e-base")
        model = SentenceTransformer('moka-ai/m3e-base')
        print(f"保存模型到本地: {local_model_path}")
        model.save(local_model_path)
    # 强制使用GPU
    try:
        model = model.to('cuda')
        print('模型已转移到GPU')
    except Exception as e:
        print('未检测到可用GPU，使用CPU')
    return model

def chunk_document(text, max_chars=500, overlap=100, is_excel=False, file_name=None):
    """
    将长文档切分成较小的块，使用滑动窗口确保上下文连贯性
    
    参数:
        text: 要切分的文本
        max_chars: 每个块的最大字符数
        overlap: 相邻块之间的重叠字符数
        is_excel: 是否为Excel文件
        file_name: 文件名
    
    返回:
        chunks: 切分后的文本块列表
    """
    # 根据文件类型设置分块参数
    if is_excel:
        max_chars = 1000
        overlap = 0
    
    # 文件名前缀
    file_prefix = ""
    if file_name:
        file_prefix = f"[文件：{file_name}]\n"
    
    if len(text) <= max_chars:
        # 如果整个文本小于一个块的大小，直接添加文件名前缀返回
        if file_name and not text.startswith(file_prefix):
            return [f"{file_prefix}{text}"]
        return [text]
    
    # 移除可能存在的文件名前缀，以避免重复
    if file_name and text.startswith(file_prefix):
        text = text[len(file_prefix):]
    
    chunks = []
    start = 0
    last_end = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        
        if end < len(text):
            sentence_ends = [
                m.end() for m in re.finditer(r'[。！？.!?]\s*', text[start:end])
            ]
            
            if sentence_ends:
                end = start + sentence_ends[-1]
            else:  # 如果没有找到，尝试在单词或标点处切分
                last_space = text[start:end].rfind(' ')
                last_punct = max(text[start:end].rfind('，'), text[start:end].rfind(','))
                cut_point = max(last_space, last_punct)
                
                if cut_point > 0:
                    end = start + cut_point + 1
        
        # 每个chunk都添加文件名前缀
        chunk = text[start:end]
        if file_name:
            chunk = f"{file_prefix}{chunk}"
        
        chunks.append(chunk)
        
        if end <= last_end:
            end = min(last_end + 1, len(text))
            chunk = text[start:end]
            if file_name:
                chunk = f"{file_prefix}{chunk}"
            chunks[-1] = chunk
            
            if end >= len(text):
                break
        
        last_end = end
        start = end - overlap
        
        if start < 0:
            start = 0
            
        if start >= end:
            start = end
            
        if start >= len(text):
            break
    
    return chunks

# 添加缓存相关的常量和变量
CACHE_DIR = Path("cache")
VECTOR_CACHE_FILE = CACHE_DIR / "vector_cache.pkl"
EMBEDDING_CACHE_FILE = CACHE_DIR / "embedding_cache.pkl"

# 确保缓存目录存在
CACHE_DIR.mkdir(exist_ok=True)

class VectorCache:
    def __init__(self):
        self.cache: Dict[str, np.ndarray] = {}
        self.load_cache()

    def load_cache(self):
        """加载缓存"""
        if VECTOR_CACHE_FILE.exists():
            try:
                with open(VECTOR_CACHE_FILE, 'rb') as f:
                    self.cache = pickle.load(f)
            except Exception:
                self.cache = {}

    def save_cache(self):
        """保存缓存"""
        try:
            with open(VECTOR_CACHE_FILE, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception:
            pass

    def get(self, key: str) -> Optional[np.ndarray]:
        """获取缓存的向量"""
        return self.cache.get(key)

    def set(self, key: str, vector: np.ndarray):
        """设置向量缓存"""
        self.cache[key] = vector

class EmbeddingCache:
    def __init__(self):
        self.cache: Dict[str, np.ndarray] = {}
        self.load_cache()

    def load_cache(self):
        """加载缓存"""
        if EMBEDDING_CACHE_FILE.exists():
            try:
                with open(EMBEDDING_CACHE_FILE, 'rb') as f:
                    self.cache = pickle.load(f)
            except Exception:
                self.cache = {}

    def save_cache(self):
        """保存缓存"""
        try:
            with open(EMBEDDING_CACHE_FILE, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception:
            pass

    def get(self, key: str) -> Optional[np.ndarray]:
        """获取缓存的嵌入向量"""
        return self.cache.get(key)

    def set(self, key: str, vector: np.ndarray):
        """设置嵌入向量缓存"""
        self.cache[key] = vector

# 创建缓存实例
vector_cache = VectorCache()
embedding_cache = EmbeddingCache()

@lru_cache(maxsize=1000)
def get_cached_embeddings(text: str) -> np.ndarray:
    """缓存文本的嵌入向量"""
    # 首先检查持久化缓存
    cache_key = hashlib.md5(text.encode()).hexdigest()
    cached_vector = embedding_cache.get(cache_key)
    if cached_vector is not None:
        return cached_vector

    # 如果缓存中没有，计算新的嵌入向量
    vector = model.encode(text, convert_to_tensor=False)
    
    # 保存到持久化缓存
    embedding_cache.set(cache_key, vector)
    return vector

def get_query_hash(query):
    """生成查询的哈希值作为缓存键"""
    return hashlib.md5(query.encode()).hexdigest()

def get_embeddings(model, texts):
    """
    获取文本嵌入向量
    
    参数:
        model: 模型实例
        texts: 文本列表
    
    返回:
        numpy数组形式的嵌入向量
    """
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.array(embeddings)

def create_or_load_index(model, all_chunks, document_to_chunks, chunks_to_document):
    """创建或加载FAISS索引（已弃用，保留用于兼容性）"""
    index_path = 'faiss_index.faiss'
    chunks_map_path = 'chunks_mapping.npy'  

    if os.path.exists(index_path):
        print(f"从本地加载索引: {index_path}")
        index = faiss.read_index(index_path)
        return index
    else:
        print("索引文件不存在，需要重新初始化系统")
        return None

# 添加系统状态管理类
class SystemState:
    def __init__(self):
        self.initialized = False
        self.last_check = 0
        self.check_interval = 60  # 60秒检查一次
    
    def needs_check(self):
        current_time = time.time()
        return not self.initialized or (current_time - self.last_check) > self.check_interval
    
    def mark_initialized(self):
        self.initialized = True
        self.last_check = time.time()

# 创建系统状态实例
system_state = SystemState()

# 优化查询嵌入缓存
@lru_cache(maxsize=1000)
def get_cached_query_embedding(query_hash: str, query: str) -> np.ndarray:
    """缓存查询的嵌入向量"""
    return model.encode(query, convert_to_tensor=False)

def get_query_embedding_cached(query: str) -> np.ndarray:
    """获取缓存的查询嵌入向量"""
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return get_cached_query_embedding(query_hash, query)

def get_related_documents(model, index, query, all_chunks, k=5):
    """优化的文档检索函数"""
    if not all_chunks:
        return [], []
        
    # 确保k不超过chunks数量
    k = min(k, len(all_chunks))
    
    try:
        # 使用缓存的嵌入向量计算
        query_embedding = get_query_embedding_cached(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # 优化FAISS搜索参数
        if hasattr(index, 'nprobe') and hasattr(index, 'ntotal'):
            # 动态调整nprobe以平衡速度和准确性
            index.nprobe = min(32, max(1, index.ntotal // 100))
        
        # 使用faiss搜索最相似的向量
        distances, indices = index.search(query_embedding, k=k)
        
        # 检查indices是否为空
        if len(indices) == 0 or len(indices[0]) == 0:
            return [], []
            
        # 批量处理文档块，减少循环开销
        context = []
        valid_indices = []
        processed_files = set()  # 用于跟踪已处理的Excel文件
        
        # 直接处理索引，避免复杂的批量逻辑
        for idx in indices[0]:
            if not (0 <= idx < len(all_chunks)):
                continue
                
            chunk = all_chunks[idx]
            
            # 检查是否包含Excel文件标记
            if "[文件：" in chunk and any(ext in chunk for ext in ['.xlsx', '.xls']):
                # 提取文件名
                file_match = re.search(r'\[文件：(.*?)\]', chunk)
                if file_match:
                    file_name = file_match.group(1)
                    if file_name not in processed_files:
                        # 如果是Excel文件且未处理过，加载文件内容
                        try:
                            file_path = os.path.join('uploads', file_name)
                            doc_info = document_loader.load_document(file_path)
                            if doc_info['file_type'] == 'excel':
                                # 简化Excel处理，只取第一个工作表
                                if doc_info['sheet_contents']:
                                    first_sheet = list(doc_info['sheet_contents'].items())[0]
                                    context.append(f"工作表: {first_sheet[0]}\n{first_sheet[1]}")
                                valid_indices.append(idx)
                                processed_files.add(file_name)
                        except Exception:
                            # 如果加载失败，使用原始块
                            context.append(chunk)
                            valid_indices.append(idx)
                    else:
                        # 非Excel文件，直接使用原始块
                        context.append(chunk)
                        valid_indices.append(idx)
            else:
                # 非Excel文件，使用原始块
                context.append(chunk)
                valid_indices.append(idx)
                
        return context, valid_indices
    except Exception as e:
        print(f"文档检索错误: {e}")
        return [], []

async def web_search(query: str) -> str:
    """执行网络搜索"""
    try:
        # 构建请求
        api_url = BOCHA_CONFIG["api_base"]
        headers = {
            "Authorization": f"Bearer {BOCHA_CONFIG['api_key']}",
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "summary": True,
            "count": 10,
            "page": 1
        }
        
        # 发送请求
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload, timeout=BOCHA_CONFIG["timeout"]) as response:
                if response.status != 200:
                    return f"搜索失败，状态码：{response.status}"
                    
                data = await response.json()
                
                # 检查数据结构
                if "data" not in data:
                    return "未找到搜索结果"
                
                if "webPages" not in data["data"]:
                    return "未找到搜索结果"
                
                if "value" not in data["data"]["webPages"]:
                    return "未找到搜索结果"
                    
                results = data["data"]["webPages"]["value"]
                if not results:
                    return "未找到搜索结果"
                    
                # 处理搜索结果
                formatted_results = []
                for i, result in enumerate(results[:5], 1):
                    title = result.get('name', f'搜索结果 {i}')
                    content = result.get('snippet', result.get('summary', '暂无内容摘要'))
                    
                    # 更安全地获取URL
                    result_url = ''
                    if 'url' in result:
                        url_value = result['url']
                        if isinstance(url_value, str):
                            result_url = url_value
                        elif isinstance(url_value, dict) and 'href' in url_value:
                            result_url = url_value['href']
                    
                    # 格式化输出，直接生成HTML链接以便在新标签页打开
                    if result_url and result_url.startswith('http'):
                        formatted_results.append(f"{i}. **{title}**\n{content}\n链接：<a href=\"{result_url}\" target=\"_blank\" rel=\"noopener noreferrer\">{result_url}</a>\n")
                    else:
                        formatted_results.append(f"{i}. **{title}**\n{content}\n")
                
                final_result = "相关搜索结果：\n" + "\n".join(formatted_results)
                return final_result
                
    except asyncio.TimeoutError:
        return "网络搜索请求超时，请稍后重试"
    except Exception as e:
        return f"网络搜索失败：{str(e)}"

async def initialize_mcp_tools():
    """初始化MCP工具列表"""
    global mcp_tools_cache
    try:
        conn = sqlite3.connect('history_messages.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 获取所有MCP服务器和工具
        cursor.execute('''
            SELECT s.id as server_id, s.name as server_name, s.url as server_url,
                   t.id as tool_id, t.name as tool_name, t.description as tool_description,
                   t.input_schema
            FROM mcp_servers s
            LEFT JOIN mcp_tools t ON s.id = t.server_id
            WHERE s.id IS NOT NULL
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        # 组织工具数据
        mcp_tools_cache = {}
        for row in results:
            server_id = row['server_id']
            if server_id not in mcp_tools_cache:
                mcp_tools_cache[server_id] = {
                    'server_name': row['server_name'],
                    'server_url': row['server_url'],
                    'tools': []
                }
            
            if row['tool_id']:  # 如果有工具
                # 解析input_schema JSON字符串
                try:
                    input_schema = json.loads(row['input_schema']) if row['input_schema'] else {}
                except (json.JSONDecodeError, TypeError):
                    input_schema = row['input_schema'] if row['input_schema'] else {}
                
                mcp_tools_cache[server_id]['tools'].append({
                    'id': row['tool_id'],
                    'name': row['tool_name'],
                    'description': row['tool_description'],
                    'input_schema': input_schema
                })
        
        print(f"已初始化 {len(mcp_tools_cache)} 个MCP服务器的工具")
        
    except Exception as e:
        print(f"初始化MCP工具失败: {str(e)}")
        mcp_tools_cache = {}

async def call_mcp_service(server_url: str, tool_name: str, parameters: dict) -> str:
    """调用MCP服务"""
    try:
        print(f"🔧 调用MCP服务: {server_url}, 工具: {tool_name}, 参数: {parameters}")
        
        from fastmcp import Client
        from fastmcp.client.transports import SSETransport
        
        async with Client(SSETransport(server_url)) as client:
            result = await client.call_tool(tool_name, parameters)
            print(f"✅ FastMCP调用成功: {result}")
            return str(result)
            
    except Exception as e:
        print(f"❌ FastMCP调用失败: {str(e)}")
        return f"调用MCP服务失败: {str(e)}"

async def generate_mcp_answer(client, query, conversation_context="", model_name="glm-4-plus"):
    """使用MCP服务生成答案"""
    global mcp_tools_cache
    
    print(f"🤖 开始MCP答案生成，查询: {query}")
    print(f"📊 MCP工具缓存状态: {len(mcp_tools_cache) if mcp_tools_cache else 0} 个服务器")
    
    # 简化的缓存内容调试输出
    if mcp_tools_cache:
        for server_id, server_info in mcp_tools_cache.items():
            server_name = server_info.get('server_name', 'Unknown')
            tool_count = len(server_info.get('tools', []))
            print(f"  📡 {server_name}: {tool_count} 个工具")
    else:
        print("❌ MCP工具缓存为空!")
    
    # 构建工具描述
    tool_descriptions = []
    for server_id, server_info in mcp_tools_cache.items():
        server_name = server_info['server_name']
        server_url = server_info['server_url']
        for tool in server_info['tools']:
            tool_desc = f"服务器: {server_name} ({server_url})\n工具名称: {tool['name']}\n描述: {tool['description']}\n参数结构: {tool['input_schema']}"
            tool_descriptions.append(tool_desc)
    
    tools_text = "\n\n".join(tool_descriptions) if tool_descriptions else "暂无可用工具"
    print(f"🔧 工具描述准备完成，共 {len(tool_descriptions)} 个工具")
    
    # 构建完整的上下文查询
    # 如果有历史对话，需要结合上下文理解用户的真实意图
    contextual_query = query
    if conversation_context.strip():
        # 让大模型先理解完整的上下文，生成更准确的查询
        context_understanding_prompt = f"""基于以下历史对话和当前问题，请理解用户的真实查询意图，并生成一个完整、准确的查询描述。

历史对话：
{conversation_context}

当前问题：{query}

请生成一个完整的查询描述，能够准确表达用户想要查询的内容："""
        
        try:
            context_response = await client.chat.completions.create(
                model=MODEL_CONFIG[model_name]["model"],
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的上下文理解助手。请根据历史对话和当前问题，生成一个完整、准确的查询描述。"
                    },
                    {
                        "role": "user",
                        "content": context_understanding_prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            contextual_query = context_response.choices[0].message.content.strip()
            print(f"🔍 上下文理解后的查询: {contextual_query}")
        except Exception as e:
            print(f"⚠️ 上下文理解失败，使用原始查询: {str(e)}")
            contextual_query = query
    
    # 第一步：让大模型选择工具
    tool_selection_prompt = f"""你是一个专业的智能助手，能够根据用户的问题，灵活选择合适的工具进行操作。  
你的目标是高效、准确地解决用户的需求。请严格按照以下规则执行：

1. 工具调用规范  
+- 如果需要调用工具，请严格返回如下 JSON 格式（不要包含多余内容）：
```json
{{
  "server_url": "server_url",
  "tool_name": "tool_name",
  "parameters": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}
```
+- 只返回 JSON，不要添加任何解释或多余文本。
+- 重要：在构建工具参数时，请使用完整的上下文查询内容，而不是简短的用户输入。

2. 直接回答规范  
+- 如果不需要调用工具，直接返回最终的中文回答内容，不要包含任何 JSON 或格式化内容。

3. 工具选择原则  
+- 仔细分析用户问题，优先选择最合适的工具。
+- 工具参数要参照工具描述，使用完整的查询内容。
+- 如果用户问题不明确，可以适当追问补充信息，但要简洁。

4. 角色设定  
+- 你是一个专业、耐心、善于分析的助手，善于用简洁明了的语言解释复杂问题。

5. 其他注意事项  
+- 不要编造不存在的工具或参数。
+- 不要输出任何与工具调用无关的内容。
+- 保持输出内容的格式严格符合要求。

历史对话：
+{conversation_context}

用户原始问题：{query}

完整上下文查询：{contextual_query}

可用工具列表及描述：
+{tools_text}

注意：在调用工具时，请使用"完整上下文查询"的内容作为查询参数，这样能确保工具能够准确理解用户的真实需求。
"""

    try:
        # 第一次调用：工具选择
        response = await client.chat.completions.create(
            model=MODEL_CONFIG[model_name]["model"],
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的工具选择助手。根据用户问题选择合适的工具，或直接回答问题。"
                },
                {
                    "role": "user",
                    "content": tool_selection_prompt
                }
            ],
            temperature=0.1,
            max_tokens=4096
        )
        
        tool_response = response.choices[0].message.content.strip()
        print(f"🧠 大模型工具选择响应: {tool_response}")
        
        # 尝试解析JSON
        try:
            # 处理可能包含markdown代码块的响应
            json_content = tool_response.strip()
            
            # 如果包含```json代码块，提取其中的JSON内容
            if json_content.startswith('```json') and json_content.endswith('```'):
                json_content = json_content[7:-3].strip()  # 移除```json和```
            elif json_content.startswith('```') and json_content.endswith('```'):
                json_content = json_content[3:-3].strip()  # 移除```和```
            
            tool_call = json.loads(json_content)
            print(f"✅ 成功解析工具调用JSON: {tool_call}")
            if "server_url" in tool_call and "tool_name" in tool_call and "parameters" in tool_call:
                print(f"🎯 准备调用工具: {tool_call['tool_name']}")
                
                # 根据服务器名称查找实际的URL
                server_name = tool_call["server_url"]
                actual_server_url = None
                for server_id, server_info in mcp_tools_cache.items():
                    if server_info['server_name'] == server_name:
                        actual_server_url = server_info['server_url']
                        break
                
                if actual_server_url is None:
                    print(f"❌ 未找到服务器 {server_name} 的URL配置")
                    return f"错误：未找到服务器 {server_name} 的配置"
                
                print(f"🔗 服务器名称: {server_name}, 实际URL: {actual_server_url}")
                
                # 调用MCP服务
                mcp_result = await call_mcp_service(
                    actual_server_url,
                    tool_call["tool_name"],
                    tool_call["parameters"]
                )
                print(f"📋 MCP调用结果长度: {len(str(mcp_result))} 字符")
                
                # 第二次调用：基于MCP结果生成最终答案
                final_prompt = f"""基于以下信息回答用户问题：

历史对话：
+{conversation_context}

MCP服务调用结果：
+{mcp_result}

用户问题：{query}

请基于以上信息，用准确、易懂的中文回答用户问题。"""
                
                final_response = await client.chat.completions.create(
                    model=MODEL_CONFIG[model_name]["model"],
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个专业的企业知识库问答助手。请用准确、易懂的中文回答用户问题。"
                        },
                        {
                            "role": "user",
                            "content": final_prompt
                        }
                    ],
                    temperature=0.3,
                    stream=True,
                    max_tokens=8192
                )
                
                return final_response
                
        except json.JSONDecodeError as json_error:
            # 如果不是JSON，说明是直接回答
            print(f"📝 大模型选择直接回答模式，JSON解析失败: {str(json_error)}")
            print(f"📄 原始响应内容: {tool_response[:200]}...")
        
        # 直接返回流式响应
        direct_response = await client.chat.completions.create(
            model=MODEL_CONFIG[model_name]["model"],
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的企业知识库问答助手。请用准确、易懂的中文回答用户问题。"
                },
                {
                    "role": "user",
                    "content": f"历史对话：\n{conversation_context}\n\n当前问题：{query}\n\n请回答："
                }
            ],
            temperature=0.3,
            stream=True,
            max_tokens=8192
        )
        
        return direct_response
        
    except Exception as e:
        raise

async def generate_answer(client, query, context, conversation_context="", model_name="glm-4-plus", web_search_context=""):
    """生成答案"""
    # 根据是否有网络搜索结果判断使用场景
    is_web_search_mode = bool(web_search_context and not context)
    
    if is_web_search_mode:
        # 联网搜索模式：只使用网络搜索结果和历史对话
        if conversation_context:
            prompt = f"""基于以下历史对话和网络搜索结果回答问题：

历史对话：
+{conversation_context}

网络搜索结果：
+{web_search_context}

当前问题：{query}

请基于以上信息，用准确、易懂的中文回答用户问题。回答："""
        else:
            prompt = f"""基于以下网络搜索结果回答问题：

网络搜索结果：
+{web_search_context}

问题：{query}

请基于以上网络搜索信息，用准确、易懂的中文回答用户问题。回答："""
    else:
        # 本地文档模式：使用本地文档和历史对话
        if conversation_context:
            prompt = f"""基于以下历史对话和相关信息回答问题：

历史对话：
+{conversation_context}

相关信息：
+{context}

当前问题：{query}

请基于以上信息，用准确、易懂的中文回答用户问题。回答："""
        else:
            prompt = f"""基于以下信息回答问题：

+{context}

问题：{query}

请用准确、易懂的中文回答用户问题。回答："""
    
    try:
        # 优化模型参数
        response = await client.chat.completions.create(
            model=MODEL_CONFIG[model_name]["model"],
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的企业知识库问答助手。请用准确、易懂的中文回答用户问题。如果上下文有链接，请在回答中提供链接。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # 降低温度，使输出更确定
            top_p=0.8,        # 调整采样范围
            frequency_penalty=0.5,  # 添加频率惩罚，避免重复
            presence_penalty=0.5,   # 添加存在惩罚，鼓励多样性
            stream=True,      # 启用流式输出
            max_tokens=8192,  # 设置为模型支持的最大值
            stream_options={"include_usage": False}  # 减少响应数据量
        )
        
        return response
        
    except Exception as e:
        raise

# 全局变量存储模型和索引
model = None
index = None
all_chunks = None
client = None
doc_sources = None
chunks_to_document = None  # 添加 chunks_to_document 到全局变量
mcp_tools_cache = {}  # 缓存MCP工具列表

async def initialize_system():
    """初始化系统，加载模型和文档"""
    global model, index, all_chunks, client, doc_sources, chunks_to_document
    
    # 加载模型
    model = load_model()
    
    # 检查是否存在映射文件，如果存在则直接加载
    chunks_map_path = 'chunks_mapping.npy'
    index_path = 'faiss_index.faiss'
    
    if os.path.exists(chunks_map_path) and os.path.exists(index_path):
        print("发现已存在的索引和映射文件，直接加载...")
        
        # 加载映射文件
        try:
            mapping_data = np.load(chunks_map_path, allow_pickle=True).item()
            chunks_to_document = mapping_data.get('chunks_to_document', {})
            document_to_chunks = mapping_data.get('document_to_chunks', {})
            
            # 加载索引
            index = faiss.read_index(index_path)
            
            # 重建all_chunks和doc_sources
            documents, doc_sources = load_docs()
            if not documents:
                print("错误: 未能加载任何文档，请确保文档目录存在且包含有效文件")
                return False
            
            all_chunks = []
            for i, (doc, source) in enumerate(zip(documents, doc_sources)):
                if i in document_to_chunks:
                    # 获取文件名
                    file_name = os.path.basename(source)
                    # 检查是否为Excel文件
                    is_excel = source.lower().endswith(('.xlsx', '.xls'))
                    chunks = chunk_document(doc, is_excel=is_excel, file_name=file_name)
                    all_chunks.extend(chunks)
                else:
                    print(f"警告: 文档 {source} 在映射中未找到")
            
            print(f"成功加载已有索引，包含 {len(all_chunks)} 个文档块")
            
        except Exception as e:
            print(f"加载已有索引失败: {str(e)}，将重新处理文档")
            # 如果加载失败，删除损坏的文件并重新处理
            if os.path.exists(chunks_map_path):
                os.remove(chunks_map_path)
            if os.path.exists(index_path):
                os.remove(index_path)
            return await initialize_system()  # 递归调用重新处理
    else:
        print("未发现索引文件，开始处理文档...")
        
        # 加载文档
        documents, doc_sources = load_docs()
        
        if not documents:
            print("错误: 未能加载任何文档，请确保文档目录存在且包含有效文件")
            return False

        # 处理文档分块
        document_to_chunks = {}
        chunks_to_document = {}
        all_chunks = []
        
        # 创建 chunks 目录
        chunks_dir = Path('chunks')
        chunks_dir.mkdir(exist_ok=True)
        
        print("开始处理文档分块")
        for i, (doc, source) in enumerate(zip(documents, doc_sources)):
            # 获取文件名
            file_name = os.path.basename(source)
            # 检查是否为Excel文件
            is_excel = source.lower().endswith(('.xlsx', '.xls'))
            chunks = chunk_document(doc, is_excel=is_excel, file_name=file_name)
            document_to_chunks[i] = chunks
            
            # 保存文档块到 JSON 文件
            chunks_data = {
                "file_name": file_name,
                "file_path": source,
                "total_chunks": len(chunks),
                "chunks": [
                    {
                        "chunk_id": j,
                        "content": chunk,
                        "length": len(chunk)
                    }
                    for j, chunk in enumerate(chunks)
                ]
            }
            
            # 将文件名中的特殊字符替换为下划线，作为 JSON 文件名
            safe_filename = re.sub(r'[^\w\-_.]', '_', file_name)
            json_path = chunks_dir / f"{safe_filename}.json"
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)
            
            for chunk in chunks:
                chunks_to_document[len(all_chunks)] = i
                all_chunks.append(chunk)
        
        if not all_chunks:
            print("错误: 没有生成任何文档块")
            return False
            
        print(f"成功生成 {len(all_chunks)} 个文档块")

        # 创建新的索引
        print("创建新的索引")
        embeddings = get_embeddings(model, all_chunks)
        
        # 使用IVF索引类型，提高搜索效率
        dimension = embeddings.shape[1]
        nlist = min(100, len(all_chunks) // 10)  # 聚类中心数量
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        # 训练索引
        if not index.is_trained and len(all_chunks) > nlist:
            print("训练索引...")
            index.train(embeddings)
        
        # 添加向量到索引
        index.add(embeddings)
        
        # 保存索引和映射
        print(f"保存索引到本地: {index_path}")
        faiss.write_index(index, index_path)
        np.save(chunks_map_path, {
            'document_to_chunks': document_to_chunks,
            'chunks_to_document': chunks_to_document
        })

    # 初始化默认的OpenAI客户端（使用GLM-4-PLUS）
    client = create_optimized_client("glm-4-plus")
    
    # 初始化MCP工具
    await initialize_mcp_tools()
    
    return True

def reprocess_documents():
    """重新处理所有文档"""
    global model, index, all_chunks, doc_sources, chunks_to_document
    
    print("开始重新处理文档")
    
    # 确保模型已加载
    if model is None:
        model = load_model()
        if model is None:
            print("错误: 无法加载模型")
            return False
    
    # 加载文档
    documents, doc_sources = load_docs()
    
    if not documents:
        print("错误: 未能加载任何文档，请确保文档目录存在且包含有效文件")
        return False
    
    # 处理文档分块
    document_to_chunks = {}
    chunks_to_document = {}
    all_chunks = []
    
    # 创建 chunks 目录
    chunks_dir = Path('chunks')
    chunks_dir.mkdir(exist_ok=True)
    
    print("开始处理文档分块")
    for i, (doc, source) in enumerate(zip(documents, doc_sources)):
        # 获取文件名
        file_name = os.path.basename(source)
        # 检查是否为Excel文件
        is_excel = source.lower().endswith(('.xlsx', '.xls'))
        chunks = chunk_document(doc, is_excel=is_excel, file_name=file_name)
        document_to_chunks[i] = chunks
        
        # 保存文档块到 JSON 文件
        chunks_data = {
            "file_name": file_name,
            "file_path": source,
            "total_chunks": len(chunks),
            "chunks": [
                {
                    "chunk_id": j,
                    "content": chunk,
                    "length": len(chunk)
                }
                for j, chunk in enumerate(chunks)
            ]
        }
        
        # 将文件名中的特殊字符替换为下划线，作为 JSON 文件名
        safe_filename = re.sub(r'[^\w\-_.]', '_', file_name)
        json_path = chunks_dir / f"{safe_filename}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        for chunk in chunks:
            chunks_to_document[len(all_chunks)] = i
            all_chunks.append(chunk)
    
    if not all_chunks:
        print("错误: 没有生成任何文档块")
        return False
        
    print(f"成功生成 {len(all_chunks)} 个文档块")
    
    # 创建新的索引
    print("创建新的索引")
    embeddings = get_embeddings(model, all_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    # 保存索引和映射
    index_path = 'faiss_index.faiss'
    chunks_map_path = 'chunks_mapping.npy'
    print(f"保存索引到本地: {index_path}")
    faiss.write_index(index, index_path)
    np.save(chunks_map_path, {
        'document_to_chunks': document_to_chunks,
        'chunks_to_document': chunks_to_document
    })
    
    return True

def add_document_to_index(model, index, all_chunks, chunks_to_document, doc_path, doc_source_idx):
    print("开始处理新上传的文档")
    # 加载文档内容
    loader = DocumentLoader()
    doc_info = loader.load_document(doc_path)
    file_name = os.path.basename(doc_path)
    is_excel = file_name.lower().endswith(('.xlsx', '.xls'))
    chunks = chunk_document(doc_info['content'], is_excel=is_excel, file_name=file_name)
    if not chunks:
        print("未生成任何文档块，跳过")
        return False
    # 计算新块的向量
    new_embeddings = get_embeddings(model, chunks)
    # 更新索引
    index.add(new_embeddings)
    # 更新all_chunks和chunks_to_document
    start_idx = len(all_chunks)
    all_chunks.extend(chunks)
    for i in range(len(chunks)):
        chunks_to_document[start_idx + i] = doc_source_idx
    # 保存索引和映射
    faiss.write_index(index, 'faiss_index.faiss')
    # 需要重新构建document_to_chunks映射
    document_to_chunks = {}
    for chunk_idx, doc_idx in chunks_to_document.items():
        if doc_idx not in document_to_chunks:
            document_to_chunks[doc_idx] = []
        # 这里简化处理，实际应该根据chunk内容重建
    
    np.save('chunks_mapping.npy', {
        'chunks_to_document': chunks_to_document,
        'document_to_chunks': document_to_chunks
    })
    print("增量入库完成")
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期管理"""
    # 启动时初始化
    try:
        if await initialize_system():
            print("系统初始化成功")
        else:
            print("系统初始化失败，但应用仍可启动")
    except Exception as e:
        print(f"系统初始化出错: {str(e)}，但应用仍可启动")
    
    yield
    
    # 关闭时清理
    try:
        await db.close()
        # 保存缓存
        embedding_cache.save_cache()
        vector_cache.save_cache()
    except Exception as e:
        print(f"清理资源时出错: {str(e)}")

# 修改 FastAPI 应用创建
app = FastAPI(lifespan=lifespan)

# 添加MCP路由
app.include_router(mcp_router)


# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加处理 Chrome 开发者工具请求的路由
@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools():
    return JSONResponse(
        content={"status": "ok"},
        status_code=200
    )

# 创建文档加载器实例
document_loader = DocumentLoader()

# 创建上传文件目录
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 获取当前文件所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 设置模板目录
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    返回主页
    功能：加载并显示聊天界面，包括历史对话列表
    参数：
        - request: 请求对象
    返回：
        - HTML页面，包含历史对话列表和聊天界面
    """
    try:
        conversations = await db.get_conversations()
        history_items = [
            {
                "id": conv[0],
                "title": conv[1],
                "timestamp": conv[2]
            }
            for conv in conversations
        ]
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "historyItems": history_items,
                "chatMessages": [],
                "files": []
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "historyItems": [],
                "chatMessages": [],
                "files": []
            }
        )

@app.get("/files")
async def get_files():
    """
    获取可用的文件列表
    功能：返回uploads目录下所有支持的文件名列表
    支持的文件类型：.txt, .pdf, .docx, .md, .xlsx, .xls
    返回：
        - 文件名列表
    """
    try:
        files = []
        for ext in ['.txt', '.pdf', '.docx', '.md', '.xlsx', '.xls']:
            files.extend(glob.glob(os.path.join('uploads', f'*{ext}')))
        return [os.path.basename(f) for f in files]
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"获取文件列表失败: {str(e)}"}
        )

def log_performance(step_name, start_time):
    """记录性能日志"""
    end_time = time.time()
    duration = end_time - start_time
    print(f"[性能日志] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {step_name}: {duration:.2f}秒")

@app.get("/query/{question}")
async def query_endpoint(
    question: str,
    file: str = Query(None, description="指定要查询的文件名"),
    conversation_id: str = Query(None, description="会话ID"),
    model_name: str = Query("glm-4-plus", description="选择的大模型名称"),
    web_search_enabled: bool = Query(False, description="是否启用联网搜索"),
    mcp_service_enabled: bool = Query(False, description="是否启用MCP服务"),
    request: Request = None
):
    # 性能调试：记录总开始时间
    total_start_time = time.time()
    print(f"🚀 [性能调试] 请求开始: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    
    global model, index, all_chunks, client, doc_sources, chunks_to_document
    
    # URL解码问题
    question = urllib.parse.unquote(question)
    print(f"📝 [性能调试] 问题解码完成: {time.time() - total_start_time:.3f}s")
    
    # 系统初始化检查
    init_start = time.time()
    if not await quick_system_check():
        return JSONResponse(
            status_code=500,
            content={"error": "系统初始化失败，请检查日志"}
        )
    print(f"🔧 [性能调试] 系统初始化检查: {time.time() - init_start:.3f}s")
    
    try:
        # 检查模型名称是否有效
        model_check_start = time.time()
        if model_name not in MODEL_CONFIG:
            return JSONResponse(
                status_code=400,
                content={"error": f"不支持的模型名称: {model_name}"}
            )
        
        # 更新客户端配置
        client = create_optimized_client(model_name)
        print(f"🤖 [性能调试] 模型配置和客户端初始化: {time.time() - model_check_start:.3f}s")
        
        # 获取对话上下文
        context_start = time.time()
        conversation_context = ""
        if conversation_id:
            # 获取历史对话上下文
            conversation_context, is_overflow = await db.get_conversation_context(conversation_id)
            if is_overflow:
                return JSONResponse(
                    status_code=400,
                    content={"error": "该对话内容过多，请开启新对话"}
                )
        else:
            # 创建新对话
            conversation_id = await db.create_conversation(question)
        print(f"💬 [性能调试] 对话上下文获取: {time.time() - context_start:.3f}s")
        
        # 保存用户问题
        save_msg_start = time.time()
        await db.add_message(conversation_id, question, "user")
        print(f"💾 [性能调试] 保存用户消息: {time.time() - save_msg_start:.3f}s")
        
        # 根据启用的服务类型决定数据源
        service_start = time.time()
        if mcp_service_enabled:
            print(f"🔧 [性能调试] 选择MCP服务模式")
            # 启用MCP服务时，使用MCP服务处理
            response = await generate_mcp_answer(client, question, conversation_context, model_name)
            context = ""  # 不使用本地文档上下文
            web_search_context = ""  # 不使用网络搜索
        elif web_search_enabled:
            print(f"🌐 [性能调试] 选择联网搜索模式")
            # 启用联网搜索时，只使用网络搜索结果，不检索本地文档
            web_search_start = time.time()
            web_search_context = await web_search(question)
            print(f"🔍 [性能调试] 网络搜索完成: {time.time() - web_search_start:.3f}s")
            context = ""  # 不使用本地文档上下文
            
            generate_start = time.time()
            response = await generate_answer(client, question, context, conversation_context, model_name, web_search_context)
            print(f"🤖 [性能调试] 生成答案调用完成: {time.time() - generate_start:.3f}s")
        else:
            print(f"📚 [性能调试] 选择本地文档模式")
            # 未启用联网搜索时，使用本地文档检索
            doc_search_start = time.time()
            context, indices = get_related_documents(model, index, question, all_chunks)
            print(f"📖 [性能调试] 文档检索完成: {time.time() - doc_search_start:.3f}s")
            
            if not context:
                return JSONResponse(
                    status_code=404,
                    content={"error": "未找到相关文档"}
                )
            
            # 如果指定了文件，过滤相关文档
            if file:
                filter_start = time.time()
                filtered_context = []
                filtered_indices = []
                for ctx, idx in zip(context, indices):
                    if idx >= len(chunks_to_document):
                        continue
                        
                    doc_idx = chunks_to_document[idx]
                    if doc_idx >= len(doc_sources):
                        continue
                        
                    source = doc_sources[doc_idx]
                    if os.path.basename(source) == file:
                        filtered_context.append(ctx)
                        filtered_indices.append(idx)
                
                if filtered_context:
                    context = filtered_context
                    indices = filtered_indices
                else:
                    return JSONResponse(
                        status_code=404,
                        content={"error": f"在文件 {file} 中未找到相关信息"}
                    )
                print(f"🔍 [性能调试] 文件过滤完成: {time.time() - filter_start:.3f}s")
            
            web_search_context = ""  # 不使用网络搜索
            generate_start = time.time()
            response = await generate_answer(client, question, context, conversation_context, model_name, web_search_context)
            print(f"🤖 [性能调试] 生成答案调用完成: {time.time() - generate_start:.3f}s")
        
        print(f"⚡ [性能调试] 服务处理总耗时: {time.time() - service_start:.3f}s")
        print(f"🎯 [性能调试] 请求预处理总耗时: {time.time() - total_start_time:.3f}s")
        
        async def generate():
            try:
                stream_start = time.time()
                first_chunk_time = None
                chunk_count = 0
                full_answer = ""
                buffer = ""  # 添加缓冲区
                buffer_size = 10  # 增加批量发送字符数到10
                last_send_time = stream_start
                send_interval = 0.1  # 100ms强制发送间隔
                
                print(f"📡 [性能调试] 开始流式响应: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
                
                async for chunk in response:
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        print(f"⚡ [性能调试] 首个数据块到达: {first_chunk_time - stream_start:.3f}s")
                    
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_answer += content
                        buffer += content
                        chunk_count += 1
                        
                        current_time = time.time()
                        # 优化触发条件：缓冲区满、流结束或时间间隔到达
                        should_send = (
                            len(buffer) >= buffer_size or 
                            chunk.choices[0].finish_reason or
                            (current_time - last_send_time) >= send_interval
                        )
                        
                        if should_send and buffer:
                            yield f"data: {json.dumps({'content': buffer}, ensure_ascii=False, separators=(',', ':'))}\n\n"
                            buffer = ""
                            last_send_time = current_time
                
                # 发送剩余缓冲区内容
                if buffer:
                    yield f"data: {json.dumps({'content': buffer}, ensure_ascii=False, separators=(',', ':'))}\n\n"
                
                stream_end = time.time()
                print(f"📊 [性能调试] 流式响应完成:")
                print(f"   - 总耗时: {stream_end - stream_start:.3f}s")
                print(f"   - 首块延迟: {(first_chunk_time - stream_start) if first_chunk_time else 0:.3f}s")
                print(f"   - 数据块数量: {chunk_count}")
                print(f"   - 响应长度: {len(full_answer)} 字符")
                print(f"   - 平均速度: {len(full_answer) / (stream_end - stream_start):.1f} 字符/秒")
                
                # 异步保存完整答案到数据库
                db_save_start = time.time()
                await db.add_message(conversation_id, full_answer, "system")
                print(f"💾 [性能调试] 保存AI回复: {time.time() - db_save_start:.3f}s")
                
                # 发送完成信号
                yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id}, ensure_ascii=False, separators=(',', ':'))}\n\n"
                
                total_end = time.time()
                print(f"🏁 [性能调试] 请求完全结束，总耗时: {total_end - total_start_time:.3f}s")
                
            except Exception as e:
                print(f"❌ [性能调试] 流式响应错误: {str(e)}")
                yield f"data: {json.dumps({'error': '生成答案时出错，请重试'}, ensure_ascii=False, separators=(',', ':'))}\n\n"
                yield f"data: {json.dumps({'done': True}, ensure_ascii=False, separators=(',', ':'))}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Credentials": "true"
            }
        )
        
    except Exception as e:
        error_time = time.time()
        print(f"💥 [性能调试] 请求异常，总耗时: {error_time - total_start_time:.3f}s, 错误: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"处理查询时出错: {str(e)}"}
        )

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    处理文件上传
    功能：接收并处理上传的文件，进行增量向量化和索引更新
    参数：
        - file: 上传的文件
    支持的文件类型：
        - .pdf, .docx, .doc, .xlsx, .xls, .txt
    返回：
        - 上传处理结果
    """
    global model, index, all_chunks, doc_sources, chunks_to_document
    allowed_types = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.xls': 'application/vnd.ms-excel',
        '.txt': 'text/plain'
    }
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_types:
        return JSONResponse(
            status_code=400,
            content={'status': 'error', 'message': f'不支持的文件类型: {file_ext}'}
        )
    try:
        upload_dir = Path('uploads')
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / file.filename
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        # 增量向量化和索引更新
        if model is None or index is None or all_chunks is None or chunks_to_document is None:
            return JSONResponse(
                status_code=500,
                content={'status': 'error', 'message': '系统未初始化，无法增量插入'}
            )
        # 维护doc_sources
        if doc_sources is None:
            doc_sources = []
        doc_sources.append(str(file_path))
        doc_source_idx = len(doc_sources) - 1
        ok = add_document_to_index(model, index, all_chunks, chunks_to_document, str(file_path), doc_source_idx)
        if ok:
            return JSONResponse(content={'status': 'success', 'message': '文件上传并增量入库成功'})
        else:
            # 如果处理失败，删除上传的文件
            if file_path.exists():
                file_path.unlink()
            return JSONResponse(
                status_code=500,
                content={'status': 'error', 'message': '文档增量入库失败'}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'status': 'error', 'message': f'文件处理失败: {str(e)}'}
        )

@app.get("/documents")
async def list_documents():
    """
    获取所有已上传文档的列表
    功能：返回uploads目录下所有已处理的文档信息
    返回：
        - 文档列表，包含文件路径和类型
    """
    try:
        docs = document_loader.load_directory(str(UPLOAD_DIR))
        return {
            "status": "success",
            "documents": [
                {
                    "file_path": doc["file_path"],
                    "file_type": doc["file_type"]
                }
                for doc in docs
            ]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"获取文档列表失败: {str(e)}"
        }

@app.get("/api/conversations")
async def get_conversations():
    """
    获取对话列表
    功能：返回所有对话的列表，按收藏状态和更新时间排序
    返回：
        - 对话列表，包含每个对话的基本信息
    """
    try:
        conversations = await db.get_conversations()
        return JSONResponse(content={
            "status": "success",
            "conversations": [
                {
                    "conversation_id": conv[0],
                    "title": conv[1],
                    "starred": int(conv[2]),  # 确保转换为整数
                    "created_at": conv[3],
                    "updated_at": conv[4],
                    "messages": conv[5] if conv[5] else ""
                }
                for conv in conversations
            ]
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    获取指定对话的详细信息
    功能：返回指定对话ID的所有消息记录
    参数：
        - conversation_id: 对话ID
    返回：
        - 消息列表，包含消息内容、类型、创建时间等信息
    """
    try:
        messages = await db.get_messages(conversation_id)
        return JSONResponse(content={
            "status": "success",
            "messages": [
                {
                    "id": msg[0],
                    "content": msg[1],
                    "type": msg[2],
                    "created_at": msg[3]
                }
                for msg in messages
            ]
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    删除对话
    功能：删除指定ID的对话及其所有相关消息
    参数：
        - conversation_id: 对话ID
    返回：
        - 删除操作结果
    """
    try:
        success = await db.delete_conversation(conversation_id)
        if success:
            return JSONResponse(content={"status": "success"})
        else:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "删除对话失败"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/docs_manage", response_class=HTMLResponse)
async def docs_page(request: Request):
    """
    返回文档管理页面
    功能：加载并显示文档管理界面
    参数：
        - request: 请求对象
    返回：
        - 文档管理页面HTML
    """
    try:
        return templates.TemplateResponse(
            "docs.html",
            {
                "request": request
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "加载文档管理页面失败"}
        )

@app.get("/mcp.html", response_class=HTMLResponse)
async def mcp_page(request: Request):
    """
    返回MCP管理页面
    功能：加载并显示MCP服务管理界面
    参数：
        - request: 请求对象
    返回：
        - MCP管理页面HTML
    """
    try:
        print(f"尝试加载MCP页面，模板目录: {os.path.join(BASE_DIR, 'templates')}")
        print(f"MCP模板文件存在: {os.path.exists(os.path.join(BASE_DIR, 'templates', 'mcp.html'))}")
        
        return templates.TemplateResponse(
            "mcp.html",
            {
                "request": request
            }
        )
    except Exception as e:
        print(f"加载MCP管理页面失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"加载MCP管理页面失败: {str(e)}"}
        )

@app.post("/api/conversations")
async def create_conversation(request: Request):
    """
    创建新对话
    功能：创建新的对话记录
    参数：
        - request: 包含标题的请求体
    返回：
        - 新创建的对话ID和标题
    """
    try:
        data = await request.json()
        title = data.get('title', 'New conversation')
        conversation_id = await db.create_conversation(title)
        return JSONResponse(content={
            "status": "success",
            "conversation_id": conversation_id,
            "title": title
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.put("/api/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, request: Request):
    """
    更新对话标题
    功能：修改指定对话的标题
    参数：
        - conversation_id: 对话ID
        - request: 包含新标题的请求体
    返回：
        - 更新操作结果
    """
    try:
        data = await request.json()
        new_title = data.get('title')
        if not new_title:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "标题不能为空"}
            )
        
        # 更新数据库中的标题
        success = await db.update_conversation_title(conversation_id, new_title)
        if success:
            return JSONResponse(content={
                "status": "success",
                "message": "标题更新成功"
            })
        else:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "更新标题失败"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.put("/api/conversations/{conversation_id}/star")
async def update_conversation_starred(conversation_id: str, request: Request):
    """
    更新对话的收藏状态
    功能：设置或取消对话的收藏状态
    参数：
        - conversation_id: 对话ID
        - request: 包含starred状态(1或0)的请求体
    返回：
        - 更新操作结果
    """
    try:
        data = await request.json()
        starred = data.get('starred', 0)
        
        success = await db.update_conversation_starred(conversation_id, starred)
        if success:
            return JSONResponse(content={
                "status": "success",
                "message": "收藏状态更新成功"
            })
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "更新收藏状态失败"
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

async def quick_system_check():
    """快速系统检查"""
    global model, index, all_chunks, client
    
    if not system_state.needs_check():
        return True
    
    if model is None or index is None or all_chunks is None:
        return await initialize_system()
    
    system_state.mark_initialized()
    return True

def create_optimized_client(model_name: str) -> AsyncOpenAI:
    """创建优化的OpenAI客户端"""
    return AsyncOpenAI(
        api_key=MODEL_CONFIG[model_name]["api_key"],
        base_url=MODEL_CONFIG[model_name]["api_base"],
        timeout=60.0,  # 设置60秒超时
        max_retries=2,  # 最大重试2次
        http_client=None  # 使用默认HTTP客户端
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)




