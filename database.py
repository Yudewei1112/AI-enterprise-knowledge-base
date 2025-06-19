import sqlite3
from datetime import datetime
import json
import uuid
import aiosqlite
import asyncio
from typing import List, Tuple, Optional
from contextlib import asynccontextmanager

class Database:
    def __init__(self, db_name="history_messages.db"):
        self.db_name = db_name
        self._init_db_sync()  # 同步初始化数据库
        self._connection_pool = asyncio.Queue(maxsize=5)  # 创建连接池
        self._initialized = False

    def _init_db_sync(self):
        """同步初始化数据库表"""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            
            # 创建对话历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    starred INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建消息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                )
            ''')
            # Create MCP servers table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS mcp_servers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                description TEXT,
                auth_type TEXT,
                auth_value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create MCP tools table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS mcp_tools (
                id TEXT PRIMARY KEY,
                server_id TEXT,
                name TEXT NOT NULL,
                description TEXT,
                input_schema TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (server_id) REFERENCES mcp_servers(id)
            )
            ''')

            
            conn.commit()

    async def _initialize_pool(self):
        """初始化连接池"""
        if not self._initialized:
            for _ in range(5):  # 创建5个连接
                conn = await aiosqlite.connect(self.db_name)
                await self._connection_pool.put(conn)
            self._initialized = True

    @asynccontextmanager
    async def get_connection(self):
        """获取数据库连接的上下文管理器"""
        if not self._initialized:
            await self._initialize_pool()
        
        conn = await self._connection_pool.get()
        try:
            yield conn
        finally:
            await self._connection_pool.put(conn)

    async def create_conversation(self, title: str) -> str:
        """异步创建新的对话"""
        conversation_id = str(uuid.uuid4())
        async with self.get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                "INSERT INTO conversations (conversation_id, title, starred) VALUES (?, ?, 0)",
                (conversation_id, title)
            )
            await conn.commit()
            return conversation_id

    async def add_message(self, conversation_id: str, content: str, message_type: str) -> int:
        """异步添加新消息"""
        async with self.get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                "INSERT INTO messages (conversation_id, content, type) VALUES (?, ?, ?)",
                (conversation_id, content, message_type)
            )
            # 更新对话的更新时间
            await cursor.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
                (conversation_id,)
            )
            await conn.commit()
            return cursor.lastrowid

    async def get_conversations(self, limit: int = 50) -> List[Tuple]:
        """异步获取最近的对话列表"""
        async with self.get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute('''
                WITH FirstMessages AS (
                    SELECT 
                        conversation_id,
                        MIN(created_at) as first_message_time
                    FROM messages
                    WHERE type = 'user'
                    GROUP BY conversation_id
                )
                SELECT 
                    c.conversation_id,
                    c.title,
                    c.starred,
                    c.created_at,
                    c.updated_at,
                    GROUP_CONCAT(m.content || '|' || m.type || '|' || m.created_at, '||') as messages
                FROM conversations c
                LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                LEFT JOIN FirstMessages fm ON c.conversation_id = fm.conversation_id
                GROUP BY c.conversation_id
                ORDER BY c.starred DESC, c.updated_at DESC
                LIMIT ?
            ''', (limit,))
            return await cursor.fetchall()

    async def get_messages(self, conversation_id: str) -> List[Tuple]:
        """异步获取指定对话的所有消息"""
        async with self.get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute('''
                SELECT id, content, type, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
            ''', (conversation_id,))
            return await cursor.fetchall()

    async def delete_conversation(self, conversation_id: str) -> bool:
        """异步删除对话及其所有消息"""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.cursor()
                # 首先删除相关的消息
                await cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
                # 然后删除对话
                await cursor.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
                await conn.commit()
                return True
        except Exception as e:
            print(f"删除对话失败: {str(e)}")
            return False

    async def search_conversations(self, keyword: str) -> List[Tuple]:
        """异步搜索对话"""
        async with self.get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute('''
                SELECT DISTINCT c.conversation_id, c.title, c.created_at, c.updated_at,
                       GROUP_CONCAT(m.content || '|' || m.type || '|' || m.created_at, '||') as messages
                FROM conversations c
                LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                WHERE c.title LIKE ? OR m.content LIKE ?
                GROUP BY c.conversation_id
                ORDER BY c.updated_at DESC
            ''', (f'%{keyword}%', f'%{keyword}%'))
            return await cursor.fetchall()

    async def update_conversation_title(self, conversation_id: str, new_title: str) -> bool:
        """异步更新会话标题"""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.cursor()
                await cursor.execute(
                    "UPDATE conversations SET title = ? WHERE conversation_id = ?",
                    (new_title, conversation_id)
                )
                await conn.commit()
                return True
        except Exception as e:
            print(f"更新会话标题失败: {str(e)}")
            return False

    async def update_conversation_starred(self, conversation_id: str, starred: int) -> bool:
        """异步更新对话的收藏状态"""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.cursor()
                await cursor.execute(
                    "UPDATE conversations SET starred = ? WHERE conversation_id = ?",
                    (starred, conversation_id)
                )
                await conn.commit()
                return True
        except Exception as e:
            print(f"更新对话收藏状态失败: {str(e)}")
            return False

    async def close(self):
        """关闭所有数据库连接"""
        if self._initialized:
            while not self._connection_pool.empty():
                conn = await self._connection_pool.get()
                await conn.close()
            self._initialized = False

    async def get_conversation_history(self, conversation_id: str, max_tokens: int = 100000) -> Tuple[List[Tuple], bool]:
        """
        获取对话历史记录
        参数:
            conversation_id: 对话ID
            max_tokens: 最大token数（默认100k）
        返回:
            Tuple[List[Tuple], bool]: (消息列表, 是否超出限制)
        """
        async with self.get_connection() as conn:
            cursor = await conn.cursor()
            # 获取所有消息，按时间排序
            await cursor.execute('''
                SELECT content, type, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
            ''', (conversation_id,))
            messages = await cursor.fetchall()
            
            # 计算总字符数
            total_chars = sum(len(msg[0]) for msg in messages)
            
            # 如果超过限制，返回True表示超出限制
            if total_chars > max_tokens:
                return messages, True
            
            return messages, False

    async def get_conversation_context(self, conversation_id: str, max_tokens: int = 100000) -> Tuple[str, bool]:
        """
        获取对话上下文
        参数:
            conversation_id: 对话ID
            max_tokens: 最大token数（默认100k）
        返回:
            Tuple[str, bool]: (上下文字符串, 是否超出限制)
        """
        messages, is_overflow = await self.get_conversation_history(conversation_id, max_tokens)
        
        if is_overflow:
            return "", True
        
        # 构建上下文字符串
        context = []
        for content, msg_type, _ in messages:
            if msg_type == "user":
                context.append(f"用户: {content}")
            else:
                context.append(f"助手: {content}")
        
        return "\n".join(context), False

# 创建数据库实例
db = Database() 