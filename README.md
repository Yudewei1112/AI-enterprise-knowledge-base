# 企业文档知识库系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

## 📖 项目简介

这是一个基于AI技术的企业文档知识库系统，支持多种文档格式的智能检索和问答。系统采用向量化技术对企业文档进行语义理解，提供精准的文档检索和智能问答服务。

**🎯 技术特色**：本系统AI应用部分完全纯原生实现，**未使用任何第三方AI框架**（如LangChain、LlamaIndex等），具有**高度自主性**和**灵活配置能力**，可根据业务需求进行深度定制和优化。

**🌟 开源特性**：
- ✅ 完全开源，MIT许可证
- ✅ 零第三方AI框架依赖
- ✅ 模块化设计，易于扩展
- ✅ 详细文档和示例
- ✅ 生产就绪的代码质量
## 📱 界面预览

### 主界面
![主界面-可任意切换大模型](images\1.模型切换.png)

### 本地知识库查询
![本地知识库查询](images\3.本地知识库文档查询&对话界面.png)

## ✨ 核心功能

### 📄 文档管理
- **多格式支持**: 支持PDF、Word(.docx/.doc)、Excel(.xlsx/.xls)、TXT、CSV等多种文档格式
- **批量上传**: 支持批量文档上传和处理
![文档上传页面](images\9-文档管理页面.png)
- **智能解析**: 自动提取文档内容并进行结构化处理
- **向量化存储**: 使用M3E模型将文档转换为向量表示

### 🔍 智能检索
- **语义搜索**: 基于FAISS向量数据库的高效语义检索
- **相似度匹配**: 智能匹配用户查询与文档内容的语义相似度
- **缓存优化**: 查询结果缓存机制，提升检索性能
- **🎯 自主实现**: 向量化和检索算法完全自主开发，无第三方框架依赖
![联网搜索](images\6-联网搜索.png)

### 🤖 AI问答
- **多模型支持**: 集成GLM-4-Plus、DeepSeek、Qwen、Claude等多个大语言模型
- **上下文理解**: 结合检索结果提供准确的问答服务
- **流式输出**: 实时流式响应，提升用户体验
- **对话历史**: 支持多轮对话和历史记录管理，支持历史对话删除、修改名称、收藏等操作
- **🔧 原生架构**: 基于底层API直接实现，无框架束缚，配置灵活度极高
![本地知识库查询](images\3.本地知识库文档查询&对话界面.png)
![本地知识库查询-多轮对话](images\4.本地知识库复杂表格文档查询+验证多轮对话.png)
![本地知识库查询-验证复杂表格——完全命中](images\5-查询的源文件（完全命中正确答案）.png)

### 🔧 MCP集成
- **模型上下文协议**: 支持MCP(Model Context Protocol)标准
- **工具扩展**: 内置天气查询、订单管理等示例工具
- **服务管理**: 支持MCP服务器的动态管理和配置
![MCP天气查询](images\7-MCP-天气查询.png)
![MCP服务](images\10-MCP管理页面.png)

## 🏗️ 系统架构

### 技术栈
- **后端框架**: FastAPI + Python 3.11
- **AI模型**: Sentence-Transformers (M3E-base)
- **向量数据库**: FAISS
- **关系数据库**: SQLite (异步支持)
- **前端**: Vue.js 3 + 原生JavaScript
- **文档处理**: PyPDF2, python-docx, openpyxl, pandas
- **AI架构**: 🚀 **纯原生实现**，无第三方AI框架依赖，完全自主可控

### 项目结构

```
├── main.py                 # 🚀 主应用入口，FastAPI服务器
├── database.py            # 💾 数据库操作和连接管理
├── document_loader.py     # 📄 文档加载和解析模块
├── file_loads.py          # 🔧 文件加载工具函数
├── mcp_api.py            # 🔌 MCP协议API接口
├── mcp_server/           # 🌐 MCP服务器实现
│   ├── __init__.py       # 模块初始化
│   └── weather_service.py # 天气查询服务示例
├── templates/            # 🎨 前端模板文件
│   ├── index.html        # 主界面
│   ├── docs.html         # 文档管理界面
│   └── mcp.html          # MCP服务管理界面
├── requirements.txt      # 📦 Python依赖包列表
├── setup.py             # ⚙️ 项目初始化脚本
├── .env.example         # 🔑 环境变量配置模板
├── .gitignore           # 📝 Git忽略文件配置
├── README.md            # 📖 项目说明文档
├── uploads/             # 📁 文档上传目录 (运行时创建)
├── chunks/              # 🧩 文档分块存储 (运行时创建)
├── cache/               # 💨 缓存文件 (运行时创建)
└── local_m3e_model/     # 🤖 本地AI模型 (首次运行时下载)
```

## 🚀 快速开始

### 环境要求
- Python 3.11+
- CUDA 12.1+ (可选，用于GPU加速)
- 支持Windows/Linux/macOS

### 快速安装

1. **克隆项目**
```bash
git clone https://github.com/your-username/ai-enterprise-knowledge-base.git
cd ai-enterprise-knowledge-base
```

2. **自动初始化**
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境 (Windows)
venv\Scripts\activate
# 激活虚拟环境 (Linux/macOS)
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 初始化项目目录
python setup.py
```

3. **配置环境变量**
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑.env文件，配置你的API密钥
# 至少需要配置一个模型的API密钥
```

4. **启动服务**
```bash
python main.py
```

5. **访问系统**
- 主界面: `http://localhost:8000`
- 文档管理: `http://localhost:8000/docs`
- MCP服务: `http://localhost:8000/mcp`
- API文档: `http://localhost:8000/docs` (FastAPI自动生成)

## 📚 使用指南

### 文档上传
1. 在Web界面点击"上传文档"按钮
2. 选择支持的文档格式文件
3. 系统自动处理并向量化文档内容
4. 上传完成后可在文档列表中查看

### 智能问答
1. 在对话框中输入问题
2. 系统自动检索相关文档
3. 结合AI模型生成准确答案
4. 支持多轮对话和上下文理解

### MCP服务管理
1. 在Web页面上点击"MCP服务"按钮访问 `/mcp` 页面管理MCP服务
2. 添加新的MCP服务器
3. 配置工具和资源
4. 测试服务连接

## ⚙️ 配置说明

### 模型配置
系统支持多个AI模型，可在 `.env` 文件中配置：
- **GLM-4-Plus**: 智谱AI的大语言模型
- **DeepSeek**: DeepSeek的代码和推理模型
- **Qwen**: 阿里云的通义千问模型
- **Claude**: Anthropic的Claude模型

### 性能优化
- **GPU加速**: 自动检测并使用CUDA加速
- **缓存机制**: 查询结果和向量缓存
- **连接池**: 数据库连接池优化
- **批量处理**: 文档批量处理和索引

## 📊 性能特性

### 已实施的优化
- **文档检索性能提升 50-70%**: 查询嵌入缓存和FAISS优化
- **系统初始化优化 90%**: 状态缓存和快速检查机制
- **API调用优化 20-30%**: 客户端优化和Prompt精简
- **前端性能提升 30-50%**: 打字机效果和DOM更新优化

### 调试功能
- 详细的性能日志记录
- 响应时间统计
- 内存使用监控
- 错误追踪和报告

## 🔧 开发指南

### 开发环境设置

1. **Fork项目**
```bash
# Fork到你的GitHub账户，然后克隆
git clone https://github.com/your-username/ai-enterprise-knowledge-base.git
cd ai-enterprise-knowledge-base
```

2. **开发环境配置**
```bash
# 创建开发分支
git checkout -b feature/your-feature-name

# 安装开发依赖
pip install -r requirements.txt

# 运行项目初始化
python setup.py
```

3. **代码规范**
- 遵循PEP 8代码风格
- 添加适当的类型注解
- 编写函数级注释
- 保持代码简洁和可读性

### 添加新的文档格式支持
1. 在 `DocumentLoader` 类中添加新的解析方法
2. 更新 `supported_extensions` 字典
3. 实现对应的文档内容提取逻辑

### 集成新的AI模型
1. 在 `get_model_config()` 函数中添加模型配置
2. 更新环境变量模板
3. 实现模型特定的API调用逻辑

## 🛠️ 故障排除

### 常见问题

**Q: 模型加载失败**
A: 检查网络连接，确保能访问HuggingFace模型库，或使用本地模型

**Q: GPU不可用**
A: 确保安装了正确版本的PyTorch和CUDA驱动

**Q: 文档上传失败**
A: 检查文件格式是否支持，确保文件没有损坏

**Q: API调用超时**
A: 检查网络连接和API密钥配置，调整超时设置

### 日志查看
系统日志保存在控制台输出中，包含：
- 文档处理状态
- 模型加载信息
- API调用记录
- 错误详情

## 🚀 部署指南

### Docker部署 (推荐)

```bash
# 构建镜像
docker build -t ai-knowledge-base .

# 运行容器
docker run -d -p 8000:8000 \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/uploads:/app/uploads \
  ai-knowledge-base
```

### 生产环境部署

1. **使用Gunicorn**
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

2. **使用Nginx反向代理**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 如何贡献

1. **报告问题**
   - 使用GitHub Issues报告bug
   - 提供详细的错误信息和复现步骤

2. **功能建议**
   - 在Issues中提出新功能建议
   - 描述功能的用途和实现思路

3. **代码贡献**
   ```bash
   # 1. Fork项目
   # 2. 创建功能分支
   git checkout -b feature/amazing-feature
   
   # 3. 提交更改
   git commit -m 'Add some amazing feature'
   
   # 4. 推送到分支
   git push origin feature/amazing-feature
   
   # 5. 创建Pull Request
   ```

### 开发规范

- 遵循现有代码风格
- 添加适当的测试
- 更新相关文档
- 确保所有测试通过

## 🌟 致谢

感谢所有为这个项目做出贡献的开发者！

## 📞 联系方式

- 📧 项目维护者: [yudewei1112@gmail.com](mailto:yudewei1112@gmail.com)
- 💬 参与 [Discussions](https://github.com/your-username/ai-enterprise-knowledge-base/discussions)
- � 提交 [GitHub Issue](https://github.com/your-username/ai-enterprise-knowledge-base/issues)
- � 安全问题: [security@your-domain.com](mailto:security@your-domain.com)

## ⚠️ 重要提醒

- 🔐 **API密钥安全**: 请妥善保护你的API密钥，不要提交到版本控制系统
- 📁 **数据隐私**: 上传的文档仅在本地处理，不会发送到第三方服务
- 🔄 **定期更新**: 建议定期更新依赖包以获得最新的安全补丁
- 💾 **数据备份**: 生产环境请定期备份数据库和重要文件

---

**⭐ 如果这个项目对你有帮助，请给我们一个Star！**