import os
from document_loader import DocumentLoader
from pathlib import Path

def load_docs():
    """
    加载uploads目录下的所有文档
    
    Returns:
        tuple: (documents, doc_sources)
            - documents: 文档内容列表
            - doc_sources: 文档源文件路径列表
    """
    # 创建文档加载器实例
    loader = DocumentLoader()
    
    # 获取uploads目录路径
    uploads_dir = Path("uploads")
    if not uploads_dir.exists():
        uploads_dir.mkdir(exist_ok=True)
        print(f"创建uploads目录: {uploads_dir}")
        return [], []
    
    # 加载所有文档
    try:
        docs = loader.load_directory(str(uploads_dir))
        documents = [doc["content"] for doc in docs]
        doc_sources = [doc["file_path"] for doc in docs]
        return documents, doc_sources
    except Exception as e:
        print(f"加载文档时出错: {str(e)}")
        return [], [] 