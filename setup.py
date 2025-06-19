#!/usr/bin/env python3
# setup.py - 项目初始化脚本
import os

def setup_project():
    """初始化项目目录和配置"""
    # 创建必要目录
    dirs = ['uploads', 'cache', 'chunks']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ 创建目录: {dir_name}")
    
    # 检查.env文件
    if not os.path.exists('.env'):
        print("⚠️  请复制.env.example为.env并配置API密钥")
    
    print("🚀 项目初始化完成！")

if __name__ == "__main__":
    setup_project()