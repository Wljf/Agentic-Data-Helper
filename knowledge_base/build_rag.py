"""
构建离线知识库（ChromaDB）的脚本。

功能：
- 读取当前目录下的 definitions.md 文件
- 将其中的业务口径定义作为文本语料
- 使用本地 HuggingFace Embeddings 将文档向量化，并持久化到本地 ChromaDB


使用方式：
1. 在 .env 中配置 CHROMA_PERSIST_DIR（可选，默认 ./chroma_db）
2. 确保已经安装依赖：chromadb、langchain-community、sentence-transformers
3. 在项目根目录执行：

   python -m knowledge_base.build_rag
"""

import os
import shutil

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


from config import config


def load_definitions_markdown() -> str:
    """
    加载 definitions.md 文件的全部内容。

    简化处理：
    - 这里不做复杂的分段与标题解析，而是将整个文件视为一个文本块
    - 对于 Demo 场景已经足够使用，如需更精细的召回，可以自行按业务口径拆分成多条文档
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(base_dir, "definitions.md")
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"未找到知识库文件：{md_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()


def load_business_definitions_markdown() -> str:
    """
    加载 business_definitions.md 的完整内容（新老客、客单价、GMV 等业务口径）。
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(base_dir, "business_definitions.md")
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"未找到知识库文件：{md_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()


def build_chroma_from_markdown() -> None:
    """
    从 definitions.md 与 business_definitions.md 构建 / 覆盖本地 ChromaDB 知识库。
    两份文档均会向量化并存入同一 Chroma 集合，RAG 检索时可同时命中。
    """
    if not config.OPENAI_API_KEY:
        raise ValueError("未在环境变量中找到 OPENAI_API_KEY，请在 .env 中配置。")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    texts: list = []
    metadatas: list = []

    if os.path.exists(os.path.join(base_dir, "definitions.md")):
        text_def = load_definitions_markdown()
        texts.append(text_def)
        metadatas.append({"source": "definitions.md"})

    text_biz = load_business_definitions_markdown()
    texts.append(text_biz)
    metadatas.append({"source": "business_definitions.md"})

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 若需完全重建，先删除已有持久化目录，避免重复追加
    if os.path.isdir(config.CHROMA_PERSIST_DIR):
        shutil.rmtree(config.CHROMA_PERSIST_DIR)

    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )

    vector_store.persist()

    print("Chroma 知识库构建完成（已包含 definitions.md 与 business_definitions.md）。")
    print(f"持久化路径：{config.CHROMA_PERSIST_DIR}")


def main():
    build_chroma_from_markdown()


if __name__ == "__main__":
    main()

