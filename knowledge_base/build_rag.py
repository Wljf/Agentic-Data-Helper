"""
构建离线知识库（ChromaDB）的脚本。

改进点：
- 对 Markdown 做分块（RecursiveCharacterTextSplitter），避免整文件单一向量导致召回粗糙。
- 同步写入 kb_manifest.json，供 BM25 与混合检索使用。

使用：在项目根目录执行  python -m knowledge_base.build_rag
"""

from __future__ import annotations

import json
import os
import shutil
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import config

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_community.text_splitter import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "520")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "100")),
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", " ", ""],
        length_function=len,
    )


def _load_file(base_dir: str, name: str) -> str:
    path = os.path.join(base_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到知识库文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _documents_from_markdown(
    text: str, source_name: str, start_index: int
) -> tuple[List[Document], List[dict]]:
    splitter = _splitter()
    docs = splitter.create_documents(
        [text],
        metadatas=[{"source": source_name}],
    )
    manifest_rows: List[dict] = []
    out_docs: List[Document] = []
    for i, d in enumerate(docs):
        idx = start_index + i
        chunk_id = f"{source_name.replace('.md', '')}_{idx}"
        d.metadata["chunk_id"] = chunk_id
        d.metadata["chunk_index"] = i
        out_docs.append(d)
        manifest_rows.append(
            {
                "chunk_id": chunk_id,
                "text": d.page_content,
                "metadata": {
                    "source": source_name,
                    "chunk_index": i,
                },
            }
        )
    return out_docs, manifest_rows


def build_chroma_from_markdown() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    all_documents: List[Document] = []
    manifest: List[dict] = []
    global_idx = 0

    for fname in ("definitions.md", "business_definitions.md"):
        path = os.path.join(base_dir, fname)
        if fname == "definitions.md" and not os.path.exists(path):
            continue
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到：{path}")
        text = _load_file(base_dir, fname)
        docs, rows = _documents_from_markdown(text, fname, global_idx)
        global_idx += len(docs)
        all_documents.extend(docs)
        manifest.extend(rows)

    if not all_documents:
        raise ValueError("没有可用的知识库文档分块，请检查 definitions.md / business_definitions.md")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    persist = os.path.abspath(config.CHROMA_PERSIST_DIR)
    if os.path.isdir(persist):
        shutil.rmtree(persist)

    ids = [d.metadata["chunk_id"] for d in all_documents]
    vector_store = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
        ids=ids,
    )
    vector_store.persist()

    os.makedirs(persist, exist_ok=True)
    manifest_path = os.path.join(persist, "kb_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(
        f"Chroma 知识库构建完成：共 {len(all_documents)} 个 chunk，"
        f"manifest：{manifest_path}"
    )
    print(f"持久化路径：{config.CHROMA_PERSIST_DIR}")


def main() -> None:
    build_chroma_from_markdown()


if __name__ == "__main__":
    main()
