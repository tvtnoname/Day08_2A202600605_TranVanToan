"""Cấu hình chung cho Group RAG Chatbot.

File này do Thành viên 1 quản lý.

Nhiệm vụ:
- Khai báo đường dẫn dữ liệu.
- Khai báo cấu hình chunking/retrieval/generation.
- Khai báo các config A/B để Thành viên 3 dùng khi evaluation.
"""

from __future__ import annotations

from pathlib import Path


# group_project/src/config.py
# parents[0] = src
# parents[1] = group_project
# parents[2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUP_PROJECT_DIR = PROJECT_ROOT / "group_project"

DATA_DIR = PROJECT_ROOT / "data"
ROOT_STANDARDIZED_DIR = DATA_DIR / "standardized"

GROUP_INDEX_DIR = GROUP_PROJECT_DIR / "index"
CHUNKS_PATH = GROUP_INDEX_DIR / "chunks.json"


def find_standardized_dirs() -> list[Path]:
    """
    Tìm toàn bộ thư mục data/standardized trong repo.

    Hỗ trợ cả 2 trường hợp:
    1. Dữ liệu nằm ngay tại: data/standardized
    2. Dữ liệu nằm trong personal_project của từng thành viên:
       personal_project/**/data/standardized
    """
    dirs: list[Path] = []

    if ROOT_STANDARDIZED_DIR.exists():
        dirs.append(ROOT_STANDARDIZED_DIR)

    for path in PROJECT_ROOT.rglob("data/standardized"):
        if path.exists() and path.is_dir() and path not in dirs:
            dirs.append(path)

    return dirs


STANDARDIZED_DIRS = find_standardized_dirs()


# Chunking config
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


# Retrieval config
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.15


# Generation config
MAX_CONTEXT_CHUNKS = 5
MAX_SENTENCES_PER_ANSWER = 4


# Các mode để Thành viên 3 chạy A/B testing
RETRIEVAL_CONFIGS = {
    "hybrid_rerank": {
        "use_semantic": True,
        "use_lexical": True,
        "use_rerank": True,
    },
    "hybrid_no_rerank": {
        "use_semantic": True,
        "use_lexical": True,
        "use_rerank": False,
    },
    "dense_only": {
        "use_semantic": True,
        "use_lexical": False,
        "use_rerank": True,
    },
    "lexical_only": {
        "use_semantic": False,
        "use_lexical": True,
        "use_rerank": True,
    },
}

DEFAULT_CONFIG_NAME = "hybrid_rerank"