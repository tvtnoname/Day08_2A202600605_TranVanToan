"""Cấu hình chung cho Group RAG Chatbot.

Hỗ trợ cả Core RAG của Thành viên 1 và cấu hình nạp động của Thành viên 2.
"""

from __future__ import annotations
import sys
import importlib.util
from pathlib import Path
from dotenv import load_dotenv

# --- Khởi tạo nạp động của Thành viên 2 ---
# Thêm đường dẫn của personal_project vào sys.path để có thể import các task cá nhân
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSONAL_PROJECT_DIR = PROJECT_ROOT / "personal_project" / "2A202600605-TranVanToan"
PERSONAL_SRC_DIR = PERSONAL_PROJECT_DIR / "src"

# Đăng ký personal_project/src dưới tên 'personal_src' để tránh xung đột với group_project/src
if "personal_src" not in sys.modules:
    # Thêm PERSONAL_PROJECT_DIR vào sys.path để các import bên trong hoạt động
    if str(PERSONAL_PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PERSONAL_PROJECT_DIR))
        
    spec = importlib.util.spec_from_file_location("personal_src", str(PERSONAL_SRC_DIR / "__init__.py"))
    personal_src = importlib.util.module_from_spec(spec)
    sys.modules["personal_src"] = personal_src
    spec.loader.exec_module(personal_src)

# Nạp biến môi trường từ file .env của personal_project
ENV_PATH = PERSONAL_PROJECT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Các tham số mặc định của RAG động (Thành viên 2)
DEFAULT_SCORE_THRESHOLD_DYNAMIC = 0.3
DEFAULT_RERANK_METHOD = "cross_encoder"

# --- Cấu hình chung Core RAG (Thành viên 1) ---
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
