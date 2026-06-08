import sys
import importlib.util
from pathlib import Path
from dotenv import load_dotenv

# Thêm đường dẫn của personal_project vào sys.path để có thể import các task cá nhân
PROJECT_ROOT = Path(__file__).parent.parent.parent
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

# Các tham số mặc định của RAG
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.3
DEFAULT_RERANK_METHOD = "cross_encoder"
