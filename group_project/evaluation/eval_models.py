"""
Custom judge model cho DeepEval — trỏ vào router OpenAI-compatible của nhóm
(không cần OpenAI API key thật).

Đọc cấu hình từ .env (ở project root hoặc group_project/):
    OPENAI_BASE_URL  — endpoint router (vd http://localhost:20128/v1)
    OPENAI_API_KEY   — bearer key của router
    JUDGE_MODEL      — model dùng làm giám khảo (mặc định claude-haiku, chấp nhận temperature=0)

Vì sao haiku? Nhanh + rẻ cho hàng trăm lệnh chấm, và chấp nhận temperature=0
(deterministic) — model opus trên router này chỉ cho temperature=1.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Tìm .env: ưu tiên project root, rồi group_project/
_HERE = Path(__file__).resolve()
for _cand in (_HERE.parents[2] / ".env", _HERE.parents[1] / ".env"):
    if _cand.exists():
        load_dotenv(_cand)
        break

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "cc/claude-haiku-4-5-20251001")
JUDGE_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:20128/v1")
JUDGE_API_KEY = os.getenv("OPENAI_API_KEY", "")


def get_judge_model():
    """Trả về DeepEval LocalModel dùng làm giám khảo (temperature=0, JSON mode)."""
    from deepeval.models import LocalModel

    return LocalModel(
        model=JUDGE_MODEL,
        base_url=JUDGE_BASE_URL,
        api_key=JUDGE_API_KEY,
        temperature=0,
        format="json",  # ép JSON để DeepEval parse structured output ổn định
    )
