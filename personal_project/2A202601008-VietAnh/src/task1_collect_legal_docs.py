"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Yêu cầu:
    - Tải tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ nguồn chính thống.
    - Lưu vào data/landing/legal/ với tên file rõ ràng, không dấu, có năm ban hành.

Nguồn dùng: cổng dữ liệu Chính phủ (datafiles.chinhphu.vn) — link tải PDF trực tiếp,
ổn định, không cần đăng nhập. Script này có thể chạy lại để tải tự động (idempotent:
bỏ qua file đã có).

Cài đặt:
    pip install requests
"""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Danh mục văn bản: filename -> {url, description}
# ---------------------------------------------------------------------------
LEGAL_SOURCES = {
    "luat-phong-chong-ma-tuy-2021.pdf": {
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/73luat.pdf",
        "description": (
            "Luật Phòng, chống ma tuý 2021 (Luật số 73/2021/QH15) — luật khung quy "
            "định phòng ngừa, kiểm soát ma tuý, quản lý người nghiện, cai nghiện."
        ),
    },
    "nghi-dinh-105-2021.pdf": {
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2021/12/105.signed_02.pdf",
        "description": (
            "Nghị định 105/2021/NĐ-CP — quy định chi tiết và hướng dẫn thi hành một số "
            "điều của Luật Phòng, chống ma tuý."
        ),
        # Nguồn dự phòng (mirror) nếu link chính lỗi:
        "mirror": "https://congan.daklak.gov.vn/laws/detail/Nghi-dinh-105-2021-ND-CP-huong-dan-Luat-Phong-chong-ma-tuy-630/?download=1&id=0",
    },
    "bo-luat-hinhsu-so-100-2015-sua-doi-2017-qh13-chuong-XX.pdf": {
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/9/135-vbhn-vpqh.pdf",
        "description": (
            "Bộ luật Hình sự 2015 (sửa đổi, bổ sung 2017) — văn bản hợp nhất 135/VBHN-VPQH; "
            "Chương XX: Các tội phạm về ma tuý (Điều 247–259)."
        ),
    },
}


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str) -> bool:
    """Tải 1 file về DATA_DIR. Trả True nếu thành công (hoặc đã tồn tại)."""
    filepath = DATA_DIR / filename
    if filepath.exists() and filepath.stat().st_size > 1024:
        print(f"  • Bỏ qua (đã có): {filename}")
        return True
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        resp.raise_for_status()
        filepath.write_bytes(resp.content)
        print(f"  ✓ Đã tải: {filename}  ({len(resp.content)//1024} KB)")
        return True
    except Exception as e:
        print(f"  ✗ Lỗi tải {filename}: {e}")
        return False


def collect_all():
    """Tải toàn bộ văn bản trong LEGAL_SOURCES."""
    setup_directory()
    ok = 0
    for filename, info in LEGAL_SOURCES.items():
        print(f"\n→ {info['description']}")
        if download_file(info["url"], filename):
            ok += 1
        elif info.get("mirror"):
            print("  ↻ Thử nguồn mirror...")
            if download_file(info["mirror"], filename):
                ok += 1
    print(f"\n✓ Hoàn tất: {ok}/{len(LEGAL_SOURCES)} văn bản trong {DATA_DIR}")


if __name__ == "__main__":
    collect_all()
