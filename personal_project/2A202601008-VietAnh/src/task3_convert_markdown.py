"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft (https://github.com/microsoft/markitdown).

Hướng dẫn:
    1. Scan file trong data/landing/legal (PDF/DOCX) và data/landing/news (JSON).
    2. Convert sang Markdown.
    3. Lưu vào data/standardized/{legal,news}/ giữ nguyên cấu trúc.

Ghi chú:
    - File .doc (binary cũ) và PDF scan (ảnh, không có text layer) MarkItDown không
      đọc được → cảnh báo và bỏ qua. Với PDF scan cần OCR riêng (vd: unstructured.io,
      tesseract) — xem README. Ở đây ưu tiên các văn bản có text layer.
    - Tránh trùng: nếu một văn bản có cả .pdf và .doc, chỉ convert bản .pdf.

Cài đặt:
    pip install markitdown
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

MIN_CHARS = 200  # ngưỡng tối thiểu để coi là convert thành công


def convert_legal_docs():
    """Convert PDF/DOCX trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    ok, skipped = 0, []

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".docx"):
            continue  # bỏ .doc (không hỗ trợ), .gitkeep, ...

        print(f"Converting: {filepath.name}")
        try:
            text = md.convert(str(filepath)).text_content
        except Exception as e:
            print(f"  ✗ Lỗi convert: {e}")
            skipped.append(filepath.name)
            continue

        if len(text) < MIN_CHARS:
            print(f"  ⚠ Chỉ {len(text)} chars — có thể là PDF scan (cần OCR). Bỏ qua.")
            skipped.append(filepath.name)
            continue

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(text, encoding="utf-8")
        ok += 1
        print(f"  ✓ Saved: {output_path.name}  ({len(text)} chars)")

    if skipped:
        print(f"  → Cần xử lý thêm (OCR): {skipped}")
    return ok


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        print(f"Converting: {filepath.name}")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ✗ Lỗi đọc JSON: {e}")
            continue

        # Header metadata để giữ nguồn cho citation về sau.
        header = (
            f"# {data.get('title', 'Unknown')}\n\n"
            f"**Source:** {data.get('url', 'N/A')}\n"
            f"**Published:** {data.get('published_date', 'N/A')}\n"
            f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
        )
        content = header + data.get("content_markdown", "")

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content, encoding="utf-8")
        ok += 1
        print(f"  ✓ Saved: {output_path.name}  ({len(content)} chars)")
    return ok


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    n_legal = convert_legal_docs()

    print("\n--- News Articles ---")
    n_news = convert_news_articles()

    print(f"\n✓ Done! {n_legal} legal + {n_news} news → {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
