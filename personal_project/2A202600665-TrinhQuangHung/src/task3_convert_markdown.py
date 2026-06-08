"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import shutil
from pathlib import Path

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _ensure_output_parent(source_path: Path) -> Path:
    relative_path = source_path.relative_to(LANDING_DIR)
    target_path = OUTPUT_DIR / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return target_path


def _write_markdown(source_path: Path, markdown_text: str):
    target_path = _ensure_output_parent(source_path).with_suffix(".md")
    target_path.write_text(markdown_text, encoding="utf-8")
    print(f"  ✓ Saved: {target_path}")


def convert_legal_docs():
    """Copy existing markdown legal docs or convert PDF/DOCX files when present."""
    legal_dir = LANDING_DIR / "legal"
    md = MarkItDown() if MarkItDown is not None else None

    for filepath in legal_dir.iterdir():
        if not filepath.is_file():
            continue

        suffix = filepath.suffix.lower()
        if suffix == ".md":
            print(f"Copying: {filepath.name}")
            target_path = _ensure_output_parent(filepath).with_suffix(".md")
            shutil.copy2(filepath, target_path)
            print(f"  ✓ Saved: {target_path}")
        elif suffix in {".pdf", ".docx", ".doc"}:
            if md is None:
                raise ImportError(
                    "markitdown is required to convert PDF/DOCX legal files"
                )
            print(f"Converting: {filepath.name}")
            result = md.convert(str(filepath))
            _write_markdown(filepath, result.text_content)


def convert_news_articles():
    """Convert crawled JSON articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"

    for filepath in news_dir.iterdir():
        if not filepath.is_file() or filepath.suffix.lower() != ".json":
            continue

        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        title = data.get("title", "Unknown")
        url = data.get("url", "N/A")
        date_crawled = data.get("date_crawled", "N/A")
        content = data.get("content_markdown", "")

        header = f"# {title}\n\n"
        header += f"**Source:** {url}\n"
        header += f"**Crawled:** {date_crawled}\n\n---\n\n"

        markdown_content = header + content.strip() + "\n"
        _write_markdown(filepath, markdown_content)


def convert_other_markdown_files():
    """Copy any other existing .md files under data/landing/ as-is."""
    for filepath in LANDING_DIR.rglob("*.md"):
        if filepath.parent.name in {"legal", "news"}:
            continue
        if not filepath.is_file():
            continue
        print(f"Copying: {filepath.relative_to(LANDING_DIR)}")
        target_path = _ensure_output_parent(filepath).with_suffix(".md")
        shutil.copy2(filepath, target_path)
        print(f"  ✓ Saved: {target_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    convert_other_markdown_files()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
