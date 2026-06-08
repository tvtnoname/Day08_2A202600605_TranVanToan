"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Yêu cầu (theo README):
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Lưu output vào data/landing/news/
    3. Mỗi bài lưu 1 file JSON với metadata: url, title, date_crawled, content_markdown.

Ghi chú lựa chọn công cụ:
    README gợi ý Crawl4AI, nhưng Crawl4AI cần Playwright + Chromium (~400MB, dễ lỗi
    trên môi trường server không có GUI). Ở đây dùng `requests` + `trafilatura`:
        - requests: tải HTML (kèm User-Agent giả lập trình duyệt để tránh bị chặn).
        - trafilatura: bộ trích xuất nội dung bài báo (boilerplate removal) rất tốt,
          xuất thẳng ra Markdown + metadata (title, date, author).
    Output JSON giữ đúng schema yêu cầu nên hoàn toàn tương thích với Task 3.

Cài đặt:
    pip install requests trafilatura
"""

import json
from datetime import datetime
from pathlib import Path

import requests
import trafilatura

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)

# Danh sách URL bài báo về nghệ sĩ Việt liên quan ma tuý (nguồn: VnExpress, VietNamNet, VOV).
ARTICLE_URLS = [
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://vnexpress.net/su-nghiep-long-nhat-truoc-khi-bi-bat-vi-lien-quan-ma-tuy-5076081.html",
    "https://vnexpress.net/su-nghiep-cua-miu-le-truoc-khi-bi-bat-qua-tang-dung-ma-tuy-5072762.html",
    "https://vnexpress.net/ca-si-miu-le-chua-bi-khoi-to-trong-vu-an-dung-ma-tuy-o-dao-cat-ba-5074052.html",
    "https://ngoisao.vnexpress.net/nhung-nghe-si-viet-nga-ngua-vi-ma-tuy-4816068.html",
    "https://vietnamnet.vn/sao-viet-bi-bat-ngoi-tu-mat-danh-tieng-vi-chat-cam-2513746.html",
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
]


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "published_date": str | None,
            "content_markdown": str
        }
    """
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    html = resp.text

    # Trích nội dung chính dạng markdown.
    content_md = trafilatura.extract(
        html, output_format="markdown", include_comments=False, include_tables=False
    ) or ""

    # Trích metadata (title, date, author...).
    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else "Unknown")
    published = (meta.date if meta and meta.date else None)

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "published_date": published,
        "content_markdown": content_md,
    }


def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS, lưu mỗi bài thành 1 file JSON."""
    setup_directory()
    saved = 0
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = crawl_article(url)
        except Exception as e:
            print(f"  ✗ Lỗi: {e}")
            continue

        if len(article["content_markdown"]) < 200:
            print(f"  ⚠ Nội dung quá ngắn ({len(article['content_markdown'])} chars), bỏ qua.")
            continue

        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        saved += 1
        print(f"  ✓ Saved: {filepath.name}  | title: {article['title'][:60]}")

    print(f"\n✓ Done! Đã lưu {saved}/{len(ARTICLE_URLS)} bài vào {DATA_DIR}")
    if saved < 5:
        print("⚠ Chưa đủ 5 bài — hãy thêm URL vào ARTICLE_URLS hoặc kiểm tra mạng.")


if __name__ == "__main__":
    crawl_all()
