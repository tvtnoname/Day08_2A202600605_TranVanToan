"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    "https://vov.vn/phap-luat/truy-to-ca-si-chi-dan-nguoi-mau-an-tay-co-tien-tu-thien-truc-phuong-post1280703.vov",
    "https://vov.vn/phap-luat/vu-an-ma-tuy-lien-quan-den-ca-si-chi-dan-vi-sao-nguoi-mau-an-tay-bi-truy-to-post1281931.vov",
    "https://dantri.com.vn/phap-luat/dien-vien-hai-huu-tin-khai-su-dung-ma-tuy-do-to-mo-20230428133813927.htm",
    "https://vtv.vn/phap-luat/khoi-to-nu-dien-vien-phim-truyen-hinh-noi-tieng-mot-thoi-vi-toi-mua-ban-ma-tuy-20230423183114918.htm",
    "https://vov.vn/vu-an/xu-phuc-tham-chau-viet-cuong-duoc-giam-an-2-nam-tu-941987.vov"
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler
    from bs4 import BeautifulSoup

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        
        title = ""
        if result.metadata and isinstance(result.metadata, dict):
            title = result.metadata.get("title") or ""
        
        if not title and result.html:
            try:
                soup = BeautifulSoup(result.html, "html.parser")
                title_tag = soup.find("title")
                if title_tag:
                    title = title_tag.get_text().strip()
                if not title:
                    h1_tag = soup.find("h1")
                    if h1_tag:
                        title = h1_tag.get_text().strip()
            except Exception:
                pass
                
        if not title:
            title = "News Article"

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown or "",
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
