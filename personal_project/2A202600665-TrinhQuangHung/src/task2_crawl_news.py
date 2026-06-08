"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Có thể dùng Crawl4AI hoặc một cách crawl tương đương.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Mặc định script sẽ đọc danh sách URL từ links.csv trong cùng thư mục.
"""

import asyncio
import csv
import html
import json
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
LINKS_FILE = DATA_DIR / "links.csv"


class VisibleTextParser(HTMLParser):
    """Trích xuất text nhìn thấy từ HTML, bỏ qua script/style."""

    def __init__(self):
        super().__init__()
        self._ignore_stack = []
        self._chunks = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._ignore_stack.append(tag)

    def handle_endtag(self, tag):
        if self._ignore_stack and self._ignore_stack[-1] == tag:
            self._ignore_stack.pop()

    def handle_data(self, data):
        if not self._ignore_stack:
            text = data.strip()
            if text:
                self._chunks.append(text)

    def text(self) -> str:
        return " ".join(self._chunks)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _slugify(text: str, fallback: str) -> str:
    normalized = html.unescape(text).lower()
    normalized = re.sub(r"[^a-z0-9\u00c0-\u1ef9]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return normalized or fallback


def _read_urls_from_csv(path: Path) -> list[str]:
    if not path.exists():
        return []

    urls: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            candidate = row[-1].strip()
            if candidate.startswith("http"):
                urls.append(candidate)
    return urls


def load_article_urls() -> list[str]:
    """Tải URL từ links.csv, hoặc dùng ARTICLE_URLS nếu được cấu hình sẵn."""
    if ARTICLE_URLS:
        return ARTICLE_URLS
    return _read_urls_from_csv(LINKS_FILE)


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    # Có thể để trống để script tự đọc links.csv.
]


def _extract_title(html_text: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
    if match:
        title = html.unescape(match.group(1))
        return _normalize_whitespace(title)
    return "Unknown title"


def _extract_visible_text(html_text: str) -> str:
    parser = VisibleTextParser()
    parser.feed(html_text)
    text = parser.text()
    return _normalize_whitespace(html.unescape(text))


def _build_markdown(title: str, url: str, visible_text: str) -> str:
    source = urlparse(url).netloc
    return f"# {title}\n\n- Source: {source}\n- URL: {url}\n\n{visible_text}"


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
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    html_text = response.text
    title = _extract_title(html_text)
    visible_text = _extract_visible_text(html_text)
    content_markdown = _build_markdown(title, url, visible_text)

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "source_domain": urlparse(url).netloc,
        "status_code": response.status_code,
        "content_markdown": content_markdown,
        "raw_html": html_text,
    }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    urls = load_article_urls()
    if len(urls) < 5:
        raise ValueError(f"Cần tối thiểu 5 URL để crawl, hiện có {len(urls)}")

    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as exc:
            print(f"  ! Crawl failed: {exc}")
            article = {
                "url": url,
                "title": f"Failed to crawl: {url}",
                "date_crawled": datetime.now().isoformat(),
                "source_domain": urlparse(url).netloc,
                "status_code": None,
                "content_markdown": (
                    f"# Failed to crawl\n\n- URL: {url}\n- Error: {exc}\n"
                ),
                "raw_html": "",
            }

        for old_file in DATA_DIR.glob(f"article_{i:02d}-*.json"):
            old_file.unlink()

        # Lưu file JSON
        filename = f"article_{i:02d}-{_slugify(article['title'], f'article-{i:02d}')}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not load_article_urls():
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tạo links.csv hoặc điền ARTICLE_URLS trong src/task2_crawl_news.py")
    else:
        asyncio.run(crawl_all())
